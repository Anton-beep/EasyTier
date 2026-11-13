pub mod discovery;

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    future::Future,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    sync::{Arc, Mutex, Weak},
    time::Duration,
};

use async_trait::async_trait;
use dashmap::DashSet;
use percent_encoding::percent_decode_str;
use quanta::Instant;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use tokio::task::JoinSet;
use tokio_util::{sync::CancellationToken, task::AbortOnDropHandle};
use url::Url;

use crate::connectivity::hole_punch::policy::BackOff;
use crate::foundation::time::Instant as DeadlineInstant;
use crate::foundation::token_bucket::TokenBucket;
use crate::peers::conn::peer_conn::PeerConnId;
use crate::peers::context::PeerContextEventSubscriber;
use crate::tunnel::ring::RingTunnelRegistry;
use crate::{
    config::PeerId,
    connectivity::{
        protocol::{ClientProtocolUpgrader, ProtocolTransport, protocol_transport},
        transport::{self, ConnectedByteStream, ConnectedTransport, UdpSessionMode},
    },
    events::{CoreEvent, CoreEventSink},
    host::dns::{DnsQuery, DnsResolver},
    peers::{PeerConnectionOrigin, peer_manager::PeerManagerCore},
    proto::common::TunnelInfo,
    socket::{
        IpVersion, SocketContext,
        tcp::{TcpBindOptions, TcpSocketPurpose, VirtualTcpSocketFactory},
        udp::{UdpBindOptions, VirtualUdpSocketFactory},
    },
    tunnel::{SplitTunnel, Tunnel, TunnelError},
};

const MANUAL_PREFLIGHT_DEFAULT_PORT: u16 = 1000;
const MAX_MANUAL_ENDPOINT_HOPS: usize = 16;

/// Base retry delays; jitter is applied around each step by the retry schedule.
const MANUAL_RETRY_BACKOFF_MS: [u64; 7] = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 60_000];
const MANUAL_RETRY_JITTER_RATIO: f64 = 0.2;
/// A connection must survive this long before its retry counter is reset.
const MANUAL_STABLE_CONN_THRESHOLD: Duration = Duration::from_secs(30);
const MANUAL_MAX_CONCURRENT_DIALS: usize = 4;
const MANUAL_DIALS_PER_SECOND: u64 = 5;
/// One permit refills per this interval; it must match the token bucket fill rate.
const MANUAL_DIAL_PERMIT_REFILL: Duration = Duration::from_millis(1_000 / MANUAL_DIALS_PER_SECOND);

fn manual_default_port(url: &Url) -> u16 {
    crate::connectivity::protocol::protocol_default_port(url.scheme()).unwrap_or(11010)
}

fn is_manual_endpoint_scheme(scheme: &str) -> bool {
    matches!(scheme, "http" | "https" | "txt" | "srv")
}

/// Checks supported connector and discovery schemes without resolving the URL.
pub fn validate_manual_url(url: &Url) -> anyhow::Result<()> {
    if ManualTransport::from_url(url).is_ok() || is_manual_endpoint_scheme(url.scheme()) {
        Ok(())
    } else {
        anyhow::bail!("unsupported core manual connector URL: {url}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ManualTransport {
    Tcp(TcpSocketPurpose),
    Udp(UdpSessionMode),
    ByteStream,
}

impl ManualTransport {
    pub(crate) fn from_url(url: &Url) -> anyhow::Result<Self> {
        match protocol_transport(url.scheme()) {
            Some(ProtocolTransport::Tcp) => Ok(Self::Tcp(TcpSocketPurpose::ManualConnect)),
            Some(ProtocolTransport::FakeTcp) => Ok(Self::Tcp(TcpSocketPurpose::FakeTcp)),
            Some(ProtocolTransport::Udp(mode)) => Ok(Self::Udp(mode)),
            None if matches!(url.scheme(), "ring" | "unix") => Ok(Self::ByteStream),
            None => anyhow::bail!("unsupported core manual connector URL: {url}"),
        }
    }

    fn is_udp(self) -> bool {
        matches!(self, Self::Udp(_))
    }

    fn supports_interface_bind(self) -> bool {
        !matches!(
            self,
            Self::Tcp(TcpSocketPurpose::FakeTcp) | Self::ByteStream
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManualInterfaceAddrs {
    pub interface_ipv4s: Vec<Ipv4Addr>,
    pub interface_ipv6s: Vec<Ipv6Addr>,
    pub public_ipv6: Option<Ipv6Addr>,
}

#[async_trait]
pub trait ManualConnectorHost: VirtualTcpSocketFactory + VirtualUdpSocketFactory {
    fn supports_external_tunnel(&self, _scheme: &str) -> bool {
        false
    }

    async fn connect_external_tunnel(&self, _url: &Url) -> anyhow::Result<Option<Box<dyn Tunnel>>> {
        Ok(None)
    }

    async fn local_addr_for_remote(
        &self,
        remote_addr: SocketAddr,
        context: SocketContext,
    ) -> anyhow::Result<SocketAddr>;

    async fn interface_addrs(&self) -> anyhow::Result<ManualInterfaceAddrs>;

    async fn connect_byte_stream(
        &self,
        url: &Url,
    ) -> anyhow::Result<ConnectedByteStream<<Self as VirtualTcpSocketFactory>::Socket>> {
        anyhow::bail!("host does not support external byte stream: {url}")
    }
}

#[async_trait]
pub trait ExternalTunnelConnector: Send + Sync + 'static {
    fn supports_scheme(&self, scheme: &str) -> bool;

    async fn connect(&self, url: &Url) -> anyhow::Result<Box<dyn Tunnel>>;
}

#[async_trait]
pub(crate) trait ManualEndpointResolver: Send + Sync + 'static {
    async fn resolve_endpoint(&self, url: &Url) -> anyhow::Result<Url>;
}

/// Connects one endpoint without owning peer-manager retry or admission policy.
///
/// This is intended for application protocols such as standalone RPC and the
/// configuration client. The host creates transports, while core resolves the
/// endpoint chain and upgrades the resulting socket or session into a tunnel.
pub struct ManualTunnelConnector<H>
where
    H: ManualConnectorHost,
{
    host: Arc<H>,
    dns: Arc<dyn DnsResolver>,
    endpoint_resolver: Arc<dyn ManualEndpointResolver>,
    protocol: Arc<dyn ClientProtocolUpgrader<<H as VirtualTcpSocketFactory>::Socket>>,
    options: ManualConnectorOptions,
    ring_registry: Option<Arc<RingTunnelRegistry>>,
}

impl<H> ManualTunnelConnector<H>
where
    H: ManualConnectorHost,
{
    pub(crate) fn new(
        host: Arc<H>,
        dns: Arc<dyn DnsResolver>,
        endpoint_resolver: Arc<dyn ManualEndpointResolver>,
        protocol: Arc<dyn ClientProtocolUpgrader<<H as VirtualTcpSocketFactory>::Socket>>,
        options: ManualConnectorOptions,
    ) -> Self {
        Self {
            host,
            dns,
            endpoint_resolver,
            protocol,
            options,
            ring_registry: None,
        }
    }

    pub(crate) fn with_ring_registry(mut self, ring_registry: Arc<RingTunnelRegistry>) -> Self {
        self.ring_registry = Some(ring_registry);
        self
    }

    pub async fn connect(
        &self,
        requested_url: Url,
        ip_version: IpVersion,
    ) -> anyhow::Result<Box<dyn Tunnel>> {
        validate_manual_url(&requested_url)?;
        let endpoint = resolve_manual_endpoint(
            self.endpoint_resolver.as_ref(),
            convert_idn_to_ascii(requested_url.clone())?,
        )
        .await?;
        if endpoint.url.scheme() == "ring" {
            let registry = self
                .ring_registry
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("ring registry is not configured"))?;
            let tunnel = connect_ring_tunnel(registry, &endpoint.url)?;
            return Ok(apply_resolved_endpoint_info(
                tunnel,
                requested_url,
                endpoint.tunnel_prefixes,
            ));
        }

        if let Some(tunnel) = self.host.connect_external_tunnel(&endpoint.url).await? {
            return Ok(apply_resolved_endpoint_info(
                tunnel,
                requested_url,
                endpoint.tunnel_prefixes,
            ));
        }

        if !self.protocol.supports_scheme(endpoint.url.scheme()) {
            anyhow::bail!(
                "unsupported client protocol upgrader: {}",
                endpoint.url.scheme()
            );
        }

        let transport = ManualTransport::from_url(&endpoint.url)?;
        let connected = if transport == ManualTransport::ByteStream {
            ConnectedTransport::ByteStream(self.host.connect_byte_stream(&endpoint.url).await?)
        } else {
            let remote_addr = resolve_url_addrs(
                &endpoint.url,
                manual_default_port(&endpoint.url),
                self.options.socket_context(transport, ip_version),
                self.dns.as_ref(),
            )
            .await?
            .choose(&mut rand::thread_rng())
            .copied()
            .ok_or(TunnelError::NoDnsRecordFound(ip_version))?;
            connect_resolved(
                self.host.clone(),
                transport,
                remote_addr,
                Vec::new(),
                self.options.tcp_bind.clone(),
                self.options.udp_bind.clone(),
            )
            .await?
        };

        let tunnel = self
            .protocol
            .upgrade_client(connected, endpoint.url)
            .await?;
        Ok(apply_resolved_endpoint_info(
            tunnel,
            requested_url,
            endpoint.tunnel_prefixes,
        ))
    }
}

struct ResolvedManualTunnel {
    inner: Box<dyn Tunnel>,
    info: TunnelInfo,
}

impl Tunnel for ResolvedManualTunnel {
    fn split(&self) -> SplitTunnel {
        self.inner.split()
    }

    fn info(&self) -> Option<TunnelInfo> {
        Some(self.info.clone())
    }
}

#[derive(Debug)]
struct ResolvedManualEndpoint {
    url: Url,
    tunnel_prefixes: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManualConnectorStatus {
    Connected,
    Disconnected,
    Connecting,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManualConnectorSnapshot {
    pub url: Url,
    pub status: ManualConnectorStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManualConnectorOptions {
    pub reconnect_interval: Duration,
    pub connect_timeout: Duration,
    pub endpoint_discovery_timeout: Duration,
    pub bind_device: bool,
    pub allow_interface_bind: bool,
    pub tcp_bind: TcpBindOptions,
    pub udp_bind: UdpBindOptions,
}

impl Default for ManualConnectorOptions {
    fn default() -> Self {
        Self {
            reconnect_interval: Duration::from_secs(1),
            connect_timeout: Duration::from_secs(2),
            endpoint_discovery_timeout: Duration::from_secs(20),
            bind_device: false,
            allow_interface_bind: true,
            tcp_bind: TcpBindOptions::default(),
            udp_bind: UdpBindOptions::direct_connect(),
        }
    }
}

impl ManualConnectorOptions {
    fn connect_timeout<TcpSocket: 'static>(
        &self,
        url: &Url,
        protocol: &dyn ClientProtocolUpgrader<TcpSocket>,
    ) -> Duration {
        if is_manual_endpoint_scheme(url.scheme()) {
            self.endpoint_discovery_timeout
        } else {
            protocol
                .connect_timeout(url.scheme())
                .unwrap_or(self.connect_timeout)
        }
    }

    pub(crate) fn socket_context(
        &self,
        transport: ManualTransport,
        ip_version: IpVersion,
    ) -> SocketContext {
        let context = match transport {
            ManualTransport::Tcp(_) => self.tcp_bind.context.clone(),
            ManualTransport::Udp(_) => self.udp_bind.context.clone(),
            ManualTransport::ByteStream => SocketContext::default(),
        };
        context.with_ip_version(ip_version)
    }
}

#[derive(Default)]
struct ManualConnectorState {
    connectors: DashSet<Url>,
    reconnecting: DashSet<Url>,
    removed: DashSet<Url>,
    state_lock: Mutex<()>,
}

/// Jitter factor in `[-1, 1]`, scaled by [`MANUAL_RETRY_JITTER_RATIO`].
type ManualRetryJitter = Box<dyn Fn() -> f64 + Send + Sync>;

fn manual_retry_jitter() -> f64 {
    rand::random::<f64>() * 2.0 - 1.0
}

fn jittered_backoff(base_ms: u64, jitter_factor: f64) -> Duration {
    let scale = 1.0 + MANUAL_RETRY_JITTER_RATIO * jitter_factor.clamp(-1.0, 1.0);
    Duration::from_millis((base_ms as f64 * scale).round() as u64)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ManualAdmittedConn {
    peer_id: PeerId,
    conn_id: PeerConnId,
    admitted_at: DeadlineInstant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ManualConnectorRetry {
    /// Waiting for a dial slot; no timer is pending.
    Idle,
    /// A dial attempt for this URL is in flight.
    Dialing,
    /// The next dial may start at `deadline`.
    Backoff { deadline: DeadlineInstant },
    /// The URL is connected; the connection is watched until it is stable.
    Active(ManualAdmittedConn),
    /// The new connection died while the peer stayed connected elsewhere.
    Refused { peer_id: PeerId },
}

struct ManualDialEntry {
    retry: ManualConnectorRetry,
    backoff: BackOff,
}

/// Per-URL retry schedule for manual connectors.
///
/// The manager loop owns all I/O and feeds this schedule dial outcomes and peer
/// events, so a URL is dialled on a computed deadline or peer-connection change
/// instead of a fixed tick.
struct ManualDialSchedule {
    entries: BTreeMap<Url, ManualDialEntry>,
    jitter: ManualRetryJitter,
}

impl ManualDialSchedule {
    fn new(jitter: ManualRetryJitter) -> Self {
        Self {
            entries: BTreeMap::new(),
            jitter,
        }
    }

    fn with_random_jitter() -> Self {
        Self::new(Box::new(manual_retry_jitter))
    }

    fn new_backoff() -> BackOff {
        BackOff::new(MANUAL_RETRY_BACKOFF_MS.to_vec())
    }

    fn register(&mut self, url: &Url) {
        self.entries
            .entry(url.clone())
            .or_insert_with(|| ManualDialEntry {
                retry: ManualConnectorRetry::Idle,
                backoff: Self::new_backoff(),
            });
    }

    fn unregister(&mut self, url: &Url) {
        self.entries.remove(url);
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    /// Dials interrupted by a manager stop are retried when it starts again.
    fn requeue_interrupted(&mut self) {
        for entry in self.entries.values_mut() {
            if entry.retry == ManualConnectorRetry::Dialing {
                entry.retry = ManualConnectorRetry::Idle;
            }
        }
    }

    fn next_deadline(&self) -> Option<DeadlineInstant> {
        self.entries
            .values()
            .filter_map(|entry| match entry.retry {
                ManualConnectorRetry::Backoff { deadline } => Some(deadline),
                _ => None,
            })
            .min()
    }

    fn next_ready(&self, now: DeadlineInstant) -> Option<Url> {
        self.entries
            .iter()
            .find_map(|(url, entry)| match entry.retry {
                ManualConnectorRetry::Idle => Some(url.clone()),
                ManualConnectorRetry::Backoff { deadline } if deadline <= now => Some(url.clone()),
                _ => None,
            })
    }

    fn begin_dial(&mut self, url: &Url) {
        if let Some(entry) = self.entries.get_mut(url) {
            entry.retry = ManualConnectorRetry::Dialing;
        }
    }

    fn admit(&mut self, url: &Url, admitted: ManualAdmittedConn) {
        if let Some(entry) = self.entries.get_mut(url) {
            entry.retry = ManualConnectorRetry::Active(admitted);
        }
    }

    fn fail_dial(&mut self, url: &Url, now: DeadlineInstant) {
        let jitter = (self.jitter)();
        let Some(entry) = self.entries.get_mut(url) else {
            return;
        };
        let deadline = now + jittered_backoff(entry.backoff.next_backoff(), jitter);
        entry.retry = ManualConnectorRetry::Backoff { deadline };
    }

    /// Applies the peer manager's current view of admitted connections.
    ///
    /// A server refusal is only observable as our admitted connection dying
    /// while the peer stays connected through its other conns, so that URL stays
    /// parked until the peer loses them. A connection that dies while the peer
    /// is gone resumes through the next backoff step, never as an immediate
    /// retry, and only a connection that outlived the stability window resets
    /// the retry sequence.
    fn reconcile(
        &mut self,
        now: DeadlineInstant,
        conn_alive: &dyn Fn(PeerId, PeerConnId) -> bool,
        peer_alive: &dyn Fn(PeerId) -> bool,
    ) {
        let jitter = (self.jitter)();
        let urls = self.entries.keys().cloned().collect::<Vec<_>>();
        for url in &urls {
            let Some(entry) = self.entries.get_mut(url) else {
                continue;
            };
            let ManualConnectorRetry::Active(admitted) = entry.retry else {
                continue;
            };
            if conn_alive(admitted.peer_id, admitted.conn_id) {
                continue;
            }
            entry.retry = if now.saturating_duration_since(admitted.admitted_at)
                >= MANUAL_STABLE_CONN_THRESHOLD
            {
                entry.backoff = Self::new_backoff();
                ManualConnectorRetry::Idle
            } else {
                ManualConnectorRetry::Refused {
                    peer_id: admitted.peer_id,
                }
            };
        }
        for url in &urls {
            let Some(entry) = self.entries.get_mut(url) else {
                continue;
            };
            let ManualConnectorRetry::Refused { peer_id } = entry.retry else {
                continue;
            };
            if peer_alive(peer_id) {
                continue;
            }
            let deadline = now + jittered_backoff(entry.backoff.next_backoff(), jitter);
            entry.retry = ManualConnectorRetry::Backoff { deadline };
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ManualDialPermit {
    Begin,
    AtCapacity,
    RateLimited,
}

/// Fleet-wide guard so a peer event that makes every URL due at once cannot
/// become a synchronized dial storm.
struct ManualDialGuard {
    permits: Arc<TokenBucket>,
    in_flight: usize,
}

impl ManualDialGuard {
    fn new() -> Self {
        Self {
            permits: TokenBucket::new(MANUAL_DIALS_PER_SECOND, MANUAL_DIALS_PER_SECOND),
            in_flight: 0,
        }
    }

    #[cfg(test)]
    fn with_permits(permits: Arc<TokenBucket>) -> Self {
        Self {
            permits,
            in_flight: 0,
        }
    }

    fn try_begin(&mut self) -> ManualDialPermit {
        if self.in_flight >= MANUAL_MAX_CONCURRENT_DIALS {
            return ManualDialPermit::AtCapacity;
        }
        if !self.permits.try_consume(1) {
            return ManualDialPermit::RateLimited;
        }
        self.in_flight += 1;
        ManualDialPermit::Begin
    }

    /// Frees a concurrency slot once its attempt ended or never started.
    fn release(&mut self) {
        self.in_flight = self.in_flight.saturating_sub(1);
    }
}

struct ManualConnectorData<H>
where
    H: ManualConnectorHost,
{
    state: Arc<ManualConnectorState>,
    schedule: Mutex<ManualDialSchedule>,
    dial_guard: Mutex<ManualDialGuard>,
    /// Bumped by connector add/remove; a watch value is stored, so a wake is
    /// never lost while the manager loop is between polls.
    wake: tokio::sync::watch::Sender<u64>,
    peer_manager: Weak<PeerManagerCore>,
    host: Arc<H>,
    dns: Arc<dyn DnsResolver>,
    endpoint_resolver: Arc<dyn ManualEndpointResolver>,
    protocol: Arc<dyn ClientProtocolUpgrader<<H as VirtualTcpSocketFactory>::Socket>>,
    ring_registry: Arc<RingTunnelRegistry>,
    events: Arc<dyn CoreEventSink>,
    options: ManualConnectorOptions,
}

pub(crate) struct ManualConnectorManager<H>
where
    H: ManualConnectorHost,
{
    data: Arc<ManualConnectorData<H>>,
    task: Mutex<Option<ManualConnectorTask>>,
}

struct ManualConnectorTask {
    cancel: CancellationToken,
    handle: AbortOnDropHandle<()>,
}

struct ReconnectReservation {
    state: Arc<ManualConnectorState>,
    url: Url,
}

impl ReconnectReservation {
    fn new(state: Arc<ManualConnectorState>, url: Url) -> Self {
        Self { state, url }
    }
}

impl Drop for ReconnectReservation {
    fn drop(&mut self) {
        let _state_guard = self
            .state
            .state_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if finish_reconnect_attempt(&self.state, &self.url) {
            tracing::warn!(url = %self.url, "manual connector removed after reconnect");
        }
    }
}

impl<H> ManualConnectorManager<H>
where
    H: ManualConnectorHost,
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        peer_manager: Arc<PeerManagerCore>,
        host: Arc<H>,
        dns: Arc<dyn DnsResolver>,
        endpoint_resolver: Arc<dyn ManualEndpointResolver>,
        protocol: Arc<dyn ClientProtocolUpgrader<<H as VirtualTcpSocketFactory>::Socket>>,
        ring_registry: Arc<RingTunnelRegistry>,
        options: ManualConnectorOptions,
        events: Arc<dyn CoreEventSink>,
    ) -> Self {
        let data = Arc::new(ManualConnectorData {
            state: Arc::new(ManualConnectorState::default()),
            schedule: Mutex::new(ManualDialSchedule::with_random_jitter()),
            dial_guard: Mutex::new(ManualDialGuard::new()),
            wake: tokio::sync::watch::channel(0).0,
            peer_manager: Arc::downgrade(&peer_manager),
            host,
            dns,
            endpoint_resolver,
            protocol,
            ring_registry,
            events,
            options,
        });
        Self {
            data,
            task: Mutex::new(None),
        }
    }

    pub fn start(&self) {
        let mut task_slot = self.task.lock().unwrap();
        if task_slot
            .as_ref()
            .is_some_and(|task| !task.handle.is_finished())
        {
            return;
        }
        let cancel = CancellationToken::new();
        task_slot.replace(ManualConnectorTask {
            handle: AbortOnDropHandle::new(tokio::spawn(Self::run(
                self.data.clone(),
                cancel.clone(),
            ))),
            cancel,
        });
    }

    pub async fn stop(&self) {
        let task = self.task.lock().unwrap().take();
        if let Some(task) = task {
            task.cancel.cancel();
            let _ = task.handle.await;
        }
        let _state_guard = self.data.state.state_lock.lock().unwrap();
        restore_interrupted_connectors(&self.data.state);
    }

    pub fn add_connector(&self, url: Url) -> anyhow::Result<()> {
        validate_manual_url(&url)?;
        let state_guard = self.data.state.state_lock.lock().unwrap();
        self.data.state.removed.remove(&url);
        if !self.data.state.reconnecting.contains(&url) {
            self.data.state.connectors.insert(url.clone());
        }
        self.data
            .schedule
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .register(&url);
        drop(state_guard);
        notify_connector_change(&self.data);
        Ok(())
    }

    pub fn remove_connector(&self, url: &Url) -> bool {
        let state_guard = self.data.state.state_lock.lock().unwrap();
        let removed = if self.data.state.connectors.remove(url).is_some() {
            tracing::warn!(%url, "manual connector removed");
            true
        } else if self.data.state.reconnecting.contains(url) {
            self.data.state.removed.insert(url.clone());
            true
        } else {
            false
        };
        if removed {
            self.data
                .schedule
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .unregister(url);
        }
        drop(state_guard);
        if removed {
            notify_connector_change(&self.data);
        }
        removed
    }

    pub fn clear_connectors(&self) {
        let state_guard = self.data.state.state_lock.lock().unwrap();
        self.data.state.connectors.clear();
        for url in self.data.state.reconnecting.iter() {
            self.data.state.removed.insert(url.key().clone());
        }
        self.data
            .schedule
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        drop(state_guard);
        notify_connector_change(&self.data);
    }

    pub fn list_connectors(&self) -> Vec<ManualConnectorSnapshot> {
        let _state_guard = self.data.state.state_lock.lock().unwrap();
        let peer_manager = self.data.peer_manager.upgrade();
        let mut snapshots = self
            .data
            .state
            .connectors
            .iter()
            .map(|entry| {
                let url = entry.key().clone();
                let connected = peer_manager
                    .as_ref()
                    .is_some_and(|peer_manager| client_url_is_alive(peer_manager, &url));
                ManualConnectorSnapshot {
                    url,
                    status: if connected {
                        ManualConnectorStatus::Connected
                    } else {
                        ManualConnectorStatus::Disconnected
                    },
                }
            })
            .collect::<Vec<_>>();
        snapshots.extend(self.data.state.reconnecting.iter().map(|entry| {
            ManualConnectorSnapshot {
                url: entry.key().clone(),
                status: ManualConnectorStatus::Connecting,
            }
        }));
        snapshots
    }

    async fn run(data: Arc<ManualConnectorData<H>>, cancel: CancellationToken) {
        let mut attempts = JoinSet::new();
        let mut attempt_urls = HashMap::new();
        let mut peer_events = data
            .peer_manager
            .upgrade()
            .and_then(|peer_manager| peer_manager.subscribe_peer_events());
        schedule_mutex(&data).requeue_interrupted();
        *dial_guard_mutex(&data) = ManualDialGuard::new();
        let mut wake = data.wake.subscribe();

        loop {
            while let Some(finished) = attempts.try_join_next_with_id() {
                finish_attempt(&data, &mut attempt_urls, finished);
            }
            reconcile_schedule(&data);
            let now = DeadlineInstant::now();
            let permit_retry_at = start_due_attempts(&data, &mut attempts, &mut attempt_urls, now);
            let deadline =
                earliest_deadline(schedule_mutex(&data).next_deadline(), permit_retry_at);

            tokio::select! {
                _ = cancel.cancelled() => break,
                _ = wake.changed() => {}
                _ = recv_peer_event(&mut peer_events) => {}
                _ = sleep_until_deadline(deadline) => {}
                finished = attempts.join_next_with_id(), if !attempts.is_empty() => {
                    if let Some(finished) = finished {
                        finish_attempt(&data, &mut attempt_urls, finished);
                    }
                }
            }
        }

        attempts.abort_all();
        while attempts.join_next().await.is_some() {}

        let _state_guard = data.state.state_lock.lock().unwrap();
        restore_interrupted_connectors(&data.state);
    }
}

fn notify_connector_change<H>(data: &ManualConnectorData<H>)
where
    H: ManualConnectorHost,
{
    data.wake
        .send_modify(|version| *version = version.wrapping_add(1));
}

fn schedule_mutex<H>(data: &ManualConnectorData<H>) -> std::sync::MutexGuard<'_, ManualDialSchedule>
where
    H: ManualConnectorHost,
{
    data.schedule
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn dial_guard_mutex<H>(data: &ManualConnectorData<H>) -> std::sync::MutexGuard<'_, ManualDialGuard>
where
    H: ManualConnectorHost,
{
    data.dial_guard
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn earliest_deadline(
    left: Option<DeadlineInstant>,
    right: Option<DeadlineInstant>,
) -> Option<DeadlineInstant> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (left, right) => left.or(right),
    }
}

/// Starts every dial the schedule and the fleet-wide guard allow, and returns
/// when a rate-limited dial may be retried.
fn start_due_attempts<H>(
    data: &Arc<ManualConnectorData<H>>,
    attempts: &mut JoinSet<(Url, anyhow::Result<ManualAdmittedConn>)>,
    attempt_urls: &mut HashMap<tokio::task::Id, Url>,
    now: DeadlineInstant,
) -> Option<DeadlineInstant>
where
    H: ManualConnectorHost,
{
    let mut permit_retry_at = None;
    loop {
        let Some(url) = schedule_mutex(data).next_ready(now) else {
            break;
        };
        match dial_guard_mutex(data).try_begin() {
            ManualDialPermit::Begin => {}
            ManualDialPermit::AtCapacity => break,
            ManualDialPermit::RateLimited => {
                permit_retry_at = Some(now + MANUAL_DIAL_PERMIT_REFILL);
                break;
            }
        }
        let Some(reservation) = claim_connector_for_dial(&data.state, &url) else {
            // The connector was re-added while its previous attempt still owns it;
            // wait for that attempt to report back instead of dialling twice.
            dial_guard_mutex(data).release();
            schedule_mutex(data).begin_dial(&url);
            continue;
        };
        schedule_mutex(data).begin_dial(&url);
        let task_data = data.clone();
        let task_url = url.clone();
        let handle = attempts.spawn(async move {
            let result = reconnect(task_data, task_url.clone()).await;
            drop(reservation);
            (task_url, result)
        });
        attempt_urls.insert(handle.id(), url);
    }
    permit_retry_at
}

fn finish_attempt<H>(
    data: &ManualConnectorData<H>,
    attempt_urls: &mut HashMap<tokio::task::Id, Url>,
    finished: Result<
        (tokio::task::Id, (Url, anyhow::Result<ManualAdmittedConn>)),
        tokio::task::JoinError,
    >,
) where
    H: ManualConnectorHost,
{
    dial_guard_mutex(data).release();
    match finished {
        Ok((task_id, (url, Ok(admitted)))) => {
            attempt_urls.remove(&task_id);
            schedule_mutex(data).admit(&url, admitted);
        }
        Ok((task_id, (url, Err(error)))) => {
            attempt_urls.remove(&task_id);
            tracing::warn!(%url, ?error, "manual reconnect attempt failed");
            schedule_mutex(data).fail_dial(&url, DeadlineInstant::now());
        }
        Err(error) => {
            let url = attempt_urls.remove(&error.id());
            tracing::error!(?error, ?url, "manual reconnect task failed");
            if let Some(url) = url {
                schedule_mutex(data).fail_dial(&url, DeadlineInstant::now());
            }
        }
    }
}

/// Re-applies the peer manager's current connection view to admitted URLs.
fn reconcile_schedule<H>(data: &ManualConnectorData<H>)
where
    H: ManualConnectorHost,
{
    let Some(peer_manager) = data.peer_manager.upgrade() else {
        return;
    };
    let peer_map = peer_manager.get_peer_map();
    let conn_alive = |peer_id: PeerId, conn_id: PeerConnId| {
        peer_map
            .get_directly_connections_by_peer_id(peer_id)
            .contains(&conn_id)
    };
    let peer_alive = |peer_id: PeerId| {
        peer_map
            .get_peer_by_id(peer_id)
            .is_some_and(|peer| peer.has_directly_connected_conn())
    };
    schedule_mutex(data).reconcile(DeadlineInstant::now(), &conn_alive, &peer_alive);
}

async fn recv_peer_event(peer_events: &mut Option<PeerContextEventSubscriber>) {
    let Some(receiver) = peer_events.as_mut() else {
        return std::future::pending().await;
    };
    match receiver.recv().await {
        Ok(_) | Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
        Err(tokio::sync::broadcast::error::RecvError::Closed) => {
            *peer_events = None;
        }
    }
}

async fn sleep_until_deadline(deadline: Option<DeadlineInstant>) {
    match deadline {
        Some(deadline) => crate::foundation::time::sleep_until(deadline).await,
        None => std::future::pending().await,
    }
}

fn finish_reconnect_attempt(state: &ManualConnectorState, url: &Url) -> bool {
    if state.reconnecting.remove(url).is_none() {
        return false;
    }
    if state.removed.remove(url).is_some() {
        return true;
    }
    state.connectors.insert(url.clone());
    false
}

fn restore_interrupted_connectors(state: &ManualConnectorState) {
    let interrupted = state
        .reconnecting
        .iter()
        .map(|entry| entry.key().clone())
        .collect::<Vec<_>>();
    for url in interrupted {
        finish_reconnect_attempt(state, &url);
    }
}

/// Claims one registered connector for a dial attempt.
///
/// The connector moves from `connectors` to `reconnecting`, and the returned
/// reservation restores it unless the URL was removed while the attempt ran.
fn claim_connector_for_dial(
    state: &Arc<ManualConnectorState>,
    url: &Url,
) -> Option<ReconnectReservation> {
    let _state_guard = state
        .state_lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    state.connectors.remove(url)?;
    let inserted = state.reconnecting.insert(url.clone());
    debug_assert!(inserted);
    Some(ReconnectReservation::new(state.clone(), url.clone()))
}

fn client_url_is_alive(peer_manager: &PeerManagerCore, url: &Url) -> bool {
    peer_manager.get_peer_map().is_client_url_alive(url)
        || peer_manager
            .get_foreign_network_client()
            .is_client_url_alive(url)
}

async fn resolve_manual_endpoint(
    resolver: &dyn ManualEndpointResolver,
    requested_url: Url,
) -> anyhow::Result<ResolvedManualEndpoint> {
    let mut url = requested_url;
    let mut tunnel_prefixes = Vec::new();
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(url.clone()) {
            anyhow::bail!("manual endpoint resolution cycle detected at {url}");
        }
        if ManualTransport::from_url(&url).is_ok() {
            return Ok(ResolvedManualEndpoint {
                url,
                tunnel_prefixes,
            });
        }
        if !is_manual_endpoint_scheme(url.scheme()) {
            anyhow::bail!("unsupported resolved manual connector URL: {url}");
        }
        if tunnel_prefixes.len() >= MAX_MANUAL_ENDPOINT_HOPS {
            anyhow::bail!("manual endpoint resolution exceeded {MAX_MANUAL_ENDPOINT_HOPS} hops");
        }
        tunnel_prefixes.push(url.scheme().to_owned());
        url = convert_idn_to_ascii(resolver.resolve_endpoint(&url).await?)?;
    }
}

fn apply_resolved_endpoint_info(
    tunnel: Box<dyn Tunnel>,
    requested_url: Url,
    tunnel_prefixes: Vec<String>,
) -> Box<dyn Tunnel> {
    if tunnel_prefixes.is_empty() {
        return tunnel;
    }
    let inner_info = tunnel.info().unwrap_or_default();
    let tunnel_type = format!("{}-{}", tunnel_prefixes.join("-"), inner_info.tunnel_type);
    Box::new(ResolvedManualTunnel {
        inner: tunnel,
        info: TunnelInfo {
            local_addr: inner_info.local_addr,
            remote_addr: Some(requested_url.into()),
            resolved_remote_addr: inner_info.resolved_remote_addr.or(inner_info.remote_addr),
            tunnel_type,
        },
    })
}

fn connect_ring_tunnel(
    registry: &RingTunnelRegistry,
    endpoint: &Url,
) -> anyhow::Result<Box<dyn Tunnel>> {
    let remote_id = endpoint
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("ring URL has no peer id: {endpoint}"))?
        .parse()?;
    Ok(registry.connect(remote_id)?.into_tunnel())
}

async fn resolve_reconnect_ip_versions(
    url: &Url,
    connect_timeout: Duration,
    context: SocketContext,
    dns: &dyn DnsResolver,
) -> anyhow::Result<Vec<IpVersion>> {
    if matches!(
        ManualTransport::from_url(url),
        Ok(ManualTransport::ByteStream)
    ) {
        return Ok(vec![IpVersion::Both]);
    }
    if matches!(url.scheme(), "txt" | "srv") {
        return Ok(vec![IpVersion::Both]);
    }

    let addrs = with_timeout_budget(
        "resolve",
        Instant::now(),
        connect_timeout,
        resolve_url_addrs(
            url,
            MANUAL_PREFLIGHT_DEFAULT_PORT,
            context.with_ip_version(IpVersion::Both),
            dns,
        ),
    )
    .await?;
    tracing::info!(?addrs, %url, "manual preflight resolve done");

    let mut ip_versions = Vec::new();
    if addrs.iter().any(SocketAddr::is_ipv4) {
        ip_versions.push(IpVersion::V4);
    }
    if addrs.iter().any(SocketAddr::is_ipv6) {
        ip_versions.push(IpVersion::V6);
    }
    Ok(ip_versions)
}

async fn reconnect<H>(
    data: Arc<ManualConnectorData<H>>,
    url: Url,
) -> anyhow::Result<ManualAdmittedConn>
where
    H: ManualConnectorHost,
{
    validate_manual_url(&url)?;
    let connect_timeout = data.options.connect_timeout(&url, data.protocol.as_ref());
    tracing::info!(%url, "manual reconnect start");
    let normalized_url = match convert_idn_to_ascii(url.clone()) {
        Ok(url) => url,
        Err(error) => {
            emit_connect_error(&data, &url, IpVersion::Both, &error);
            return Err(error);
        }
    };
    let ip_versions = match if data.host.supports_external_tunnel(normalized_url.scheme()) {
        Ok(vec![IpVersion::Both])
    } else {
        resolve_reconnect_ip_versions(
            &normalized_url,
            connect_timeout,
            ManualTransport::from_url(&normalized_url)
                .ok()
                .map(|transport| data.options.socket_context(transport, IpVersion::Both))
                .unwrap_or_default(),
            data.dns.as_ref(),
        )
        .await
    } {
        Ok(ip_versions) => ip_versions,
        Err(error) => {
            emit_connect_error(&data, &url, IpVersion::Both, &error);
            return Err(error);
        }
    };

    let mut last_error = anyhow::anyhow!("cannot get IP from URL");
    for ip_version in ip_versions {
        let started_at = Instant::now();
        match reconnect_with_ip_version(
            data.clone(),
            url.clone(),
            ip_version,
            started_at,
            connect_timeout,
        )
        .await
        {
            Ok(admitted) => return Ok(admitted),
            Err(error) => {
                emit_connect_error(&data, &url, ip_version, &error);
                last_error = error;
            }
        }
    }
    Err(last_error)
}

async fn reconnect_with_ip_version<H>(
    data: Arc<ManualConnectorData<H>>,
    requested_url: Url,
    ip_version: IpVersion,
    started_at: Instant,
    connect_timeout: Duration,
) -> anyhow::Result<ManualAdmittedConn>
where
    H: ManualConnectorHost,
{
    let endpoint = with_timeout_budget(
        "discover",
        started_at,
        connect_timeout,
        resolve_manual_endpoint(
            data.endpoint_resolver.as_ref(),
            convert_idn_to_ascii(requested_url.clone())?,
        ),
    )
    .await?;
    let uses_external_tunnel = data.host.supports_external_tunnel(endpoint.url.scheme());
    if endpoint.url.scheme() != "ring"
        && !uses_external_tunnel
        && !data.protocol.supports_scheme(endpoint.url.scheme())
    {
        anyhow::bail!(
            "unsupported client protocol upgrader: {}",
            endpoint.url.scheme()
        );
    }
    let transport = (endpoint.url.scheme() != "ring" && !uses_external_tunnel)
        .then(|| ManualTransport::from_url(&endpoint.url))
        .transpose()?;
    let resolved = match transport {
        None | Some(ManualTransport::ByteStream) => None,
        Some(transport) => Some(
            with_timeout_budget("resolve", started_at, connect_timeout, async {
                let peer_manager = data.peer_manager.upgrade().ok_or_else(|| {
                    anyhow::anyhow!("peer manager is gone, cannot resolve connector")
                })?;
                let remote_addr = resolve_remote_addr(
                    peer_manager.as_ref(),
                    data.host.as_ref(),
                    data.dns.as_ref(),
                    &endpoint.url,
                    manual_default_port(&endpoint.url),
                    data.options.socket_context(transport, ip_version),
                )
                .await?;
                let bind_addrs = if data.options.bind_device
                    && data.options.allow_interface_bind
                    && transport.supports_interface_bind()
                {
                    collect_bind_addrs(
                        peer_manager.as_ref(),
                        data.host.as_ref(),
                        transport.is_udp(),
                        remote_addr,
                    )
                    .await?
                } else {
                    Vec::new()
                };
                Ok((remote_addr, bind_addrs))
            })
            .await?,
        ),
    };
    data.events.emit(CoreEvent::ManualConnecting {
        url: requested_url.clone(),
    });

    let tunnel = with_timeout_budget("connect", started_at, connect_timeout, async {
        if endpoint.url.scheme() == "ring" {
            return connect_ring_tunnel(&data.ring_registry, &endpoint.url);
        }
        if uses_external_tunnel {
            return data
                .host
                .connect_external_tunnel(&endpoint.url)
                .await?
                .ok_or_else(|| {
                    anyhow::anyhow!("host did not provide external tunnel for {}", endpoint.url)
                });
        }
        let transport = transport.expect("non-Ring endpoint should have a transport");
        let connected = match resolved {
            Some((remote_addr, bind_addrs)) => {
                connect_resolved(
                    data.host.clone(),
                    transport,
                    remote_addr,
                    bind_addrs,
                    data.options.tcp_bind.clone(),
                    data.options.udp_bind.clone(),
                )
                .await?
            }
            None => {
                ConnectedTransport::ByteStream(data.host.connect_byte_stream(&endpoint.url).await?)
            }
        };
        data.protocol.upgrade_client(connected, endpoint.url).await
    })
    .await?;
    let tunnel =
        apply_resolved_endpoint_info(tunnel, requested_url.clone(), endpoint.tunnel_prefixes);
    let peer_manager = data
        .peer_manager
        .upgrade()
        .ok_or_else(|| anyhow::anyhow!("peer manager is gone, cannot reconnect"))?;
    let (peer_id, conn_id) =
        with_timeout_budget("handshake", started_at, connect_timeout, async move {
            peer_manager
                .add_client_tunnel_with_peer_id_hint(tunnel, PeerConnectionOrigin::Manual, None)
                .await
                .map_err(anyhow::Error::from)
        })
        .await?;
    tracing::info!(peer_id, %conn_id, %requested_url, "manual reconnect succeeded");
    Ok(ManualAdmittedConn {
        peer_id,
        conn_id,
        admitted_at: DeadlineInstant::now(),
    })
}

pub(crate) async fn connect_resolved<H>(
    host: Arc<H>,
    transport: ManualTransport,
    remote_addr: SocketAddr,
    bind_addrs: Vec<SocketAddr>,
    tcp_bind: TcpBindOptions,
    udp_bind: UdpBindOptions,
) -> anyhow::Result<ConnectedTransport<<H as VirtualTcpSocketFactory>::Socket>>
where
    H: ManualConnectorHost,
{
    match transport {
        ManualTransport::Tcp(purpose) => {
            transport::connect_tcp(host, remote_addr, bind_addrs, tcp_bind, purpose)
                .await
                .map(ConnectedTransport::Tcp)
        }
        ManualTransport::Udp(mode) => {
            transport::connect_udp(host, remote_addr, bind_addrs, udp_bind, mode)
                .await
                .map(ConnectedTransport::Udp)
        }
        ManualTransport::ByteStream => {
            anyhow::bail!("external byte streams do not use an IP transport address")
        }
    }
}

pub(crate) async fn resolve_remote_addr<H>(
    peer_manager: &PeerManagerCore,
    host: &H,
    dns: &dyn DnsResolver,
    url: &Url,
    default_port: u16,
    context: SocketContext,
) -> anyhow::Result<SocketAddr>
where
    H: ManualConnectorHost,
{
    let ip_version = context.ip_version;
    let addrs = resolve_url_addrs(url, default_port, context.clone(), dns).await?;
    let mut usable = Vec::new();
    let mut rejected_reason = None;
    for addr in addrs {
        let SocketAddr::V6(v6_addr) = addr else {
            usable.push(addr);
            continue;
        };
        if peer_manager.is_easytier_managed_ipv6(v6_addr.ip()).await {
            rejected_reason = Some(format!(
                "{url} resolves to EasyTier-managed IPv6 {}",
                v6_addr.ip()
            ));
            continue;
        }
        match host.local_addr_for_remote(addr, context.clone()).await {
            Ok(SocketAddr::V6(local_addr))
                if peer_manager.is_easytier_managed_ipv6(local_addr.ip()).await =>
            {
                rejected_reason = Some(format!(
                    "{url} would use EasyTier-managed IPv6 {} as local source for {v6_addr}",
                    local_addr.ip()
                ));
            }
            Ok(_) => usable.push(addr),
            Err(error) => return Err(error),
        }
    }

    if usable.is_empty() {
        if let Some(reason) = rejected_reason {
            anyhow::bail!("{reason}, refusing overlay-backed underlay connection");
        }
        return Err(TunnelError::NoDnsRecordFound(ip_version).into());
    }
    usable
        .choose(&mut rand::thread_rng())
        .copied()
        .ok_or_else(|| TunnelError::NoDnsRecordFound(ip_version).into())
}

pub(crate) async fn collect_bind_addrs<H>(
    peer_manager: &PeerManagerCore,
    host: &H,
    is_udp: bool,
    remote_addr: SocketAddr,
) -> anyhow::Result<Vec<SocketAddr>>
where
    H: ManualConnectorHost,
{
    if is_udp && remote_addr.is_ipv6() {
        return Ok(Vec::new());
    }

    let addrs = host.interface_addrs().await?;
    if remote_addr.is_ipv4() {
        return Ok(addrs
            .interface_ipv4s
            .into_iter()
            .map(|addr| SocketAddr::new(IpAddr::V4(addr), 0))
            .collect());
    }

    let mut ipv6s = addrs.interface_ipv6s;
    ipv6s.extend(addrs.public_ipv6);
    let mut ret = Vec::new();
    for addr in ipv6s {
        if !peer_manager.is_easytier_managed_ipv6(&addr).await {
            ret.push(SocketAddr::new(IpAddr::V6(addr), 0));
        }
    }
    Ok(ret)
}

pub(crate) async fn resolve_url_addrs(
    url: &Url,
    default_port: u16,
    context: SocketContext,
    dns: &dyn DnsResolver,
) -> anyhow::Result<Vec<SocketAddr>> {
    let ip_version = context.ip_version;
    let host = url
        .host()
        .ok_or_else(|| anyhow::anyhow!("URL has no host: {url}"))?;
    let port = url.port().unwrap_or(default_port);
    let addrs = match host {
        url::Host::Ipv4(addr) => vec![SocketAddr::new(IpAddr::V4(addr), port)],
        url::Host::Ipv6(addr) => vec![SocketAddr::new(IpAddr::V6(addr), port)],
        url::Host::Domain(host) => dns
            .resolve(DnsQuery::new(host, context))
            .await?
            .into_iter()
            .map(|addr| SocketAddr::new(addr, port))
            .collect(),
    };
    let addrs = addrs
        .into_iter()
        .filter(|addr| match ip_version {
            IpVersion::V4 => addr.is_ipv4(),
            IpVersion::V6 => addr.is_ipv6(),
            IpVersion::Both => true,
        })
        .collect::<Vec<_>>();
    if addrs.is_empty() {
        return Err(TunnelError::NoDnsRecordFound(ip_version).into());
    }
    Ok(addrs)
}

pub(crate) fn convert_idn_to_ascii(mut url: Url) -> anyhow::Result<Url> {
    if url.is_special() {
        return Ok(url);
    }
    if let Some(domain) = url.domain() {
        let domain = percent_decode_str(domain).decode_utf8()?;
        let domain = idna::domain_to_ascii(&domain)?;
        url.set_host(Some(&domain))?;
    }
    Ok(url)
}

fn emit_connect_error<H>(
    data: &ManualConnectorData<H>,
    url: &Url,
    ip_version: IpVersion,
    error: &anyhow::Error,
) where
    H: ManualConnectorHost,
{
    data.events.emit(CoreEvent::ManualConnectError {
        url: url.clone(),
        ip_version,
        error: format!("{error:#?}"),
    });
}

async fn with_timeout_budget<T, F>(
    stage: &'static str,
    started_at: Instant,
    total_timeout: Duration,
    future: F,
) -> anyhow::Result<T>
where
    F: Future<Output = anyhow::Result<T>>,
{
    let remaining = total_timeout
        .checked_sub(started_at.elapsed())
        .filter(|remaining| !remaining.is_zero())
        .ok_or_else(|| anyhow::anyhow!("{stage} timeout after {:?}", started_at.elapsed()))?;
    crate::foundation::time::timeout(remaining, future)
        .await
        .map_err(|_| anyhow::anyhow!("{stage} timeout after {remaining:?}"))?
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use crate::socket::udp::UdpSessionProtocol;

    use super::*;

    fn reserve_pending_connector(
        state: &Arc<ManualConnectorState>,
        url: &Url,
    ) -> ReconnectReservation {
        let _state_guard = state.state_lock.lock().unwrap();
        assert!(state.connectors.remove(url).is_some());
        assert!(state.reconnecting.insert(url.clone()));
        ReconnectReservation::new(state.clone(), url.clone())
    }

    fn assert_connector_is_pending(state: &ManualConnectorState, url: &Url) {
        assert!(state.connectors.contains(url));
        assert!(!state.reconnecting.contains(url));
        assert!(!state.removed.contains(url));
    }

    #[test]
    fn idn_normalization_covers_connector_schemes_and_url_round_trips() {
        let cases = [
            ("example.com", "example.com"),
            ("test.org:8080/path", "test.org:8080/path"),
            ("räksmörgås.nu", "xn--rksmrgs-5wao1o.nu"),
            ("中文.测试", "xn--fiq228c.xn--0zwm56d"),
            ("räksmörgås.nu:8080", "xn--rksmrgs-5wao1o.nu:8080"),
            ("例子.测试/path", "xn--fsqu00a.xn--0zwm56d/path"),
            ("中文.测试:9000/api", "xn--fiq228c.xn--0zwm56d:9000/api"),
            ("räksmörgås.nu:8080/path", "xn--rksmrgs-5wao1o.nu:8080/path"),
            (
                "中文.测试:8000/用户/管理",
                "xn--fiq228c.xn--0zwm56d:8000/%E7%94%A8%E6%88%B7/%E7%AE%A1%E7%90%86",
            ),
            ("[2001:db8::1]:8080", "[2001:db8::1]:8080"),
            ("[2001:db8::1]/path", "[2001:db8::1]/path"),
            (
                "[2001:db8::1]/路径/资源",
                "[2001:db8::1]/%E8%B7%AF%E5%BE%84/%E8%B5%84%E6%BA%90",
            ),
        ];
        let schemes = ["tcp", "udp", "ws", "wss", "wg", "quic", "http", "https"];

        for (host_part, expected_host_part) in cases {
            for scheme in schemes {
                for round_trip in [false, true] {
                    let input = Url::parse(&format!("{scheme}://{host_part}")).unwrap();
                    let input = if round_trip {
                        input.to_string().parse().unwrap()
                    } else {
                        input
                    };
                    let actual = convert_idn_to_ascii(input.clone()).unwrap().to_string();
                    let mut expected = format!("{scheme}://{expected_host_part}");
                    if input.is_special()
                        && actual.ends_with('/')
                        && !expected_host_part.ends_with('/')
                    {
                        expected.push('/');
                    }
                    assert_eq!(actual, expected, "scheme={scheme}, input={host_part}");
                }
            }
        }
    }

    struct StaticDnsResolver {
        ips: Vec<IpAddr>,
        queries: Mutex<Vec<DnsQuery>>,
    }

    struct ChainedEndpointResolver;

    struct CyclingEndpointResolver;

    #[async_trait]
    impl ManualEndpointResolver for ChainedEndpointResolver {
        async fn resolve_endpoint(&self, url: &Url) -> anyhow::Result<Url> {
            match url.scheme() {
                "http" => Ok("txt://discovery.example".parse().unwrap()),
                "txt" => Ok("tcp://peer.example:12000".parse().unwrap()),
                scheme => anyhow::bail!("unexpected endpoint scheme: {scheme}"),
            }
        }
    }

    #[async_trait]
    impl ManualEndpointResolver for CyclingEndpointResolver {
        async fn resolve_endpoint(&self, url: &Url) -> anyhow::Result<Url> {
            Ok(url.clone())
        }
    }

    #[async_trait]
    impl DnsResolver for StaticDnsResolver {
        async fn resolve(&self, query: DnsQuery) -> anyhow::Result<Vec<IpAddr>> {
            self.queries.lock().unwrap().push(query);
            Ok(self.ips.clone())
        }
    }

    #[test]
    fn manual_transport_maps_ip_protocols_to_their_socket_boundary() {
        let cases = [
            (
                "tcp://127.0.0.1:1",
                ManualTransport::Tcp(TcpSocketPurpose::ManualConnect),
            ),
            (
                "ws://127.0.0.1:1",
                ManualTransport::Tcp(TcpSocketPurpose::ManualConnect),
            ),
            (
                "wss://127.0.0.1:1",
                ManualTransport::Tcp(TcpSocketPurpose::ManualConnect),
            ),
            (
                "faketcp://127.0.0.1:1",
                ManualTransport::Tcp(TcpSocketPurpose::FakeTcp),
            ),
            (
                "udp://127.0.0.1:1",
                ManualTransport::Udp(UdpSessionMode::EasyTierMux),
            ),
            (
                "wg://127.0.0.1:1",
                ManualTransport::Udp(UdpSessionMode::Classified(UdpSessionProtocol::WireGuard)),
            ),
            (
                "quic://127.0.0.1:1",
                ManualTransport::Udp(UdpSessionMode::Classified(UdpSessionProtocol::Quic)),
            ),
            ("ring://local", ManualTransport::ByteStream),
            ("unix:///tmp/easytier.sock", ManualTransport::ByteStream),
        ];

        for (url, expected) in cases {
            assert_eq!(
                ManualTransport::from_url(&url.parse().unwrap()).unwrap(),
                expected
            );
        }
        assert!(ManualTransport::from_url(&"http://127.0.0.1:1".parse().unwrap()).is_err());
        assert!(validate_manual_url(&"http://127.0.0.1:1".parse().unwrap()).is_ok());
    }

    #[tokio::test]
    async fn manual_endpoint_resolution_preserves_the_discovery_chain() {
        let endpoint = resolve_manual_endpoint(
            &ChainedEndpointResolver,
            "http://discovery.example".parse().unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(endpoint.url.as_str(), "tcp://peer.example:12000");
        assert_eq!(endpoint.tunnel_prefixes, ["http", "txt"]);
    }

    #[tokio::test]
    async fn manual_endpoint_resolution_rejects_cycles() {
        let error = resolve_manual_endpoint(
            &CyclingEndpointResolver,
            "http://discovery.example".parse().unwrap(),
        )
        .await
        .unwrap_err();

        assert!(error.to_string().contains("cycle detected"));
    }

    #[tokio::test]
    async fn txt_discovery_does_not_require_address_records() {
        let resolver = StaticDnsResolver {
            ips: Vec::new(),
            queries: Mutex::new(Vec::new()),
        };
        let versions = resolve_reconnect_ip_versions(
            &"txt://discovery.example".parse().unwrap(),
            Duration::from_secs(1),
            SocketContext::default(),
            &resolver,
        )
        .await
        .unwrap();

        assert_eq!(versions, [IpVersion::Both]);
        assert!(resolver.queries.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn external_byte_stream_does_not_require_address_records() {
        let resolver = StaticDnsResolver {
            ips: Vec::new(),
            queries: Mutex::new(Vec::new()),
        };
        let versions = resolve_reconnect_ip_versions(
            &"ring://local".parse().unwrap(),
            Duration::from_secs(1),
            SocketContext::default(),
            &resolver,
        )
        .await
        .unwrap();

        assert_eq!(versions, [IpVersion::Both]);
        assert!(resolver.queries.lock().unwrap().is_empty());
    }

    #[test]
    fn external_protocol_and_discovery_timeouts_are_explicit() {
        struct Protocol;

        #[async_trait]
        impl ClientProtocolUpgrader<()> for Protocol {
            fn supports_scheme(&self, scheme: &str) -> bool {
                scheme == "external"
            }

            fn connect_timeout(&self, scheme: &str) -> Option<Duration> {
                (scheme == "external").then_some(Duration::from_secs(20))
            }

            async fn upgrade_client(
                &self,
                _connected: ConnectedTransport<()>,
                _requested_url: Url,
            ) -> anyhow::Result<Box<dyn Tunnel>> {
                unreachable!()
            }
        }

        let options = ManualConnectorOptions::default();
        assert_eq!(
            options.connect_timeout(&"external://127.0.0.1".parse().unwrap(), &Protocol),
            Duration::from_secs(20)
        );
        assert_eq!(
            options.connect_timeout(&"txt://example.com".parse().unwrap(), &Protocol),
            Duration::from_secs(20)
        );
        assert_eq!(
            options.connect_timeout(&"tcp://127.0.0.1".parse().unwrap(), &Protocol),
            Duration::from_secs(2)
        );
    }

    #[tokio::test]
    async fn resolver_receives_instance_socket_context_and_filters_family() {
        let resolver = StaticDnsResolver {
            ips: vec![IpAddr::from([127, 0, 0, 1]), Ipv6Addr::LOCALHOST.into()],
            queries: Mutex::new(Vec::new()),
        };
        let url: Url = "udp://example.com:12000".parse().unwrap();

        let addrs = resolve_url_addrs(
            &url,
            crate::connectivity::protocol::protocol_default_port("tcp").unwrap(),
            SocketContext::default()
                .with_ip_version(IpVersion::V6)
                .with_socket_mark(Some(7)),
            &resolver,
        )
        .await
        .unwrap();

        assert_eq!(addrs, vec!["[::1]:12000".parse().unwrap()]);
        assert_eq!(
            resolver.queries.lock().unwrap().as_slice(),
            &[DnsQuery::new(
                "example.com",
                SocketContext {
                    ip_version: IpVersion::V6,
                    socket_mark: Some(7),
                    netns: None,
                }
            )]
        );
    }

    #[tokio::test]
    async fn timeout_budget_reports_the_active_stage() {
        let error = with_timeout_budget(
            "connect",
            Instant::now(),
            Duration::from_millis(1),
            std::future::pending::<anyhow::Result<()>>(),
        )
        .await
        .unwrap_err();

        assert!(error.to_string().contains("connect timeout after"));
    }

    #[test]
    fn interrupted_reconnects_return_to_the_pending_set_unless_removed() {
        let state = ManualConnectorState::default();
        let retained: Url = "tcp://127.0.0.1:11010".parse().unwrap();
        let deleted: Url = "udp://127.0.0.1:11010".parse().unwrap();
        state.reconnecting.insert(retained.clone());
        state.reconnecting.insert(deleted.clone());
        state.removed.insert(deleted.clone());

        restore_interrupted_connectors(&state);
        restore_interrupted_connectors(&state);

        assert!(state.connectors.contains(&retained));
        assert!(!state.connectors.contains(&deleted));
        assert!(state.reconnecting.is_empty());
        assert!(state.removed.is_empty());
    }

    #[test]
    fn reconnect_reservation_restores_connector_on_normal_drop() {
        let state = Arc::new(ManualConnectorState::default());
        let url: Url = "tcp://127.0.0.1:11010".parse().unwrap();
        state.connectors.insert(url.clone());

        let reservation = reserve_pending_connector(&state, &url);
        assert!(!state.connectors.contains(&url));
        assert!(state.reconnecting.contains(&url));

        drop(reservation);

        assert_connector_is_pending(&state, &url);
    }

    #[tokio::test]
    async fn reconnect_reservation_restores_connector_after_attempt_panic() {
        let state = Arc::new(ManualConnectorState::default());
        let url: Url = "tcp://127.0.0.1:11010".parse().unwrap();
        state.connectors.insert(url.clone());
        let reservation = reserve_pending_connector(&state, &url);

        let task = tokio::spawn(async move {
            let _reservation = reservation;
            panic!("reconnect attempt panic");
        });
        assert!(task.await.unwrap_err().is_panic());
        assert_connector_is_pending(&state, &url);

        let retry_reservation = reserve_pending_connector(&state, &url);
        drop(retry_reservation);
        assert_connector_is_pending(&state, &url);
    }

    #[tokio::test]
    async fn reconnect_reservation_restores_connector_after_attempt_abort() {
        let state = Arc::new(ManualConnectorState::default());
        let url: Url = "tcp://127.0.0.1:11010".parse().unwrap();
        state.connectors.insert(url.clone());
        let reservation = reserve_pending_connector(&state, &url);
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();

        let task = tokio::spawn(async move {
            let _reservation = reservation;
            let _ = started_tx.send(());
            std::future::pending::<()>().await;
        });
        started_rx.await.unwrap();
        task.abort();

        assert!(task.await.unwrap_err().is_cancelled());
        assert_connector_is_pending(&state, &url);

        let retry_reservation = reserve_pending_connector(&state, &url);
        drop(retry_reservation);
        assert_connector_is_pending(&state, &url);
    }

    #[test]
    fn reconnect_reservation_does_not_restore_removed_connector() {
        let state = Arc::new(ManualConnectorState::default());
        let url: Url = "tcp://127.0.0.1:11010".parse().unwrap();
        state.connectors.insert(url.clone());
        let reservation = reserve_pending_connector(&state, &url);
        state.removed.insert(url.clone());

        drop(reservation);

        assert!(!state.connectors.contains(&url));
        assert!(!state.reconnecting.contains(&url));
        assert!(!state.removed.contains(&url));
    }

    fn retry_url(port: u16) -> Url {
        format!("tcp://127.0.0.1:{port}").parse().unwrap()
    }

    fn admitted_conn(peer_id: PeerId, admitted_at: DeadlineInstant) -> ManualAdmittedConn {
        ManualAdmittedConn {
            peer_id,
            conn_id: uuid::Uuid::from_u128(u128::from(peer_id) + 1),
            admitted_at,
        }
    }

    fn retry_state(schedule: &ManualDialSchedule, url: &Url) -> ManualConnectorRetry {
        schedule.entries.get(url).expect("registered URL").retry
    }

    #[tokio::test(start_paused = true)]
    async fn retry_backoff_doubles_from_one_second_up_to_a_capped_minute() {
        let url = retry_url(11010);
        let mut schedule = ManualDialSchedule::new(Box::new(|| 0.0));
        schedule.register(&url);
        let now = DeadlineInstant::now();

        let mut steps = Vec::new();
        for _ in 0..8 {
            schedule.begin_dial(&url);
            schedule.fail_dial(&url, now);
            let ManualConnectorRetry::Backoff { deadline } = retry_state(&schedule, &url) else {
                panic!("failed dial must schedule a backoff");
            };
            steps.push(deadline - now);
        }

        assert_eq!(
            steps,
            [1, 2, 4, 8, 16, 32, 60, 60]
                .map(Duration::from_secs)
                .to_vec()
        );
    }

    #[tokio::test(start_paused = true)]
    async fn retry_backoff_jitter_stays_within_twenty_percent_of_each_step() {
        for (jitter, expected) in [(0.0, 1000), (1.0, 1200), (-1.0, 800), (0.5, 1100)] {
            let url = retry_url(11010);
            let mut schedule = ManualDialSchedule::new(Box::new(move || jitter));
            schedule.register(&url);
            let now = DeadlineInstant::now();
            schedule.begin_dial(&url);
            schedule.fail_dial(&url, now);
            let ManualConnectorRetry::Backoff { deadline } = retry_state(&schedule, &url) else {
                panic!("failed dial must schedule a backoff");
            };
            assert_eq!(deadline - now, Duration::from_millis(expected));
        }

        for _ in 0..512 {
            let factor = manual_retry_jitter();
            assert!((-1.0..=1.0).contains(&factor), "jitter factor {factor}");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn retry_attempts_reset_only_after_a_connection_stays_alive_past_thirty_seconds() {
        let url = retry_url(11010);
        let mut schedule = ManualDialSchedule::new(Box::new(|| 0.0));
        schedule.register(&url);
        let now = DeadlineInstant::now();
        let conn_gone = |_: PeerId, _: PeerConnId| false;
        let peer_gone = |_: PeerId| false;

        for _ in 0..3 {
            schedule.begin_dial(&url);
            schedule.fail_dial(&url, now);
        }

        // A connection that dies before the stability window keeps escalating.
        schedule.begin_dial(&url);
        schedule.admit(&url, admitted_conn(7, now));
        schedule.reconcile(now + Duration::from_secs(20), &conn_gone, &peer_gone);
        let ManualConnectorRetry::Backoff { deadline } = retry_state(&schedule, &url) else {
            panic!("a refused connector must resume through backoff");
        };
        assert_eq!(
            deadline - (now + Duration::from_secs(20)),
            Duration::from_secs(8)
        );

        // Admission alone never resets: the same connection dying at 20s was still
        // one backoff step further along than the third failure.
        let later = now + Duration::from_secs(40);
        schedule.begin_dial(&url);
        schedule.admit(&url, admitted_conn(7, later));
        schedule.reconcile(later + Duration::from_secs(31), &conn_gone, &peer_gone);
        assert_eq!(retry_state(&schedule, &url), ManualConnectorRetry::Idle);
        assert_eq!(
            schedule.next_ready(later + Duration::from_secs(31)),
            Some(url.clone())
        );

        schedule.begin_dial(&url);
        schedule.fail_dial(&url, later + Duration::from_secs(31));
        let ManualConnectorRetry::Backoff { deadline } = retry_state(&schedule, &url) else {
            panic!("failed dial must schedule a backoff");
        };
        assert_eq!(
            deadline - (later + Duration::from_secs(31)),
            Duration::from_secs(1)
        );
    }

    #[tokio::test(start_paused = true)]
    async fn refused_connector_waits_for_the_peer_connection_to_disappear_before_dialling_once() {
        let url = retry_url(11010);
        let mut schedule = ManualDialSchedule::new(Box::new(|| 0.0));
        schedule.register(&url);
        let now = DeadlineInstant::now();
        let conn_gone = |_: PeerId, _: PeerConnId| false;
        let peer_gone = |_: PeerId| false;
        let peer_connected = |_: PeerId| true;

        // The server refused the dial: the admitted conn dies while the peer stays
        // connected through its other conns.
        schedule.begin_dial(&url);
        schedule.admit(&url, admitted_conn(7, now));
        schedule.reconcile(now, &conn_gone, &peer_connected);
        assert_eq!(
            retry_state(&schedule, &url),
            ManualConnectorRetry::Refused { peer_id: 7 }
        );
        assert_eq!(schedule.next_ready(now), None);
        assert_eq!(schedule.next_deadline(), None);

        schedule.reconcile(now + Duration::from_secs(5), &conn_gone, &peer_connected);
        assert_eq!(schedule.next_ready(now + Duration::from_secs(5)), None);

        // The peer's remaining connection is finally removed: one dial, after backoff.
        schedule.reconcile(now + Duration::from_secs(5), &conn_gone, &peer_gone);
        let deadline = schedule.next_deadline().expect("resumed through backoff");
        assert_eq!(
            deadline - (now + Duration::from_secs(5)),
            Duration::from_secs(1)
        );
        assert_eq!(schedule.next_ready(now + Duration::from_secs(5)), None);
        assert_eq!(schedule.next_ready(deadline), Some(url.clone()));
        schedule.begin_dial(&url);
        assert_eq!(schedule.next_ready(deadline), None);
    }

    #[tokio::test(start_paused = true)]
    async fn connector_waits_for_its_backoff_deadline_instead_of_a_fixed_tick() {
        let url = retry_url(11010);
        let mut schedule = ManualDialSchedule::new(Box::new(|| 0.0));
        schedule.register(&url);
        let now = DeadlineInstant::now();

        schedule.begin_dial(&url);
        schedule.fail_dial(&url, now);

        let deadline = schedule.next_deadline().expect("backoff deadline");
        assert_eq!(deadline, now + Duration::from_secs(1));
        assert_eq!(
            schedule.next_ready(deadline - Duration::from_millis(1)),
            None
        );
        assert_eq!(schedule.next_ready(deadline), Some(url));
    }

    #[tokio::test]
    async fn due_connector_burst_never_exceeds_four_concurrent_dials() {
        let now = DeadlineInstant::now();
        let mut schedule = ManualDialSchedule::new(Box::new(|| 0.0));
        for offset in 0..12 {
            schedule.register(&retry_url(11010 + offset));
        }

        let mut guard = ManualDialGuard::with_permits(TokenBucket::new(1_000, 1_000));
        let mut started = Vec::new();
        let start_due = |schedule: &mut ManualDialSchedule,
                         guard: &mut ManualDialGuard,
                         started: &mut Vec<Url>| {
            while let Some(url) = schedule.next_ready(now) {
                match guard.try_begin() {
                    ManualDialPermit::Begin => {
                        schedule.begin_dial(&url);
                        started.push(url);
                    }
                    ManualDialPermit::AtCapacity | ManualDialPermit::RateLimited => break,
                }
            }
        };

        start_due(&mut schedule, &mut guard, &mut started);
        assert_eq!(started.len(), 4);
        assert_eq!(guard.try_begin(), ManualDialPermit::AtCapacity);

        // A finished dial frees one slot for the next due connector.
        guard.release();
        start_due(&mut schedule, &mut guard, &mut started);
        assert_eq!(started.len(), 5);
    }

    #[test]
    fn dial_guard_limits_a_burst_to_five_attempts_per_second() {
        let mut guard = ManualDialGuard::new();
        for _ in 0..5 {
            assert_eq!(guard.try_begin(), ManualDialPermit::Begin);
            guard.release();
        }
        assert_eq!(guard.try_begin(), ManualDialPermit::RateLimited);

        // One permit refills every 200ms at five dials per second.
        std::thread::sleep(Duration::from_millis(250));

        assert_eq!(guard.try_begin(), ManualDialPermit::Begin);
    }

    #[test]
    fn interrupted_dialling_returns_to_idle_when_the_manager_restarts() {
        let url = retry_url(11010);
        let mut schedule = ManualDialSchedule::new(Box::new(|| 0.0));
        schedule.register(&url);
        schedule.begin_dial(&url);

        schedule.requeue_interrupted();

        let now = DeadlineInstant::now();
        assert_eq!(schedule.next_ready(now), Some(url));
    }

    #[test]
    fn dial_claim_moves_a_connector_between_the_pending_and_reconnecting_sets_once() {
        let state = Arc::new(ManualConnectorState::default());
        let url = retry_url(11010);
        state.connectors.insert(url.clone());

        let claim = claim_connector_for_dial(&state, &url).expect("pending connector");
        assert!(!state.connectors.contains(&url));
        assert!(state.reconnecting.contains(&url));
        assert!(claim_connector_for_dial(&state, &url).is_none());

        drop(claim);

        assert_connector_is_pending(&state, &url);
    }
}
