//! Opt-in scheduler test instrumentation.
//!
//! This module does not assemble batches, select members, release admission,
//! alter job phases, or implement terminal semantics.
//!
//! It supplies:
//! - a monotonic manual clock;
//! - acknowledged insertion events;
//! - owner checkpoints outside serving locks;
//! - observations of the production assembly and launch path.
//!
//! Event retention is bounded. Overflow is a test failure, never permission
//! to silently omit evidence.
//!
//! One handle may be attached to one Service only.

use super::{lock, Code, Job};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Condvar, Mutex,
    },
    time::{Duration, Instant},
};
use tokio::sync::watch;

const EVENT_LIMIT: usize = 4096;
const MAX_ADVANCE: Duration = Duration::from_secs(86_400);

#[derive(Debug)]
struct State {
    now: Instant,
    events: Vec<Value>,
    overflow: bool,
    pending_checkpoint: Option<usize>,
    released: BTreeSet<usize>,
    cleanup_open: bool,
}

#[derive(Debug)]
struct Inner {
    origin: Instant,
    claimed: AtomicBool,
    state: Mutex<State>,
    progress: Condvar,
    clock: watch::Sender<Instant>,
    changed: watch::Sender<u64>,
}

#[derive(Clone, Debug)]
pub struct SchedulerControl {
    inner: Arc<Inner>,
}

impl Default for SchedulerControl {
    fn default() -> Self {
        Self::new()
    }
}

impl SchedulerControl {
    /// Manual time starts at offset zero and never follows wall time.
    pub fn new() -> Self {
        let origin = Instant::now();
        let (clock, _) = watch::channel(origin);
        let (changed, _) = watch::channel(0);
        Self {
            inner: Arc::new(Inner {
                origin,
                claimed: AtomicBool::new(false),
                state: Mutex::new(State {
                    now: origin,
                    events: Vec::new(),
                    overflow: false,
                    pending_checkpoint: None,
                    released: BTreeSet::new(),
                    cleanup_open: false,
                }),
                progress: Condvar::new(),
                clock,
                changed,
            }),
        }
    }

    pub(super) fn claim(&self) -> Result<(), Code> {
        self.inner
            .claimed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| Code::InvalidRequest)
    }

    pub(super) fn now(&self) -> Instant {
        lock(&self.inner.state).now
    }

    pub fn now_us(&self) -> u64 {
        self.offset(self.now())
    }

    fn offset(&self, time: Instant) -> u64 {
        u64::try_from(
            time.saturating_duration_since(self.inner.origin)
                .as_micros(),
        )
        .expect("bounded scheduler test clock")
    }

    /// Advance to an absolute microsecond offset. Backwards movement fails.
    pub fn advance_to_us(&self, offset: u64) -> Result<(), Code> {
        let duration = Duration::from_micros(offset);
        if duration > MAX_ADVANCE {
            return Err(Code::InvalidRequest);
        }
        let next = self
            .inner
            .origin
            .checked_add(duration)
            .ok_or(Code::InvalidRequest)?;
        let mut state = lock(&self.inner.state);
        if next < state.now {
            return Err(Code::InvalidRequest);
        }
        state.now = next;
        self.inner.clock.send_replace(next);
        Ok(())
    }

    pub fn advance_by_us(&self, increment: u64) -> Result<(), Code> {
        let next = self
            .now_us()
            .checked_add(increment)
            .ok_or(Code::InvalidRequest)?;
        self.advance_to_us(next)
    }

    pub(super) async fn sleep_until(&self, deadline: Instant) {
        let mut receiver = self.inner.clock.subscribe();
        loop {
            if *receiver.borrow_and_update() >= deadline {
                return;
            }
            if receiver.changed().await.is_err() {
                return;
            }
        }
    }

    fn push_locked(&self, state: &mut State, mut event: Value) -> Option<usize> {
        if state.events.len() == EVENT_LIMIT {
            state.overflow = true;
            self.inner.changed.send_modify(|n| *n += 1);
            return None;
        }
        let sequence = state.events.len();
        event["seq"] = json!(sequence);
        event["now_us"] = json!(self.offset(state.now));
        state.events.push(event);
        self.inner.changed.send_modify(|n| *n += 1);
        Some(sequence)
    }

    pub(super) fn emit(&self, event: Value) {
        let mut state = lock(&self.inner.state);
        self.push_locked(&mut state, event);
    }

    pub(super) fn insertion(&self, job: &Job) -> Value {
        json!({
            "kind": "inserted",
            "request_id": job.request_id,
            "case_id": job.case_id,
            "B": job.batches.len(),
            "S": job.batches.iter().map(Vec::len).max().unwrap_or(0),
            "inserted_us": self.offset(job.inserted),
            "deadline_us": self.offset(job.deadline)
        })
    }

    fn members(&self, jobs: &[Job], boundary: Option<Instant>) -> Value {
        json!({
            "request_ids": jobs.iter().map(|job| job.request_id).collect::<Vec<_>>(),
            "B": jobs.iter().map(|job| job.batches.len()).sum::<usize>(),
            "S": jobs.iter()
                .flat_map(|job| job.batches.iter().map(Vec::len))
                .max().unwrap_or(0),
            "boundary_us": boundary.map(|time| self.offset(time))
        })
    }

    pub(super) fn assembly(&self, jobs: &[Job], boundary: Instant) {
        let mut event = self.members(jobs, Some(boundary));
        event["kind"] = json!("assembly");
        self.emit(event);
    }

    pub(super) fn waiting(&self, boundary: Instant) {
        self.emit(json!({
            "kind": "assembly_receive",
            "boundary_us": self.offset(boundary)
        }));
    }

    pub(super) fn launch(&self, jobs: &[Job]) {
        let mut event = self.members(jobs, None);
        event["kind"] = json!("launch");
        self.emit(event);
    }

    /// Called only by the owner, outside core/job/admission locks.
    pub(super) fn checkpoint(&self, stage: &'static str, jobs: &[Job], boundary: Option<Instant>) {
        let mut event = self.members(jobs, boundary);
        event["kind"] = json!("checkpoint");
        event["stage"] = json!(stage);

        let mut state = lock(&self.inner.state);
        let Some(sequence) = self.push_locked(&mut state, event) else {
            // Overflow is observable and makes acceptance fail. Do not strand
            // an owner merely because its diagnostic buffer is exhausted.
            return;
        };
        state.pending_checkpoint = Some(sequence);
        while !state.cleanup_open && !state.released.remove(&sequence) {
            state = self
                .inner
                .progress
                .wait(state)
                .unwrap_or_else(|error| error.into_inner());
        }
        state.pending_checkpoint = None;
    }

    /// Release exactly the currently observed owner checkpoint.
    pub fn release(&self, sequence: usize) -> Result<(), Code> {
        let mut state = lock(&self.inner.state);
        if state.cleanup_open || state.pending_checkpoint != Some(sequence) {
            return Err(Code::InvalidRequest);
        }
        state.released.insert(sequence);
        self.inner.progress.notify_all();
        Ok(())
    }

    /// Irreversibly open progress gates for cleanup.
    ///
    /// Does not reset time, change lifecycle, release permits, clear quarantine,
    /// stop a worker, or constitute a join.
    pub fn open_for_cleanup(&self) {
        let mut state = lock(&self.inner.state);
        state.cleanup_open = true;
        self.inner.progress.notify_all();
    }

    pub fn snapshot(&self) -> Result<Vec<Value>, Code> {
        let state = lock(&self.inner.state);
        if state.overflow {
            return Err(Code::InferenceFailed);
        }
        Ok(state.events.clone())
    }

    /// Wait for a retained event by index. Caller supplies a real watchdog.
    pub async fn event(&self, index: usize) -> Result<Value, Code> {
        let mut changed = self.inner.changed.subscribe();
        loop {
            changed.borrow_and_update();
            {
                let state = lock(&self.inner.state);
                if state.overflow {
                    return Err(Code::InferenceFailed);
                }
                if let Some(event) = state.events.get(index) {
                    return Ok(event.clone());
                }
            }
            changed.changed().await.map_err(|_| Code::NotReady)?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_is_monotonic_and_single_service_only() {
        let control = SchedulerControl::new();
        assert_eq!(control.now_us(), 0);
        control.advance_to_us(999).unwrap();
        assert_eq!(control.now_us(), 999);
        assert!(control.advance_to_us(998).is_err());
        control.advance_to_us(1000).unwrap();
        assert!(control.claim().is_ok());
        assert!(control.claim().is_err());
    }

    #[tokio::test]
    async fn retained_events_are_not_lost_before_subscription() {
        let control = SchedulerControl::new();
        control.emit(json!({"kind": "example"}));
        assert_eq!(control.event(0).await.unwrap()["kind"], "example");
    }
}
