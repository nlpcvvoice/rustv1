#!/usr/bin/env python3
from pathlib import Path

path = Path("src/http_serve.rs")
source = path.read_text()

def replace_once(old, new):
    global source
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"expected one occurrence, found {count}: {old[:100]!r}")
    source = source.replace(old, new, 1)

def transform_region(start, end, transform):
    global source
    a = source.index(start)
    b = source.index(end, a)
    source = source[:a] + transform(source[a:b]) + source[b:]

replace_once(
    'pub const MODEL_ID: &str =',
    '''#[cfg(feature = "scheduler-test-seam")]
#[path = "http_serve_deterministic.rs"]
pub mod deterministic;

pub const MODEL_ID: &str ='''
)

replace_once(
    '    pub batching: Option<BatchedExecution>,\n',
    '''    pub batching: Option<BatchedExecution>,
    #[cfg(feature = "scheduler-test-seam")]
    pub scheduler_control: Option<deterministic::SchedulerControl>,
'''
)

replace_once(
    '            batching: None,\n',
    '''            batching: None,
            #[cfg(feature = "scheduler-test-seam")]
            scheduler_control: None,
'''
)

replace_once(
    'impl Shared {\n',
    '''impl Shared {
    #[inline]
    fn now(&self) -> Instant {
        #[cfg(feature = "scheduler-test-seam")]
        if let Some(control) = &self.config.scheduler_control {
            return control.now();
        }
        Instant::now()
    }

    async fn sleep_until(&self, deadline: Instant) {
        #[cfg(feature = "scheduler-test-seam")]
        if let Some(control) = &self.config.scheduler_control {
            control.sleep_until(deadline).await;
            return;
        }
        sleep(deadline.saturating_duration_since(Instant::now())).await;
    }

    async fn request_timeout_at<F: Future>(
        &self,
        deadline: Instant,
        future: F,
    ) -> Result<F::Output, Code> {
        #[cfg(feature = "scheduler-test-seam")]
        if let Some(control) = &self.config.scheduler_control {
            return tokio::select! {
                biased;
                result = future => Ok(result),
                _ = control.sleep_until(deadline) => Err(Code::DeadlineExceeded),
            };
        }
        timeout_at(TokioInstant::from_std(deadline), future)
            .await
            .map_err(|_| Code::DeadlineExceeded)
    }

    async fn assembly_receive(
        &self,
        receiver: &mut mpsc::Receiver<Message>,
        boundary: Instant,
    ) -> Result<Option<Message>, Code> {
        #[cfg(feature = "scheduler-test-seam")]
        if let Some(control) = &self.config.scheduler_control {
            control.waiting(boundary);
            return tokio::select! {
                biased;
                _ = control.sleep_until(boundary) => Err(Code::DeadlineExceeded),
                message = receiver.recv() => Ok(message),
            };
        }
        timeout_at(TokioInstant::from_std(boundary), receiver.recv())
            .await
            .map_err(|_| Code::DeadlineExceeded)
    }

'''
)

# The original insertion critical section is retained.
def infer_region(s):
    s = s.replace("Instant::now()", "self.shared.now()")
    old = '''            self.tx
                .try_send(Message::Infer(job))'''
    new = '''            #[cfg(feature = "scheduler-test-seam")]
            let insertion = self.shared.config.scheduler_control
                .as_ref().map(|control| control.insertion(&job));
            self.tx
                .try_send(Message::Infer(job))'''
    assert s.count(old) == 1
    s = s.replace(old, new)
    old = '''                    mpsc::error::TrySendError::Closed(_) => Code::ExecutorPoisoned,
                })?;
        }

        tokio::select!'''
    new = '''                    mpsc::error::TrySendError::Closed(_) => Code::ExecutorPoisoned,
                })?;
            #[cfg(feature = "scheduler-test-seam")]
            if let (Some(control), Some(event)) =
                (&self.shared.config.scheduler_control, insertion)
            {
                control.emit(event);
            }
        }

        tokio::select!'''
    assert s.count(old) == 1
    s = s.replace(old, new)
    old = '_ = sleep(deadline.saturating_duration_since(self.shared.now())) =>'
    assert s.count(old) == 1
    s = s.replace(old, '_ = self.shared.sleep_until(deadline) =>')
    return s

transform_region("    async fn infer(\n", "    async fn command(\n", infer_region)

# Keep wall-clock logging; only the request's semantic time origin is virtual.
def request_region(s):
    old = '        let deadline = started + self.config.request_timeout;\n'
    new = '''        #[cfg(feature = "scheduler-test-seam")]
        let started = if self.config.scheduler_control.is_some() {
            self.backend.shared.now()
        } else {
            started
        };
        let deadline = started + self.config.request_timeout;
'''
    assert s.count(old) == 1
    s = s.replace(old, new)
    s = s.replace("Instant::now()", "self.backend.shared.now()")
    old = '''        let tokenizer_permit = timeout_at(
            TokioInstant::from_std(deadline),
            self.tokenizers.clone().acquire_owned(),
        )'''
    new = '''        let tokenizer_permit = self.backend.shared.request_timeout_at(
            deadline,
            self.tokenizers.clone().acquire_owned(),
        )'''
    assert s.count(old) == 1
    s = s.replace(old, new)
    old = 'timeout_at(TokioInstant::from_std(deadline), work)'
    assert s.count(old) == 1
    s = s.replace(old, 'self.backend.shared.request_timeout_at(deadline, work)')
    return s

transform_region("    async fn request(\n", "    pub async fn control(", request_region)

replace_once(
    '        let config = Arc::new(config);\n',
    '''        #[cfg(feature = "scheduler-test-seam")]
        if let Some(control) = &config.scheduler_control {
            control.claim()?;
        }
        let config = Arc::new(config);
'''
)

transform_region(
    "fn queued_error(",
    "fn worker_loop",
    lambda s: s.replace("Instant::now()", "shared.now()")
)

def worker_region(s):
    # Control-message deadlines deliberately remain real-time watchdogs.
    s = s.replace(
        'while b < policy.max_batch_size && Instant::now() < boundary {',
        'while b < policy.max_batch_size && shared.now() < boundary {'
    )
    old = '''                        let next = runtime.block_on(async {
                            timeout_at(TokioInstant::from_std(boundary), receiver.recv()).await
                        });'''
    new = '''                        let next =
                            runtime.block_on(shared.assembly_receive(&mut receiver, boundary));'''
    assert s.count(old) == 1
    s = s.replace(old, new)
    s = s.replace(
        'if Instant::now() >= boundary {',
        'if shared.now() >= boundary {'
    )
    old = '                let mut jobs = vec![first];\n'
    new = '''                #[cfg(feature = "scheduler-test-seam")]
                if let Some(control) = &shared.config.scheduler_control {
                    control.checkpoint(
                        "assembly_start", std::slice::from_ref(&first), None,
                    );
                }
                let mut jobs = vec![first];
'''
    assert s.count(old) == 1
    s = s.replace(old, new)

    old = '''                    .min(jobs[0].deadline);

                    while'''
    new = '''                    .min(jobs[0].deadline);

                    #[cfg(feature = "scheduler-test-seam")]
                    if let Some(control) = &shared.config.scheduler_control {
                        control.assembly(&jobs, boundary);
                    }

                    while'''
    assert s.count(old) == 1
    s = s.replace(old, new)

    old = '                                jobs.push(job);\n'
    new = '''                                jobs.push(job);
                                #[cfg(feature = "scheduler-test-seam")]
                                if let Some(control) = &shared.config.scheduler_control {
                                    control.assembly(&jobs, boundary);
                                }
'''
    assert s.count(old) == 1
    s = s.replace(old, new)

    old = '''                    }
                }
                run_batch(backend, shared, jobs);'''
    new = '''                    }
                    #[cfg(feature = "scheduler-test-seam")]
                    if let Some(control) = &shared.config.scheduler_control {
                        control.checkpoint("assembled", &jobs, Some(boundary));
                    }
                }
                run_batch(backend, shared, jobs);'''
    assert s.count(old) == 1
    s = s.replace(old, new)
    return s

transform_region("fn worker_loop", "fn run_batch", worker_region)

def batch_region(s):
    s = s.replace("Instant::now()", "shared.now()")
    old = '''            job.queue_slot.take();
        }
    }

    // Catch operational panics'''
    new = '''            job.queue_slot.take();
        }
        #[cfg(feature = "scheduler-test-seam")]
        if let Some(control) = &shared.config.scheduler_control {
            control.launch(&live);
        }
    }

    // Catch operational panics'''
    assert s.count(old) == 1
    return s.replace(old, new)

transform_region(
    "fn run_batch",
    "#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]",
    batch_region
)

cargo = Path("Cargo.toml")
manifest = cargo.read_text()
anchor = 'cuda = ["candle-core/cuda", "candle-nn/cuda"]\n'
if manifest.count(anchor) != 1 or "scheduler-test-seam" in manifest:
    raise SystemExit("unexpected Cargo feature section")
manifest = manifest.replace(anchor, anchor + "scheduler-test-seam = []\n")

# Write only after all source transformations have succeeded.
path.write_text(source)
cargo.write_text(manifest)
