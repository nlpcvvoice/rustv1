use crate::model::{EngineConfig, Model};
use anyhow::{anyhow, ensure, Result};
use candle_core::Device;
use std::{
    path::PathBuf,
    sync::mpsc::{self, Receiver, SyncSender},
    thread,
    time::{Duration, Instant},
};
use tokio::sync::oneshot;

pub struct Job {
    pub ids: Vec<u32>,
    pub reply: oneshot::Sender<Result<Vec<f32>, String>>,
}

pub fn start(
    dir: PathBuf,
    cfg: EngineConfig,
    cpu: bool,
) -> Result<SyncSender<Job>> {
    ensure!(cfg.max_seqs > 0, "max_seqs must be positive");
    ensure!(cfg.max_tokens >= cfg.max_len, "one sequence must fit");

    let (tx, rx) = mpsc::sync_channel::<Job>(128);
    let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(), String>>(1);

    thread::Builder::new().name("rustv1-gpu".into()).spawn(move || {
        let load = || -> Result<Model> {
            let device = if cpu { Device::Cpu } else { Device::new_cuda(0)? };
            let dtype = if cpu { candle_core::DType::F32 } else { candle_core::DType::BF16 };
            Model::load(&dir, device, cfg.clone(), dtype)
        };

        let model = match load() {
            Ok(m) => m,
            Err(e) => {
                let _ = ready_tx.send(Err(format!("{e:#}")));
                return;
            }
        };

        let _ = ready_tx.send(Ok(()));
        run(model, cfg, rx);
    })?;

    ready_rx.recv().map_err(|e| anyhow!(e))?
        .map_err(|e| anyhow!(e))?;
    Ok(tx)
}

fn run(model: Model, cfg: EngineConfig, rx: Receiver<Job>) {
    let mut carry: Option<Job> = None;

    loop {
        let first = match carry.take().or_else(|| rx.recv().ok()) {
            Some(job) => job,
            None => break,
        };
        if first.reply.is_closed() {
            continue;
        }

        let mut tokens = first.ids.len();
        let mut max_len = first.ids.len();
        let mut jobs = vec![first];
        let deadline = Instant::now() + Duration::from_micros(cfg.wait_us);

        while jobs.len() < cfg.max_seqs {
            let next = match rx.try_recv() {
                Ok(job) => Some(job),
                Err(mpsc::TryRecvError::Disconnected) => None,
                Err(mpsc::TryRecvError::Empty) => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        None
                    } else {
                        rx.recv_timeout(remaining).ok()
                    }
                }
            };

            let Some(job) = next else { break };
            if job.reply.is_closed() {
                continue;
            }

            let next_tokens = tokens + job.ids.len();
            let next_max = max_len.max(job.ids.len());
            let padded_tokens = (jobs.len() + 1) * next_max;

            if next_tokens > cfg.max_tokens || padded_tokens > cfg.max_tokens {
                carry = Some(job);
                break;
            }

            tokens = next_tokens;
            max_len = next_max;
            jobs.push(job);
        }

        jobs.retain(|j| !j.reply.is_closed());
        if jobs.is_empty() {
            continue;
        }

        // Deliberate starter copy; production packs directly into pinned slabs.
        let inputs: Vec<_> = jobs.iter().map(|j| j.ids.clone()).collect();

        match model.forward(&inputs) {
            Ok(outputs) => {
                for (job, output) in jobs.into_iter().zip(outputs) {
                    let _ = job.reply.send(Ok(output));
                }
            }
            Err(error) => {
                let message = format!("{error:#}");
                for job in jobs {
                    let _ = job.reply.send(Err(message.clone()));
                }
            }
        }
    }
}
