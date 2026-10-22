use anyhow::{ensure, Result};
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::{Request, Uri};
use hyper_util::{
    client::legacy::{connect::HttpConnector, Client},
    rt::TokioExecutor,
};
use serde_json::json;
use std::{sync::Arc, time::Instant};
use tokio::sync::Barrier;

type HttpClient = Client<HttpConnector, Full<Bytes>>;

async fn call(client: &HttpClient, uri: &Uri, body: Bytes) -> Result<()> {
    let request = Request::post(uri.clone())
        .header("content-type", "application/json")
        .body(Full::new(body))?;

    let response = client.request(request).await?;
    let status = response.status();
    let body = response.into_body().collect().await?.to_bytes();
    ensure!(status.is_success(), "HTTP {status}: {}", String::from_utf8_lossy(&body));

    let response: serde_json::Value = serde_json::from_slice(&body)?;
    ensure!(
        response["data"][0]["embedding"].as_array().map(Vec::len) == Some(2048),
        "invalid response"
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // bench URL corpus.json concurrency iterations-per-worker
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 5, "bench URL corpus.json concurrency iterations");

    let uri: Uri = args[1].parse()?;
    let texts: Vec<String> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let concurrency: usize = args[3].parse()?;
    let iterations: usize = args[4].parse()?;
    ensure!(!texts.is_empty() && concurrency > 0 && iterations > 0, "invalid args");

    let bodies: Arc<Vec<Bytes>> = Arc::new(texts.into_iter().map(|text| {
        Bytes::from(serde_json::to_vec(&json!({
            "model": "nvidia/Nemotron-3-Embed-1B",
            "input": text,
            "encoding_format": "float"
        })).unwrap())
    }).collect());

    let client: HttpClient = Client::builder(TokioExecutor::new()).build_http();
    let barrier = Arc::new(Barrier::new(concurrency + 1));
    let mut tasks = Vec::new();

    for worker in 0..concurrency {
        let client = client.clone();
        let uri = uri.clone();
        let bodies = bodies.clone();
        let barrier = barrier.clone();

        tasks.push(tokio::spawn(async move {
            // Warm connections and execution before starting the clock.
            // Keep the barrier reachable even when warmup fails.
            let warmup = call(&client, &uri, bodies[worker % bodies.len()].clone()).await;
            barrier.wait().await;
            warmup?;

            let mut samples = Vec::with_capacity(iterations);
            for n in 0..iterations {
                let body = bodies[(worker + n) % bodies.len()].clone();
                let start = Instant::now();
                call(&client, &uri, body).await?;
                samples.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            Ok::<_, anyhow::Error>(samples)
        }));
    }

    barrier.wait().await;
    let start = Instant::now();

    let mut samples = Vec::with_capacity(concurrency * iterations);
    for task in tasks {
        samples.extend(task.await??);
    }

    let seconds = start.elapsed().as_secs_f64();
    samples.sort_by(f64::total_cmp);
    let percentile = |p: f64| {
        let index = ((samples.len() as f64 * p).ceil() as usize)
            .saturating_sub(1)
            .min(samples.len() - 1);
        samples[index]
    };

    println!(
        "requests={} req/s={:.2} p50_ms={:.2} p90_ms={:.2} p99_ms={:.2}",
        samples.len(),
        samples.len() as f64 / seconds,
        percentile(0.50),
        percentile(0.90),
        percentile(0.99),
    );
    Ok(())
}
