//! Compare the GPU and CPU alignment arms on identical recorded input.
//!
//! A live replay cannot do this. The slower arm misses the scan budget, drops
//! scans, and each dropped scan leaves a staler prior, which scores worse,
//! which fails the convergence gate -- so a live run measures a collapse rather
//! than a speed ratio. Here every arm sees the same (scan, initial guess)
//! sequence, so equal work is structural rather than something to check
//! afterwards, and the only difference left is how long an alignment takes.
//!
//! Input comes from `scripts/testing/localization/export_ndt_frames.py`.
//!
//! ```text
//! cargo run --release -p ndt_cuda --example offline_bench -- tmp/ndt-frames.bin
//! cargo run --release -p ndt_cuda --example offline_bench -- tmp/ndt-frames.bin --arms gpu,cpu --repeats 3 --csv out.csv
//! ```
use std::env;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::time::Instant;

use nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion};
use ndt_cuda::NdtScanMatcher;

struct Frame {
    pose: Isometry3<f64>,
    points: Vec<[f32; 3]>,
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_f64(r: &mut impl Read) -> std::io::Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

fn read_points(r: &mut impl Read, n: usize) -> std::io::Result<Vec<[f32; 3]>> {
    let mut raw = vec![0u8; n * 12];
    r.read_exact(&mut raw)?;
    Ok(raw
        .chunks_exact(12)
        .map(|c| {
            [
                f32::from_le_bytes(c[0..4].try_into().unwrap()),
                f32::from_le_bytes(c[4..8].try_into().unwrap()),
                f32::from_le_bytes(c[8..12].try_into().unwrap()),
            ]
        })
        .collect())
}

fn load(path: &str) -> std::io::Result<(Vec<[f32; 3]>, Vec<Frame>)> {
    let mut r = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    assert_eq!(&magic, b"NDTB", "not an NDT frame dump: {path}");
    let version = read_u32(&mut r)?;
    assert_eq!(version, 1, "unsupported dump version {version}");

    let n_map = read_u32(&mut r)? as usize;
    let map = read_points(&mut r, n_map)?;

    let n_frames = read_u32(&mut r)? as usize;
    let mut frames = Vec::with_capacity(n_frames);
    for _ in 0..n_frames {
        let _stamp = read_f64(&mut r)?;
        let mut p = [0f64; 7];
        for v in p.iter_mut() {
            *v = read_f64(&mut r)?;
        }
        let n = read_u32(&mut r)? as usize;
        let points = read_points(&mut r, n)?;
        let rotation =
            UnitQuaternion::from_quaternion(Quaternion::new(p[6], p[3], p[4], p[5]));
        frames.push(Frame {
            pose: Isometry3::from_parts(Translation3::new(p[0], p[1], p[2]), rotation),
            points,
        });
    }
    Ok((map, frames))
}

struct Sample {
    ms: f64,
    iterations: usize,
    score: f64,
    nvtl: f64,
    x: f64,
    y: f64,
    z: f64,
}

fn run_arm(
    name: &str,
    use_gpu: bool,
    map: &[[f32; 3]],
    frames: &[Frame],
    warmup: usize,
) -> anyhow::Result<Vec<Sample>> {
    let mut matcher = NdtScanMatcher::builder()
        .resolution(2.0)
        .max_iterations(30)
        .transformation_epsilon(0.01)
        .step_size(0.1)
        .use_line_search(true)
        .use_gpu(use_gpu)
        .build()?;

    let t0 = Instant::now();
    matcher.set_target(map)?;
    eprintln!(
        "[{name}] target set: {} map points in {:.1} s",
        map.len(),
        t0.elapsed().as_secs_f64()
    );

    // The first alignments pay for lazy GPU pipeline setup and cache warming,
    // which would otherwise land entirely on whichever arm runs first.
    for f in frames.iter().take(warmup) {
        let _ = matcher.align(&f.points, f.pose);
    }

    let mut out = Vec::with_capacity(frames.len());
    for f in frames {
        let start = Instant::now();
        let result = matcher.align(&f.points, f.pose);
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        match result {
            Ok(r) => {
                let nvtl = matcher.evaluate_nvtl(&f.points, &r.pose).unwrap_or(f64::NAN);
                let t = r.pose.translation.vector;
                out.push(Sample {
                    ms,
                    iterations: r.iterations,
                    score: r.score,
                    nvtl,
                    x: t.x,
                    y: t.y,
                    z: t.z,
                });
            }
            Err(e) => eprintln!("[{name}] alignment failed: {e}"),
        }
    }
    Ok(out)
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[((sorted.len() as f64 * p) as usize).min(sorted.len() - 1)]
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: offline_bench <frames.bin> [--arms gpu,cpu] [--repeats N] [--warmup N] [--csv PATH]"
        );
        std::process::exit(2);
    }
    let path = &args[1];
    let mut arms = vec!["gpu".to_string(), "cpu".to_string()];
    let mut repeats = 1usize;
    let mut warmup = 5usize;
    let mut csv: Option<String> = None;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--arms" => {
                arms = args[i + 1].split(',').map(|s| s.to_string()).collect();
                i += 2;
            }
            "--repeats" => {
                repeats = args[i + 1].parse()?;
                i += 2;
            }
            "--warmup" => {
                warmup = args[i + 1].parse()?;
                i += 2;
            }
            "--csv" => {
                csv = Some(args[i + 1].clone());
                i += 2;
            }
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
    }

    let (map, frames) = load(path)?;

    // Scoring probe: NVTL from each arm at the *same* pose, so any difference
    // is the scoring code rather than the alignment that preceded it. The gate
    // that decides whether a pose gets published is a fixed threshold, so the
    // arms disagreeing here rejects on one what it accepts on the other.
    if let Ok(n) = env::var("NVTL_PROBE").map(|v| v.parse::<usize>().unwrap_or(5)) {
        // Both matchers built identically apart from the arm, so this compares
        // exactly what production runs: evaluate_nvtl prefers the GPU kernel and
        // silently falls back to compute_nvtl_simple when there is no GPU.
        let build = |use_gpu: bool| -> anyhow::Result<NdtScanMatcher> {
            let mut m = NdtScanMatcher::builder()
                .resolution(2.0)
                .max_iterations(30)
                .transformation_epsilon(0.01)
                .step_size(0.1)
                .use_line_search(true)
                .use_gpu(use_gpu)
                .build()?;
            m.set_target(&map)?;
            Ok(m)
        };
        let gpu_m = build(true)?;
        let cpu_m = build(false)?;
        println!("NVTL at the identical recorded pose, per arm\n");
        println!("{:>6} {:>10} {:>10} {:>9}", "frame", "gpu nvtl", "cpu nvtl", "cpu/gpu");
        for (k, f) in frames.iter().take(n).enumerate() {
            let g = gpu_m.evaluate_nvtl(&f.points, &f.pose).unwrap_or(f64::NAN);
            let c = cpu_m.evaluate_nvtl(&f.points, &f.pose).unwrap_or(f64::NAN);
            println!("{k:>6} {g:>10.4} {c:>10.4} {:>9.4}", c / g);
        }
        return Ok(());
    }

    println!(
        "offline NDT benchmark: {} map points, {} frames, {} point(s) per scan (first), \
         warmup {warmup}, repeats {repeats}",
        map.len(),
        frames.len(),
        frames.first().map(|f| f.points.len()).unwrap_or(0)
    );
    println!("every arm sees the identical (scan, initial guess) sequence\n");

    let mut results: Vec<(String, Vec<Sample>)> = Vec::new();
    for arm in &arms {
        let use_gpu = match arm.as_str() {
            "gpu" => true,
            "cpu" => false,
            other => {
                eprintln!("unknown arm '{other}', expected gpu or cpu");
                std::process::exit(2);
            }
        };
        let mut best: Option<Vec<Sample>> = None;
        for rep in 0..repeats {
            let s = run_arm(arm, use_gpu, &map, &frames, warmup)?;
            let m = mean(&s.iter().map(|x| x.ms).collect::<Vec<_>>());
            eprintln!("[{arm}] repeat {} of {repeats}: mean {m:.3} ms", rep + 1);
            // Keep the fastest repeat: a slower one is contention or thermal
            // drift, not the matcher being intermittently better.
            if best
                .as_ref()
                .map(|b| m < mean(&b.iter().map(|x| x.ms).collect::<Vec<_>>()))
                .unwrap_or(true)
            {
                best = Some(s);
            }
        }
        results.push((arm.clone(), best.unwrap()));
    }

    println!(
        "{:>6} {:>7} {:>9} {:>8} {:>8} {:>8} {:>7} {:>8} {:>8}",
        "arm", "aligns", "mean ms", "p50", "p95", "max", "iters", "score", "NVTL"
    );
    println!("{}", "-".repeat(76));
    for (arm, s) in &results {
        let mut ms: Vec<f64> = s.iter().map(|x| x.ms).collect();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "{arm:>6} {:>7} {:>9.3} {:>8.3} {:>8.3} {:>8.3} {:>7.2} {:>8.1} {:>8.3}",
            s.len(),
            mean(&ms),
            pct(&ms, 0.5),
            pct(&ms, 0.95),
            ms.last().copied().unwrap_or(f64::NAN),
            mean(&s.iter().map(|x| x.iterations as f64).collect::<Vec<_>>()),
            mean(&s.iter().map(|x| x.score).collect::<Vec<_>>()),
            mean(&s.iter().map(|x| x.nvtl).collect::<Vec<_>>()),
        );
    }

    if let (Some((_, g)), Some((_, c))) = (
        results.iter().find(|(a, _)| a == "gpu"),
        results.iter().find(|(a, _)| a == "cpu"),
    ) {
        let gm = mean(&g.iter().map(|x| x.ms).collect::<Vec<_>>());
        let cm = mean(&c.iter().map(|x| x.ms).collect::<Vec<_>>());
        println!("\nGPU vs CPU on identical input: {:.2}x ({cm:.3} -> {gm:.3} ms)", cm / gm);

        // Same inputs should mean the same answers; say so or say how far off.
        let n = g.len().min(c.len());
        let mut worst = 0.0f64;
        let mut d_iter = 0.0f64;
        let mut d_nvtl = 0.0f64;
        for k in 0..n {
            let d = ((g[k].x - c[k].x).powi(2)
                + (g[k].y - c[k].y).powi(2)
                + (g[k].z - c[k].z).powi(2))
            .sqrt();
            worst = worst.max(d);
            d_iter += (g[k].iterations as i64 - c[k].iterations as i64).abs() as f64;
            d_nvtl += (g[k].nvtl - c[k].nvtl).abs();
        }
        println!(
            "agreement over {n} frames: worst pose difference {worst:.4} m, \
             mean |Δiterations| {:.2}, mean |ΔNVTL| {:.4}",
            d_iter / n as f64,
            d_nvtl / n as f64
        );
        if worst > 0.05 {
            println!(
                "  WARNING the arms disagree by more than 5 cm somewhere; the timing \
                 ratio above compares different work"
            );
        }
    }

    if let Some(path) = csv {
        let mut f = File::create(&path)?;
        writeln!(f, "arm,frame,ms,iterations,score,nvtl,x,y,z")?;
        for (arm, s) in &results {
            for (k, x) in s.iter().enumerate() {
                writeln!(
                    f,
                    "{arm},{k},{:.6},{},{:.4},{:.6},{:.6},{:.6},{:.6}",
                    x.ms, x.iterations, x.score, x.nvtl, x.x, x.y, x.z
                )?;
            }
        }
        println!("\nper-frame data: {path}");
    }
    Ok(())
}
