mod callbacks;
mod degeneracy;
mod init;
mod processing;
mod publishers;
mod services;
mod state;

pub(crate) use state::NdtScanMatcherNode;

/// Stage timing helpers for the callback path, active when `NDT_PROFILE=1`.
///
/// The scoring passes around the alignment are not free on the CUDA path --
/// each is a full pass over every source point -- so they are timed separately
/// from the alignment they bracket.
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

static PROFILE_ON: AtomicBool = AtomicBool::new(false);
static PROFILE_INIT: std::sync::Once = std::sync::Once::new();

pub(crate) fn profile_enabled() -> bool {
    PROFILE_INIT.call_once(|| {
        let on = std::env::var("NDT_PROFILE").is_ok_and(|v| v != "0" && !v.is_empty());
        PROFILE_ON.store(on, Ordering::Relaxed);
    });
    PROFILE_ON.load(Ordering::Relaxed)
}

pub(crate) fn profile_stage() -> Option<Instant> {
    profile_enabled().then(Instant::now)
}

pub(crate) fn profile_ms(t: Option<Instant>) -> f64 {
    t.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0)
}

pub(crate) fn profile_emit_scores(tp_before_ms: f64, nvtl_before_ms: f64, align_ms: f64) {
    if !profile_enabled() {
        return;
    }
    eprintln!(
        "[ndt_scores] tp_before={tp_before_ms:.2}ms nvtl_before={nvtl_before_ms:.2}ms align={align_ms:.2}ms"
    );
}
