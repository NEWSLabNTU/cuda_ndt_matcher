//! Divergence detection for `scan_matching_status`, latched until re-initialization.
//!
//! The scan matcher's own convergence gate decides, one scan at a time, whether
//! to publish a pose. It says nothing about whether localization as a whole has
//! gone wrong, and two failures get past it on the Ming-Da replays
//! (superproject `docs/roadmaps/8-autonomous-driving.md`, "L6 design (desk)"):
//!
//! 1. **Skips.** After a 3 m lateral jump every scan hit the 30-iteration cap
//!    and nothing was published for 55 s, while `scan_matching_status` stayed
//!    OK. Upstream Autoware raises ERROR once `skipping_publish_num` reaches
//!    `validation.skipping_publish_num`; this matcher never did.
//! 2. **Confident wrong fit.** After a 5 m jump along the aisle NDT settled
//!    2.5-4 m off with NVTL 2.0-2.76, comfortably above the publish gate, and
//!    the EKF accepted it. Per scan nothing is wrong; the median NVTL over a
//!    window is what separates it from a healthy run.
//!
//! Either one **latches**. `use_emergency_holding` is false in this stack, so a
//! leaf that clears lets the cart drive on; once diverged, the leaf stays ERROR
//! until a successful `ndt_align` (the pose initializer's re-initialization)
//! calls [`DivergenceMonitor::reset`].
//!
//! Pure state, no ROS: the node feeds it one scan at a time.

use std::{collections::VecDeque, fmt};

use super::diagnostics::{DiagnosticCategory, DiagnosticLevel};

/// Thresholds for the two detectors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DivergenceConfig {
    /// Detector 1 fires when this many consecutive scans were not published.
    /// `validation.skipping_publish_num`, as upstream uses it. 0 or less disables.
    pub(crate) max_consecutive_skips: i32,
    /// Detector 2's window, in scans. Fires only once the window is full.
    pub(crate) nvtl_window_size: usize,
    /// Detector 2 fires when the window's median NVTL is below this.
    /// 0.0 or less disables: the gate is per map and per input, so there is no
    /// default that is right everywhere.
    pub(crate) nvtl_median_gate: f64,
}

/// Which detector fired, with the values that made it fire.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Divergence {
    Skips {
        count: i32,
        limit: i32,
    },
    NvtlMedian {
        median: f64,
        gate: f64,
        window: usize,
    },
}

impl fmt::Display for Divergence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Skips { count, limit } => write!(
                f,
                "NDT diverged: skipping_publish_num {count} >= {limit} \
                 (latched until re-initialization)"
            ),
            Self::NvtlMedian {
                median,
                gate,
                window,
            } => write!(
                f,
                "NDT diverged: median NVTL of last {window} scans {median:.3} < {gate:.3} \
                 (latched until re-initialization)"
            ),
        }
    }
}

/// Detectors 1 and 2 plus the latch.
#[derive(Debug, Clone)]
pub(crate) struct DivergenceMonitor {
    config: DivergenceConfig,
    window: VecDeque<f64>,
    latched: Option<Divergence>,
}

impl DivergenceMonitor {
    pub(crate) fn new(config: DivergenceConfig) -> Self {
        let config = DivergenceConfig {
            nvtl_window_size: config.nvtl_window_size.max(1),
            ..config
        };
        Self {
            window: VecDeque::with_capacity(config.nvtl_window_size),
            config,
            latched: None,
        }
    }

    /// Feed one aligned scan: its consecutive-skip count (0 when published)
    /// and its NVTL. Returns the divergence if one is latched, newly or not.
    ///
    /// The first detector to fire is the one kept; later scans do not
    /// overwrite the cause.
    pub(crate) fn observe(&mut self, skipping_publish_num: i32, nvtl: f64) -> Option<Divergence> {
        // A non-finite score is the worst score, not a missing one: NaN would
        // compare false against the gate and the detector would never fire.
        let nvtl = if nvtl.is_finite() { nvtl } else { 0.0 };
        if self.window.len() == self.config.nvtl_window_size {
            self.window.pop_front();
        }
        self.window.push_back(nvtl);

        if self.latched.is_none() {
            self.latched = self.check(skipping_publish_num);
        }
        self.latched
    }

    fn check(&self, skipping_publish_num: i32) -> Option<Divergence> {
        let limit = self.config.max_consecutive_skips;
        if limit > 0 && skipping_publish_num >= limit {
            return Some(Divergence::Skips {
                count: skipping_publish_num,
                limit,
            });
        }
        let gate = self.config.nvtl_median_gate;
        if gate > 0.0
            && let Some(median) = self.window_median()
            && median < gate
        {
            return Some(Divergence::NvtlMedian {
                median,
                gate,
                window: self.config.nvtl_window_size,
            });
        }
        None
    }

    /// Median NVTL of the window, `None` until it holds `nvtl_window_size` scans.
    pub(crate) fn window_median(&self) -> Option<f64> {
        if self.window.len() < self.config.nvtl_window_size {
            return None;
        }
        Some(median(self.window.iter().copied()))
    }

    pub(crate) fn latched(&self) -> Option<Divergence> {
        self.latched
    }

    /// Successful re-initialization: clear the latch and the window, so scores
    /// from the diverged pose cannot re-fire detector 2 against the new one.
    pub(crate) fn reset(&mut self) {
        self.latched = None;
        self.window.clear();
    }

    /// Add the detector state to `diag` and, while latched, force it to ERROR.
    /// Run after `ScanMatchingDiagnostics::apply_to`, which sets the
    /// per-scan level this may override; an existing ERROR message is kept.
    pub(crate) fn apply_to(&self, diag: &mut DiagnosticCategory) {
        diag.add_key_value("divergence_latched", self.latched.is_some());
        diag.add_key_value(
            "nvtl_window_median",
            self.window_median()
                .map_or_else(|| "nan".to_string(), |m| format!("{m:.6}")),
        );
        diag.add_key_value(
            "nvtl_window_median_gate",
            format!("{:.3}", self.config.nvtl_median_gate),
        );
        if let Some(divergence) = self.latched {
            let message = if diag.level() == DiagnosticLevel::Error {
                format!("{divergence}; {}", diag.message())
            } else {
                divergence.to_string()
            };
            diag.set_level_and_message(DiagnosticLevel::Error, &message);
        }
    }
}

fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut v: Vec<f64> = values.collect();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn monitor(skips: i32, window: usize, gate: f64) -> DivergenceMonitor {
        DivergenceMonitor::new(DivergenceConfig {
            max_consecutive_skips: skips,
            nvtl_window_size: window,
            nvtl_median_gate: gate,
        })
    }

    #[test]
    fn median_odd_even_and_unsorted() {
        assert_eq!(median([3.0, 1.0, 2.0].into_iter()), 2.0);
        assert_eq!(median([4.0, 1.0, 3.0, 2.0].into_iter()), 2.5);
    }

    #[test]
    fn window_median_none_until_full_then_slides() {
        let mut m = monitor(0, 3, 0.0);
        m.observe(0, 1.0);
        m.observe(0, 2.0);
        assert_eq!(m.window_median(), None);
        m.observe(0, 3.0);
        assert_eq!(m.window_median(), Some(2.0));
        // Oldest (1.0) drops out: window is [2, 3, 10].
        m.observe(0, 10.0);
        assert_eq!(m.window_median(), Some(3.0));
    }

    /// One outlier scan does not move a median; that is why it is a median.
    #[test]
    fn single_bad_scan_does_not_fire_nvtl_detector() {
        let mut m = monitor(0, 10, 2.5);
        for _ in 0..9 {
            assert_eq!(m.observe(0, 2.9), None);
        }
        assert_eq!(m.observe(0, 0.5), None);
        assert_eq!(m.latched(), None);
    }

    #[test]
    fn nvtl_detector_needs_full_window() {
        let mut m = monitor(0, 10, 2.5);
        for _ in 0..9 {
            assert_eq!(m.observe(0, 2.0), None);
        }
        let fired = m.observe(0, 2.0);
        assert_eq!(
            fired,
            Some(Divergence::NvtlMedian {
                median: 2.0,
                gate: 2.5,
                window: 10
            })
        );
    }

    /// The 5 m case from the replay: NVTL 2.0-2.76, median 2.31, above the
    /// 1.3/2.0 publish gates, below 2.5.
    #[test]
    fn nvtl_detector_fires_on_wrong_convergence_scores() {
        let mut m = monitor(5, 10, 2.5);
        let scores = [2.31, 2.76, 2.0, 2.4, 2.3, 2.1, 2.5, 2.2, 2.6, 2.35];
        let mut fired = None;
        for s in scores {
            fired = m.observe(0, s);
        }
        assert!(matches!(fired, Some(Divergence::NvtlMedian { median, .. }) if median < 2.5));
    }

    #[test]
    fn nvtl_detector_disabled_at_zero_gate() {
        let mut m = monitor(0, 10, 0.0);
        for _ in 0..20 {
            assert_eq!(m.observe(0, 0.1), None);
        }
    }

    #[test]
    fn healthy_scores_never_fire() {
        // Healthy Ming-Da NVTL never drops below 2.57.
        let mut m = monitor(5, 10, 2.5);
        for i in 0..100 {
            let s = 2.57 + 0.3 * ((i % 7) as f64 / 7.0);
            assert_eq!(m.observe(0, s), None);
        }
    }

    #[test]
    fn skip_detector_fires_at_limit_not_before() {
        let mut m = monitor(5, 10, 0.0);
        for n in 1..5 {
            assert_eq!(m.observe(n, 2.8), None, "fired early at {n}");
        }
        assert_eq!(
            m.observe(5, 2.8),
            Some(Divergence::Skips { count: 5, limit: 5 })
        );
    }

    #[test]
    fn skip_detector_disabled_at_zero_limit() {
        let mut m = monitor(0, 10, 0.0);
        assert_eq!(m.observe(1000, 2.8), None);
    }

    /// A non-consecutive run of skips is reset by the node's counter to 0 on
    /// every published scan, so it never reaches the limit.
    #[test]
    fn interrupted_skips_do_not_fire() {
        let mut m = monitor(5, 10, 0.0);
        for n in [1, 2, 3, 4, 0, 1, 2, 3, 4, 0] {
            assert_eq!(m.observe(n, 2.8), None);
        }
    }

    #[test]
    fn latch_holds_after_scores_recover_and_keeps_first_cause() {
        let mut m = monitor(5, 3, 2.5);
        m.observe(5, 2.8);
        let cause = Divergence::Skips { count: 5, limit: 5 };
        assert_eq!(m.latched(), Some(cause));
        // Recovered scans, and a later NVTL breach, change nothing.
        for _ in 0..20 {
            assert_eq!(m.observe(0, 3.0), Some(cause));
        }
        for _ in 0..5 {
            assert_eq!(m.observe(0, 1.0), Some(cause));
        }
    }

    #[test]
    fn reset_clears_latch_and_window() {
        let mut m = monitor(5, 3, 2.5);
        for _ in 0..3 {
            m.observe(0, 1.0);
        }
        assert!(m.latched().is_some());
        m.reset();
        assert_eq!(m.latched(), None);
        assert_eq!(m.window_median(), None);
        // Stale low scores are gone: two good scans after reset do not fire,
        // and a full window of good scans stays clear.
        for _ in 0..3 {
            assert_eq!(m.observe(0, 3.0), None);
        }
    }

    #[test]
    fn non_finite_nvtl_counts_as_worst() {
        let mut m = monitor(0, 3, 2.5);
        m.observe(0, f64::NAN);
        m.observe(0, f64::NAN);
        assert!(m.observe(0, 3.0).is_some());
    }

    #[test]
    fn apply_to_forces_error_only_while_latched() {
        let mut m = monitor(5, 3, 2.5);
        let mut cat = DiagnosticCategory::new("scan_matching_status", "ndt");
        cat.set_level_and_message(DiagnosticLevel::Ok, "OK");
        m.observe(0, 3.0);
        m.apply_to(&mut cat);
        assert_eq!(cat.level(), DiagnosticLevel::Ok);

        m.observe(5, 3.0);
        let mut cat = DiagnosticCategory::new("scan_matching_status", "ndt");
        cat.set_level_and_message(DiagnosticLevel::Warn, "Oscillation detected");
        m.apply_to(&mut cat);
        assert_eq!(cat.level(), DiagnosticLevel::Error);
        assert!(cat.message().contains("skipping_publish_num 5 >= 5"));
        assert!(cat.message().contains("latched"));
    }

    #[test]
    fn apply_to_keeps_existing_error_message() {
        let mut m = monitor(5, 3, 2.5);
        m.observe(6, 3.0);
        let mut cat = DiagnosticCategory::new("scan_matching_status", "ndt");
        cat.set_level_and_message(DiagnosticLevel::Error, "Failed to transform sensor points");
        m.apply_to(&mut cat);
        assert_eq!(cat.level(), DiagnosticLevel::Error);
        assert!(cat.message().contains("skipping_publish_num 6 >= 5"));
        assert!(cat.message().contains("Failed to transform sensor points"));
    }

    #[test]
    fn nvtl_message_names_value_and_gate() {
        let d = Divergence::NvtlMedian {
            median: 2.31,
            gate: 2.5,
            window: 10,
        };
        let s = d.to_string();
        assert!(
            s.contains("median NVTL of last 10 scans 2.310 < 2.500"),
            "{s}"
        );
    }

    #[test]
    fn zero_window_is_clamped_to_one() {
        let mut m = monitor(0, 0, 2.5);
        assert!(m.observe(0, 1.0).is_some());
    }
}
