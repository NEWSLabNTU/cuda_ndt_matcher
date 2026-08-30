//! Degeneracy monitoring for the NDT registration problem.
//!
//! A scan matcher can converge, report a good score, and still be badly wrong,
//! because the geometry it was given did not constrain every degree of freedom.
//! A corridor constrains everything except travel along it; a tunnel or an open
//! field can leave more than one direction free. The optimizer has no way to say
//! so from its score: the score measures how well the points it *did* match fit,
//! not whether the remaining directions were pinned down at all.
//!
//! What does say so is the Hessian, which is the information matrix of the
//! problem. A direction the scan does not constrain shows up as a small
//! eigenvalue, and its eigenvector names the direction. This is the standard
//! treatment of degeneracy in registration-based state estimation, and it costs
//! a 6x6 symmetric eigendecomposition per frame -- nothing next to the alignment
//! it is describing.
//!
//! Motivation here is the Seyond Robin-W: a 120 degree forward field of view
//! sees far less of the scene than the spinning sensors this stack was tuned on,
//! so it is far likelier to meet geometry that constrains only some axes. See
//! `docs/research/localization/restricted-fov-ndt.md` in the superproject.
//!
//! # Why translation and rotation are reported separately
//!
//! The obvious summary of a Hessian's conditioning is the ratio of its largest
//! to smallest eigenvalue. **For this matrix that number is meaningless**, and
//! reporting it would be worse than reporting nothing.
//!
//! The 6x6 Hessian mixes units. Its translation block carries inverse metres
//! squared and its rotation block inverse radians squared, so the eigenvalues of
//! the whole matrix are not commensurable and their ratio depends on the choice
//! of length unit. Rescaling the map from metres to centimetres would change the
//! "condition number" by a factor of ten thousand while describing exactly the
//! same geometry, and any threshold set on it would silently mean something
//! different on a different map.
//!
//! So the 3x3 diagonal blocks are decomposed separately. Within a block the
//! units are uniform and the ratio is a real statement about anisotropy. The
//! blocks' absolute eigenvalues still scale with the number of correspondences,
//! which is why the ratio is the primary signal and the minimum is reported
//! beside it as context rather than as something to threshold blindly.

use nalgebra::{Matrix3, Matrix6, Vector3};

/// Conditioning of one 3x3 block of the Hessian.
pub(crate) struct BlockConditioning {
    /// Smallest eigenvalue: how well the worst-constrained axis is pinned down.
    /// Scales with correspondence count, so it is context, not a threshold.
    pub(crate) min_eigenvalue: f64,
    /// Largest over smallest. Unit-free within the block, and the number to
    /// watch: it says how much better the best-constrained axis is than the
    /// worst, regardless of how many points were matched.
    pub(crate) anisotropy: f64,
    /// Eigenvector of the smallest eigenvalue, in the frame the Hessian was
    /// built in. Names the direction that is poorly observed.
    pub(crate) weakest_axis: Vector3<f64>,
}

/// Degeneracy report for one alignment.
pub(crate) struct Degeneracy {
    pub(crate) translation: BlockConditioning,
    pub(crate) rotation: BlockConditioning,
}

fn condition_block(block: Matrix3<f64>) -> BlockConditioning {
    // Symmetrise before decomposing. The Hessian is symmetric in exact
    // arithmetic but is accumulated on the GPU in f32 and converted, so the two
    // triangles differ in the last bits. `symmetric_eigen` reads only one
    // triangle and would silently use whichever it picked.
    let sym = (block + block.transpose()) * 0.5;
    let eigen = sym.symmetric_eigen();

    let (mut lo, mut hi, mut lo_idx) = (f64::INFINITY, f64::NEG_INFINITY, 0usize);
    for i in 0..3 {
        let v = eigen.eigenvalues[i];
        if v < lo {
            lo = v;
            lo_idx = i;
        }
        if v > hi {
            hi = v;
        }
    }

    // A non-positive or denormal minimum means the direction is not merely
    // poorly constrained but unconstrained, and the ratio would be infinite or
    // negative. Report it as infinite anisotropy rather than emitting a value
    // that compares wrongly against a threshold.
    let anisotropy = if lo > f64::MIN_POSITIVE {
        hi / lo
    } else {
        f64::INFINITY
    };

    BlockConditioning {
        min_eigenvalue: lo,
        anisotropy,
        weakest_axis: eigen.eigenvectors.column(lo_idx).into(),
    }
}

/// Decompose an NDT Hessian into per-block conditioning.
///
/// Takes the row-major array the alignment result carries. The convention that
/// rows 0..3 are translation and 3..6 rotation follows the optimizer that filled
/// it; the cross blocks are deliberately ignored, because a block-diagonal read
/// is the only one whose units are consistent.
///
/// **The input is negated.** NDT maximises its score, so the Hessian it reports
/// is negative definite and the information matrix is `-H`. `covariance.rs` does
/// the same negation for the Laplace covariance and says so. Reading the raw
/// Hessian instead makes every eigenvalue negative, which sends every frame down
/// the "unconstrained" branch below and reports infinite anisotropy always --
/// measured, on both a full-circle and a 120 degree run, before this negation
/// was added.
pub(crate) fn analyze(hessian: &[[f64; 6]; 6]) -> Degeneracy {
    let h = Matrix6::from_fn(|i, j| -hessian[i][j]);
    Degeneracy {
        translation: condition_block(h.fixed_view::<3, 3>(0, 0).into_owned()),
        rotation: condition_block(h.fixed_view::<3, 3>(3, 3).into_owned()),
    }
}

/// Name the dominant axis of an eigenvector, for a log line a human can act on.
///
/// "Weakly constrained along x" is actionable; six floats are not.
pub(crate) fn dominant_axis(v: &Vector3<f64>, labels: [&'static str; 3]) -> &'static str {
    let mut best = 0usize;
    for i in 1..3 {
        if v[i].abs() > v[best].abs() {
            best = i;
        }
    }
    labels[best]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a Hessian in NDT's own sign convention from the information blocks
    /// a caller thinks in. Negated, because `analyze` negates back.
    fn hessian_from_blocks(t: Matrix3<f64>, r: Matrix3<f64>) -> [[f64; 6]; 6] {
        let mut h = [[0.0f64; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] = -t[(i, j)];
                h[i + 3][j + 3] = -r[(i, j)];
            }
        }
        h
    }

    #[test]
    fn a_negative_definite_hessian_is_not_read_as_degenerate() {
        // The bug this guards: NDT maximises, so its Hessian is negative
        // definite, and reading it unnegated made every frame report infinite
        // anisotropy on a well-constrained full-circle scan.
        let raw = [[0.0f64; 6]; 6];
        let mut h = raw;
        for i in 0..6 {
            h[i][i] = -3.0;
        }
        let d = analyze(&h);
        assert!(d.translation.min_eigenvalue > 0.0);
        assert!(d.translation.anisotropy.is_finite());
        assert!(d.rotation.min_eigenvalue > 0.0);
    }

    #[test]
    fn isotropic_geometry_has_unit_anisotropy() {
        let h = hessian_from_blocks(Matrix3::identity() * 5.0, Matrix3::identity() * 2.0);
        let d = analyze(&h);
        assert!((d.translation.anisotropy - 1.0).abs() < 1e-9);
        assert!((d.translation.min_eigenvalue - 5.0).abs() < 1e-9);
        assert!((d.rotation.anisotropy - 1.0).abs() < 1e-9);
    }

    #[test]
    fn corridor_shows_as_one_weak_translation_axis() {
        // Well constrained across the corridor and vertically, barely
        // constrained along it. This is the case the monitor exists for.
        let t = Matrix3::from_diagonal(&Vector3::new(0.01, 100.0, 100.0));
        let d = analyze(&hessian_from_blocks(t, Matrix3::identity()));
        assert!((d.translation.anisotropy - 10000.0).abs() < 1e-6);
        assert_eq!(
            dominant_axis(&d.translation.weakest_axis, ["x", "y", "z"]),
            "x"
        );
    }

    #[test]
    fn unconstrained_axis_reports_infinite_anisotropy() {
        // Zero, not merely small: the ratio must not come back as a finite
        // number that would compare as "fine" against a threshold.
        let t = Matrix3::from_diagonal(&Vector3::new(0.0, 1.0, 1.0));
        let d = analyze(&hessian_from_blocks(t, Matrix3::identity()));
        assert!(d.translation.anisotropy.is_infinite());
    }

    #[test]
    fn asymmetry_from_f32_accumulation_does_not_change_the_answer() {
        let mut h = hessian_from_blocks(
            Matrix3::from_diagonal(&Vector3::new(1.0, 4.0, 9.0)),
            Matrix3::identity(),
        );
        let clean = analyze(&h).translation.anisotropy;
        h[0][1] -= 1e-7;
        assert!((analyze(&h).translation.anisotropy - clean).abs() < 1e-3);
    }
}
