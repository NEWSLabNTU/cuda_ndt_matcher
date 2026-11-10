//! Which CUDA toolkit's NVRTC this binary should load at runtime.
//!
//! Shared by `build.rs`, which acts on the answer, and the crate's tests, which
//! check it. Everything here is pure; probing the host is build.rs's job.

use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

/// A toolkit on the build host: its root, its CUDA version, and the
/// architectures its nvcc (and so its NVRTC, from the same release) can target.
pub struct Toolkit {
    pub root: PathBuf,
    pub version: (u32, u32),
    pub arches: BTreeSet<u32>,
}

/// What the binary should do about NVRTC.
#[derive(Debug, PartialEq, Eq)]
pub enum Choice {
    /// The library the loader already finds can compile for the device.
    KeepDefault,
    /// Embed this toolkit's library directory as the binary's rpath.
    Use(PathBuf),
    /// No installed toolkit can compile for the device under this driver.
    NoneFits,
}

/// Choose the NVRTC for a device of compute capability `device` (e.g. 120).
///
/// `driver` is the newest CUDA the installed driver supports, when known.
/// `default_root` is the toolkit whose `libnvrtc.so` the dynamic loader would
/// find with no help, when that could be identified.
pub fn choose(
    device: u32,
    driver: Option<(u32, u32)>,
    default_root: Option<&Path>,
    toolkits: &[Toolkit],
) -> Choice {
    // A toolkit newer than the driver emits PTX the driver refuses to load, so
    // knowing the architecture is not enough.
    let usable = |toolkit: &&Toolkit| {
        toolkit.arches.contains(&device) && driver.is_none_or(|driver| toolkit.version <= driver)
    };
    if let Some(root) = default_root
        && toolkits
            .iter()
            .filter(usable)
            .any(|toolkit| toolkit.root == root)
    {
        return Choice::KeepDefault;
    }
    match toolkits
        .iter()
        .filter(usable)
        .max_by_key(|toolkit| toolkit.version)
    {
        Some(toolkit) => Choice::Use(toolkit.root.clone()),
        None => Choice::NoneFits,
    }
}

/// `12.0` (nvidia-smi's compute_cap) as 120.
pub fn parse_compute_cap(text: &str) -> Option<u32> {
    let (major, minor) = text.trim().split_once('.')?;
    Some(major.parse::<u32>().ok()? * 10 + minor.parse::<u32>().ok()?)
}

/// `X.Y` at the start of `text`, as `(X, Y)`.
fn parse_version(text: &str) -> Option<(u32, u32)> {
    let (major, rest) = text.trim_start().split_once('.')?;
    let minor: String = rest.chars().take_while(char::is_ascii_digit).collect();
    Some((major.parse().ok()?, minor.parse().ok()?))
}

/// The `CUDA Version: 13.1` field of nvidia-smi's banner.
pub fn parse_driver_cuda(banner: &str) -> Option<(u32, u32)> {
    parse_version(banner.split_once("CUDA Version:")?.1)
}

/// `release 12.8` from `nvcc --version`.
pub fn parse_nvcc_release(text: &str) -> Option<(u32, u32)> {
    parse_version(text.split_once("release ")?.1)
}

/// The `compute_NN` lines of `nvcc --list-gpu-arch`.
pub fn parse_gpu_arches(text: &str) -> BTreeSet<u32> {
    text.lines()
        .filter_map(|line| line.trim().strip_prefix("compute_"))
        .filter_map(|digits| digits.parse().ok())
        .collect()
}

fn run(program: &Path, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(program)
        .args(args)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

fn probe_toolkit(root: &Path) -> Option<Toolkit> {
    let nvcc = root.join("bin").join("nvcc");
    let version = parse_nvcc_release(&run(&nvcc, &["--version"])?)?;
    let arches = parse_gpu_arches(&run(&nvcc, &["--list-gpu-arch"])?);
    let has_nvrtc = std::fs::read_dir(root.join("lib64"))
        .ok()?
        .flatten()
        .any(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with("libnvrtc.so")
        });
    (has_nvrtc && !arches.is_empty()).then(|| Toolkit {
        root: root.to_path_buf(),
        version,
        arches,
    })
}

/// Every toolkit under /usr/local plus CUDA_PATH / CUDA_HOME, by canonical root.
fn discover_toolkits() -> Vec<Toolkit> {
    let mut roots: Vec<PathBuf> = ["CUDA_PATH", "CUDA_HOME"]
        .iter()
        .filter_map(|var| std::env::var_os(var).map(PathBuf::from))
        .collect();
    roots.extend(
        std::fs::read_dir("/usr/local")
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("cuda"))
            }),
    );
    let mut seen = BTreeSet::new();
    roots
        .into_iter()
        .filter_map(|root| std::fs::canonicalize(root).ok())
        .filter(|root| seen.insert(root.clone()))
        .filter_map(|root| probe_toolkit(&root))
        .collect()
}

/// The toolkit whose `libnvrtc.so` the loader finds unaided, as cudarc asks for
/// it: LD_LIBRARY_PATH first, then the ld.so cache.
fn default_root(toolkits: &[Toolkit]) -> Option<PathBuf> {
    let from_env = std::env::var("LD_LIBRARY_PATH")
        .unwrap_or_default()
        .split(':')
        .filter(|dir| !dir.is_empty())
        .map(|dir| Path::new(dir).join("libnvrtc.so"))
        .find(|lib| lib.exists());
    let lib = from_env.or_else(|| {
        run(Path::new("ldconfig"), &["-p"])
            .or_else(|| run(Path::new("/sbin/ldconfig"), &["-p"]))?
            .lines()
            .filter(|line| line.trim_start().starts_with("libnvrtc.so "))
            .find_map(|line| {
                line.split_once("=> ")
                    .map(|(_, path)| PathBuf::from(path.trim()))
            })
    })?;
    let dir = std::fs::canonicalize(lib.parent()?).ok()?;
    toolkits
        .iter()
        .find(|toolkit| std::fs::canonicalize(toolkit.root.join("lib64")).ok() == Some(dir.clone()))
        .map(|toolkit| toolkit.root.clone())
}

/// For build.rs: make the binary load an NVRTC that can compile for this GPU.
///
/// CubeCL compiles every kernel at runtime with NVRTC for the device's exact
/// architecture (`sm_120a` on an RTX 5090), and cudarc loads NVRTC by bare
/// name, so the process gets whichever `libnvrtc.so` the loader finds first --
/// on a desktop with `/usr/local/cuda` at 12.6 that is a library which rejects
/// `sm_120` and every kernel launch panics. A running process cannot change
/// its own library search path, and neither cudarc nor CubeCL takes a path or
/// an architecture override, so the choice is made here: when the default
/// cannot serve the build host's GPU, the newest toolkit that can (and that
/// the driver accepts) is embedded as a DT_RPATH, which the loader searches
/// before LD_LIBRARY_PATH. cuSOLVER and cudart then come from the same
/// toolkit, which is also the one cuda_ffi built its kernels with.
///
/// When the default suffices (the AGX Orin), or there is no GPU or toolkit to
/// ask, nothing is emitted and the binary behaves exactly as before.
pub fn emit_rpath() {
    println!("cargo:rerun-if-env-changed=LD_LIBRARY_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    let smi = Path::new("nvidia-smi");
    let Some(device) = run(smi, &["--query-gpu=compute_cap", "--format=csv,noheader"])
        .and_then(|out| parse_compute_cap(out.lines().next()?))
    else {
        return;
    };
    let driver = run(smi, &[]).and_then(|banner| parse_driver_cuda(&banner));
    let toolkits = discover_toolkits();
    let default = default_root(&toolkits);

    match choose(device, driver, default.as_deref(), &toolkits) {
        Choice::KeepDefault => {}
        Choice::Use(root) => {
            let lib = root.join("lib64");
            println!(
                "cargo:warning=loading NVRTC from {} at runtime: the default cannot compile \
                 for this GPU (sm_{device})",
                lib.display()
            );
            println!(
                "cargo:rustc-link-arg=-Wl,--disable-new-dtags,-rpath,{}",
                lib.display()
            );
        }
        Choice::NoneFits => println!(
            "cargo:warning=no installed CUDA toolkit can compile for sm_{device} under this \
             driver; CubeCL kernels will fail to compile at runtime"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toolkit(version: (u32, u32), arches: &[u32]) -> Toolkit {
        Toolkit {
            root: PathBuf::from(format!("/usr/local/cuda-{}.{}", version.0, version.1)),
            version,
            arches: arches.iter().copied().collect(),
        }
    }

    /// This development desktop: /usr/local/cuda is 12.6, 12.8 also installed.
    fn desktop() -> Vec<Toolkit> {
        vec![
            toolkit((12, 8), &[75, 80, 86, 87, 89, 90, 100, 101, 120]),
            toolkit((12, 6), &[75, 80, 86, 87, 89, 90]),
            toolkit((12, 3), &[75, 80, 86, 87, 89, 90]),
        ]
    }

    #[test]
    fn rtx5090_gets_the_toolkit_that_knows_sm_120() {
        let choice = choose(
            120,
            Some((13, 1)),
            Some(Path::new("/usr/local/cuda-12.6")),
            &desktop(),
        );
        assert_eq!(choice, Choice::Use(PathBuf::from("/usr/local/cuda-12.8")));
    }

    #[test]
    fn adequate_default_is_left_alone() {
        // AGX Orin: compute 8.7 on CUDA 12.6, nothing else installed.
        let orin = vec![toolkit((12, 6), &[72, 75, 80, 86, 87, 89, 90])];
        let choice = choose(
            87,
            Some((12, 6)),
            Some(Path::new("/usr/local/cuda-12.6")),
            &orin,
        );
        assert_eq!(choice, Choice::KeepDefault);
        // Desktop with an older card: 12.6 compiles sm_89, keep it.
        let choice = choose(
            89,
            Some((13, 1)),
            Some(Path::new("/usr/local/cuda-12.6")),
            &desktop(),
        );
        assert_eq!(choice, Choice::KeepDefault);
    }

    #[test]
    fn toolkit_newer_than_the_driver_is_skipped() {
        // Its PTX would be rejected by the driver even though NVRTC knows the arch.
        let mut toolkits = desktop();
        toolkits.insert(0, toolkit((13, 2), &[75, 80, 86, 87, 89, 90, 100, 120]));
        let choice = choose(
            120,
            Some((13, 1)),
            Some(Path::new("/usr/local/cuda-12.6")),
            &toolkits,
        );
        assert_eq!(choice, Choice::Use(PathBuf::from("/usr/local/cuda-12.8")));
        // And a default newer than the driver is not "adequate" either.
        let choice = choose(
            89,
            Some((12, 7)),
            Some(Path::new("/usr/local/cuda-12.8")),
            &desktop(),
        );
        assert_eq!(choice, Choice::Use(PathBuf::from("/usr/local/cuda-12.6")));
    }

    #[test]
    fn newest_fitting_toolkit_wins_regardless_of_order() {
        let mut toolkits = desktop();
        toolkits.reverse();
        let choice = choose(89, None, None, &toolkits);
        assert_eq!(choice, Choice::Use(PathBuf::from("/usr/local/cuda-12.8")));
    }

    #[test]
    fn nothing_fits() {
        let old = vec![toolkit((12, 6), &[75, 80, 86, 87, 89, 90])];
        let choice = choose(
            120,
            Some((13, 1)),
            Some(Path::new("/usr/local/cuda-12.6")),
            &old,
        );
        assert_eq!(choice, Choice::NoneFits);
    }

    #[test]
    fn parses_tool_output() {
        assert_eq!(parse_compute_cap("12.0\n"), Some(120));
        assert_eq!(parse_compute_cap(" 8.7"), Some(87));
        assert_eq!(parse_compute_cap("[N/A]"), None);
        let banner =
            "| NVIDIA-SMI 590.48.01   Driver Version: 590.48.01   CUDA Version: 13.1     |";
        assert_eq!(parse_driver_cuda(banner), Some((13, 1)));
        assert_eq!(parse_driver_cuda("no version here"), None);
        let nvcc = "nvcc: NVIDIA (R) Cuda compiler driver\n\
                    Cuda compilation tools, release 12.8, V12.8.61\n";
        assert_eq!(parse_nvcc_release(nvcc), Some((12, 8)));
        assert_eq!(
            parse_gpu_arches("compute_87\ncompute_120\n"),
            BTreeSet::from([87, 120])
        );
    }
}
