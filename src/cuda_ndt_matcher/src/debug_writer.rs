//! Centralized debug JSONL file I/O.
//!
//! All debug output goes through these functions to avoid duplicating
//! the env var lookup and file open/append logic.

#[cfg(feature = "debug-output")]
use std::fs::OpenOptions;
#[cfg(feature = "debug-output")]
use std::io::Write;

/// Default path for the debug JSONL file.
#[cfg(feature = "debug-output")]
const DEFAULT_DEBUG_FILE: &str = "/tmp/ndt_cuda_debug.jsonl";

/// Return the debug file path from `NDT_DEBUG_FILE` env var, or the default.
#[cfg(feature = "debug-output")]
pub(crate) fn debug_file_path() -> String {
    std::env::var("NDT_DEBUG_FILE").unwrap_or_else(|_| DEFAULT_DEBUG_FILE.to_string())
}

/// Truncate the debug file and write a `run_start` header.
/// Called once at node startup.
#[cfg(feature = "debug-output")]
pub(crate) fn clear_debug_file() {
    let path = debug_file_path();
    if let Ok(mut file) = std::fs::File::create(&path) {
        use std::time::SystemTime;
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let _ = writeln!(
            file,
            r#"{{"run_start": true, "unix_timestamp": {timestamp}}}"#
        );
    }
}

/// Append a single JSON line to the debug file.
#[cfg(feature = "debug-output")]
pub(crate) fn append_debug_line(json: &str) {
    let path = debug_file_path();
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(file, "{json}");
    }
}

/// Write an init-to-tracking transition time entry.
#[cfg(feature = "debug-output")]
pub(crate) fn write_init_to_tracking(elapsed_ms: f64) {
    let json = format!(r#"{{"type":"init_to_tracking","elapsed_ms":{elapsed_ms:.2}}}"#);
    append_debug_line(&json);
}
