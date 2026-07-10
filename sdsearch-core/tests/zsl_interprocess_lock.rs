//! Validates that the write-lock excludes writers in separate PROCESSES (not just threads).
//! Uses std::process::Command (cross-platform, no fork/libc) to spawn the
//! `stream_writer hold-lock` example as real child processes over the same index dir.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn temp_kb() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("sdsearch_iplock_{}_{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).unwrap();
    let src = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/zsl_index_kb"
    ));
    for f in std::fs::read_dir(&src).unwrap() {
        let f = f.unwrap().path();
        let name = f.file_name().unwrap().to_string_lossy();
        if name.contains("lock") || name.ends_with(".sti") {
            continue;
        }
        std::fs::copy(&f, dir.join(f.file_name().unwrap())).unwrap();
    }
    dir
}

/// Path to the already-compiled example binary (cargo test leaves it in target/<profile>/examples).
fn example_bin() -> PathBuf {
    // CARGO_BIN_EXE_ does not apply to examples; use the test exe's dir.
    let mut p = std::env::current_exe().unwrap();
    p.pop(); // .../deps
    if p.ends_with("deps") {
        p.pop();
    }
    p.join(if cfg!(windows) {
        "examples\\stream_writer.exe"
    } else {
        "examples/stream_writer"
    })
}

#[test]
fn write_lock_excludes_a_second_process() {
    let dir = temp_kb();
    let bin = example_bin();
    assert!(
        bin.is_file(),
        "build the example: cargo build -p sdsearch-core --example stream_writer"
    );

    // Process A: takes the lock and holds it for 1500 ms.
    let mut a = Command::new(&bin)
        .args(["hold-lock", dir.to_str().unwrap(), "1500"])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();

    // Wait for A to print LOCK_ACQUIRED (handshake over stdout, no Unix signals).
    let mut a_out = a.stdout.take().unwrap();
    let mut buf = [0u8; 32];
    let nread = a_out.read(&mut buf).unwrap();
    assert!(
        String::from_utf8_lossy(&buf[..nread]).contains("LOCK_ACQUIRED"),
        "A did not take the lock"
    );

    // Process B: tries to take the lock while A holds it → must fail with exit 3.
    let b = Command::new(&bin)
        .args(["hold-lock", dir.to_str().unwrap(), "0"])
        .stdout(Stdio::piped())
        .output()
        .unwrap();
    assert_eq!(
        b.status.code(),
        Some(3),
        "B should receive WouldBlock (exit 3), stdout={}",
        String::from_utf8_lossy(&b.stdout)
    );
    assert!(String::from_utf8_lossy(&b.stdout).contains("LOCK_WOULDBLOCK"));

    // A releases on exit; then C must be able to take it.
    a.wait().unwrap();
    let c = Command::new(&bin)
        .args(["hold-lock", dir.to_str().unwrap(), "0"])
        .stdout(Stdio::piped())
        .output()
        .unwrap();
    assert_eq!(
        c.status.code(),
        Some(0),
        "C should take the lock after A releases it"
    );
    assert!(String::from_utf8_lossy(&c.stdout).contains("LOCK_ACQUIRED"));

    std::fs::remove_dir_all(&dir).ok();
}
