//! Written by Gemini 3 to handle Linux perf integration.
//! TODO: this is still breaking and requires debugging.
#![allow(dead_code)]

use std::env;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::os::unix::io::AsRawFd;
use std::process;
use std::sync::Mutex;

// --- JITDUMP Binary Specifications ---

const JITHEADER_MAGIC: u32 = 0x4A695444; // "JiTD"
const JITHEADER_VERSION: u32 = 1;
const JIT_CODE_LOAD: u32 = 0;

#[cfg(target_arch = "x86_64")]
const ELF_MACH: u32 = 62; // EM_X86_64

#[cfg(target_arch = "aarch64")]
const ELF_MACH: u32 = 183; // EM_AARCH64

#[repr(C)]
struct JitHeader {
    magic: u32,
    version: u32,
    total_size: u32,
    elf_mach: u32,
    pad1: u32,
    pid: u32,
    timestamp: u64,
    flags: u64,
}

#[repr(C)]
struct JrPrefix {
    id: u32,
    total_size: u32,
    timestamp: u64,
}

#[repr(C)]
struct JrCodeLoad {
    p: JrPrefix,
    pid: u32,
    tid: u32,
    vma: u64,
    code_addr: u64,
    code_size: u64,
    code_index: u64,
}

// Helper to safely cast structs to byte slices for writing
unsafe fn as_bytes<T>(data: &T) -> &[u8] {
    std::slice::from_raw_parts((data as *const T) as *const u8, std::mem::size_of::<T>())
}

// Fetch the monotonic clock time exactly how `perf` expects it
fn get_timestamp() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    (ts.tv_sec as u64) * 1_000_000_000 + (ts.tv_nsec as u64)
}

// --- Safe Rust Abstraction ---

/// A JITDUMP profiler integration for Linux `perf`.
/// The existence of this struct guarantees the `.dump` file is open and mmap'd.
pub struct PerfJitProfiler {
    file: Mutex<File>,
    code_index: Mutex<u64>,
    _mmap_ptr: *mut libc::c_void,
    mmap_size: usize,
}

impl PerfJitProfiler {
    /// Attempts to initialize the Perf JITDUMP Profiler.
    /// Returns `Ok(Some(PerfJitDumpProfiler))` if `ENABLE_PERF_JITDUMP` is "1".
    pub fn new() -> io::Result<Option<Self>> {
        if env::var("ENABLE_PERF_JIT").unwrap_or_default() != "1" {
            return Ok(None);
        }

        let pid = process::id();
        let path = format!("/tmp/jit-{}.dump", pid);

        let mut file =
            OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&path)?;

        let header = JitHeader {
            magic: JITHEADER_MAGIC,
            version: JITHEADER_VERSION,
            total_size: std::mem::size_of::<JitHeader>() as u32,
            elf_mach: ELF_MACH,
            pad1: 0,
            pid,
            timestamp: get_timestamp(),
            flags: 0,
        };

        // Write the header to disk
        file.write_all(unsafe { as_bytes(&header) })?;
        file.flush()?;

        // CRITICAL STEP: We must mmap the header so `perf` detects the JIT engine.
        let fd = file.as_raw_fd();
        let mmap_size = std::mem::size_of::<JitHeader>();

        let mmap_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mmap_size,
                libc::PROT_READ | libc::PROT_EXEC,
                libc::MAP_PRIVATE,
                fd,
                0,
            )
        };

        if mmap_ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        Ok(Some(Self {
            file: Mutex::new(file),
            code_index: Mutex::new(0),
            _mmap_ptr: mmap_ptr,
            mmap_size,
        }))
    }

    /// Registers a newly JIT-compiled function with `perf`, writing the raw bytes.
    pub fn register_function(
        &self,
        name: &str,
        start_address: *const u8,
        size: usize,
    ) -> io::Result<()> {
        let mut file = self.file.lock().unwrap();
        let mut index = self.code_index.lock().unwrap();
        *index += 1;

        // The size of the record includes the struct + the string + null terminator + the code bytes
        let name_bytes = name.as_bytes();
        let name_len = name_bytes.len() + 1; // +1 for null terminator
        let total_size = std::mem::size_of::<JrCodeLoad>() as u32 + name_len as u32 + size as u32;

        let record = JrCodeLoad {
            p: JrPrefix { id: JIT_CODE_LOAD, total_size, timestamp: get_timestamp() },
            pid: process::id(),
            tid: unsafe { libc::syscall(libc::SYS_gettid) as u32 }, // Get OS thread ID
            vma: start_address as u64,
            code_addr: start_address as u64,
            code_size: size as u64,
            code_index: *index,
        };

        // 1. Write the record header
        file.write_all(unsafe { as_bytes(&record) })?;

        // 2. Write the null-terminated name
        file.write_all(name_bytes)?;
        file.write_all(&[0])?;

        // 3. Write the actual machine code bytes
        let code_slice = unsafe { std::slice::from_raw_parts(start_address, size) };
        file.write_all(code_slice)?;

        file.flush()?;
        Ok(())
    }
}

impl Drop for PerfJitProfiler {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self._mmap_ptr, self.mmap_size);
        }
    }
}
