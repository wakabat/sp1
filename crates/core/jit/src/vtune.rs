//! Written by Gemini 3 to handle VTune integration.
#![allow(dead_code)]

use std::ffi::CString;
use std::os::raw::{c_char, c_uint, c_void};

// --- FFI Bindings to VTune JIT Profiling API ---
#[repr(C)]
#[allow(non_camel_case_types)]
enum iJIT_jvm_event {
    iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED = 13,
    iJVM_EVENT_TYPE_METHOD_UNLOAD_START = 14,
}

#[repr(C)]
#[allow(non_camel_case_types)]
struct iJIT_Method_Load {
    method_id: c_uint,
    method_name: *mut c_char,
    method_load_address: *mut c_void,
    method_size: c_uint,
    line_number_size: c_uint,
    line_number_table: *mut c_void,
    class_id: c_uint,
    class_file_name: *mut c_char,
    source_file_name: *mut c_char,
}

#[repr(C)]
#[allow(non_camel_case_types)]
enum iJIT_IsProfilingActiveFlags {
    iJIT_NOTHING_RUNNING = 0x0000,
    iJIT_SAMPLING_ON = 0x0001,
}

#[link(name = "jitprofiling")]
extern "C" {
    fn iJIT_NotifyEvent(
        event_type: iJIT_jvm_event,
        event_specific_data: *mut c_void,
    ) -> std::os::raw::c_int;
    fn iJIT_IsProfilingActive() -> iJIT_IsProfilingActiveFlags;
    fn iJIT_GetNewMethodID() -> c_uint;
}

// --- Safe Rust Abstraction ---

/// A zero-sized struct acting as the VTune JIT Profiler interface.
/// The existence of this struct guarantees profiling is active.
pub struct VtuneJitProfiler;

impl VtuneJitProfiler {
    /// Attempts to initialize the VTune profiler.
    /// Returns `Some(VtuneJitProfiler)` if VTune is attached and sampling.
    /// Returns `None` otherwise.
    pub fn new() -> Option<Self> {
        let status = unsafe { iJIT_IsProfilingActive() };
        if matches!(status, iJIT_IsProfilingActiveFlags::iJIT_SAMPLING_ON) {
            Some(Self)
        } else {
            None
        }
    }

    /// Registers a newly JIT-compiled function with VTune.
    /// Assumes profiling is active.
    pub fn register_function(
        &self,
        name: &str,
        start_address: *const u8,
        size: usize,
    ) -> Option<u32> {
        let method_id = unsafe { iJIT_GetNewMethodID() };
        let c_name =
            CString::new(name).unwrap_or_else(|_| CString::new("unknown_jit_func").unwrap());

        let mut method_load = iJIT_Method_Load {
            method_id,
            method_name: c_name.as_ptr() as *mut c_char,
            method_load_address: start_address as *mut c_void,
            method_size: size as c_uint,
            line_number_size: 0,
            line_number_table: std::ptr::null_mut(),
            class_id: 0,
            class_file_name: std::ptr::null_mut(),
            source_file_name: std::ptr::null_mut(),
        };

        let result = unsafe {
            iJIT_NotifyEvent(
                iJIT_jvm_event::iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED,
                &mut method_load as *mut _ as *mut c_void,
            )
        };

        if result == 1 {
            Some(method_id)
        } else {
            None
        }
    }

    /// Unregisters a JIT-compiled function.
    /// Assumes profiling is active.
    pub fn unregister_function(&self, method_id: u32) {
        let mut id = method_id;
        unsafe {
            iJIT_NotifyEvent(
                iJIT_jvm_event::iJVM_EVENT_TYPE_METHOD_UNLOAD_START,
                &mut id as *mut _ as *mut c_void,
            );
        }
    }
}
