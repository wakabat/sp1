use serde::{Deserialize, Serialize};

/// Cached JIT compilation result that can be saved to disk and reloaded,
/// or embedded in a static binary for AOT execution.
///
/// The code bytes contain x86_64 machine code with embedded absolute function
/// pointer addresses that need relocation when loaded at a different address.
/// The `fn_relocations` field records the offsets of these 8-byte immediates.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct JitCache {
    /// Raw x86_64 machine code bytes.
    pub code: Vec<u8>,
    /// Offsets into `code` for each RISC-V instruction's entry point.
    pub jump_table_offsets: Vec<usize>,
    /// Size of the VM memory buffer (in bytes).
    pub memory_size: usize,
    /// Starting program counter value.
    pub pc_start: u64,
    /// Offsets within `code` where 8-byte function pointer immediates reside.
    /// These must be patched with the actual ecall handler address on load.
    pub fn_relocations: Vec<usize>,
}

impl JitCache {
    /// Generate Rust source code that embeds this cache as static data for AOT compilation.
    ///
    /// The generated module exports:
    /// - `CODE: &[u8]` - the machine code bytes
    /// - `JUMP_TABLE_OFFSETS: &[usize]` - jump table offsets
    /// - `FN_RELOCATIONS: &[usize]` - relocation offsets
    /// - `MEMORY_SIZE: usize`
    /// - `PC_START: u64`
    ///
    /// Usage: write the returned string to a `.rs` file, then `include!()` it
    /// or add it as a module in your build script.
    #[must_use]
    pub fn generate_aot_source(&self) -> String {
        let mut s = String::new();

        s.push_str("/// Auto-generated AOT JIT cache. Do not edit.\n\n");

        // Code bytes
        s.push_str(&format!("pub static CODE: &[u8] = &{:?};\n\n", self.code.as_slice()));

        // Jump table offsets
        s.push_str("pub static JUMP_TABLE_OFFSETS: &[usize] = &[\n");
        for offset in &self.jump_table_offsets {
            s.push_str(&format!("    {offset},\n"));
        }
        s.push_str("];\n\n");

        // Fn relocations
        s.push_str("pub static FN_RELOCATIONS: &[usize] = &[\n");
        for offset in &self.fn_relocations {
            s.push_str(&format!("    {offset},\n"));
        }
        s.push_str("];\n\n");

        s.push_str(&format!("pub const MEMORY_SIZE: usize = {};\n\n", self.memory_size));
        s.push_str(&format!("pub const PC_START: u64 = {};\n", self.pc_start));

        s
    }
}

#[cfg(sp1_native_executor_available)]
mod load {
    use super::JitCache;
    use crate::{EcallHandler, JitFunction, JitMemory};
    use memmap2::MmapMut;
    use std::io;

    impl JitCache {
        /// Load cached code into executable memory, patching relocations with the
        /// given ecall handler, and return a ready-to-use [`JitFunction`].
        pub fn load<M: JitMemory>(
            &self,
            ecall_handler: EcallHandler,
        ) -> io::Result<JitFunction<M>> {
            Self::load_code(
                &self.code,
                &self.jump_table_offsets,
                &self.fn_relocations,
                self.memory_size,
                self.pc_start,
                ecall_handler,
            )
        }

        /// Load from static AOT data (e.g. from `generate_aot_source()` output).
        pub fn load_static<M: JitMemory>(
            code: &[u8],
            jump_table_offsets: &[usize],
            fn_relocations: &[usize],
            memory_size: usize,
            pc_start: u64,
            ecall_handler: EcallHandler,
        ) -> io::Result<JitFunction<M>> {
            Self::load_code(
                code,
                jump_table_offsets,
                fn_relocations,
                memory_size,
                pc_start,
                ecall_handler,
            )
        }

        fn load_code<M: JitMemory>(
            code: &[u8],
            jump_table_offsets: &[usize],
            fn_relocations: &[usize],
            memory_size: usize,
            pc_start: u64,
            ecall_handler: EcallHandler,
        ) -> io::Result<JitFunction<M>> {
            let mut mmap = MmapMut::map_anon(code.len())?;
            mmap.copy_from_slice(code);

            let handler_bytes = (ecall_handler as usize).to_le_bytes();
            for &offset in fn_relocations {
                mmap[offset..offset + 8].copy_from_slice(&handler_bytes);
            }

            let exec_mmap = mmap.make_exec()?;

            JitFunction::from_cached(exec_mmap, jump_table_offsets.to_vec(), memory_size, pc_start)
        }
    }
}
