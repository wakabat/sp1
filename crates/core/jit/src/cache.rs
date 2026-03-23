//! JIT cache: save and load transpiled machine code.
//!
//! A [`JitArtifact`] captures everything needed to re-execute previously transpiled code
//! without running the transpiler again: the raw x86-64 bytes, the jump-table offsets, and
//! an instruction map that records which RISC-V PC generated each byte range.
//!
//! The artifact can be:
//! - serialized/deserialized with bincode for on-disk caching
//! - loaded back into a [`JitFunction`] for re-execution
//! - exported as an ELF object file with DWARF debug line info (feature `cache`)

use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{self, Write},
    path::Path,
};

/// Magic bytes for the cache file header.
const MAGIC: &[u8; 8] = b"SP1JIT\x00\x01";

/// A single entry mapping a RISC-V instruction to its x86-64 byte range.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct InstrMapEntry {
    /// Index of the instruction in the program.
    pub instr_index: u32,
    /// The RISC-V PC of this instruction.
    pub riscv_pc: u64,
    /// Start byte offset in the code buffer (inclusive).
    pub start_offset: u32,
    /// End byte offset in the code buffer (exclusive).
    pub end_offset: u32,
}

/// A serializable artifact capturing a finalized JIT compilation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitArtifact {
    /// Format version for forward compatibility.
    pub format_version: u32,
    /// The raw x86-64 machine code bytes.
    pub code: Vec<u8>,
    /// Jump table as offsets into `code`.
    pub jump_table_offsets: Vec<u32>,
    /// Mapping from RISC-V instructions to x86-64 byte ranges.
    pub instr_map: Vec<InstrMapEntry>,
    /// VM memory buffer size in bytes.
    pub memory_size: u64,
    /// Starting PC of the program.
    pub pc_start: u64,
    /// Base PC for address calculations.
    pub pc_base: u64,
    /// Maximum trace size used during compilation.
    pub max_trace_size: u64,
    /// Clock bump per instruction.
    pub clk_bump: u64,
    /// Whether the code contains embedded absolute host function pointers.
    ///
    /// When true, the artifact is **not portable** across process restarts because
    /// ASLR may relocate the host functions. The ecall handler is always loaded
    /// indirectly from the JitContext and is therefore safe.
    pub has_host_call_relocations: bool,
}

impl JitArtifact {
    /// Save the artifact to a file using bincode serialization.
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut file = fs::File::create(path)?;
        file.write_all(MAGIC)?;
        let encoded = bincode::serialize(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        file.write_all(&encoded)?;
        Ok(())
    }

    /// Load an artifact from a file.
    pub fn load_from_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let data = fs::read(path)?;
        if data.len() < 8 || &data[..8] != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid JIT cache magic"));
        }
        bincode::deserialize(&data[8..])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Create a `JitFunction` from this cached artifact.
    ///
    /// The code is mapped into executable memory and the jump table is reconstructed.
    #[cfg(sp1_native_executor_available)]
    pub fn into_jit_function<M: crate::JitMemory>(self) -> io::Result<crate::JitFunction<M>> {
        use memmap2::MmapOptions;

        // Map the code into executable memory.
        let mut mmap = MmapOptions::new().len(self.code.len()).map_anon()?;
        mmap.copy_from_slice(&self.code);
        let exec = mmap.make_exec().map_err(|e| {
            io::Error::new(io::ErrorKind::PermissionDenied, format!("mmap make_exec: {e}"))
        })?;

        let base = exec.as_ptr();
        let jump_table: Vec<*const u8> = self
            .jump_table_offsets
            .iter()
            .map(|&off| unsafe { base.add(off as usize) })
            .collect();

        let memory = M::new(self.memory_size as usize);

        Ok(crate::JitFunction {
            jump_table,
            code: crate::ExecutableCode::Mmap(exec),
            memory,
            pc: self.pc_start,
            pc_start: self.pc_start,
            clk: 1,
            global_clk: 0,
            registers: [0; 32],
            initial_memory_image: std::sync::Arc::new(hashbrown::HashMap::new()),
            input_buffer: std::collections::VecDeque::new(),
            hints: Vec::new(),
            public_values_stream: Vec::new(),
            debug_sender: None,
            exit_code: 0,
        })
    }

    /// Write the artifact as an ELF object file with DWARF debug line info.
    ///
    /// The resulting ELF contains:
    /// - `.text` section with the JIT code
    /// - A `sp1_jit_entry` symbol at offset 0
    /// - `.debug_line` section mapping x86-64 offsets to RISC-V PCs
    ///
    /// This is a debug/inspection object, not a standalone executable.
    #[cfg(feature = "cache")]
    pub fn save_as_elf(&self, path: impl AsRef<Path>) -> io::Result<()> {
        use object::write::{Object, StandardSection, Symbol, SymbolSection};
        use object::{Architecture, BinaryFormat, Endianness, SymbolFlags, SymbolKind, SymbolScope};

        let mut obj = Object::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);

        // Add .text section with the JIT code.
        let text_section = obj.section_id(StandardSection::Text);
        obj.append_section_data(text_section, &self.code, 16);

        // Add the entry point symbol.
        obj.add_symbol(Symbol {
            name: b"sp1_jit_entry".to_vec(),
            value: 0,
            size: self.code.len() as u64,
            kind: SymbolKind::Text,
            scope: SymbolScope::Dynamic,
            weak: false,
            section: SymbolSection::Section(text_section),
            flags: SymbolFlags::None,
        });

        // Add per-instruction symbols for navigation.
        for entry in &self.instr_map {
            let name = format!("riscv_pc_{:#x}", entry.riscv_pc);
            obj.add_symbol(Symbol {
                name: name.into_bytes(),
                value: entry.start_offset as u64,
                size: (entry.end_offset - entry.start_offset) as u64,
                kind: SymbolKind::Text,
                scope: SymbolScope::Compilation,
                weak: false,
                section: SymbolSection::Section(text_section),
                flags: SymbolFlags::None,
            });
        }

        // Build DWARF .debug_line section manually.
        //
        // We create a minimal DWARF v4 line number program that maps each x86-64
        // instruction range to a "line number" equal to the RISC-V PC. The synthetic
        // source file is "riscv_pc.map".
        let debug_line = build_debug_line_section(&self.instr_map);
        let debug_line_section = obj.add_section(
            Vec::new(),
            b".debug_line".to_vec(),
            object::SectionKind::Debug,
        );
        obj.append_section_data(debug_line_section, &debug_line, 1);

        let data = obj.write().map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        fs::write(path, data)
    }
}

/// Build a minimal DWARF v4 `.debug_line` section.
///
/// Maps each instruction's x86-64 byte range to a line number equal to the RISC-V PC.
/// Uses the synthetic file name `riscv_pc.map`.
#[cfg(feature = "cache")]
fn build_debug_line_section(instr_map: &[InstrMapEntry]) -> Vec<u8> {
    let mut buf = Vec::new();

    // We'll write the header, then patch the total length at the end.
    let length_pos = buf.len();
    buf.extend_from_slice(&0u32.to_le_bytes()); // placeholder for unit_length

    // DWARF version 4
    buf.extend_from_slice(&4u16.to_le_bytes());

    let header_length_pos = buf.len();
    buf.extend_from_slice(&0u32.to_le_bytes()); // placeholder for header_length

    let header_start = buf.len();

    // minimum_instruction_length
    buf.push(1);
    // maximum_operations_per_instruction (DWARF4)
    buf.push(1);
    // default_is_stmt
    buf.push(1);
    // line_base
    buf.push(0u8); // 0 as i8
    // line_range
    buf.push(1);
    // opcode_base
    buf.push(13);
    // standard_opcode_lengths (opcodes 1-12)
    buf.extend_from_slice(&[0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1]);

    // include_directories: empty (terminated by null byte)
    buf.push(0);

    // file_names: one entry "riscv_pc.map"
    buf.extend_from_slice(b"riscv_pc.map\0"); // file name
    buf.push(0); // directory index (0 = compilation dir)
    buf.push(0); // last modification time
    buf.push(0); // file size
    // terminate file names
    buf.push(0);

    // Patch header_length
    let header_length = (buf.len() - header_start) as u32;
    buf[header_length_pos..header_length_pos + 4].copy_from_slice(&header_length.to_le_bytes());

    // Line number program opcodes.
    // We use DW_LNS_set_file(1), then for each instruction:
    //   DW_LNS_extended_op -> DW_LNE_set_address(addr)
    //   DW_LNS_advance_line(delta)
    //   DW_LNS_copy

    // Set file to 1
    buf.push(4); // DW_LNS_set_file
    encode_uleb128(&mut buf, 1);

    let mut current_line: i64 = 1; // DWARF line starts at 1

    for entry in instr_map {
        // DW_LNS_extended_op: set address
        buf.push(0); // extended opcode marker
        let addr_bytes = 1 + 8; // 1 byte opcode + 8 byte address
        encode_uleb128(&mut buf, addr_bytes);
        buf.push(2); // DW_LNE_set_address
        buf.extend_from_slice(&(entry.start_offset as u64).to_le_bytes());

        // Advance line to riscv_pc. We use the PC value as the line number.
        // Line numbers must be positive, so we use (riscv_pc + 1) to avoid line 0.
        let target_line = entry.riscv_pc as i64 + 1;
        let delta = target_line - current_line;
        if delta != 0 {
            buf.push(3); // DW_LNS_advance_line
            encode_sleb128(&mut buf, delta);
        }
        current_line = target_line;

        // DW_LNS_copy - emit a row
        buf.push(1);
    }

    // End sequence
    buf.push(0); // extended opcode marker
    encode_uleb128(&mut buf, 1);
    buf.push(1); // DW_LNE_end_sequence

    // Patch unit_length (total length minus the 4-byte length field itself)
    let unit_length = (buf.len() - length_pos - 4) as u32;
    buf[length_pos..length_pos + 4].copy_from_slice(&unit_length.to_le_bytes());

    buf
}

/// Encode a u64 as ULEB128.
#[cfg(feature = "cache")]
fn encode_uleb128(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Encode an i64 as SLEB128.
#[cfg(feature = "cache")]
fn encode_sleb128(buf: &mut Vec<u8>, mut value: i64) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        let more = !(((value == 0) && (byte & 0x40 == 0)) || ((value == -1) && (byte & 0x40 != 0)));
        if more {
            byte |= 0x80;
        }
        buf.push(byte);
        if !more {
            break;
        }
    }
}
