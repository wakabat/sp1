#![allow(clippy::fn_to_numeric_cast)]

use super::{TranspilerBackend, CONTEXT};
use crate::{
    cache::InstrMapEntry, DebugFn, EcallHandler, ExternFn, JitFunction, JitMemory, RiscOperand,
    RiscRegister, RiscvTranspiler,
};
use dynasmrt::{
    dynasm,
    x64::{Rq, X64Relocation},
    DynasmApi, VecAssembler,
};
use std::io;

impl RiscvTranspiler for TranspilerBackend {
    fn new(
        program_size: usize,
        memory_size: usize,
        max_trace_size: u64,
        pc_start: u64,
        pc_base: u64,
        clk_bump: u64,
    ) -> Result<Self, std::io::Error> {
        if pc_start < pc_base {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "pc_start must be greater than pc_base",
            ));
        }

        let mut this = Self {
            inner: VecAssembler::<X64Relocation>::new(0),
            jump_table: Vec::with_capacity(program_size),
            memory_size,
            has_instructions: false,
            pc_base,
            pc_start,
            // Register a dummy ecall handler.
            ecall_handler: super::ecallk as _,
            control_flow_instruction_inserted: false,
            instruction_started: false,
            clk_bump,
            max_trace_size,
            may_early_exit: false,
            instr_map: Vec::with_capacity(program_size),
            has_embedded_host_calls: false,
        };

        // Handle calling conventions and save anything were gonna clobber.
        this.prologue();

        Ok(this)
    }

    fn register_ecall_handler(&mut self, handler: EcallHandler) {
        self.ecall_handler = handler;
    }

    fn start_instr(&mut self) {
        // We dont want to compile without a single jumpdest, otherwise we will sigsegv.
        self.has_instructions = true;

        // If the instruction has already started, then we are in a bad state.
        if self.instruction_started {
            panic!("start_instr called without calling end_instr");
        }

        // Push the offset of the jumpdest for this instruction.
        let offset = self.inner.offset();
        let instr_index = self.jump_table.len() as u32;
        self.jump_table.push(offset.0);

        // Record the start of this instruction's byte range.
        self.instr_map.push(InstrMapEntry {
            instr_index,
            riscv_pc: self.pc_base + (instr_index as u64) * 4,
            start_offset: offset.0 as u32,
            end_offset: offset.0 as u32, // updated in end_instr
        });

        // We are now "within" an instruction.
        self.instruction_started = true;
    }

    fn end_instr(&mut self) {
        // Add the base amount of cycles for the instruction.
        self.bump_clk();

        // If the instruction is branch, jal, jalr or ecall then we need to emit a jump to pc
        if self.control_flow_instruction_inserted {
            // If we have a control flow instruction that may early exit, we need to check if the
            // trace size has been exceeded.
            if self.may_early_exit {
                self.exit_if_trace_exceeds(self.max_trace_size);
            }

            self.jump_to_pc();
        } else {
            self.bump_pc(4);

            // We dont have a control flow insruction so we need to bump the pc first.
            if self.may_early_exit {
                self.exit_if_trace_exceeds(self.max_trace_size);
            }
        }

        // Record the end of this instruction's byte range.
        let end = self.inner.offset().0 as u32;
        if let Some(entry) = self.instr_map.last_mut() {
            entry.end_offset = end;
        }

        self.may_early_exit = false;
        self.control_flow_instruction_inserted = false;
        self.instruction_started = false;
    }

    fn finalize<M: JitMemory>(mut self) -> io::Result<JitFunction<M>> {
        self.epilogue();

        let code = self.inner.finalize().expect("failed to finalize x86 backend");

        debug_assert!(!code.is_empty(), "Got empty x86 code buffer");

        JitFunction::from_bytes(code, self.jump_table, self.memory_size, self.pc_start)
    }

    fn call_extern_fn(&mut self, fn_ptr: ExternFn) {
        self.has_embedded_host_calls = true;

        // Load the JitContext pointer into the argument register.
        dynasm! {
            self;
            .arch x64;
            mov rdi, Rq(CONTEXT)
        };

        self.call_extern_fn_raw(fn_ptr as _);
    }

    fn inspect_register(&mut self, reg: RiscRegister, handler: DebugFn) {
        self.has_embedded_host_calls = true;

        // Load into the argument register for the function call.
        self.emit_risc_operand_load(RiscOperand::Register(reg), Rq::RDI as u8);

        // Call the handler with the value of the register.
        self.call_extern_fn_raw(handler as _);
    }

    fn inspect_immediate(&mut self, imm: u64, handler: DebugFn) {
        self.has_embedded_host_calls = true;
        dynasm! {
            self;
            .arch x64;

            mov rdi, imm as i32
        }

        self.call_extern_fn_raw(handler as _);
    }
}

impl TranspilerBackend {
    /// Consume the backend and return both a [`JitFunction`] and a [`crate::cache::JitArtifact`].
    ///
    /// This is equivalent to calling [`RiscvTranspiler::finalize`] and then
    /// [`JitFunction::to_artifact`], but avoids a redundant copy of the jump table.
    pub fn finalize_with_artifact<M: JitMemory>(
        mut self,
    ) -> io::Result<(JitFunction<M>, crate::cache::JitArtifact)> {
        self.epilogue();

        let instr_map = std::mem::take(&mut self.instr_map);
        let has_host_calls = self.has_embedded_host_calls;
        let pc_base = self.pc_base;
        let max_trace_size = self.max_trace_size;
        let clk_bump = self.clk_bump;

        let code = self.inner.finalize().expect("failed to finalize x86 backend");
        debug_assert!(!code.is_empty(), "Got empty x86 code buffer");

        let func: JitFunction<M> =
            JitFunction::from_bytes(code, self.jump_table, self.memory_size, self.pc_start)?;
        let artifact =
            func.to_artifact(instr_map, pc_base, max_trace_size, clk_bump, has_host_calls);
        Ok((func, artifact))
    }

    /// Get the instruction map (RISC-V PC → x86-64 byte range).
    pub fn instr_map(&self) -> &[InstrMapEntry] {
        &self.instr_map
    }

    /// Whether the code has embedded absolute host function pointers.
    pub fn has_embedded_host_calls(&self) -> bool {
        self.has_embedded_host_calls
    }
}
