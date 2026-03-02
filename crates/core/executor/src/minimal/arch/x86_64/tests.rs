use memmap2::MmapMut;
use sp1_jit::{
    memory::AnonymousMemory, trace_capacity, ComputeInstructions, JitContext, MemoryInstructions,
    RiscOperand, RiscRegister, RiscvTranspiler, SystemInstructions,
};

// Import the actual sp1_ecall_handler from the minimal executor
use crate::minimal::ecall::sp1_ecall_handler;

// Helper function to create a new backend for testing
fn new_backend() -> sp1_jit::backends::x86::TranspilerBackend {
    sp1_jit::backends::x86::TranspilerBackend::new(0, 1024 * 2, 1000, 100, 100, 8).unwrap()
}

// Finalize the function and call it.
fn run_test(assembler: sp1_jit::backends::x86::TranspilerBackend) {
    let mut func = assembler.finalize::<AnonymousMemory>().expect("Failed to finalize function");

    let mut trace_buf = MmapMut::map_anon(trace_capacity(Some(1000))).expect("create mmap buf");
    let trace_buf_ptr = trace_buf.as_mut_ptr();

    unsafe {
        func.call(trace_buf_ptr);
    }
}

#[test]
fn test_write_syscall_to_public_values() {
    let mut backend = new_backend();

    backend.register_ecall_handler(sp1_ecall_handler);

    backend.start_instr();

    // FD_PUBLIC_VALUES from sp1_primitives
    const FD_PUBLIC_VALUES: u32 = 13;

    // Store some data at address 0x10
    backend.add(RiscRegister::X1, RiscOperand::Immediate(0x12345678), RiscOperand::Immediate(0));
    backend.sw(RiscRegister::X0, RiscRegister::X1, 0x10);
    backend.add(
        RiscRegister::X1,
        RiscOperand::Immediate(0x9ABCDEF0u32 as i32),
        RiscOperand::Immediate(0),
    );
    backend.sw(RiscRegister::X0, RiscRegister::X1, 0x14);

    // Set up WRITE syscall for public values
    backend.add(RiscRegister::X5, RiscOperand::Immediate(0x02), RiscOperand::Immediate(0));
    backend.add(
        RiscRegister::X10,
        RiscOperand::Immediate(FD_PUBLIC_VALUES as i32),
        RiscOperand::Immediate(0),
    );
    backend.add(RiscRegister::X11, RiscOperand::Immediate(0x10), RiscOperand::Immediate(0));
    backend.add(RiscRegister::X12, RiscOperand::Immediate(8), RiscOperand::Immediate(0)); // 8 bytes

    backend.ecall();

    // Verify the public values were written to the stream
    extern "C" fn check_public_values(ctx: *mut JitContext) {
        let ctx = unsafe { &mut *ctx };
        let public_values = unsafe { ctx.public_values_stream() };
        assert_eq!(public_values.len(), 8);
        // Check the written values (little endian)
        assert_eq!(&public_values[0..4], &0x12345678u32.to_le_bytes());
        assert_eq!(&public_values[4..8], &0x9ABCDEF0u32.to_le_bytes());
    }
    backend.call_extern_fn(check_public_values);

    run_test(backend);
}

#[test]
fn test_write_syscall_to_hint() {
    let mut backend = new_backend();

    backend.register_ecall_handler(sp1_ecall_handler);

    backend.start_instr();

    // FD_HINT from sp1_primitives
    const FD_HINT: u32 = 14;

    // Store hint data at address 0x10
    backend.add(
        RiscRegister::X1,
        RiscOperand::Immediate(0xDEADBEEFu32 as i32),
        RiscOperand::Immediate(0),
    );
    backend.sw(RiscRegister::X0, RiscRegister::X1, 0x10);

    // Set up WRITE syscall for hint buffer
    backend.add(RiscRegister::X5, RiscOperand::Immediate(0x02), RiscOperand::Immediate(0));
    backend.add(
        RiscRegister::X10,
        RiscOperand::Immediate(FD_HINT as i32),
        RiscOperand::Immediate(0),
    );
    backend.add(RiscRegister::X11, RiscOperand::Immediate(0x10), RiscOperand::Immediate(0));
    backend.add(RiscRegister::X12, RiscOperand::Immediate(4), RiscOperand::Immediate(0));

    backend.ecall();

    // Verify the hint was added to the input buffer
    extern "C" fn check_hint_buffer(ctx: *mut JitContext) {
        let ctx = unsafe { &mut *ctx };
        let input_buffer = unsafe { ctx.input_buffer() };
        assert_eq!(input_buffer.len(), 1);
        let hint_data = &input_buffer[0];
        assert_eq!(hint_data.len(), 4);
        assert_eq!(&hint_data[0..4], &0xDEADBEEFu32.to_le_bytes());
    }
    backend.call_extern_fn(check_hint_buffer);

    run_test(backend);
}

mod debug {
    use super::*;
    use crate::{disassembler::transpile, MinimalTranspiler, Program};
    use dynasmrt::DynasmApi;
    use sp1_jit::{
        ComputeInstructions, ControlFlowInstructions, DebugBackend, DebugFn, EcallHandler,
        ExternFn, JitFunction, JitMemory, MemoryInstructions, SystemInstructions, TraceCollector,
        TranspilerBackend,
    };
    use std::ops::Deref;

    trait TranspilerWithOffset: RiscvTranspiler {
        fn offset(&self) -> usize;
    }

    impl TranspilerWithOffset for TranspilerBackend {
        fn offset(&self) -> usize {
            self.inner.offset().0
        }
    }

    impl<T: TranspilerWithOffset> TranspilerWithOffset for DebugBackend<T> {
        fn offset(&self) -> usize {
            self.backend.offset()
        }
    }

    struct DumpTransipler<T> {
        inner: T,
        start_offsets: Vec<usize>,
        end_offsets: Vec<usize>,
    }

    impl<T: RiscvTranspiler> TraceCollector for DumpTransipler<T> {
        fn trace_registers(&mut self) {
            self.inner.trace_registers()
        }

        fn trace_mem_value(&mut self, rs1: RiscRegister, imm: u64) {
            self.inner.trace_mem_value(rs1, imm)
        }

        fn trace_pc_start(&mut self) {
            self.inner.trace_pc_start()
        }

        fn trace_clk_start(&mut self) {
            self.inner.trace_clk_start()
        }

        fn trace_clk_end(&mut self) {
            self.inner.trace_clk_end()
        }
    }

    impl<T: RiscvTranspiler> ComputeInstructions for DumpTransipler<T> {
        fn add(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.add(rd, rs1, rs2)
        }

        fn sub(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.sub(rd, rs1, rs2)
        }

        fn xor(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.xor(rd, rs1, rs2)
        }

        fn or(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.or(rd, rs1, rs2)
        }

        fn and(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.and(rd, rs1, rs2)
        }

        fn sll(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.sll(rd, rs1, rs2)
        }

        fn srl(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.srl(rd, rs1, rs2)
        }

        fn sra(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.sra(rd, rs1, rs2)
        }

        fn slt(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.slt(rd, rs1, rs2)
        }

        fn sltu(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.sltu(rd, rs1, rs2)
        }

        fn mul(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.mul(rd, rs1, rs2)
        }

        fn mulh(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.mulh(rd, rs1, rs2)
        }

        fn mulhu(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.mulhu(rd, rs1, rs2)
        }

        fn mulhsu(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.mulhsu(rd, rs1, rs2)
        }

        fn div(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.div(rd, rs1, rs2)
        }

        fn divu(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.divu(rd, rs1, rs2)
        }

        fn rem(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.rem(rd, rs1, rs2)
        }

        fn remu(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.remu(rd, rs1, rs2)
        }

        fn addw(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.addw(rd, rs1, rs2)
        }

        fn subw(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.subw(rd, rs1, rs2)
        }

        fn sllw(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.sllw(rd, rs1, rs2)
        }

        fn srlw(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.srlw(rd, rs1, rs2)
        }

        fn sraw(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.sraw(rd, rs1, rs2)
        }

        fn mulw(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.mulw(rd, rs1, rs2)
        }

        fn divw(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.divw(rd, rs1, rs2)
        }

        fn divuw(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.divuw(rd, rs1, rs2)
        }

        fn remw(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.remw(rd, rs1, rs2)
        }

        fn remuw(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
            self.inner.remuw(rd, rs1, rs2)
        }

        fn auipc(&mut self, rd: RiscRegister, imm: u64) {
            self.inner.auipc(rd, imm)
        }

        fn lui(&mut self, rd: RiscRegister, imm: u64) {
            self.inner.lui(rd, imm)
        }
    }

    impl<T: RiscvTranspiler> ControlFlowInstructions for DumpTransipler<T> {
        fn beq(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u64) {
            self.inner.beq(rs1, rs2, imm)
        }

        fn bne(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u64) {
            self.inner.bne(rs1, rs2, imm)
        }

        fn blt(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u64) {
            self.inner.blt(rs1, rs2, imm)
        }

        fn bge(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u64) {
            self.inner.bge(rs1, rs2, imm)
        }

        fn bltu(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u64) {
            self.inner.bltu(rs1, rs2, imm)
        }

        fn bgeu(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u64) {
            self.inner.bgeu(rs1, rs2, imm)
        }

        fn jal(&mut self, rd: RiscRegister, imm: u64) {
            self.inner.jal(rd, imm)
        }

        fn jalr(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u64) {
            self.inner.jalr(rd, rs1, imm)
        }
    }

    impl<T: RiscvTranspiler> MemoryInstructions for DumpTransipler<T> {
        fn lb(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u64) {
            self.inner.lb(rd, rs1, imm)
        }

        fn lh(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u64) {
            self.inner.lh(rd, rs1, imm)
        }

        fn lw(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u64) {
            self.inner.lw(rd, rs1, imm)
        }

        fn lbu(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u64) {
            self.inner.lbu(rd, rs1, imm)
        }

        fn lhu(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u64) {
            self.inner.lhu(rd, rs1, imm)
        }

        fn ld(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u64) {
            self.inner.ld(rd, rs1, imm)
        }

        fn lwu(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u64) {
            self.inner.lwu(rd, rs1, imm)
        }

        fn sb(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u64) {
            self.inner.sb(rs1, rs2, imm)
        }

        fn sh(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u64) {
            self.inner.sh(rs1, rs2, imm)
        }

        fn sw(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u64) {
            self.inner.sw(rs1, rs2, imm)
        }

        fn sd(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u64) {
            self.inner.sd(rs1, rs2, imm)
        }
    }

    impl<T: RiscvTranspiler> SystemInstructions for DumpTransipler<T> {
        fn ecall(&mut self) {
            self.inner.ecall()
        }

        fn unimp(&mut self) {
            self.inner.unimp()
        }
    }

    impl<T: TranspilerWithOffset> RiscvTranspiler for DumpTransipler<T> {
        fn new(
            program_size: usize,
            memory_size: usize,
            max_trace_size: u64,
            pc_start: u64,
            pc_base: u64,
            clk_bump: u64,
        ) -> Result<Self, std::io::Error> {
            Ok(Self {
                inner: T::new(
                    program_size,
                    memory_size,
                    max_trace_size,
                    pc_start,
                    pc_base,
                    clk_bump,
                )?,
                start_offsets: Vec::new(),
                end_offsets: Vec::new(),
            })
        }

        fn register_ecall_handler(&mut self, handler: EcallHandler) {
            self.inner.register_ecall_handler(handler)
        }

        fn start_instr(&mut self) {
            self.start_offsets.push(self.inner.offset());
            self.inner.start_instr();
        }

        fn end_instr(&mut self) {
            self.inner.end_instr();
            self.end_offsets.push(self.inner.offset());
        }

        fn inspect_register(&mut self, reg: RiscRegister, handler: DebugFn) {
            self.inner.inspect_register(reg, handler)
        }

        fn inspect_immediate(&mut self, imm: u64, handler: DebugFn) {
            self.inner.inspect_immediate(imm, handler)
        }

        fn call_extern_fn(&mut self, handler: ExternFn) {
            self.inner.call_extern_fn(handler)
        }

        fn finalize<M: JitMemory>(self) -> std::io::Result<JitFunction<M>> {
            let func = self.inner.finalize()?;
            let code = func.code.deref().to_vec();

            std::fs::write("/tmp/sp1_minimal_code_full.bin", &code).expect("write");
            for (i, (s, e)) in self.start_offsets.iter().zip(self.end_offsets.iter()).enumerate() {
                let name = format!("/tmp/sp1_minimal_code_inst{}.bin", i);
                std::fs::write(name, &code[(*s)..(*e)]).expect("write");
            }

            Ok(func)
        }
    }

    #[test]
    fn test_decode_jit_dump() {
        let max_trace_size = 10;
        let is_debug = false;
        let raw_instructions = vec![
            0x00e48513, // addi a0, s1, 14
            0x01273423, // sd s2, 8(a4)
            0x01273023, // sd s2, 0(a4)
            0x234010ef, // jal ra, 0x1234
            0x010600e7, // jalr ra, 0x10(a2)
            0x01049533, // sll a0, s1, a6
            0x0124e433, // or s0, s1, s2
            0x00a5f533, // and a0, a1, a0
            0x02a58533, // mul a0, a1, a0
            0x01148603, // lb a2, 17(s1)
            0x0044a603, // lw a2, 4(s1)
            0x0064e603, // lwu a2, 6(s1)
        ];
        eprintln!(
            "Tracing: {}, debug: {is_debug}, insts: {}",
            max_trace_size > 0,
            raw_instructions.len()
        );

        let instructions = transpile(&raw_instructions).into_iter().map(|(i, _r)| i).collect();
        let program = Program::new(
            instructions,
            100, // PC start
            100, // PC base
        );
        let transpiler = MinimalTranspiler::new(
            1024 * 2, // max memory size
            is_debug,
            Some(max_trace_size),
        );
        let backend: DumpTransipler<TranspilerBackend> = DumpTransipler::new(
            program.instructions.len(),
            transpiler.max_memory_size,
            transpiler.max_trace_size,
            program.pc_start_abs,
            program.pc_base,
            8, // clk bump
        )
        .unwrap();

        let _: JitFunction<AnonymousMemory> = if transpiler.is_debug {
            transpiler.transpile_instructions(DebugBackend::new(backend), &program)
        } else {
            transpiler.transpile_instructions(backend, &program)
        };
    }
}
