use crate::MinimalExecutorRunner;
use sp1_core_executor::{ExecutionError, Program};
use sp1_core_machine::{io::SP1Stdin, utils::setup_logger};
use std::sync::Arc;
use test_artifacts::MEMORY_TESTER_ELF;

fn run(runner: &mut MinimalExecutorRunner) -> Option<ExecutionError> {
    loop {
        match runner.try_execute_chunk() {
            Ok(Some(_)) => (), // continue
            Ok(None) => return None,
            Err(e) => return Some(e),
        }
    }
}

#[test]
fn test_out_of_bound_access() {
    setup_logger();

    let program = Arc::new(Program::from(&MEMORY_TESTER_ELF).expect("parse program"));
    let mut stdin = SP1Stdin::new();
    stdin.write(&0u8);

    let mut runner = MinimalExecutorRunner::new(program, false, Some(1000), None);
    for input in &stdin.buffer {
        runner.with_input(input);
    }

    let result = run(&mut runner);
    assert!(matches!(result, Some(ExecutionError::InvalidMemoryAccess(_, _))));
}

#[test]
fn test_using_too_much_memory() {
    setup_logger();

    let program = Arc::new(Program::from(&MEMORY_TESTER_ELF).expect("parse program"));
    let mut stdin = SP1Stdin::new();
    stdin.write(&1u8);

    let mut runner =
        MinimalExecutorRunner::new(program, false, Some(16000000), Some(2 * 1024 * 1024 * 1024));
    for input in &stdin.buffer {
        runner.with_input(input);
    }

    let result = run(&mut runner);
    assert_eq!(result, Some(ExecutionError::TooMuchMemory()));
}
