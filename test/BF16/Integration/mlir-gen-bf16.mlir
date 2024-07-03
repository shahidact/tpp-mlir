// MLP without softmax (can't print packed version for now)
// RUN: mlir-gen --kernel=const --bias --relu --seed=123 --batch=10 --layers=10,10,10 --float-type=bf16 | tpp-run -e entry -entry-point-result=void
// RUN: mlir-gen --output=named --kernel=const --bias --relu --seed=123 --batch=10 --layers=10,10,10 --float-type=bf16 | tpp-run -e entry -entry-point-result=void

// Matmul only
// RUN: mlir-gen --kernel=const --bias --relu --seed=123 --batch=10 --layers=10,10 --float-type=bf16 | tpp-run -e entry -entry-point-result=void
// RUN: mlir-gen --output=named --kernel=const --bias --relu --seed=123 --batch=10 --layers=10,10 --float-type=bf16 | tpp-run -e entry -entry-point-result=void

// Kernel - matmul
// RUN: mlir-gen --kernel=args --seed=123 --float-type=bf16 --batch=10 --layers=10,10 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=GEN-MATMUL-BF16
// RUN: mlir-gen --output=named --kernel=args --seed=123 --float-type=bf16 --batch=10 --layers=10,10 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=GEN-MATMUL-BF16

// Kernel - fc
// RUN: mlir-gen --kernel=args --bias --relu --seed=123 --float-type=bf16 --batch=10 --layers=10,10 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=GEN-FC-BF16
// RUN: mlir-gen --output=named --kernel=args --bias --relu --seed=123 --float-type=bf16 --batch=10 --layers=10,10 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=GEN-FC-BF16

// BF16/VNNI execution
// RUN: mlir-gen --kernel=const --bias --relu --seed=123 --batch=10 --layers=10,10 --tiles=2,2,2 --float-type=bf16 | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF
// RUN: mlir-gen --output=named --kernel=const --bias --relu --seed=123 --batch=10 --layers=10,10 --tiles=2,2,2 --float-type=bf16 | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF
// RUN: mlir-gen --kernel=const --bias --relu --seed=123 --batch=10 --layers=10,10 --tiles=2,2,2 --float-type=bf16 | tpp-opt --pack-vnni | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF
// RUN: mlir-gen --output=named --kernel=const --bias --relu --seed=123 --batch=10 --layers=10,10 --tiles=2,2,2 --float-type=bf16 | tpp-opt --pack-vnni | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF


// GEN-MATMUL-BF16: ( 11, 11, 11, 11, 11, 11, 11, 11, 11, 11 )

// GEN-FC-BF16: ( 12, 12, 12, 12, 12, 12, 12, 12, 12, 12 )

// PERF:    {{[0-9]+}}{{.?}}{{[0-9e-]+}}
