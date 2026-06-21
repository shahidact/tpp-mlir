// RUN: tpp-opt %s -replicate-buffers-benchmark="num-layers=2" --split-input-file | FileCheck %s
// RUN: tpp-opt %s -replicate-buffers-benchmark="num-layers=2" --split-input-file | FileCheck %s --check-prefix=BF16

module {
  memref.global "private" @__wrapper_4 : memref<2x2x32x32xf32> = dense<0.0> {alignment = 128 : i64}
  memref.global "private" @__wrapper_3 : memref<64xf32> = dense<1.0> {alignment = 128 : i64}
  memref.global "private" @__wrapper_2 : memref<2x2x16x32x4xi8> = dense<1> {alignment = 128 : i64}
  memref.global "private" @__wrapper_1 : memref<64xf32> = dense<1.0> {alignment = 128 : i64}
  memref.global "private" @__wrapper_0 : memref<2x2x32x64xi8> = dense<1> {alignment = 128 : i64}

  func.func @_entry(%arg0: memref<2x2x32x64xi8>, %arg1: memref<64xf32>,
                    %arg2: memref<2x2x16x32x4xi8>, %arg3: memref<64xf32>,
                    %arg4: memref<2x2x32x32xf32>) {
    return
  }

  func.func @entry() {
    %0 = memref.get_global @__wrapper_0 : memref<2x2x32x64xi8>
    %2 = memref.get_global @__wrapper_1 : memref<64xf32>
    %4 = memref.get_global @__wrapper_2 : memref<2x2x16x32x4xi8>
    %6 = memref.get_global @__wrapper_3 : memref<64xf32>
    %8 = memref.get_global @__wrapper_4 : memref<2x2x32x32xf32>

    // Warmup
    %c1_i64 = arith.constant 1 : i64
    %warmup = perf.bench(%c1_i64 : i64) -> f64 {
      func.call @_entry(%0, %2, %4, %6, %8) : (memref<2x2x32x64xi8>, memref<64xf32>, memref<2x2x16x32x4xi8>, memref<64xf32>, memref<2x2x32x32xf32>) -> ()
    }

    // Benchmark
    %c10_i64 = arith.constant 10 : i64
    %bench = perf.bench(%c10_i64 : i64) -> f64 {
      func.call @_entry(%0, %2, %4, %6, %8) : (memref<2x2x32x64xi8>, memref<64xf32>, memref<2x2x16x32x4xi8>, memref<64xf32>, memref<2x2x32x32xf32>) -> ()
    }

    %iters_f64 = arith.uitofp %c10_i64 : i64 to f64
    %avg_time = arith.divf %bench, %iters_f64 : f64
    vector.print %avg_time : f64
    return
  }
}

// CHECK-LABEL: func.func @entry()

// Warmup
// CHECK: perf.bench({{.*}} : i64) -> f64 {
// CHECK: func.call @_entry

// Unrolled allocations: 2 separate allocs per buffer (numLayers=2)
// i8 activation buffers (2 layers)
// CHECK-DAG: memref.alloc() : memref<2x2x32x64xi8>
// CHECK-DAG: memref.alloc() : memref<2x2x32x64xi8>
// f32 scale buffers (2 layers)
// CHECK-DAG: memref.alloc() : memref<64xf32>
// CHECK-DAG: memref.alloc() : memref<64xf32>
// i8 weight buffers (2 layers)
// CHECK-DAG: memref.alloc() : memref<2x2x16x32x4xi8>
// CHECK-DAG: memref.alloc() : memref<2x2x16x32x4xi8>
// f32 scale buffers (2 layers)
// CHECK-DAG: memref.alloc() : memref<64xf32>
// CHECK-DAG: memref.alloc() : memref<64xf32>
// f32 output buffers (2 layers)
// CHECK-DAG: memref.alloc() : memref<2x2x32x32xf32>
// CHECK-DAG: memref.alloc() : memref<2x2x32x32xf32>

// Benchmark with unrolled kernel calls
// CHECK: perf.bench({{.*}} : i64) -> f64 {
// CHECK: func.call @_entry
// CHECK: func.call @_entry

// Timing adjusted for 2 layers
// CHECK: %[[NLAYERS:.*]] = arith.constant 2.000000e+00 : f64
// CHECK: arith.mulf {{.*}}, %[[NLAYERS]] : f64

// -----

// Test cache-nuke buffer replication.
module {
  memref.global "private" @__wrapper_2 : memref<2x2x32x32xf32> = dense<0.0> {alignment = 128 : i64}
  memref.global "private" @__wrapper_1 : memref<2x2x16x32x2xbf16> = dense<1.0> {alignment = 128 : i64}
  memref.global "private" @__wrapper_0 : memref<2x2x32x32xbf16> = dense<1.0> {alignment = 128 : i64}

  func.func @_entry(%arg0: memref<2x2x32x32xbf16>, %arg1: memref<2x2x16x32x2xbf16>,
                    %arg2: memref<2x2x32x32xf32>) {
    return
  }

  func.func @entry() {
    %0 = memref.get_global @__wrapper_0 : memref<2x2x32x32xbf16>
    %1 = memref.get_global @__wrapper_1 : memref<2x2x16x32x2xbf16>
    %2 = memref.get_global @__wrapper_2 : memref<2x2x32x32xf32>

    // Warmup
    %c1_i64 = arith.constant 1 : i64
    %warmup = perf.bench(%c1_i64 : i64) -> f64 {
      func.call @_entry(%0, %1, %2) : (memref<2x2x32x32xbf16>, memref<2x2x16x32x2xbf16>, memref<2x2x32x32xf32>) -> ()
    }

    // Benchmark
    %c10_i64 = arith.constant 10 : i64
    %bench = perf.bench(%c10_i64 : i64) -> f64 {
      func.call @_entry(%0, %1, %2) : (memref<2x2x32x32xbf16>, memref<2x2x16x32x2xbf16>, memref<2x2x32x32xf32>) -> ()
    }

    %iters_f64 = arith.uitofp %c10_i64 : i64 to f64
    %avg_time = arith.divf %bench, %iters_f64 : f64
    vector.print %avg_time : f64
    return
  }
}

// BF16-LABEL: func.func @entry()

// Warmup
// BF16: perf.bench({{.*}} : i64) -> f64 {
// BF16: func.call @_entry

// Unrolled allocations: 2 separate allocs per buffer (numLayers=2)
// bf16 activation buffers (2 layers)
// BF16-DAG: memref.alloc() : memref<2x2x32x32xbf16>
// BF16-DAG: memref.alloc() : memref<2x2x32x32xbf16>
// bf16 weight buffers in VNNI format (2 layers)
// BF16-DAG: memref.alloc() : memref<2x2x16x32x2xbf16>
// BF16-DAG: memref.alloc() : memref<2x2x16x32x2xbf16>
// f32 output buffers (2 layers)
// BF16-DAG: memref.alloc() : memref<2x2x32x32xf32>
// BF16-DAG: memref.alloc() : memref<2x2x32x32xf32>

// Benchmark with unrolled kernel calls
// BF16: perf.bench({{.*}} : i64) -> f64 {
// BF16: func.call @_entry
// BF16: func.call @_entry

// Timing adjusted for 2 layers
// BF16: %[[NLAYERS:.*]] = arith.constant 2.000000e+00 : f64
// BF16: arith.mulf {{.*}}, %[[NLAYERS]] : f64
