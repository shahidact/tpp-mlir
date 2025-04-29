// REQUIRES: target=x86{{.*}}

// RUN: tpp-opt %s  --vector-contract-to-amx --split-input-file | FileCheck %s


#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
module {
  memref.global "private" constant @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @entry(%arg0: memref<8x32x32x32xbf16>) -> memref<8x32x32x32xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %cst_0 = arith.constant dense<0.000000e+00> : vector<32x32xbf16>
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %0 = memref.get_global @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x32x32x32xbf16>
  %expand_shape = memref.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [8, 32, 32, 16, 2] : memref<8x32x32x32xbf16> into memref<8x32x32x16x2xbf16>
  scf.forall (%arg1, %arg2) in (8, 32) {
    %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
    vector.transfer_write %cst_0, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xbf16>, memref<32x32xbf16, strided<[32, 1], offset: ?>>
    %subview_1 = memref.subview %expand_shape[%arg1, 0, 0, 0, 0] [1, 32, 32, 16, 2] [1, 1, 1, 1, 1] : memref<8x32x32x16x2xbf16> to memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
    %1 = vector.transfer_read %subview[%c0, %c0], %cst {in_bounds = [true, true]} : memref<32x32xbf16, strided<[32, 1], offset: ?>>, vector<32x32xbf16>
    %2 = scf.for %arg3 = %c0 to %c32 step %c1 iter_args(%arg4 = %1) -> (vector<32x32xbf16>) {
      %subview_2 = memref.subview %subview_1[%arg3, 0, 0, 0] [1, 32, 16, 2] [1, 1, 1, 1] : memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>> to memref<1x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
      %subview_3 = memref.subview %0[%arg3, 0, 0, 0] [1, 16, 32, 2] [1, 1, 1, 1] : memref<32x16x32x2xbf16> to memref<1x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
      %3 = vector.transfer_read %subview_2[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>, vector<1x32x16x2xbf16>
      %4 = vector.transfer_read %subview_3[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, vector<1x16x32x2xbf16>
      %5 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %4, %arg4 : vector<1x32x16x2xbf16>, vector<1x16x32x2xbf16> into vector<32x32xbf16>
      scf.yield %5 : vector<32x32xbf16>
    }
    vector.transfer_write %2, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xbf16>, memref<32x32xbf16, strided<[32, 1], offset: ?>>
  }
  return %alloc : memref<8x32x32x32xbf16>
}

}

// CHECK-LABEL:   func.func @entry
// CHECK:             memref.alloca() : memref<32x32xf32>
// CHECK:             amx.tile_load
// CHECK:             memref.collapse_shape
// CHECK:             amx.tile_load
// CHECK-COUNT-4:     amx.tile_mulf
// CHECK:             amx.tile_store
// CHECK:             vector.transfer_read
// CHECK:             arith.truncf
// CHECK:             vector.transfer_write

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
module {
  memref.global "private" constant @__constant_16x32x64x2xbf16 : memref<16x32x64x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @entry(%arg0: memref<4x16x64x64xbf16>) -> memref<4x16x64x64xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant dense<0.000000e+00> : vector<64x64xbf16>
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_16x32x64x2xbf16 : memref<16x32x64x2xbf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x16x64x64xbf16>
    %expand_shape = memref.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [4, 16, 64, 32, 2] : memref<4x16x64x64xbf16> into memref<4x16x64x32x2xbf16>
    scf.forall (%arg1, %arg2) in (4, 16) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<4x16x64x64xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
      vector.transfer_write %cst_0, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<64x64xbf16>, memref<64x64xbf16, strided<[64, 1], offset: ?>>
      %subview_1 = memref.subview %expand_shape[%arg1, 0, 0, 0, 0] [1, 16, 64, 32, 2] [1, 1, 1, 1, 1] : memref<4x16x64x32x2xbf16> to memref<16x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>
      scf.for %arg3 = %c0 to %c64 step %c32 {
        scf.for %arg4 = %c0 to %c64 step %c32 {
          %subview_2 = memref.subview %subview[%arg3, %arg4] [32, 32] [1, 1] : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<32x32xbf16, strided<[64, 1], offset: ?>>
          %1 = vector.transfer_read %subview_2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<32x32xbf16, strided<[64, 1], offset: ?>>, vector<32x32xbf16>
          %2 = scf.for %arg5 = %c0 to %c16 step %c1 iter_args(%arg6 = %1) -> (vector<32x32xbf16>) {
            %3 = scf.for %arg7 = %c0 to %c32 step %c16 iter_args(%arg8 = %arg6) -> (vector<32x32xbf16>) {
              %subview_3 = memref.subview %subview_1[%arg5, %arg3, %arg7, 0] [1, 32, 16, 2] [1, 1, 1, 1] : memref<16x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>> to memref<1x32x16x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>
              %subview_4 = memref.subview %0[%arg5, %arg7, %arg4, 0] [1, 16, 32, 2] [1, 1, 1, 1] : memref<16x32x64x2xbf16> to memref<1x16x32x2xbf16, strided<[4096, 128, 2, 1], offset: ?>>
              %4 = vector.transfer_read %subview_3[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x32x16x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>, vector<1x32x16x2xbf16>
              %5 = vector.transfer_read %subview_4[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x32x2xbf16, strided<[4096, 128, 2, 1], offset: ?>>, vector<1x16x32x2xbf16>
              %6 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %arg8 : vector<1x32x16x2xbf16>, vector<1x16x32x2xbf16> into vector<32x32xbf16>
              scf.yield %6 : vector<32x32xbf16>
            }
            scf.yield %3 : vector<32x32xbf16>
          }
          vector.transfer_write %2, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xbf16>, memref<32x32xbf16, strided<[64, 1], offset: ?>>
        }
      }
    }
    return %alloc : memref<4x16x64x64xbf16>
  }
}

// CHECK-LABEL:   func.func @entry
// CHECK:             memref.alloca() : memref<32x32xf32>
// CHECK:             amx.tile_load
// CHECK:             amx.tile_load
// CHECK-COUNT-4:     amx.tile_mulf
// CHECK:             amx.tile_store
// CHECK:             vector.transfer_read
// CHECK:             arith.truncf
// CHECK:             vector.transfer_write

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>

memref.global "private" constant @__constant_16x32x64x2xbf16 : memref<16x32x64x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
func.func @entry(%arg0: memref<4x16x64x64xbf16>) -> memref<4x16x64x64xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %cst_0 = arith.constant dense<0.000000e+00> : vector<64x64xbf16>
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %0 = memref.get_global @__constant_16x32x64x2xbf16 : memref<16x32x64x2xbf16>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x16x64x64xbf16>
  %expand_shape = memref.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [4, 16, 64, 32, 2] : memref<4x16x64x64xbf16> into memref<4x16x64x32x2xbf16>
  scf.forall (%arg1, %arg2) in (4, 16) {
    %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<4x16x64x64xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
    vector.transfer_write %cst_0, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<64x64xbf16>, memref<64x64xbf16, strided<[64, 1], offset: ?>>
    %subview_1 = memref.subview %expand_shape[%arg1, 0, 0, 0, 0] [1, 16, 64, 32, 2] [1, 1, 1, 1, 1] : memref<4x16x64x32x2xbf16> to memref<16x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>
    scf.for %arg3 = %c0 to %c64 step %c64 {
      scf.for %arg4 = %c0 to %c64 step %c32 {
        %subview_2 = memref.subview %subview[%arg3, %arg4] [64, 32] [1, 1] : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<64x32xbf16, strided<[64, 1], offset: ?>>
        %1 = vector.transfer_read %subview_2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<64x32xbf16, strided<[64, 1], offset: ?>>, vector<64x32xbf16>
        %2 = scf.for %arg5 = %c0 to %c16 step %c1 iter_args(%arg6 = %1) -> (vector<64x32xbf16>) {
          %3 = scf.for %arg7 = %c0 to %c32 step %c16 iter_args(%arg8 = %arg6) -> (vector<64x32xbf16>) {
            %subview_3 = memref.subview %subview_1[%arg5, %arg3, %arg7, 0] [1, 64, 16, 2] [1, 1, 1, 1] : memref<16x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>> to memref<1x64x16x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>
            %subview_4 = memref.subview %0[%arg5, %arg7, %arg4, 0] [1, 16, 32, 2] [1, 1, 1, 1] : memref<16x32x64x2xbf16> to memref<1x16x32x2xbf16, strided<[4096, 128, 2, 1], offset: ?>>
            %4 = vector.transfer_read %subview_3[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x64x16x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>, vector<1x64x16x2xbf16>
            %5 = vector.transfer_read %subview_4[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x32x2xbf16, strided<[4096, 128, 2, 1], offset: ?>>, vector<1x16x32x2xbf16>
            %6 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %arg8 : vector<1x64x16x2xbf16>, vector<1x16x32x2xbf16> into vector<64x32xbf16>
            scf.yield %6 : vector<64x32xbf16>
          }
          scf.yield %3 : vector<64x32xbf16>
        }
        vector.transfer_write %2, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<64x32xbf16>, memref<64x32xbf16, strided<[64, 1], offset: ?>>
      }
    }
  }
  return %alloc : memref<4x16x64x64xbf16>
}

// CHECK-LABEL:   func.func @entry
// CHECK:             memref.alloca() : memref<64x32xf32>
// CHECK:             amx.tile_load
// CHECK:             amx.tile_load
// CHECK-COUNT-8:     amx.tile_mulf
// CHECK:             amx.tile_store
// CHECK:             vector.transfer_read
// CHECK:             arith.truncf
// CHECK:             vector.transfer_write
