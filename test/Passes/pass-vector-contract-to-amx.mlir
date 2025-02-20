// RUN: tpp-opt %s  --vector-contract-to-amx --split-input-file | FileCheck %s


#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
module {
  memref.global "private" constant @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16> = dense<1.048580e+06> {alignment = 64 : i64}
  func.func @entry(%arg0: memref<8x32x32x32xbf16>) -> memref<8x32x32x32xf32> {
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_f32 = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : vector<32x32xf32>
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x32x32x32xf32>
    %expand_shape = memref.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [8, 32, 32, 16, 2] : memref<8x32x32x32xbf16> into memref<8x32x32x16x2xbf16>
    scf.forall (%arg1, %arg2) in (8, 32) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      vector.transfer_write %cst_0, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
      %subview_1 = memref.subview %expand_shape[%arg1, 0, 0, 0, 0] [1, 32, 32, 16, 2] [1, 1, 1, 1, 1] : memref<8x32x32x16x2xbf16> to memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
      scf.for %arg3 = %c0 to %c32 step %c32 {
        scf.for %arg4 = %c0 to %c32 step %c32 {
          %subview_2 = memref.subview %subview[%arg3, %arg4] [32, 32] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<32x32xf32, strided<[32, 1], offset: ?>>
          %1 = vector.transfer_read %subview_2[%c0, %c0], %cst_f32 {in_bounds = [true, true]} : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<32x32xf32>
          %2 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %1) -> (vector<32x32xf32>) {
            %3 = scf.for %arg7 = %c0 to %c16 step %c16 iter_args(%arg8 = %arg6) -> (vector<32x32xf32>) {

              %subview_3 = memref.subview %subview_1[%arg5, %arg3, %arg7, 0] [1, 32, 16, 2] [1, 1, 1, 1] : memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>> to memref<1x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
              %subview_4 = memref.subview %0[%arg5, %arg7, %arg4, 0] [1, 16, 32, 2] [1, 1, 1, 1] : memref<32x16x32x2xbf16> to memref<1x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>

              %4 = vector.transfer_read %subview_3[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>, vector<1x32x16x2xbf16>
              %5 = vector.transfer_read %subview_4[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, vector<1x16x32x2xbf16>
              %6 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %arg8 : vector<1x32x16x2xbf16>, vector<1x16x32x2xbf16> into vector<32x32xf32>
              scf.yield %6 : vector<32x32xf32>
            }
            scf.yield %3 : vector<32x32xf32>
          }
          vector.transfer_write %2, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
        }
      }
    }
    return %alloc : memref<8x32x32x32xf32>
  }
}

// CHECK-LABEL:   memref.global "private" constant @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16> = dense<1.048580e+06> {alignment = 64 : i64}

// CHECK-LABEL:   func.func @entry(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<8x32x32x32xbf16>) -> memref<8x32x32x32xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<0.000000e+00> : vector<32x32xf32>
// CHECK:           %[[VAL_2:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = memref.get_global @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16>
// CHECK:           %[[VAL_7:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x32x32x32xf32>
// CHECK:           %[[VAL_8:.*]] = memref.expand_shape %[[VAL_0]] {{\[\[}}0], [1], [2], [3, 4]] output_shape [8, 32, 32, 16, 2] : memref<8x32x32x32xbf16> into memref<8x32x32x16x2xbf16>
// CHECK:           scf.forall (%[[VAL_9:.*]], %[[VAL_10:.*]]) in (8, 32) {
// CHECK:             %[[VAL_11:.*]] = memref.subview %[[VAL_7]]{{\[}}%[[VAL_9]], %[[VAL_10]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:             vector.transfer_write %[[VAL_1]], %[[VAL_11]]{{\[}}%[[VAL_5]], %[[VAL_5]]] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:             %[[VAL_12:.*]] = memref.subview %[[VAL_8]]{{\[}}%[[VAL_9]], 0, 0, 0, 0] [1, 32, 32, 16, 2] [1, 1, 1, 1, 1] : memref<8x32x32x16x2xbf16> to memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
// CHECK:             scf.for %[[VAL_13:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_4]] {
// CHECK:               scf.for %[[VAL_14:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_4]] {
// CHECK:                 %[[VAL_15:.*]] = memref.subview %[[VAL_11]]{{\[}}%[[VAL_13]], %[[VAL_14]]] [32, 32] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:                 %[[VAL_16:.*]] = amx.tile_load %[[VAL_15]]{{\[}}%[[VAL_5]], %[[VAL_5]]] : memref<32x32xf32, strided<[32, 1], offset: ?>> into !amx.tile<16x16xf32>
// CHECK:                 %[[VAL_17:.*]] = amx.tile_load %[[VAL_15]]{{\[}}%[[VAL_5]], %[[VAL_2]]] : memref<32x32xf32, strided<[32, 1], offset: ?>> into !amx.tile<16x16xf32>
// CHECK:                 %[[VAL_18:.*]] = amx.tile_load %[[VAL_15]]{{\[}}%[[VAL_2]], %[[VAL_5]]] : memref<32x32xf32, strided<[32, 1], offset: ?>> into !amx.tile<16x16xf32>
// CHECK:                 %[[VAL_19:.*]] = amx.tile_load %[[VAL_15]]{{\[}}%[[VAL_2]], %[[VAL_2]]] : memref<32x32xf32, strided<[32, 1], offset: ?>> into !amx.tile<16x16xf32>
// CHECK:                 %[[VAL_20:.*]]:4 = scf.for %[[VAL_21:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_3]] iter_args(%[[VAL_22:.*]] = %[[VAL_16]], %[[VAL_23:.*]] = %[[VAL_17]], %[[VAL_24:.*]] = %[[VAL_18]], %[[VAL_25:.*]] = %[[VAL_19]]) -> (!amx.tile<16x16xf32>, !amx.tile<16x16xf32>, !amx.tile<16x16xf32>, !amx.tile<16x16xf32>) {
// CHECK:                   %[[VAL_26:.*]]:4 = scf.for %[[VAL_27:.*]] = %[[VAL_5]] to %[[VAL_2]] step %[[VAL_2]] iter_args(%[[VAL_28:.*]] = %[[VAL_22]], %[[VAL_29:.*]] = %[[VAL_23]], %[[VAL_30:.*]] = %[[VAL_24]], %[[VAL_31:.*]] = %[[VAL_25]]) -> (!amx.tile<16x16xf32>, !amx.tile<16x16xf32>, !amx.tile<16x16xf32>, !amx.tile<16x16xf32>) {
// CHECK:                     %[[VAL_32:.*]] = memref.subview %[[VAL_12]]{{\[}}%[[VAL_21]], %[[VAL_13]], %[[VAL_27]], 0] [1, 32, 16, 2] [1, 1, 1, 1] : memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>> to memref<1x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
// CHECK:                     %[[VAL_33:.*]] = amx.tile_load %[[VAL_32]]{{\[}}%[[VAL_5]], %[[VAL_5]], %[[VAL_5]], %[[VAL_5]]] : memref<1x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>> into !amx.tile<16x32xbf16>
// CHECK:                     %[[VAL_34:.*]] = amx.tile_load %[[VAL_32]]{{\[}}%[[VAL_5]], %[[VAL_2]], %[[VAL_5]], %[[VAL_5]]] : memref<1x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>> into !amx.tile<16x32xbf16>
// CHECK:                     %[[VAL_35:.*]] = memref.subview %[[VAL_6]]{{\[}}%[[VAL_21]], %[[VAL_27]], %[[VAL_14]], 0] [1, 16, 32, 2] [1, 1, 1, 1] : memref<32x16x32x2xbf16> to memref<1x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
// CHECK:                     %[[VAL_36:.*]] = amx.tile_load %[[VAL_35]]{{\[}}%[[VAL_5]], %[[VAL_5]], %[[VAL_5]], %[[VAL_5]]] : memref<1x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> into !amx.tile<16x32xbf16>
// CHECK:                     %[[VAL_37:.*]] = amx.tile_load %[[VAL_35]]{{\[}}%[[VAL_5]], %[[VAL_5]], %[[VAL_2]], %[[VAL_5]]] : memref<1x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> into !amx.tile<16x32xbf16>
// CHECK:                     %[[VAL_38:.*]] = amx.tile_mulf %[[VAL_33]], %[[VAL_36]], %[[VAL_28]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK:                     %[[VAL_39:.*]] = amx.tile_mulf %[[VAL_33]], %[[VAL_37]], %[[VAL_29]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK:                     %[[VAL_40:.*]] = amx.tile_mulf %[[VAL_34]], %[[VAL_36]], %[[VAL_30]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK:                     %[[VAL_41:.*]] = amx.tile_mulf %[[VAL_34]], %[[VAL_37]], %[[VAL_31]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK:                     scf.yield %[[VAL_38]], %[[VAL_39]], %[[VAL_40]], %[[VAL_41]] : !amx.tile<16x16xf32>, !amx.tile<16x16xf32>, !amx.tile<16x16xf32>, !amx.tile<16x16xf32>
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_42:.*]]#0, %[[VAL_42]]#1, %[[VAL_42]]#2, %[[VAL_42]]#3 : !amx.tile<16x16xf32>, !amx.tile<16x16xf32>, !amx.tile<16x16xf32>, !amx.tile<16x16xf32>
// CHECK:                 }
// CHECK:                 amx.tile_store %[[VAL_15]]{{\[}}%[[VAL_5]], %[[VAL_5]]], %[[VAL_43:.*]]#0 : memref<32x32xf32, strided<[32, 1], offset: ?>>, !amx.tile<16x16xf32>
// CHECK:                 amx.tile_store %[[VAL_15]]{{\[}}%[[VAL_5]], %[[VAL_2]]], %[[VAL_43]]#1 : memref<32x32xf32, strided<[32, 1], offset: ?>>, !amx.tile<16x16xf32>
// CHECK:                 amx.tile_store %[[VAL_15]]{{\[}}%[[VAL_2]], %[[VAL_5]]], %[[VAL_43]]#2 : memref<32x32xf32, strided<[32, 1], offset: ?>>, !amx.tile<16x16xf32>
// CHECK:                 amx.tile_store %[[VAL_15]]{{\[}}%[[VAL_2]], %[[VAL_2]]], %[[VAL_43]]#3 : memref<32x32xf32, strided<[32, 1], offset: ?>>, !amx.tile<16x16xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return %[[VAL_7]] : memref<8x32x32x32xf32>
// CHECK:         }

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
module {
  memref.global "private" constant @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16> = dense<1.048580e+06> {alignment = 64 : i64}
  func.func @entry(%arg0: memref<8x32x32x32xbf16>) -> memref<8x32x32x32xf32> {
  %cst = arith.constant 0.000000e+00 : bf16
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant dense<0.000000e+00> : vector<32x32xf32>
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %0 = memref.get_global @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x32x32x32xf32>
  %expand_shape = memref.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [8, 32, 32, 16, 2] : memref<8x32x32x32xbf16> into memref<8x32x32x16x2xbf16>
  scf.forall (%arg1, %arg2) in (8, 32) {
    %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
    vector.transfer_write %cst_1, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
    %subview_2 = memref.subview %expand_shape[%arg1, 0, 0, 0, 0] [1, 32, 32, 16, 2] [1, 1, 1, 1, 1] : memref<8x32x32x16x2xbf16> to memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
    %1 = vector.transfer_read %subview[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<32x32xf32>
    %2 = scf.for %arg3 = %c0 to %c32 step %c1 iter_args(%arg4 = %1) -> (vector<32x32xf32>) {
      %subview_3 = memref.subview %subview_2[%arg3, 0, 0, 0] [1, 32, 16, 2] [1, 1, 1, 1] : memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>> to memref<1x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
      %subview_4 = memref.subview %0[%arg3, 0, 0, 0] [1, 16, 32, 2] [1, 1, 1, 1] : memref<32x16x32x2xbf16> to memref<1x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
      %3 = vector.transfer_read %subview_3[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>, vector<1x32x16x2xbf16>
      %4 = vector.transfer_read %subview_4[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, vector<1x16x32x2xbf16>
      %5 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %4, %arg4 : vector<1x32x16x2xbf16>, vector<1x16x32x2xbf16> into vector<32x32xf32>
      scf.yield %5 : vector<32x32xf32>
    }
    vector.transfer_write %2, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
  }
  return %alloc : memref<8x32x32x32xf32>
}
}

// CHECK-LABEL:   memref.global "private" constant @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16> = dense<1.048580e+06> {alignment = 64 : i64}

// CHECK-LABEL:   func.func @entry(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<8x32x32x32xbf16>) -> memref<8x32x32x32xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<0.000000e+00> : vector<32x32xf32>
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = memref.get_global @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16>
// CHECK:           %[[VAL_7:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x32x32x32xf32>
// CHECK:           %[[VAL_8:.*]] = memref.expand_shape %[[VAL_0]] {{\[\[}}0], [1], [2], [3, 4]] output_shape [8, 32, 32, 16, 2] : memref<8x32x32x32xbf16> into memref<8x32x32x16x2xbf16>
// CHECK:           scf.forall (%[[VAL_9:.*]], %[[VAL_10:.*]]) in (8, 32) {
// CHECK:             %[[VAL_11:.*]] = memref.subview %[[VAL_7]]{{\[}}%[[VAL_9]], %[[VAL_10]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:             vector.transfer_write %[[VAL_2]], %[[VAL_11]]{{\[}}%[[VAL_5]], %[[VAL_5]]] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:             %[[VAL_12:.*]] = memref.subview %[[VAL_8]]{{\[}}%[[VAL_9]], 0, 0, 0, 0] [1, 32, 32, 16, 2] [1, 1, 1, 1, 1] : memref<8x32x32x16x2xbf16> to memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
// CHECK:             %[[VAL_13:.*]] = amx.tile_load %[[VAL_11]]{{\[}}%[[VAL_5]], %[[VAL_5]]] : memref<32x32xf32, strided<[32, 1], offset: ?>> into !amx.tile<16x16xf32>
// CHECK:             %[[VAL_14:.*]] = amx.tile_load %[[VAL_11]]{{\[}}%[[VAL_5]], %[[VAL_1]]] : memref<32x32xf32, strided<[32, 1], offset: ?>> into !amx.tile<16x16xf32>
// CHECK:             %[[VAL_15:.*]] = amx.tile_load %[[VAL_11]]{{\[}}%[[VAL_1]], %[[VAL_5]]] : memref<32x32xf32, strided<[32, 1], offset: ?>> into !amx.tile<16x16xf32>
// CHECK:             %[[VAL_16:.*]] = amx.tile_load %[[VAL_11]]{{\[}}%[[VAL_1]], %[[VAL_1]]] : memref<32x32xf32, strided<[32, 1], offset: ?>> into !amx.tile<16x16xf32>
// CHECK:             %[[VAL_17:.*]]:4 = scf.for %[[VAL_18:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_3]] iter_args(%[[VAL_19:.*]] = %[[VAL_13]], %[[VAL_20:.*]] = %[[VAL_14]], %[[VAL_21:.*]] = %[[VAL_15]], %[[VAL_22:.*]] = %[[VAL_16]]) -> (!amx.tile<16x16xf32>, !amx.tile<16x16xf32>, !amx.tile<16x16xf32>, !amx.tile<16x16xf32>) {
// CHECK:               %[[VAL_23:.*]] = memref.subview %[[VAL_12]]{{\[}}%[[VAL_18]], 0, 0, 0] [1, 32, 16, 2] [1, 1, 1, 1] : memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>> to memref<1x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
// CHECK:               %[[VAL_24:.*]] = amx.tile_load %[[VAL_23]]{{\[}}%[[VAL_5]], %[[VAL_5]], %[[VAL_5]], %[[VAL_5]]] : memref<1x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>> into !amx.tile<16x32xbf16>
// CHECK:               %[[VAL_25:.*]] = amx.tile_load %[[VAL_23]]{{\[}}%[[VAL_5]], %[[VAL_1]], %[[VAL_5]], %[[VAL_5]]] : memref<1x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>> into !amx.tile<16x32xbf16>
// CHECK:               %[[VAL_26:.*]] = memref.subview %[[VAL_6]]{{\[}}%[[VAL_18]], 0, 0, 0] [1, 16, 32, 2] [1, 1, 1, 1] : memref<32x16x32x2xbf16> to memref<1x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
// CHECK:               %[[VAL_27:.*]] = amx.tile_load %[[VAL_26]]{{\[}}%[[VAL_5]], %[[VAL_5]], %[[VAL_5]], %[[VAL_5]]] : memref<1x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> into !amx.tile<16x32xbf16>
// CHECK:               %[[VAL_28:.*]] = amx.tile_load %[[VAL_26]]{{\[}}%[[VAL_5]], %[[VAL_5]], %[[VAL_1]], %[[VAL_5]]] : memref<1x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>> into !amx.tile<16x32xbf16>
// CHECK:               %[[VAL_29:.*]] = amx.tile_mulf %[[VAL_24]], %[[VAL_27]], %[[VAL_19]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK:               %[[VAL_30:.*]] = amx.tile_mulf %[[VAL_24]], %[[VAL_28]], %[[VAL_20]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK:               %[[VAL_31:.*]] = amx.tile_mulf %[[VAL_25]], %[[VAL_27]], %[[VAL_21]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK:               %[[VAL_32:.*]] = amx.tile_mulf %[[VAL_25]], %[[VAL_28]], %[[VAL_22]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK:               scf.yield %[[VAL_29]], %[[VAL_30]], %[[VAL_31]], %[[VAL_32]] : !amx.tile<16x16xf32>, !amx.tile<16x16xf32>, !amx.tile<16x16xf32>, !amx.tile<16x16xf32>
// CHECK:             }
// CHECK:             amx.tile_store %[[VAL_11]]{{\[}}%[[VAL_5]], %[[VAL_5]]], %[[VAL_33:.*]]#0 : memref<32x32xf32, strided<[32, 1], offset: ?>>, !amx.tile<16x16xf32>
// CHECK:             amx.tile_store %[[VAL_11]]{{\[}}%[[VAL_5]], %[[VAL_1]]], %[[VAL_33]]#1 : memref<32x32xf32, strided<[32, 1], offset: ?>>, !amx.tile<16x16xf32>
// CHECK:             amx.tile_store %[[VAL_11]]{{\[}}%[[VAL_1]], %[[VAL_5]]], %[[VAL_33]]#2 : memref<32x32xf32, strided<[32, 1], offset: ?>>, !amx.tile<16x16xf32>
// CHECK:             amx.tile_store %[[VAL_11]]{{\[}}%[[VAL_1]], %[[VAL_1]]], %[[VAL_33]]#3 : memref<32x32xf32, strided<[32, 1], offset: ?>>, !amx.tile<16x16xf32>
// CHECK:           }
// CHECK:           return %[[VAL_7]] : memref<8x32x32x32xf32>
// CHECK:         }

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

// CHECK-LABEL:   memref.global "private" constant @__constant_16x32x64x2xbf16 : memref<16x32x64x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}

// CHECK-LABEL:   func.func @entry(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<4x16x64x64xbf16>) -> memref<4x16x64x64xbf16> {
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<0.000000e+00> : vector<64x64xbf16>
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = memref.get_global @__constant_16x32x64x2xbf16 : memref<16x32x64x2xbf16>
// CHECK:           %[[VAL_8:.*]] = memref.alloc() {alignment = 64 : i64} : memref<4x16x64x64xbf16>
// CHECK:           %[[VAL_9:.*]] = memref.expand_shape %[[VAL_0]] {{\[\[}}0], [1], [2], [3, 4]] output_shape [4, 16, 64, 32, 2] : memref<4x16x64x64xbf16> into memref<4x16x64x32x2xbf16>
// CHECK:           scf.forall (%[[VAL_10:.*]], %[[VAL_11:.*]]) in (4, 16) {
// CHECK:             %[[VAL_12:.*]] = memref.subview %[[VAL_8]]{{\[}}%[[VAL_10]], %[[VAL_11]], 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<4x16x64x64xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
// CHECK:             vector.transfer_write %[[VAL_1]], %[[VAL_12]]{{\[}}%[[VAL_6]], %[[VAL_6]]] {in_bounds = [true, true]} : vector<64x64xbf16>, memref<64x64xbf16, strided<[64, 1], offset: ?>>
// CHECK:             %[[VAL_13:.*]] = memref.subview %[[VAL_9]]{{\[}}%[[VAL_10]], 0, 0, 0, 0] [1, 16, 64, 32, 2] [1, 1, 1, 1, 1] : memref<4x16x64x32x2xbf16> to memref<16x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>
// CHECK:             scf.for %[[VAL_14:.*]] = %[[VAL_6]] to %[[VAL_5]] step %[[VAL_4]] {
// CHECK:               scf.for %[[VAL_15:.*]] = %[[VAL_6]] to %[[VAL_5]] step %[[VAL_4]] {
// CHECK:                 %[[VAL_16:.*]] = memref.subview %[[VAL_12]]{{\[}}%[[VAL_14]], %[[VAL_15]]] [32, 32] [1, 1] : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<32x32xbf16, strided<[64, 1], offset: ?>>
// CHECK:                 %[[VAL_17:.*]] = amx.tile_load %[[VAL_16]]{{\[}}%[[VAL_6]], %[[VAL_6]]] : memref<32x32xbf16, strided<[64, 1], offset: ?>> into !amx.tile<16x16xf32>
// CHECK:                 %[[VAL_18:.*]] = amx.tile_load %[[VAL_16]]{{\[}}%[[VAL_6]], %[[VAL_3]]] : memref<32x32xbf16, strided<[64, 1], offset: ?>> into !amx.tile<16x16xf32>
// CHECK:                 %[[VAL_19:.*]] = amx.tile_load %[[VAL_16]]{{\[}}%[[VAL_3]], %[[VAL_6]]] : memref<32x32xbf16, strided<[64, 1], offset: ?>> into !amx.tile<16x16xf32>
// CHECK:                 %[[VAL_20:.*]] = amx.tile_load %[[VAL_16]]{{\[}}%[[VAL_3]], %[[VAL_3]]] : memref<32x32xbf16, strided<[64, 1], offset: ?>> into !amx.tile<16x16xf32>
// CHECK:                 %[[VAL_21:.*]]:4 = scf.for %[[VAL_22:.*]] = %[[VAL_6]] to %[[VAL_3]] step %[[VAL_2]] iter_args(%[[VAL_23:.*]] = %[[VAL_17]], %[[VAL_24:.*]] = %[[VAL_18]], %[[VAL_25:.*]] = %[[VAL_19]], %[[VAL_26:.*]] = %[[VAL_20]]) -> (!amx.tile<16x16xf32>, !amx.tile<16x16xf32>, !amx.tile<16x16xf32>, !amx.tile<16x16xf32>) {
// CHECK:                   %[[VAL_27:.*]]:4 = scf.for %[[VAL_28:.*]] = %[[VAL_6]] to %[[VAL_4]] step %[[VAL_3]] iter_args(%[[VAL_29:.*]] = %[[VAL_23]], %[[VAL_30:.*]] = %[[VAL_24]], %[[VAL_31:.*]] = %[[VAL_25]], %[[VAL_32:.*]] = %[[VAL_26]]) -> (!amx.tile<16x16xf32>, !amx.tile<16x16xf32>, !amx.tile<16x16xf32>, !amx.tile<16x16xf32>) {
// CHECK:                     %[[VAL_33:.*]] = memref.subview %[[VAL_13]]{{\[}}%[[VAL_22]], %[[VAL_14]], %[[VAL_28]], 0] [1, 32, 16, 2] [1, 1, 1, 1] : memref<16x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>> to memref<1x32x16x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>
// CHECK:                     %[[VAL_34:.*]] = amx.tile_load %[[VAL_33]]{{\[}}%[[VAL_6]], %[[VAL_6]], %[[VAL_6]], %[[VAL_6]]] : memref<1x32x16x2xbf16, strided<[4096, 64, 2, 1], offset: ?>> into !amx.tile<16x32xbf16>
// CHECK:                     %[[VAL_35:.*]] = amx.tile_load %[[VAL_33]]{{\[}}%[[VAL_6]], %[[VAL_3]], %[[VAL_6]], %[[VAL_6]]] : memref<1x32x16x2xbf16, strided<[4096, 64, 2, 1], offset: ?>> into !amx.tile<16x32xbf16>
// CHECK:                     %[[VAL_36:.*]] = memref.subview %[[VAL_7]]{{\[}}%[[VAL_22]], %[[VAL_28]], %[[VAL_15]], 0] [1, 16, 32, 2] [1, 1, 1, 1] : memref<16x32x64x2xbf16> to memref<1x16x32x2xbf16, strided<[4096, 128, 2, 1], offset: ?>>
// CHECK:                     %[[VAL_37:.*]] = amx.tile_load %[[VAL_36]]{{\[}}%[[VAL_6]], %[[VAL_6]], %[[VAL_6]], %[[VAL_6]]] : memref<1x16x32x2xbf16, strided<[4096, 128, 2, 1], offset: ?>> into !amx.tile<16x32xbf16>
// CHECK:                     %[[VAL_38:.*]] = amx.tile_load %[[VAL_36]]{{\[}}%[[VAL_6]], %[[VAL_6]], %[[VAL_3]], %[[VAL_6]]] : memref<1x16x32x2xbf16, strided<[4096, 128, 2, 1], offset: ?>> into !amx.tile<16x32xbf16>
// CHECK:                     %[[VAL_39:.*]] = amx.tile_mulf %[[VAL_34]], %[[VAL_37]], %[[VAL_29]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK:                     %[[VAL_40:.*]] = amx.tile_mulf %[[VAL_34]], %[[VAL_38]], %[[VAL_30]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK:                     %[[VAL_41:.*]] = amx.tile_mulf %[[VAL_35]], %[[VAL_37]], %[[VAL_31]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK:                     %[[VAL_42:.*]] = amx.tile_mulf %[[VAL_35]], %[[VAL_38]], %[[VAL_32]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK:                     scf.yield %[[VAL_39]], %[[VAL_40]], %[[VAL_41]], %[[VAL_42]] : !amx.tile<16x16xf32>, !amx.tile<16x16xf32>, !amx.tile<16x16xf32>, !amx.tile<16x16xf32>
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_43:.*]]#0, %[[VAL_43]]#1, %[[VAL_43]]#2, %[[VAL_43]]#3 : !amx.tile<16x16xf32>, !amx.tile<16x16xf32>, !amx.tile<16x16xf32>, !amx.tile<16x16xf32>
// CHECK:                 }
// CHECK:                 amx.tile_store %[[VAL_16]]{{\[}}%[[VAL_6]], %[[VAL_6]]], %[[VAL_44:.*]]#0 : memref<32x32xbf16, strided<[64, 1], offset: ?>>, !amx.tile<16x16xf32>
// CHECK:                 amx.tile_store %[[VAL_16]]{{\[}}%[[VAL_6]], %[[VAL_3]]], %[[VAL_44]]#1 : memref<32x32xbf16, strided<[64, 1], offset: ?>>, !amx.tile<16x16xf32>
// CHECK:                 amx.tile_store %[[VAL_16]]{{\[}}%[[VAL_3]], %[[VAL_6]]], %[[VAL_44]]#2 : memref<32x32xbf16, strided<[64, 1], offset: ?>>, !amx.tile<16x16xf32>
// CHECK:                 amx.tile_store %[[VAL_16]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_44]]#3 : memref<32x32xbf16, strided<[64, 1], offset: ?>>, !amx.tile<16x16xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return %[[VAL_8]] : memref<4x16x64x64xbf16>
// CHECK:         }

