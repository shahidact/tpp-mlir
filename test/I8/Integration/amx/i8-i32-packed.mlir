// RUN: tpp-opt --default-tpp-passes="vector-to-kernel registerBlocking=32,32,64" %s | FileCheck %s -check-prefix=IR

// RUN: tpp-run -e entry --entry-point-result=void -print --splat-to-random --init-type quant -seed 123  %s > %t.1
// RUN: tpp-opt --default-tpp-passes="vector-to-kernel registerBlocking=32,32,64" %s | tpp-run -e entry --target-feature="-avx, -avx2, +amx_tile, +amx_int8" --entry-point-result=void --print --splat-to-random -seed 123 --init-type quant %s > %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.2

// IR-COUNT-4: amx.tile_muli
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
module {
  func.func @entry(%arg0: tensor<2x2x32x64xi8>, %arg1: tensor<64xf32>, %arg2: tensor<2x2x16x32x4xi8>, %arg3: tensor<64xf32>, %arg4: tensor<2x2x32x32xi32>) -> tensor<2x2x32x32xi32> {
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<2x2x32x32xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<2x2x32x32xi32>) -> tensor<2x2x32x32xi32>
    %expanded = tensor.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [2, 2, 32, 16, 4] : tensor<2x2x32x64xi8> into tensor<2x2x32x16x4xi8>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%expanded, %arg2 : tensor<2x2x32x16x4xi8>, tensor<2x2x16x32x4xi8>) outs(%1 : tensor<2x2x32x32xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %3 = arith.extsi %in : i8 to i32
      %4 = arith.extsi %in_0 : i8 to i32
      %5 = arith.muli %3, %4 : i32
      %6 = arith.addi %out, %5 : i32
      linalg.yield %6 : i32
    } -> tensor<2x2x32x32xi32>
    return %2 : tensor<2x2x32x32xi32>
  }
}