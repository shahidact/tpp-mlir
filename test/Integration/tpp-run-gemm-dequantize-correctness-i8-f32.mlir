// RUN: tpp-run -e entry --entry-point-result=void --linalg-to-loops -print --splat-to-random --init-type quant -seed 123  %s > %t.1
// RUN: tpp-run -e entry --entry-point-result=void -print --vector-to-kernels --registerBlocking=32,32,64 --splat-to-random -seed 123 --init-type quant %s > %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.2


#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
#map3 = affine_map<(d0, d1) -> (d0)>
#map4 = affine_map<(d0, d1) -> (d1)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @entry(%arg0: tensor<2x2x32x64xi8>, %arg1: tensor<64xf32>, %arg2: tensor<2x2x16x32x4xi8>, %arg3: tensor<64xf32>, %arg4: tensor<2x2x32x32xf32>) -> tensor<2x2x32x32xf32> {
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<2x2x32x32xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<2x2x32x32xi32>) -> tensor<2x2x32x32xi32>
    %expanded = tensor.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [2, 2, 32, 16, 4] : tensor<2x2x32x64xi8> into tensor<2x2x32x16x4xi8>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%expanded, %arg2 : tensor<2x2x32x16x4xi8>, tensor<2x2x16x32x4xi8>) outs(%1 : tensor<2x2x32x32xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %8 = arith.extsi %in : i8 to i32
      %9 = arith.extsi %in_0 : i8 to i32
      %10 = arith.muli %8, %9 : i32
      %11 = arith.addi %out, %10 : i32
      linalg.yield %11 : i32
    } -> tensor<2x2x32x32xi32>
    %3 = tensor.empty() : tensor<64x64xf32>
    %4 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg3 : tensor<64xf32>, tensor<64xf32>) outs(%3 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %8 = arith.mulf %in, %in_0 : f32
      linalg.yield %8 : f32
    } -> tensor<64x64xf32>
    %5 = tensor.empty() : tensor<2x2x32x32xf32>
    %pack = linalg.pack %4 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %5 : tensor<64x64xf32> -> tensor<2x2x32x32xf32>
    %6 = linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<2x2x32x32xi32>) outs(%arg4 : tensor<2x2x32x32xf32>) {
    ^bb0(%in: i32, %out: f32):
      %8 = arith.sitofp %in : i32 to f32
      linalg.yield %8 : f32
    } -> tensor<2x2x32x32xf32>
    %7 = linalg.mul ins(%6, %pack : tensor<2x2x32x32xf32>, tensor<2x2x32x32xf32>) outs(%arg4 : tensor<2x2x32x32xf32>) -> tensor<2x2x32x32xf32>
    return %7 : tensor<2x2x32x32xf32>
  }
}