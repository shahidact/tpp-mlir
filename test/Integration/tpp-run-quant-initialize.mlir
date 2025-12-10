// RUN: tpp-run %s -e entry --entry-point-result=void --splat-to-random --init-type quant --print-input --seed 123 | FileCheck %s

// CHECK: ( -13, 67, 77 )
// CHECK: ( 3, 0, 76 )
// CHECK: ( 50, 6, -69 )
// CHECK: ( 0.00195312, 0.00390625, 0.00195312 )
// CHECK: ( -40, -59, 13, 66 )
// CHECK: ( 4, -23, -28, -40 )
// CHECK: ( -110, -99, -98, -67 )
// CHECK: ( 0.00390625, 0.00390625, 0.00390625, 0.00195312 )

!twoDimInputf32 = tensor<3x3xf32>
!twoDimWeightf32 = tensor<3x4xf32>
!twoDimInputi8 = tensor<3x3xi8>
!twoDimWeighti8 = tensor<3x4xi8>
!oneDimScaleInputf32 = tensor<3xf32>
!oneDimScaleWeightf32 = tensor<4xf32>


func.func @entry(%input : !twoDimInputi8, %iScale : !oneDimScaleInputf32, %weight : !twoDimWeighti8, %wScale : !oneDimScaleWeightf32) {
  return
}

// func.func @entry(%input : !twoDimInputf32, %weight : !twoDimWeightf32, %weight1 : !twoDimWeightf32) {
//   return
// }

// func.func @entry(%input : !twoDimInputi8, %iScale : !oneDimScaleInputf32, %weight : !twoDimWeighti8, %wScale : !oneDimScaleWeightf32) -> (!twoDimInputi8) {
//   return %iScale : !oneDimScaleInputf32
// }

// func.func @entry(%input : !twoDimInputi8, %iScale : !oneDimScaleInputf32, %weight : !twoDimWeighti8, %wScale : !oneDimScaleWeightf32) -> (!twoDimInputi8) {
//   return %weight : !twoDimWeighti8
// }

// func.func @entry(%input : !twoDimInputi8, %iScale : !oneDimScaleInputf32, %weight : !twoDimWeighti8, %wScale : !oneDimScaleWeightf32) -> (!twoDimInputi8) {
//   return %wScale : !oneDimScaleWeightf32
// }