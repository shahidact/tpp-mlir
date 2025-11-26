//===- TensorInitFloat.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Transforms/Utils/TensorInitFloat.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

TensorInitFloat::DataType
TensorInitFloat::getTensorInitDataType(mlir::Type type) {
  if (type.isBF16())
    return DataType::BF16;
  if (type.isF16())
    return DataType::FP16;
  if (type.isF32())
    return DataType::FP32;
  if (type.isF64())
    return DataType::FP64;
  return DataType::AUTO;
}

void TensorInitFloat::insert(size_t index, float value) {
  this->TensorInit::insert(index, APFloat(value));
}

void TensorInitFloat::push(float value) {
  this->TensorInit::push(APFloat(value));
}

void TensorInitFloat::convertType(llvm::APFloat &value) {
  switch (type) {
  case DataType::FP16:
    toFP16(value);
    break;
  case DataType::FP32:
    toFP32(value);
    break;
  case DataType::FP64:
    toFP64(value);
    break;
  case DataType::BF16:
    toBF16(value);
    break;
  case DataType::AUTO:
    toFP32(value);
    break;
  }
}

FailureOr<DenseElementsAttr> ConstantTensorInitFloat::get(ShapedType shape) {
  auto floatValue = APFloat(1.0F);
  if (!isTypeSupported(shape.getElementType()))
    assert(false && "Element type not supported");
  convertType(floatValue);

  // For some reason, memref global op needs dense tensor type
  // See: lib/Dialect/MemRef/IR/MemRefOps.cpp :: GlobalOp::verify
  auto tensorType =
      RankedTensorType::get(shape.getShape(), shape.getElementType());
  return mlir::DenseElementsAttr::get(tensorType, floatValue);
}

void RandomTensorInitFloat::fillData() {
  assert(buffer.size() == 0 && "Buffer not empty");
  for (size_t i = 0; i < size; i++)
    push(next());
}

void NormalTensorInitFloat::fillData() {
  assert(buffer.size() == 0 && "Buffer not empty");
  for (size_t i = 0; i < size; i++)
    push(next());
}

// Compute a simple dynamic fixed point per channel quantization scale for the
// 2-D flattened tensor data Find the max absolute value in the distribution
// channel wise and use it to determine the unbiased exponent using frexp().
std::vector<APFloat>
QuantTensorInitFloat::computeReScales(const std::vector<float> &samples) {
  llvm::errs() << "QuantTensorInitFloat::computeReScales\n";
  std::vector<APFloat> channelwiseScales;
  if (dims.size() != 1) {
    llvm::errs()
        << "QuantTensorInitInt::computeReScales only supports 1-D tensors\n";
    return channelwiseScales;
  }
  size_t channels = dims[1]; // Assuming shape is [channelSize, channels]
  size_t channelSize = dims[0];
  channelwiseScales.resize(channels, APFloat(0.0f));

  for (size_t c = 0; c < channels; c++) {
    float maxAbsValue = 0.0f;
    for (size_t i = 0; i < channelSize; i++) {
      float value = samples[i * channels + c];
      float absValue = std::abs(value);
      if (absValue > maxAbsValue)
        maxAbsValue = absValue;
    }
    int exponent;
    if (maxAbsValue == 0.0f) {
      exponent = 0; // Handle zero case
    } else {
      std::frexp(maxAbsValue, &exponent); // Get unbiased exponent
    }
    exponent = exponent - 7;
    int intBits = exponent + 1; // +1 for sign bit
    // APFloat(bitWidth, std::exp2(-(exponent)), isSigned);
    channelwiseScales[c] = APFloat(static_cast<float>(std::exp2((exponent))));
    llvm::errs() << "channelWiseScale[" << c << "]=" << channelwiseScales[c]
                 << "\n";
    llvm::errs() << "Channel " << c << ": maxAbsValue=" << maxAbsValue
                 << ", exponent=" << exponent << ", intBits=" << intBits
                 << "\n";
  }
  return channelwiseScales;
}

void QuantTensorInitFloat::fillData() {
  llvm::errs() << "floatInt" << this << "\n";
  llvm::errs() << "Buffer Size: " << buffer.size() << "\n";
  // assert(buffer.size() > 0 && "Buffer is empty");
  assert(scaleSamples.size() > 0 && "scaleSamples is empty");
  llvm::errs() << "QuantTensorInitFloat::fillData()\n";
  for (size_t i = 0; i < scaleSamples.size(); i++) {
    llvm::errs() << "QuantTensorInitFloat::fillData() buffer[" << i
                 << "]=" << scaleSamples[i] << "\n";
    // push(next());
    push(scaleSamples[i]);
  }
}

// void QuantTensorInitFloat::fillData() {
//   llvm::errs() << "QuantTensorInitFloat::fillData() implemented\n";
//   assert(buffer.size() == 0 && "Buffer not empty");

//   std::vector<float> samples;
//   llvm::errs() << "Generating " << size << " samples\n";
//   for (size_t i = 0; i < size; i++) {
//     auto p = next();
//     llvm::errs() << "Samples " << i << ": " << p << "\n";
//     samples.push_back(p);
//   }

// Print the samples row-wise where dim[1] is the inner dimension
// if (dims.size() == 2) {
//   size_t channels = dims[1];
//   size_t channelSize = dims[0];
//   for (size_t i = 0; i < channelSize; i++) {
//     llvm::errs() << "Row " << i << ": ";
//     for (size_t c = 0; c < channels; c++) {
//       llvm::errs() << samples[i * channels + c] << " ";
//     }
//     llvm::errs() << "\n";
//   }
// }

//   // std::vector<APFloat> channelwiseScales = computeReScales(samples);

//   // Quantize each sample using the channel-wise scale
//   // std::vector<APInt> quantizedValues = quantizeDFP(samples,
//   // channelwiseScales);
//   // buffer = channelwiseScales;
//   // buffer = std::vector<APInt>(size, APInt(bitWidth, 0, isSigned));
// }

void IdentityTensorInitFloat::fillData() {
  assert(buffer.size() == 0 && "Buffer not empty");
  APFloat zero = APFloat(0.0);
  convertType(zero);
  buffer.resize(size, zero);
  size_t ld = dims[0];

  // Shape is guaranteed to be "2D square" by `checkShape()`
  for (size_t i=0; i < ld; i++) {
    size_t offset = i*ld + i;
    insert(offset, APFloat(1.0));
  }
}

void ZeroTensorInitFloat::fillData() {
  assert(buffer.size() == 0 && "Buffer not empty");
  APFloat zero = APFloat(0.0);
  convertType(zero);
  buffer.resize(size, zero);
}
