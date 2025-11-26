//===- TensorInitInt.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Transforms/Utils/TensorInitInt.h"
#include "TPP/Transforms/Utils/TensorInitFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <vector>

using namespace mlir;

TensorInitInt::DataType TensorInitInt::getTensorInitDataType(mlir::Type type) {
  if (type.isSignlessInteger(8))
    return DataType::I8;
  if (type.isSignlessInteger(16))
    return DataType::I16;
  if (type.isSignlessInteger(32))
    return DataType::I32;
  if (type.isSignlessInteger(64))
    return DataType::I64;
  return DataType::AUTO;
}

unsigned TensorInitInt::getDataTypeBitWidth(TensorInitInt::DataType type) {
  switch (type) {
  case DataType::I8:
    return 8;
  case DataType::I16:
    return 16;
  case DataType::I32:
    return 32;
  case DataType::I64:
    return 64;
  case DataType::AUTO:
    return 32;
  }
  llvm_unreachable("unknown type");
}

bool TensorInitInt::isDataTypeSigned(TensorInitInt::DataType type) {
  switch (type) {
  case DataType::I8:
  case DataType::I16:
  case DataType::I32:
  case DataType::I64:
  case DataType::AUTO:
    return true;
  }
  llvm_unreachable("unknown type");
}

void TensorInitInt::insert(size_t index, uint64_t value) {
  this->TensorInit::insert(index, APInt(bitWidth, value, isSigned));
}

void TensorInitInt::push(uint64_t value) {
  this->TensorInit::push(APInt(bitWidth, value, isSigned));
}

void TensorInitInt::convertType(llvm::APInt &value) {
  assert(value.getBitWidth() == bitWidth && "Invalid element size");
}

FailureOr<DenseElementsAttr> ConstantTensorInitInt::get(ShapedType shape) {
  auto value = APInt(bitWidth, 1, isSigned);
  if (!isTypeSupported(shape.getElementType()))
    assert(false && "Element type not supported");
  convertType(value);

  // For some reason, memref global op needs dense tensor type
  // See: lib/Dialect/MemRef/IR/MemRefOps.cpp :: GlobalOp::verify
  auto tensorType =
      RankedTensorType::get(shape.getShape(), shape.getElementType());
  return mlir::DenseElementsAttr::get(tensorType, value);
}

void RandomTensorInitInt::fillData() {
  llvm::errs() << "RandomTensorInitInt::fillData() \n";
  assert(buffer.size() == 0 && "Buffer not empty");
  for (size_t i = 0; i < size; i++)
    push(next());
}

void NormalTensorInitInt::fillData() {
  llvm::errs() << "NormalTensorInitInt::fillData()\n";
  assert(buffer.size() == 0 && "Buffer not empty");
  for (size_t i = 0; i < size; i++)
    push(next());
}

// Compute a simple dynamic fixed point per channel quantization scale for the
// 2-D flattened tensor data Find the max absolute value in the distribution
// channel wise and use it to determine the unbiased exponent using frexp().
std::vector<int>
QuantTensorInitInt::computeScales(const std::vector<float> &samples) {
  llvm::errs() << "QuantTensorInitInt::computeScales\n";
  std::vector<int> channelwiseScales;
  if (dims.size() != 2) {
    llvm::errs()
        << "QuantTensorInitInt::computeScales only supports 2D tensors\n";
    return channelwiseScales;
  }
  size_t channels = dims[1]; // Assuming shape is [channelSize, channels]
  size_t channelSize = dims[0];
  channelwiseScales.resize(channels, 0);

  for (size_t c = 0; c < channels; c++) {
    float maxAbsValue = 0.0f;
    for (size_t i = 0; i < channelSize; i++) {
      float value = samples[i * channels + c];
      llvm::errs() << "computeScales::samples " << value << " \n";
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

    channelwiseScales[c] = std::exp2(-(exponent));
    llvm::errs() << "channelWiseScale[" << c << "]=" << channelwiseScales[c]
                 << "\n";
    llvm::errs() << "Channel " << c << ": maxAbsValue=" << maxAbsValue
                 << ", exponent=" << exponent << ", intBits=" << intBits
                 << "\n";
  }
  return channelwiseScales;
}

// Determine the fractional length based on the max absolute value
std::vector<APInt>
QuantTensorInitInt::quantizeDFP(const std::vector<float> &samples,
                                const std::vector<int> &channelwiseScales) {
  std::vector<APInt> quantizedValues;
  if (dims.size() != 2) {
    llvm::errs()
        << "QuantTensorInitInt::quantizeDFP only supports 2D tensors\n";
    return quantizedValues;
  }
  size_t channels = dims[1];
  size_t channelSize = dims[0];
  quantizedValues.resize(samples.size(), APInt(bitWidth, 0, isSigned));

  for (size_t c = 0; c < channels; c++) {
    int scale = channelwiseScales[c];
    for (size_t i = 0; i < channelSize; i++) {
      float value = samples[i * channels + c];
      int quantized = static_cast<int>(std::round(value * scale));
      // Clamp to the representable range
      int64_t minVal = isSigned ? -(1LL << (bitWidth - 1)) : 0;
      int64_t maxVal =
          isSigned ? ((1LL << (bitWidth - 1)) - 1) : ((1LL << bitWidth) - 1);
      if (quantized < minVal)
        quantized = minVal;
      if (quantized > maxVal)
        quantized = maxVal;

      quantizedValues[c * channelSize + i] =
          APInt(bitWidth, quantized, isSigned);
    }
  }
  return quantizedValues;
}

void QuantTensorInitInt::fillData() {
  llvm::errs() << "QuantTensorInitInt::fillData() implemented\n";
  assert(buffer.size() == 0 && "Buffer not empty");

  std::vector<float> samples;
  for (size_t i = 0; i < size; i++) {
    auto p = next();
    samples.push_back(p);
  }

  // Print the samples row-wise where dim[1] is the inner dimension
  if (dims.size() == 2) {
    size_t channels = dims[1];
    size_t channelSize = dims[0];
    for (size_t i = 0; i < channelSize; i++) {
      llvm::errs() << "Row " << i << ": ";
      for (size_t c = 0; c < channels; c++) {
        llvm::errs() << samples[i * channels + c] << " ";
      }
      llvm::errs() << "\n";
    }
  }

  std::vector<int> channelwiseScales = computeScales(samples);

  // Quantize each sample using the channel-wise scale
  std::vector<APInt> quantizedValues = quantizeDFP(samples, channelwiseScales);
  buffer = quantizedValues;

  // Update the QuantTensorInitFloat buffer
  if (floatInit) {
    llvm::errs() << "floatInt" << floatInit << "\n";
    llvm::errs() << "QuantTensorInitInt::fillData(): Update "
                    "QuantTensorInitFloat scale buffer\n";
    std::vector<llvm::APFloat> floatScales;
    for (int scale : channelwiseScales) {
      llvm::errs() << "IdentityTensorInitInt::Scale " << scale << " \n";
      floatScales.emplace_back(static_cast<float>(scale));
    }
    floatInit->updateBuffer(floatScales);
  }
}

void IdentityTensorInitInt::fillData() {
  llvm::errs() << "IdentityTensorInitInt::fillData()\n";
  assert(buffer.size() == 0 && "Buffer not empty");
  APInt zero = APInt(bitWidth, 0, isSigned);
  convertType(zero);
  buffer.resize(size, zero);
  size_t ld = dims[0];

  // Shape is guaranteed to be "2D square" by `checkShape()`
  for (size_t i=0; i < ld; i++) {
    size_t offset = i*ld + i;
    insert(offset, APInt(bitWidth, 1, isSigned));
  }
}
