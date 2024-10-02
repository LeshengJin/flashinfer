/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <flashinfer/hip/sampling.hpp>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_bf16.h>
#include "flashinfer_ops.h"
// #include "pytorch_extension_utils.h"

using namespace flashinfer;

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)

#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

#define CHECK_GE(a, b) TORCH_CHECK((a) >= (b), "CHECK_GE(" #a ", " #b ") failed. ", a, " vs ", b)

torch::Tensor sampling_from_probs(torch::Tensor probs, torch::Tensor uniform_samples,
                                  bool deterministic) {
  CHECK_INPUT(probs);
  CHECK_INPUT(uniform_samples);
  auto device = probs.device();
  CHECK_EQ(uniform_samples.device(), device);
  CHECK_DIM(2, probs);            // probs: (batch_size, vocab_size)
  CHECK_DIM(1, uniform_samples);  // uniform_samples: (batch_size)
  CHECK_EQ(probs.size(0), uniform_samples.size(0));
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  probs = probs.to(torch::kFloat32);
  uniform_samples = uniform_samples.to(torch::kFloat32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto samples = torch::empty({batch_size}, torch::dtype(torch::kInt32).device(device));

  cudaError_t status = sampling::SamplingFromProb(static_cast<float*>(probs.data_ptr()),
                                                  static_cast<float*>(uniform_samples.data_ptr()),
                                                  static_cast<int*>(samples.data_ptr()), batch_size,
                                                  vocab_size, deterministic, torch_current_stream);
  TORCH_CHECK(status == cudaSuccess, "SamplingFromProbs failed with error code " +
                                         std::string(cudaGetErrorString(status)));
  return samples;
}

std::vector<torch::Tensor> top_p_sampling_from_probs(torch::Tensor probs,
                                                     torch::Tensor uniform_samples,
                                                     std::optional<torch::Tensor> maybe_top_p_arr,
                                                     double top_p_val, bool deterministic) {
  CHECK_INPUT(probs);
  CHECK_INPUT(uniform_samples);
  auto device = probs.device();
  CHECK_EQ(uniform_samples.device(), device);
  CHECK_DIM(2, probs);            // probs: (batch_size, vocab_size)
  CHECK_DIM(2, uniform_samples);  // uniform_samples: (max_top_p_rounds, batch_size)
  CHECK_EQ(probs.size(0), uniform_samples.size(1));
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  unsigned int max_top_p_rounds = uniform_samples.size(0);
  bool has_top_p_arr = maybe_top_p_arr.has_value();
  auto top_p_arr = maybe_top_p_arr.value_or(torch::empty({0}, torch::dtype(torch::kFloat32)));
  if (has_top_p_arr) {
    CHECK_INPUT(top_p_arr);
    CHECK_DIM(1, top_p_arr);  // top_p_arr: (batch_size,)
    CHECK_EQ(top_p_arr.size(0), batch_size);
    CHECK_EQ(top_p_arr.device(), device);
  }
  probs = probs.to(torch::kFloat32);
  uniform_samples = uniform_samples.to(torch::kFloat32);
  top_p_arr = top_p_arr.to(torch::kFloat32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto samples = torch::empty({batch_size}, torch::dtype(torch::kInt32).device(device));
  auto success = torch::empty({batch_size}, torch::dtype(torch::kBool).device(device));

  cudaError_t status = sampling::TopPSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(uniform_samples.data_ptr()),
      static_cast<int*>(samples.data_ptr()), static_cast<bool*>(success.data_ptr()),
      has_top_p_arr ? static_cast<float*>(top_p_arr.data_ptr()) : nullptr, batch_size, top_p_val,
      vocab_size, max_top_p_rounds, deterministic, torch_current_stream);
  TORCH_CHECK(status == cudaSuccess, "TopPSamplingFromProbs failed with error code " +
                                         std::string(cudaGetErrorString(status)));

  return {samples, success};
}

std::vector<torch::Tensor> top_k_sampling_from_probs(torch::Tensor probs,
                                                     torch::Tensor uniform_samples,
                                                     std::optional<torch::Tensor> maybe_top_k_arr,
                                                     unsigned int top_k_val, bool deterministic) {
  CHECK_INPUT(probs);
  CHECK_INPUT(uniform_samples);
  auto device = probs.device();
  CHECK_EQ(uniform_samples.device(), device);
  CHECK_DIM(2, probs);            // probs: (batch_size, vocab_size)
  CHECK_DIM(2, uniform_samples);  // uniform_samples: (max_top_k_rounds, batch_size)
  CHECK_EQ(probs.size(0), uniform_samples.size(1));
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  unsigned int max_top_k_rounds = uniform_samples.size(0);
  bool has_top_k_arr = maybe_top_k_arr.has_value();
  auto top_k_arr = maybe_top_k_arr.value_or(torch::empty({0}, torch::dtype(torch::kInt32)));
  if (has_top_k_arr) {
    CHECK_INPUT(top_k_arr);
    CHECK_DIM(1, top_k_arr);  // top_k_arr: (batch_size,)
    CHECK_EQ(top_k_arr.size(0), batch_size);
    CHECK_EQ(top_k_arr.device(), device);
  }
  probs = probs.to(torch::kFloat32);
  uniform_samples = uniform_samples.to(torch::kFloat32);
  top_k_arr = top_k_arr.to(torch::kInt32);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto samples = torch::empty({batch_size}, torch::dtype(torch::kInt32).device(device));
  auto success = torch::empty({batch_size}, torch::dtype(torch::kBool).device(device));

  cudaError_t status = sampling::TopKSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(uniform_samples.data_ptr()),
      static_cast<int*>(samples.data_ptr()), static_cast<bool*>(success.data_ptr()),
      has_top_k_arr ? static_cast<float*>(top_k_arr.data_ptr()) : nullptr, batch_size, top_k_val,
      vocab_size, max_top_k_rounds, deterministic, torch_current_stream);
  TORCH_CHECK(status == cudaSuccess, "TopKSamplingFromProbs failed with error code " +
                                         std::string(cudaGetErrorString(status)));

  return {samples, success};
}

