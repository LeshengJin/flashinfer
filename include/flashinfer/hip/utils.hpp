// MIT License
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef UTILS_HPP
#define UTILS_HPP

// Compiling HIP on Windows includes windows.h, and this triggers many silly warnings.
#include <cstdint>
#if defined(_WIN32) && defined(__NVCC__)
    #pragma nv_diag_suppress 108 // signed bit field of length 1
    #pragma nv_diag_suppress 174 // expression has no effect
    #pragma nv_diag_suppress 1835 // attribute "dllimport" does not apply here
#endif

// rocPRIM adds a #warning about printf on NAVI.
#ifdef __clang__
    #pragma clang diagnostic ignored "-W#warnings"
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

constexpr int error_exit_code = -1;

/// \brief Checks if the provided error code is \p hipSuccess and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define HIP_CHECK(condition)                                                                \
    {                                                                                       \
        const hipError_t error = condition;                                                 \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            std::cerr << "An error encountered: \"" << hipGetErrorString(error) << "\" at " \
                      << __FILE__ << ':' << __LINE__ << std::endl;                          \
            std::exit(error_exit_code);                                                     \
        }                                                                                   \
    }

/// \brief Formats a range of elements to a pretty string.
/// \tparam BidirectionalIterator - must implement the BidirectionalIterator concept and
/// must be dereferencable in host code. Its value type must be formattable to
/// \p std::ostream.
template<class BidirectionalIterator>
inline std::string format_range(const BidirectionalIterator begin, const BidirectionalIterator end)
{
    std::stringstream sstream;
    sstream << "[ ";
    for(auto it = begin; it != end; ++it)
    {
        sstream << *it;
        if(it != std::prev(end))
        {
            sstream << ", ";
        }
    }
    sstream << " ]";
    return sstream.str();
}

/// \brief Formats a range of pairs to a pretty string. The length of the two ranges must match.
/// \tparam BidirectionalIteratorT - must implement the BidirectionalIterator concept and
/// must be dereferencable in host code. Its value type must be formattable to \p std::ostream.
/// \tparam BidirectionalIteratorU - must implement the BidirectionalIterator concept and
/// must be dereferencable in host code. Its value type must be formattable to \p std::ostream.
template<class BidirectionalIteratorT, typename BidirectionalIteratorU>
inline std::string format_pairs(const BidirectionalIteratorT begin_a,
                                const BidirectionalIteratorT end_a,
                                const BidirectionalIteratorU begin_b,
                                const BidirectionalIteratorU end_b)
{
    (void)end_b;
    assert(std::distance(begin_a, end_a) == std::distance(begin_b, end_b));

    std::stringstream sstream;
    sstream << "[ ";
    auto it_a = begin_a;
    auto it_b = begin_b;
    for(; it_a < end_a; ++it_a, ++it_b)
    {
        sstream << "(" << *it_a << ", " << *it_b << ")";

        if(it_a != std::prev(end_a))
        {
            sstream << ", ";
        }
    }
    sstream << " ]";
    return sstream.str();
}

/// \brief A function to parse a string for an int. If the string is a valid integer then return true
/// else if it has non-numeric character then return false.
inline bool parse_int_string(const std::string& str, int& out)
{
    try
    {
        size_t end;
        int    value = std::stoi(str, &end);
        if(end == str.size())
        {
            out = value;
            return true;
        }
        return false;
    }
    catch(const std::exception&)
    {
        return false;
    }
}

/// \brief A class to measures time between intervals
class HostClock
{
private:
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::duration   elapsed_time;

public:
    HostClock()
    {
        this->reset_timer();
    }

    inline void reset_timer()
    {
        this->elapsed_time = std::chrono::steady_clock::duration(0);
    }

    inline void start_timer()
    {
        this->start_time = std::chrono::steady_clock::now();
    }

    inline void stop_timer()
    {
        const auto end_time = std::chrono::steady_clock::now();
        this->elapsed_time += end_time - this->start_time;
    }

    /// @brief Returns time elapsed in Seconds
    /// @return type double that contains the elapsed time in Seconds
    inline double get_elapsed_time() const
    {
        return std::chrono::duration_cast<std::chrono::duration<double>>(this->elapsed_time)
            .count();
    }
};

/// \brief Returns <tt>ceil(dividend / divisor)</tt>, where \p dividend is an integer and
/// \p divisor is an unsigned integer.
template<typename T,
         typename U,
         std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<U>::value, int> = 0>
__host__ __device__ constexpr auto ceiling_div(const T& dividend, const U& divisor)
{
    return (dividend + divisor - 1) / divisor;
}

/// \brief Report validation results.
inline int report_validation_result(int errors)
{
    if(errors)
    {
        std::cout << "Validation failed. Errors: " << errors << std::endl;
        return error_exit_code;
    }

    std::cout << "Validation passed." << std::endl;
    return 0;
}

/// \brief Generate an identity matrix.
/// The identity matrix is a $m \times n$ matrix with ones in the main diagonal and zeros elsewhere.
template<typename T>
void generate_identity_matrix(T* A, int m, int n, size_t lda)
{
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            A[i + j * lda] = T(i == j);
        }
    }
}

/// \brief Multiply an $A$ matrix ($m \times k$) with a $B$ matrix ($k \times n$) as:
/// $C := \alpha \cdot A \cdot B + \beta \cdot C$
template<typename T>
void multiply_matrices(T        alpha,
                       T        beta,
                       int      m,
                       int      n,
                       int      k,
                       const T* A,
                       int      stride1_a,
                       int      stride2_a,
                       const T* B,
                       int      stride1_b,
                       int      stride2_b,
                       T*       C,
                       int      stride_c)
{
    for(int i1 = 0; i1 < m; ++i1)
    {
        for(int i2 = 0; i2 < n; ++i2)
        {
            T t = T(0.0);
            for(int i3 = 0; i3 < k; ++i3)
            {
                t += A[i1 * stride1_a + i3 * stride2_a] * B[i3 * stride1_b + i2 * stride2_b];
            }
            C[i1 + i2 * stride_c] = beta * C[i1 + i2 * stride_c] + alpha * t;
        }
    }
}

/// \brief Returns a string from the double \p value with specified \p precision .
inline std::string
    double_precision(const double value, const int precision, const bool fixed = false)
{
    std::stringstream ss;
    if(fixed)
    {
        ss << std::fixed;
    }
    ss << std::setprecision(precision) << value;
    return ss.str();
}

// Import ceil_div from flashinfer
template <typename T1, typename T2>
__forceinline__ __device__ __host__ T1 ceil_div(const T1 x, const T2 y) {
  return (x + y - 1) / y;
}

#define DISPATCH_ALIGNED_VEC_SIZE(aligned_vec_size, ALIGNED_VEC_SIZE, ...) \
  switch (aligned_vec_size) {                                              \
    case 16: {                                                             \
      constexpr size_t ALIGNED_VEC_SIZE = 16;                              \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 8: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 8;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 4: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 4;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 2: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 2;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 1: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 1;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    default: {                                                             \
      std::ostringstream err_msg;                                          \
      err_msg << "Unsupported aligned_vec_size: " << aligned_vec_size;     \
      throw std::invalid_argument(err_msg.str());                          \
    }                                                                      \
  }

/*
 * Copyright (c) 2023 by FlashInfer team.
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
#include <hip/hip_runtime.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// macro to turn off fp16 qk reduction to reduce binary
#ifndef FLASHINFER_ALWAYS_DISALLOW_FP16_QK_REDUCTION
#define FLASHINFER_ALWAYS_DISALLOW_FP16_QK_REDUCTION 0
#endif

#ifndef NDEBUG
#define FLASHINFER_CUDA_CALL(func, ...)                                                     \
  {                                                                                         \
    hipError_t e = (func);                                                                 \
    if (e != hipSuccess) {                                                                 \
      std::cerr << "CUDA Error: " << hipGetErrorString(e) << " (" << e << ") " << __FILE__ \
                << ": line " << __LINE__ << " at function " << STR(func) << std::endl;      \
      return e;                                                                             \
    }                                                                                       \
  }
#else
#define FLASHINFER_CUDA_CALL(func, ...) \
  {                                     \
    hipError_t e = (func);             \
    if (e != hipSuccess) {             \
      return e;                         \
    }                                   \
  }
#endif

#define DISPATCH_SPLIT_QO_INDPTR(split_qo_indptr, SPLIT_QO_INDPTR, ...) \
  if (split_qo_indptr) {                                                \
    constexpr bool SPLIT_QO_INDPTR = true;                              \
    __VA_ARGS__                                                         \
  } else {                                                              \
    constexpr bool SPLIT_QO_INDPTR = false;                             \
    __VA_ARGS__                                                         \
  }

#if FLASHINFER_ALWAYS_DISALLOW_FP16_QK_REDUCTION

#define DISPATCH_ALLOW_FP16_QK_REDUCTION(allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, ...) \
  if (allow_fp16_qk_reduction) {                                                                \
    throw std::runtime_error("FP16_QK_REDUCTION disabled at compile time");                     \
  } else {                                                                                      \
    constexpr bool ALLOW_FP16_QK_REDUCTION = false;                                             \
    __VA_ARGS__                                                                                 \
  }

#else

#define DISPATCH_ALLOW_FP16_QK_REDUCTION(allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, ...) \
  if (allow_fp16_qk_reduction) {                                                                \
    constexpr bool ALLOW_FP16_QK_REDUCTION = true;                                              \
    __VA_ARGS__                                                                                 \
  } else {                                                                                      \
    constexpr bool ALLOW_FP16_QK_REDUCTION = false;                                             \
    __VA_ARGS__                                                                                 \
  }

#endif

#define DISPATCH_PAGE_SIZE(page_size, PAGE_SIZE, ...)  \
  if (page_size == 1) {                                \
    constexpr size_t PAGE_SIZE = 1;                    \
    __VA_ARGS__                                        \
  } else if (page_size == 16) {                        \
    constexpr size_t PAGE_SIZE = 16;                   \
    __VA_ARGS__                                        \
  } else if (page_size == 32) {                        \
    constexpr size_t PAGE_SIZE = 32;                   \
    __VA_ARGS__                                        \
  } else {                                             \
    std::ostringstream err_msg;                        \
    err_msg << "Unsupported page_size: " << page_size; \
    throw std::invalid_argument(err_msg.str());        \
  }

#define DISPATCH_NUM_FRAGS_X(num_frags_x, NUM_FRAGS_X, ...) \
  if (num_frags_x == 1) {                                   \
    constexpr size_t NUM_FRAGS_X = 1;                       \
    __VA_ARGS__                                             \
  } else if (num_frags_x == 2) {                            \
    constexpr size_t NUM_FRAGS_X = 2;                       \
    __VA_ARGS__                                             \
  } else {                                                  \
    std::ostringstream err_msg;                             \
    err_msg << "Unsupported num_frags_x: " << num_frags_x;  \
    throw std::invalid_argument(err_msg.str());             \
  }

#define DISPATCH_NUM_FRAGS_Z(max_frags_z, NUM_FRAGS_Z, ...) \
  if (max_frags_z >= 4) {                                   \
    constexpr size_t NUM_FRAGS_Z = 4;                       \
    __VA_ARGS__                                             \
  } else if (max_frags_z >= 2) {                            \
    constexpr size_t NUM_FRAGS_Z = 2;                       \
    __VA_ARGS__                                             \
  } else if (max_frags_z >= 1) {                            \
    constexpr size_t NUM_FRAGS_Z = 1;                       \
    __VA_ARGS__                                             \
  } else {                                                  \
    std::ostringstream err_msg;                             \
    err_msg << "Unsupported max_frags_z: " << max_frags_z;  \
    throw std::invalid_argument(err_msg.str());             \
  }

#define DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, ...) \
  if (group_size == 1) {                                     \
    constexpr size_t GROUP_SIZE = 1;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 4) {                              \
    constexpr size_t GROUP_SIZE = 4;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 6) {                              \
    constexpr size_t GROUP_SIZE = 6;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 8) {                              \
    constexpr size_t GROUP_SIZE = 8;                         \
    __VA_ARGS__                                              \
  } else {                                                   \
    std::ostringstream err_msg;                              \
    err_msg << "Unsupported group_size: " << group_size;     \
    throw std::invalid_argument(err_msg.str());              \
  }

#define DISPATCH_CAUSAL(causal, CAUSAL, ...) \
  if (causal) {                              \
    constexpr bool CAUSAL = true;            \
    __VA_ARGS__                              \
  } else {                                   \
    constexpr bool CAUSAL = false;           \
    __VA_ARGS__                              \
  }

#define DISPATCH_LAYOUT(layout, LAYOUT, ...)            \
  switch (layout) {                                     \
    case QKVLayout::kNHD: {                             \
      constexpr QKVLayout LAYOUT = QKVLayout::kNHD;     \
      __VA_ARGS__                                       \
      break;                                            \
    }                                                   \
    case QKVLayout::kHND: {                             \
      constexpr QKVLayout LAYOUT = QKVLayout::kHND;     \
      __VA_ARGS__                                       \
      break;                                            \
    }                                                   \
    default: {                                          \
      std::ostringstream err_msg;                       \
      err_msg << "Unsupported layout: " << int(layout); \
      throw std::invalid_argument(err_msg.str());       \
    }                                                   \
  }

#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)     \
  switch (head_dim) {                                  \
    case 64: {                                         \
      constexpr size_t HEAD_DIM = 64;                  \
      __VA_ARGS__                                      \
      break;                                           \
    }                                                  \
    case 128: {                                        \
      constexpr size_t HEAD_DIM = 128;                 \
      __VA_ARGS__                                      \
      break;                                           \
    }                                                  \
    case 256: {                                        \
      constexpr size_t HEAD_DIM = 256;                 \
      __VA_ARGS__                                      \
      break;                                           \
    }                                                  \
    default: {                                         \
      std::ostringstream err_msg;                      \
      err_msg << "Unsupported head_dim: " << head_dim; \
      throw std::invalid_argument(err_msg.str());      \
    }                                                  \
  }

#define DISPATCH_POS_ENCODING_MODE(pos_encoding_mode, POS_ENCODING_MODE, ...)    \
  switch (pos_encoding_mode) {                                                   \
    case PosEncodingMode::kNone: {                                               \
      constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kNone;      \
      __VA_ARGS__                                                                \
      break;                                                                     \
    }                                                                            \
    case PosEncodingMode::kRoPELlama: {                                          \
      constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kRoPELlama; \
      __VA_ARGS__                                                                \
      break;                                                                     \
    }                                                                            \
    case PosEncodingMode::kALiBi: {                                              \
      constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kALiBi;     \
      __VA_ARGS__                                                                \
      break;                                                                     \
    }                                                                            \
    default: {                                                                   \
      std::ostringstream err_msg;                                                \
      err_msg << "Unsupported pos_encoding_mode: " << int(pos_encoding_mode);    \
      throw std::invalid_argument(err_msg.str());                                \
    }                                                                            \
  }

#define DISPATCH_ALIGNED_VEC_SIZE(aligned_vec_size, ALIGNED_VEC_SIZE, ...) \
  switch (aligned_vec_size) {                                              \
    case 16: {                                                             \
      constexpr size_t ALIGNED_VEC_SIZE = 16;                              \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 8: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 8;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 4: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 4;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 2: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 2;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 1: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 1;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    default: {                                                             \
      std::ostringstream err_msg;                                          \
      err_msg << "Unsupported aligned_vec_size: " << aligned_vec_size;     \
      throw std::invalid_argument(err_msg.str());                          \
    }                                                                      \
  }

namespace flashinfer {

inline bool is_device_ptr(const void* ptr) {
  hipPointerAttribute_t attrs;
  FLASHINFER_CUDA_CALL(hipPointerGetAttributes(&attrs, ptr));
  return attrs.type == hipMemoryTypeDevice;
}

template <typename T1, typename T2>
__forceinline__ __device__ __host__ T1 ceil_div(const T1 x, const T2 y) {
  return (x + y - 1) / y;
}

template <typename IdType>
std::tuple<IdType, IdType, std::vector<IdType>, std::vector<IdType>> split_qo_indptr(
    IdType* qo_indptr, uint32_t batch_size, uint32_t gqa_group_size, uint32_t head_dim,
    hipStream_t stream = nullptr) {
  constexpr uint32_t num_warps = 4;
  std::vector<IdType> qo_indptr_h(batch_size + 1), request_indices, tile_indices;
  if (is_device_ptr((void*)qo_indptr)) {
    hipMemcpyAsync(qo_indptr_h.data(), qo_indptr, sizeof(IdType) * (batch_size + 1),
                    hipMemcpyDeviceToHost, stream);
  } else {
    qo_indptr_h.assign(qo_indptr, qo_indptr + batch_size + 1);
  }

  const uint32_t total_q_len = qo_indptr_h[batch_size];
  const bool avg_len_greater_than_64 = total_q_len * gqa_group_size > 64 * batch_size;
  const uint32_t num_frags_x = (head_dim < 256 && avg_len_greater_than_64) ? 2 : 1;
  const uint32_t num_rows_per_cta = num_frags_x * num_warps * 16;
  uint32_t num_qo_tiles = 0;

  for (uint32_t i = 0; i < batch_size; ++i) {
    for (uint32_t j = qo_indptr_h[i] * gqa_group_size; j < qo_indptr_h[i + 1] * gqa_group_size;
         j += num_rows_per_cta) {
      request_indices.push_back(i);
      tile_indices.push_back((j - qo_indptr_h[i] * gqa_group_size) / num_rows_per_cta);
      ++num_qo_tiles;
    }
  }

  return {num_frags_x, num_qo_tiles, std::move(request_indices), std::move(tile_indices)};
}

}  // namespace flashinfer

#endif // UTILS_HPP
