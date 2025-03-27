#pragma once

#include <hip/hip_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_fp8.h>
#include <torch/all.h>

#define CUDATHROW(cmd)                                                                                                \
  do {                                                                                                                \
    cudaError_t err = cmd;                                                                                            \
    if (err != cudaSuccess) {                                                                                         \
      std::string msg = std::string("Test CUDA failure: ") + std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                        " '" + cudaGetErrorString(err) + "'";                                                         \
      throw std::runtime_error(msg);                                                                                  \
    }                                                                                                                 \
  } while (0)


template <typename T, int N>
struct vec_impl {
    using type = __attribute__((__vector_size__(N * sizeof(T)))) T;
};

template <typename T, int N>
using vec = typename vec_impl<T, N>::type;

template <typename T> struct vec_impl<T, 1> { using type = T; };
template <> struct vec_impl<__hip_fp8_storage_t, 2> { using type = __hip_fp8x2_storage_t; };
template <> struct vec_impl<__hip_fp8_storage_t, 4> { using type = int; };
template <> struct vec_impl<half, 2> { using type = __half2; };

using fp8 = __hip_fp8_storage_t;
using fp8x2 = vec<fp8, 2>;
using fp8_4 = vec<fp8, 4>;
using fp8x8 = vec<fp8, 8>;
using fp8x16 = vec<fp8, 16>;
using fp8_4x2 = vec<int, 2>;
using fp8_4x4 = vec<int, 4>;
using f32x4 = vec<float, 4>;
using uint8 = unsigned char;
using uint16 = unsigned short;
using uint32 = unsigned int;
using uint64 = unsigned long long;

// Absolute constants
#define WARPSIZE 64
#define OP_M 8
#define OP_N 16
#define OP_K 64
#define E_P_BANK 4
#define NB_BANKS 32
#define CU 304

// User defined constants
#define OPS 4

// Infered constants
#define WARPTILE_M OP_M
#define WARPTILE_K (OP_K * OPS)

// Parameters
#define B_LANES_ 5

#define A_PRODUCERS_ 2
#define B_PRODUCERS_ 6
#define CONSUMERS_ 2

#define QSIZE_ 2
#define SK 1

// Macros
#define K_BLOCKS(k, split_k) (((k / WARPTILE_K) / split_k))

#define CDIV(a, b) ((a + b - 1) / (b))
