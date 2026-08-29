// SPDX-License-Identifier: MIT
// Dense Q4_K x canonical-MMQ-Q8_1, m128n128, 16-warp experiment.
//
// The canonical Turing/Ampere MMQ kernel assigns two 16-row MMA minitiles to
// each of eight warps at N=128.  That leaves 64 F32 accumulators per thread
// and can spill on shallow-K, wide-M prefill projections.  This kernel keeps
// the canonical shared representation and arithmetic but splits each 128-row
// tile over four N-warps for each 32-row band.  Each warp therefore owns
// 32 rows x 32 columns and carries 32 accumulators.  Keeping both 16-row A
// fragments in one warp also preserves the canonical reuse of each B load.
//
// Numerical contract:
//   * canonical Q4_K nibble/scales/min unpack;
//   * canonical Q8_1 DS4 (half scale + half sum) activation blocks;
//   * identical ascending sequence of eight K32 folds per Q4_K block;
//   * the two canonical F32 accumulation statements are kept verbatim;
//   * one CTA owns the complete K reduction (no stream-K/fixup tree).

#include "ds4_mmq_q4_16warp.cuh"

#include "common.cuh"
#include "mmq.cuh"

#include <cstddef>
#include <cstdint>

namespace {
namespace q4w16 {

constexpr int kMTile       = 128;
constexpr int kNTile       = 128;
constexpr int kRowGroups   = 4;
constexpr int kColWarps    = 4;
constexpr int kWarps       = kRowGroups * kColWarps;
constexpr int kThreads     = 32 * kWarps;
constexpr int kRowFrag     = 2;
constexpr int kNFrag       = kNTile / 8;
constexpr int kNFragPerWarp = kNFrag / kColWarps;
constexpr int kMetadataWarps = kMTile / 16;
constexpr int kWeightStride = MMQ_MMA_TILE_X_K_Q8_1;
constexpr int kYStrideInts = sizeof(block_q8_1_mmq) / sizeof(int);
constexpr int kYChunks16   = sizeof(block_q8_1_mmq) / 16;

constexpr size_t kWeightTileBytes =
    (size_t)kMTile * (size_t)kWeightStride * sizeof(int);
constexpr size_t kYTileBytes =
    (size_t)kNTile * sizeof(block_q8_1_mmq);
constexpr size_t kSharedBytes = kWeightTileBytes + kYTileBytes;

static_assert(kWarps == 16, "Q4 16-warp decomposition changed");
static_assert(kThreads == 512, "Q4 16-warp CTA must have 512 threads");
static_assert(kRowGroups * kRowFrag * 16 == kMTile,
              "Q4 split-N row coverage changed");
static_assert(kNFragPerWarp == 4, "Q4 split-N fragment count changed");
static_assert(kWeightStride == 76, "canonical Q4_K MMA row stride changed");
static_assert(kYStrideInts == 36, "canonical Q8_1 DS4 stride changed");
static_assert(kYChunks16 == 9, "canonical Q8_1 DS4 block size changed");
static_assert(kSharedBytes == 57344, "Q4 16-warp shared-memory model changed");
static_assert(kSharedBytes <= 99ull * 1024ull,
              "Q4 16-warp kernel exceeds the intended opt-in shared limit");

__device__ __forceinline__ int lane_id() {
    return (int)threadIdx.x;
}

__device__ __forceinline__ int warp_id() {
    return (int)threadIdx.y;
}

__device__ __forceinline__ int linear_tid() {
    return (warp_id() << 5) | lane_id();
}

__device__ __forceinline__ void load_weight_tile(
        const block_q4_K * __restrict__ W,
        int * __restrict__ tile,
        int cta_row0,
        int blocks_per_row,
        int kb) {
    int *x_qs = tile;
    half2 *x_dm = reinterpret_cast<half2 *>(x_qs + 2 * MMQ_TILE_NE_K);
    const int lane = lane_id();
    const int warp = warp_id();

    // Canonical load_tiles_q4_K nibble expansion.  With 16 warps each warp
    // visits eight rows; all 128 rows are covered exactly once.
#pragma unroll
    for (int row = warp; row < kMTile; row += kWarps) {
        const block_q4_K *b =
            W + (uint64_t)(cta_row0 + row) * (uint64_t)blocks_per_row + kb;
        const int qs0 = get_int_b4(b->qs, lane);
        x_qs[row * kWeightStride + 16 * (lane / 8) + lane % 8 + 0] =
            (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[row * kWeightStride + 16 * (lane / 8) + lane % 8 + 8] =
            (qs0 >> 4) & 0x0F0F0F0F;
    }

    // The canonical loader uses 16 rows/warp and two lanes/row for metadata.
    // Only eight warps participate, so the extra split-N warps do not
    // duplicate any metadata row.
    if (warp < kMetadataWarps) {
        const int row = warp * 16 + lane / 2;
        const int ksc = lane & 1;
        const block_q4_K *b =
            W + (uint64_t)(cta_row0 + row) * (uint64_t)blocks_per_row + kb;
        const int *scales = reinterpret_cast<const int *>(b->scales);
        const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
        const int m32  = unpack_scales_q45_K(scales, ksc + 2);
        const uint8_t *sc8 = reinterpret_cast<const uint8_t *>(&sc32);
        const uint8_t *m8  = reinterpret_cast<const uint8_t *>(&m32);
        const half2 dm = b->dm * make_half2(1.0f, -1.0f);
#pragma unroll
        for (int l = 0; l < (int)sizeof(int); ++l) {
            x_dm[row * kWeightStride + (int)sizeof(int) * ksc + l] =
                dm * make_half2(sc8[l], m8[l]);
        }
    }
}

__device__ __forceinline__ void load_y_tile(
        const block_q8_1_mmq * __restrict__ q8,
        block_q8_1_mmq * __restrict__ tile,
        int N,
        int col0,
        int k128) {
    const int tid = linear_tid();
    constexpr int threads_per_col = kThreads / kNTile;
    static_assert(threads_per_col == 4,
                  "Q8_1 DS4 copy mapping changed");
    const int col = tid & (kNTile - 1);
#pragma unroll
    for (int chunk = tid >> 7; chunk < kYChunks16;
         chunk += threads_per_col) {
        int4 value = make_int4(0, 0, 0, 0);
        if (col0 + col < N) {
            const char *src = reinterpret_cast<const char *>(
                q8 + (uint64_t)k128 * (uint64_t)N + (uint64_t)(col0 + col));
            value = *reinterpret_cast<const int4 *>(src + chunk * 16);
        }
        char *dst = reinterpret_cast<char *>(tile + col);
        *reinterpret_cast<int4 *>(dst + chunk * 16) = value;
    }
}

template <typename TileA, typename TileB, typename TileC>
__device__ __forceinline__ void fold_y_half(
        float (&acc)[kNFragPerWarp][kRowFrag][TileC::ne],
        const int * __restrict__ x_tile,
        const block_q8_1_mmq * __restrict__ y_tile,
        int x_group0) {
    static_assert(TileC::ne == 4,
                  "expected m16n8 s32 accumulator fragment");
    const half2 *x_dm = reinterpret_cast<const half2 *>(
        x_tile + 2 * MMQ_TILE_NE_K);
    const int warp = warp_id();
    const int row0 = (warp / kColWarps) * (kRowFrag * 16);
    const int nf0 = (warp % kColWarps) * kNFragPerWarp;
    const int c0 = TileC::get_j(0);
    const int c1 = TileC::get_j(1);
    const int r0 = TileC::get_i(0);
    const int r1 = TileC::get_i(2);

    // K32-phased A loads keep only the two fragments needed for this 32-row
    // band live, instead of canonical MMQ's eight K32 phases at once.  Each
    // B fragment is reused by both A fragments exactly as in canonical MMQ.
    // For every output element folds remain in canonical group order 0..7.
#pragma unroll
    for (int local_group = 0; local_group < 4; ++local_group) {
        const int x_group = x_group0 + local_group;
        TileA A[kRowFrag];
        float2 dmA[kRowFrag][2];
#pragma unroll
        for (int nr = 0; nr < kRowFrag; ++nr) {
            const int frag_row0 = row0 + nr * 16;
            ggml_cuda_mma::load_ldmatrix(
                A[nr],
                x_tile + frag_row0 * kWeightStride + x_group * QI8_1,
                kWeightStride);
            dmA[nr][0] = __half22float2(
                x_dm[(frag_row0 + r0) * kWeightStride + x_group]);
            dmA[nr][1] = __half22float2(
                x_dm[(frag_row0 + r1) * kWeightStride + x_group]);
        }

#pragma unroll
        for (int nf = 0; nf < kNFragPerWarp; ++nf) {
            const int col_base = (nf0 + nf) * 8;
            TileB B;
            const int *b_qs = reinterpret_cast<const int *>(
                &y_tile[col_base].qs[local_group * QK8_1]);
            // Canonical NVIDIA MMQ deliberately uses load_generic for B.
            ggml_cuda_mma::load_generic(B, b_qs, kYStrideInts);

            const float2 dsB[2] = {
                __half22float2(y_tile[col_base + c0].ds4[local_group]),
                __half22float2(y_tile[col_base + c1].ds4[local_group]),
            };

            // These are the canonical vec_dot_q8_1_q8_1_mma accumulation
            // statements.  Do not fuse the min correction into the dot fold
            // or change their order: parity depends on this reduction tree.
#pragma unroll
            for (int nr = 0; nr < kRowFrag; ++nr) {
                TileC C;
                ggml_cuda_mma::mma(C, A[nr], B);
#pragma unroll
                for (int l = 0; l < TileC::ne; ++l) {
                    acc[nf][nr][l] +=
                        dmA[nr][l / 2].x * dsB[l % 2].x * C.x[l];
                    acc[nf][nr][l] +=
                        dmA[nr][l / 2].y * dsB[l % 2].y;
                }
            }
        }
    }
}

__global__ __launch_bounds__(kThreads, 1)
void dense_q4_16warp_kernel(
        const block_q4_K * __restrict__ W,
        const block_q8_1_mmq * __restrict__ q8,
        float * __restrict__ out,
        int M,
        int N,
        int K) {
#if defined(TURING_MMA_AVAILABLE)
    using tile_A = ggml_cuda_mma::tile<16, 8, int>;
    using tile_B = ggml_cuda_mma::tile<8, 8, int>;
    using tile_C = ggml_cuda_mma::tile<16, 8, int>;

    extern __shared__ __align__(16) unsigned char dynamic_smem[];
    int *x_tile = reinterpret_cast<int *>(dynamic_smem);
    block_q8_1_mmq *y_tile = reinterpret_cast<block_q8_1_mmq *>(
        dynamic_smem + kWeightTileBytes);

    const int cta_row0 = (int)blockIdx.x * kMTile;
    const int col0 = (int)blockIdx.y * kNTile;
    const int blocks_per_row = K / QK_K;
    float acc[kNFragPerWarp][kRowFrag][tile_C::ne] = {};

    for (int kb = 0; kb < blocks_per_row; ++kb) {
        load_weight_tile(W, x_tile, cta_row0, blocks_per_row, kb);
        load_y_tile(q8, y_tile, N, col0, 2 * kb + 0);
        __syncthreads();

        fold_y_half<tile_A, tile_B, tile_C>(acc, x_tile, y_tile, 0);
        __syncthreads();

        load_y_tile(q8, y_tile, N, col0, 2 * kb + 1);
        __syncthreads();

        fold_y_half<tile_A, tile_B, tile_C>(acc, x_tile, y_tile, 4);
        // Protect both shared tiles before the following K256 iteration.
        __syncthreads();
    }

    const int warp = warp_id();
    const int out_row0 =
        cta_row0 + (warp / kColWarps) * (kRowFrag * 16);
    const int out_col0 =
        col0 + (warp % kColWarps) * (kNFragPerWarp * 8);
#pragma unroll
    for (int nf = 0; nf < kNFragPerWarp; ++nf) {
#pragma unroll
        for (int nr = 0; nr < kRowFrag; ++nr) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int row = out_row0 + nr * 16 + tile_C::get_i(l);
                const int col = out_col0 + nf * 8 + tile_C::get_j(l);
                if (col < N) {
                    const float value = isfinite(acc[nf][nr][l])
                        ? acc[nf][nr][l] : 0.0f;
                    out[(uint64_t)col * (uint64_t)M + (uint64_t)row] =
                        value;
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(W, q8, out, M, N, K);
    NO_DEVICE_CODE;
#endif
}

} // namespace q4w16
} // anonymous namespace

extern "C" int ds4_mmq_q4_K_dense_16warp_available(int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) &&
           ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE;
}

extern "C" int ds4_mmq_q4_K_dense_16warp_supported(
        int cc, int M, int N, int K) {
    if (!ds4_mmq_q4_K_dense_16warp_available(cc)) {
        return 0;
    }
    return M >= 2048 && (M % q4w16::kMTile) == 0 &&
           N >= 512 && (N % q4w16::kNTile) == 0 &&
           K >= 1024 && K <= 4096 && (K % QK_K) == 0;
}

extern "C" int ds4_mmq_q4_K_dense_16warp_prepare(void) {
    using namespace q4w16;
    int device = -1;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        return -1;
    }
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess || prop.major < 8 ||
        prop.maxThreadsPerBlock < kThreads) {
        return -2;
    }
#if CUDART_VERSION >= 9000
    if ((size_t)prop.sharedMemPerBlockOptin < kSharedBytes) {
        return -2;
    }
#else
    if ((size_t)prop.sharedMemPerBlock < kSharedBytes) {
        return -2;
    }
#endif
    err = cudaFuncSetAttribute(
        dense_q4_16warp_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)kSharedBytes);
    return err == cudaSuccess ? 0 : -3;
}

extern "C" int ds4_mmq_q4_K_dense_16warp_enqueue(
        const void *W,
        const void *q8_ds4,
        float *out,
        int M,
        int N,
        int K,
        cudaStream_t stream) {
    using namespace q4w16;
    if (!W || !q8_ds4 || !out || M <= 0 || N <= 0 || K <= 0 ||
        (M % kMTile) != 0 || (K % QK_K) != 0) {
        return -1;
    }

    // Convert before adding the tile bias: N is a positive signed int, but
    // N + 127 would otherwise overflow for a (syntactically valid) INT_MAX
    // direct-enqueue request.
    const dim3 grid((unsigned)M / (unsigned)kMTile,
                    ((unsigned)N + (unsigned)kNTile - 1u) /
                        (unsigned)kNTile,
                    1);
    const dim3 block(32, kWarps, 1);
    dense_q4_16warp_kernel<<<grid, block, kSharedBytes, stream>>>(
        static_cast<const block_q4_K *>(W),
        static_cast<const block_q8_1_mmq *>(q8_ds4),
        out, M, N, K);
    const cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -4;
}
