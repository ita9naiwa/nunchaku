#pragma once

#include "gemm_base.cuh"
// #include "gemm_w4a4_block.cuh"

namespace nunchaku::kernels {

template<typename Config>
class GEMM_W4A4;

#ifndef __INTELLISENSE__
template<typename Config>
class GEMM_W4A4 : public GEMMBase<Config> {
#else
template<>
class GEMM_W4A4<GEMMConfig_W4A4_FP16> : public GEMMBase<GEMMConfig_W4A4_FP16> {
    using Config = GEMMConfig_W4A4_FP16;
#endif

public:
    IMPORT_GEMM_BASE(Config);

public:
    // micro-scales for FP4 MMA
    // each uint32_t is a 4*32 matrix of scales (for MMA of 64*32)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    static constexpr bool FP4_AVAILABLE = true;
#else
    static constexpr bool FP4_AVAILABLE = false;
#endif

    __device__ __forceinline__ 
    static void trap_no_fp4() {
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
            printf("FP4 is not available on this device\n");
        }
        __syncthreads();
        __nanosleep(1000000);
        __trap();
    }

    static_assert(WARP_N % 32 == 0);
    static_assert(WARP_M % 32 == 0);

    static constexpr int WMSCALES_PACK_SIZE = clamp(WARP_N / 32, 1, 4);
    static constexpr int WMSCALES_NUM_PACKS = ceilDiv(WARP_N / 32, WMSCALES_PACK_SIZE);
    static constexpr int WMSCALES_VALID_LANES = WARP_SIZE;

    static constexpr int AMSCALES_PACK_SIZE = clamp(WARP_M / 32, 1, 4);
    static constexpr int AMSCALES_NUM_PACKS = ceilDiv(WARP_M / 32, AMSCALES_PACK_SIZE);
    static constexpr int AMSCALES_VALID_LANES = WARP_SIZE;

    struct packed_wmscale_t {
        uint32_t data[WMSCALES_PACK_SIZE];
    };
    struct packed_amscale_t {
        uint32_t data[AMSCALES_PACK_SIZE];
    };
    using amscale_warp = std::array<packed_amscale_t, AMSCALES_NUM_PACKS>;
    using wmscale_warp = std::array<packed_wmscale_t, WMSCALES_NUM_PACKS>;

    // amscales: [M / BLOCK_M, K / group size, NUM_WARPS, AMSCALES_NUM_PACKS, WARP_SIZE] of packed_amscale_t
    __device__ __forceinline__ 
    static void load_amscale(const packed_amscale_t *ptr, int group, amscale_warp &out, bool pred) {
        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;
    #pragma unroll
        for (int i = 0; i < AMSCALES_NUM_PACKS; i++) {
            out[i] = load_pred(&ptr[(group * NUM_WARPS + warpId) * AMSCALES_NUM_PACKS * AMSCALES_VALID_LANES + i * AMSCALES_VALID_LANES + laneId], pred);
        }
    }
    // wmscales: [N / BLOCK_N, 1, K / group size, WMSCALES_NUM_PACKS, WMSCALES_VALID_LANES] of packed_wmscale_t
    __device__ __forceinline__
    static void load_wmscale(const packed_wmscale_t *ptr, int group, wmscale_warp &out, bool pred) {
        const int laneId = threadIdx.x % WARP_SIZE;
    #pragma unroll
        for (int i = 0; i < WMSCALES_NUM_PACKS; i++) {
            out[i] = load_pred(&ptr[(group * WMSCALES_NUM_PACKS + i) * WMSCALES_VALID_LANES + laneId], pred);
        }
    }

    __device__ __forceinline__
    static void quantize_w4a4_fp4_from_fpsum_warp(const packed_fpsum_t (&fpsum)[INSN_K / INSN_N], packed_act_t &output, uint32_t &output_scale, int ida) {

        constexpr int NUM_GROUPS = 4;
        static_assert(NUM_GROUPS == INSN_K / INSN_N);

        constexpr float QVALUE_MAX = 6.0f;
        constexpr float RECPI_QVALUE_MAX = 1 / QVALUE_MAX;
        constexpr float MSCALE_MAX = 448.0f;

        const int laneId = threadIdx.x % WARP_SIZE;

        // 0 for row 0-7; 1 for row 8-15
        // each half2_t represents a 8*8 matrix
        half2_t input[2][INSN_K / INSN_N * 2];
    #pragma unroll
        for (int i = 0; i < INSN_K / INSN_N; i++) {
            input[0][i * 2 + 0] = fpsum[i].data[0];
            input[0][i * 2 + 1] = fpsum[i].data[2];
            input[1][i * 2 + 0] = fpsum[i].data[1];
            input[1][i * 2 + 1] = fpsum[i].data[3];
        }
        
        auto maxabs = [](half2_t val) ALWAYSINLINE {
            val = __habs2(val);
            return __hmax(val.x, val.y);
        };

        // each half_t represents maxvalue in a 8*16 matrix
        half_t maxvalue[2][NUM_GROUPS];
    #pragma unroll
        for (int i = 0; i < NUM_GROUPS; i++) {
            maxvalue[0][i] = __hmax(maxabs(input[0][i * 2]), maxabs(input[0][i * 2 + 1]));
            maxvalue[1][i] = __hmax(maxabs(input[1][i * 2]), maxabs(input[1][i * 2 + 1]));
        }
    #pragma unroll
        for (int mask = 2; mask > 0; mask /= 2) {
    #pragma unroll
            for (int i = 0; i < NUM_GROUPS; i++) {
                maxvalue[0][i] = __hmax(maxvalue[0][i], __shfl_xor_sync(~0, maxvalue[0][i], mask));
                maxvalue[1][i] = __hmax(maxvalue[1][i], __shfl_xor_sync(~0, maxvalue[1][i], mask));
            }
        }
        // lane 0,1,2,3 / 4,5,6,7 / ...  should have identical maxvalue now

        float scale[2][NUM_GROUPS];
        float rscale[2][NUM_GROUPS];
    #pragma unroll
        for (int i = 0; i < NUM_GROUPS; i++) {
            scale[0][i]  = fminf(float(maxvalue[0][i]) * RECPI_QVALUE_MAX, MSCALE_MAX);
            scale[1][i]  = fminf(float(maxvalue[1][i]) * RECPI_QVALUE_MAX, MSCALE_MAX);
            // TODO: check whether (1 / scale) or (1 / fp8scale) is better
            rscale[0][i] = cuda_frcp(scale[0][i]);
            rscale[1][i] = cuda_frcp(scale[1][i]);
        }
        uint32_t fp8scale[2];
        fp8scale[0] = quantize_float4_fp8(make_float4(scale[0][0], scale[0][1], scale[0][2], scale[0][3]));
        fp8scale[1] = quantize_float4_fp8(make_float4(scale[1][0], scale[1][1], scale[1][2], scale[1][3]));

        /**
         * output_scale pack format: (ida=0)
         * lane 0 => row 0 if ida==0
         * lane 1 => row 8 if ida==0
         * lane 2 => row 0 if ida==1
         * lane 3 => row 8 if ida==1
         * ...
         * lane i => quad (i/4) => row (i/4+8*(i%2)) if (i%4/2==ida) => srclane i, index i%2
         */
        if (laneId % 4 / 2 == ida) {
            output_scale = (laneId % 2 == 0) ? fp8scale[0] : fp8scale[1];
        }
        
        
        uint32_t qpacks[2][INSN_K / INSN_M * 2];
    #pragma unroll
        for (int i = 0; i < INSN_K / INSN_M * 2; i++) {
    #pragma unroll
            for (int j = 0; j < 2; j++) {
                float2 fval = half22float2(input[j][i]) * make_float2(rscale[j][i / 2], rscale[j][i / 2]);
                qpacks[j][i] = quantize_float2_fp4(fval) << (laneId % 4 * 8);
            }
        }

        #pragma unroll
        for (int mask = 1; mask <= 2; mask *= 2) {
    #pragma unroll
            for (int i = 0; i < INSN_K / INSN_M * 2; i++) {
    #pragma unroll
                for (int j = 0; j < 2; j++) {
                    qpacks[j][i] |= __shfl_xor_sync(~0, qpacks[j][i], mask);
                }
            }
        }
        // lane 0,1,2,3 / 4,5,6,7 / ...  should have identical qpacks now
    #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (laneId % 4 == i) {
                output.x = qpacks[0][0 + i];
                output.y = qpacks[1][0 + i];
                output.z = qpacks[0][4 + i];
                output.w = qpacks[1][4 + i];
            }
        }
    }

    // m16n16k64 MMA
    // ida, idb in {0, 1}
    __device__ __forceinline__
    static packed_f32psum_t mma_fp4(packed_act_t act, packed_wgt_t wgt, packed_f32psum_t psum, uint32_t amscale, uint32_t wmscale, int ida, int idb) {
        packed_f32psum_t out;
        asm volatile (
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13}, "
            "{%14}, {%15, %16}, "
            "{%17}, {%18, %19};" 
            :
            "=f"(out.data[0]), "=f"(out.data[1]), "=f"(out.data[2]), "=f"(out.data[3])
            : 
            "r"(act.x), "r"(act.y), "r"(act.z), "r"(act.w),
            "r"(wgt.x), "r"(wgt.y),
            "f"(psum.data[0]), "f"(psum.data[1]), "f"(psum.data[2]), "f"(psum.data[3]),
            "r"(amscale), "n"(0), "h"((short)ida),
            "r"(wmscale), "n"(0), "h"((short)(idb * 2))
        );
        asm volatile (
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13}, "
            "{%14}, {%15, %16}, "
            "{%17}, {%18, %19};" 
            :
            "=f"(out.data[4]), "=f"(out.data[5]), "=f"(out.data[6]), "=f"(out.data[7])
            : 
            "r"(act.x), "r"(act.y), "r"(act.z), "r"(act.w),
            "r"(wgt.z), "r"(wgt.w),
            "f"(psum.data[4]), "f"(psum.data[5]), "f"(psum.data[6]), "f"(psum.data[7]),
            "r"(amscale), "n"(0), "h"((short)ida),
            "r"(wmscale), "n"(0), "h"((short)(idb * 2 + 1))
        );
        return out;
    }

    __device__ __forceinline__
    static void compute_fp4(act_warp A, wgt_warp W, amscale_warp amscale, wmscale_warp wmscale, f32psum_warp &psum) {
        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;

    #pragma unroll
        for (int j = 0; j < WARP_N_TILES; j++) {
    #pragma unroll
            for (int i = 0; i < WARP_M_TILES; i++) {
                psum[i * WARP_N_TILES + j] = mma_fp4(
                    A[i], W[j], psum[i * WARP_N_TILES + j], 
                    amscale[i / 2 / AMSCALES_PACK_SIZE].data[i / 2 % AMSCALES_PACK_SIZE], 
                    wmscale[j / 2 / WMSCALES_PACK_SIZE].data[j / 2 % WMSCALES_PACK_SIZE], 
                    i % 2, j % 2    
                );
            }
        }
    }

    template<typename Epilogue, bool USE_ALPHA>
    __device__ __forceinline__
    static void gemm_w4a4_fp4_block(
        const BlockInfo binfo,
        const packed_act_t *act,
        const packed_wgt_t *wgt,
        const packed_amscale_t *ascales,
        const packed_wmscale_t *wscales,
        float alpha,  // per-tensor scale of weight
        int M, int N, int K, 
        Epilogue::Arguments epilogueArgs,
        bool alwaysfalse)
    {
        constexpr int NUM_STAGES = 2;

        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;

        act_warp A[NUM_STAGES];  // 8 * 2
        wgt_warp W[NUM_STAGES];  // 32 * 2
        amscale_warp amscale[NUM_STAGES];  // 1 * 2
        wmscale_warp wmscale[NUM_STAGES];  // 4 * 2
        f32psum_warp fpsum;   // 128

        for (int k = 0; k < NUM_STAGES - 1; k++) {
            load_act(act, k, K, A[k], true);
            load_wgt(wgt, k, K, W[k], true);
            load_amscale(ascales, k, amscale[k], true);
            load_wmscale(wscales, k, wmscale[k], true);
        }

    #pragma unroll
        for (auto &pack : fpsum) {
    #pragma unroll
            for (int i = 0; i < 8; i++) {
                pack.data[i] = 0;
            }
        }
        
        int dummy = 0;

        for (int k1 = 0; k1 < K / WARP_K; k1 += NUM_STAGES) {
    #pragma unroll
            for (int k2 = 0; k2 < NUM_STAGES; k2++) {
                int nextk = k1 + k2 + NUM_STAGES - 1;
                int idx = (k2 + NUM_STAGES - 1) % NUM_STAGES;
                bool pred = nextk < K / WARP_K;
                load_act(act, nextk, K, A[idx], pred);
                load_wgt(wgt, nextk, K, W[idx], pred);
                load_amscale(ascales, nextk, amscale[idx], pred);
                load_wmscale(wscales, nextk, wmscale[idx], pred);

                // __syncthreads();
                // if (alwaysfalse) {
                //     dummy = clock();
                // }

                compute_fp4(A[k2], W[k2], amscale[k2], wmscale[k2], fpsum);

                if (alwaysfalse) {
                    dummy = clock();
                }

                // asm volatile ("membar.cta;");
            }
        }

        unused_var(dummy, alwaysfalse);

        if constexpr (USE_ALPHA) {
    #pragma unroll
            for (auto &pack : fpsum) {
    #pragma unroll
                for (int i = 0; i < 8; i++) {
                    pack.data[i] *= alpha;
                }
            }
        }
    

        auto f16psum = packed_fp32_to_fp16(fpsum);

        CHECK_NAN(f16psum, "f16psum");

        Epilogue()(binfo, f16psum, M, N, K, epilogueArgs);
    }

    template<typename Epilogue, bool USE_ALPHA>
    struct gemm_w4a4_fp4_kernel {
        __device__
        void operator()(
            const packed_act_t *act,
            const packed_wgt_t *wgt,
            const packed_amscale_t *ascales,
            const packed_wmscale_t *wscales,
            float alpha,
            int M, int N, int K, 
            Epilogue::Arguments epilogueArgs,
            bool swapBlockXY,
            bool alwaysfalse)
        {
            BlockInfo binfo = {
                .bm = (int)blockIdx.x,
                .bn = (int)blockIdx.y,
                .numBlocksM = (int)gridDim.x,
                .numBlocksN = (int)gridDim.y,
            };

            if (swapBlockXY) {
                std::swap(binfo.bm, binfo.bn);
                std::swap(binfo.numBlocksM, binfo.numBlocksN);
            }

            const int bm = binfo.bm;
            const int bn = binfo.bn;

            if constexpr (FP4_AVAILABLE) {
                gemm_w4a4_fp4_block<Epilogue, USE_ALPHA>(
                    binfo,
                    act + bm * (K / WARP_K) * NUM_WARPS * WARP_M_TILES * WARP_SIZE,
                    wgt + bn * (K / WARP_K) * WARP_N_TILES * WARP_SIZE,
                    ascales + bm * (K / WARP_K) * NUM_WARPS * AMSCALES_NUM_PACKS * AMSCALES_VALID_LANES,
                    wscales + bn * (K / WARP_K) * WMSCALES_NUM_PACKS * WMSCALES_VALID_LANES,
                    alpha,
                    M, N, K,
                    epilogueArgs,
                    alwaysfalse
                );

            } else {
                trap_no_fp4();
            }

            
        }
    };

public:
    template<bool ACT_UNSIGNED>
    __device__ __forceinline__
    static packed_psum_t mma(packed_act_t act, packed_wgt_t wgt) {
        packed_psum_t psum;

        if constexpr (!ACT_UNSIGNED) {
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
                "{%0,  %1,  %2,  %3},"
                "{%4,  %5,  %6,  %7},"
                "{%8,  %9},"
                "{%10,  %11,  %12,  %13};\n"
                : 
                "=r"(psum.data[0]), "=r"(psum.data[1]), "=r"(psum.data[2]), "=r"(psum.data[3])
                : 
                "r"(act.x), "r"(act.y), "r"(act.z), "r"(act.w),
                "r"(wgt.x), "r"(wgt.y),
                "r"(0), "r"(0), "r"(0), "r"(0)
                // "r"(psum.data[0]), "r"(psum.data[1]), "r"(psum.data[2]), "r"(psum.data[3])
            );
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
                "{%0,  %1,  %2,  %3},"
                "{%4,  %5,  %6,  %7},"
                "{%8,  %9},"
                "{%10,  %11,  %12,  %13};\n"
                : 
                "=r"(psum.data[4]), "=r"(psum.data[5]), "=r"(psum.data[6]), "=r"(psum.data[7])
                : 
                "r"(act.x), "r"(act.y), "r"(act.z), "r"(act.w),
                "r"(wgt.z), "r"(wgt.w),
                "r"(0), "r"(0), "r"(0), "r"(0)
                // "r"(psum.data[4]), "r"(psum.data[5]), "r"(psum.data[6]), "r"(psum.data[7])
            );
        }

        if constexpr (ACT_UNSIGNED) {
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32 "
                "{%0,  %1,  %2,  %3},"
                "{%4,  %5,  %6,  %7},"
                "{%8,  %9},"
                "{%10,  %11,  %12,  %13};\n"
                : 
                "=r"(psum.data[0]), "=r"(psum.data[1]), "=r"(psum.data[2]), "=r"(psum.data[3])
                : 
                "r"(act.x), "r"(act.y), "r"(act.z), "r"(act.w),
                "r"(wgt.x), "r"(wgt.y),
                "r"(0), "r"(0), "r"(0), "r"(0)
                // "r"(psum.data[0]), "r"(psum.data[1]), "r"(psum.data[2]), "r"(psum.data[3])
            );
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32 "
                "{%0,  %1,  %2,  %3},"
                "{%4,  %5,  %6,  %7},"
                "{%8,  %9},"
                "{%10,  %11,  %12,  %13};\n"
                : 
                "=r"(psum.data[4]), "=r"(psum.data[5]), "=r"(psum.data[6]), "=r"(psum.data[7])
                : 
                "r"(act.x), "r"(act.y), "r"(act.z), "r"(act.w),
                "r"(wgt.z), "r"(wgt.w),
                "r"(0), "r"(0), "r"(0), "r"(0)
                // "r"(psum.data[4]), "r"(psum.data[5]), "r"(psum.data[6]), "r"(psum.data[7])
            );
        }
        
        return psum;
    }

    // template<bool si>
    template<bool use_unsigned>
    __device__ __forceinline__
    static void quantize_w4a4_from_fpsum_warp(const packed_fpsum_t (&fpsum)[INSN_K / INSN_N], packed_act_t &output, half_t *output_scale) {
        const int laneId = threadIdx.x % WARP_SIZE;

        constexpr float QVALUE_MAX_SIGNED = 7.0f;
        constexpr float QVALUE_MAX_UNSIGNED = 15.0f;
        constexpr float RECPI_QVALUE_MAX_SIGNED = 1 / QVALUE_MAX_SIGNED;
        constexpr float RECPI_QVALUE_MAX_UNSIGNED = 1 / QVALUE_MAX_UNSIGNED;

        constexpr float QVALUE_MAX = use_unsigned ? QVALUE_MAX_UNSIGNED : QVALUE_MAX_SIGNED;
        constexpr float RECPI_QVALUE_MAX = use_unsigned ? RECPI_QVALUE_MAX_UNSIGNED : RECPI_QVALUE_MAX_SIGNED;
        // constexpr int QUANTIZE_BITMASK = 0xf;

        // 0 for row 0-7; 1 for row 8-15
        half2_t input[2][INSN_K / INSN_N * 2];

    #pragma unroll
        for (int i = 0; i < INSN_K / INSN_N; i++) {
            input[0][i * 2 + 0] = fpsum[i].data[0];
            input[0][i * 2 + 1] = fpsum[i].data[2];
            input[1][i * 2 + 0] = fpsum[i].data[1];
            input[1][i * 2 + 1] = fpsum[i].data[3];
        }

        half_t maxvalue[2];
        maxvalue[0] = 0;
        maxvalue[1] = 0;
    #pragma unroll
        for (int i = 0; i < INSN_K / INSN_M * 2; i++) {
            half2_t abs0 = __habs2(input[0][i]);
            half2_t abs1 = __habs2(input[1][i]);
            maxvalue[0] = __hmax(maxvalue[0], __hmax(abs0.x, abs0.y));
            maxvalue[1] = __hmax(maxvalue[1], __hmax(abs1.x, abs1.y));
        }
    #pragma unroll
        for (int mask = 2; mask > 0; mask /= 2) {
            maxvalue[0] = __hmax(maxvalue[0], __shfl_xor_sync(~0, maxvalue[0], mask));
            maxvalue[1] = __hmax(maxvalue[1], __shfl_xor_sync(~0, maxvalue[1], mask));
        }
        maxvalue[0] = __shfl_sync(~0, maxvalue[0], laneId / 4 * 4);
        maxvalue[1] = __shfl_sync(~0, maxvalue[1], laneId / 4 * 4);

        float scale[2];
        // scale[0] = float(maxvalue[0]) / QVALUE_MAX;
        // scale[1] = float(maxvalue[1]) / QVALUE_MAX;
        scale[0] = float(maxvalue[0]) * RECPI_QVALUE_MAX;
        scale[1] = float(maxvalue[1]) * RECPI_QVALUE_MAX;
        if (laneId % 4 == 0) {
            output_scale[laneId / 4] = half_t(scale[0]);
            output_scale[laneId / 4 + 8] = half_t(scale[1]);
        }

        float rscale[2];
        // rscale[0] = QVALUE_MAX / float(maxvalue[0]);
        // rscale[1] = QVALUE_MAX / float(maxvalue[1]);
        rscale[0] = cuda_frcp(scale[0]);
        rscale[1] = cuda_frcp(scale[1]);

        uint32_t qpacks[2][INSN_K / INSN_M * 2];
    #pragma unroll
        for (int i = 0; i < INSN_K / INSN_M * 2; i++) {
    #pragma unroll
            for (int j = 0; j < 2; j++) {
                // half2_t hval = __hmul2(input[j][i], half2_t(rscale[j], rscale[j]));
                // float2 fval = half22float2(hval);
                float2 fval = half22float2(input[j][i]) * make_float2(rscale[j], rscale[j]);
                qpacks[j][i] = quantize_float2<4, use_unsigned>(fval) << (laneId % 4 * 8);
            }
        }
        
        // 2 * 8 * 2 = 32 instructions => 256 cycles
    #pragma unroll
        for (int mask = 1; mask <= 2; mask *= 2) {
    #pragma unroll
            for (int i = 0; i < INSN_K / INSN_M * 2; i++) {
    #pragma unroll
                for (int j = 0; j < 2; j++) {
                    qpacks[j][i] |= __shfl_xor_sync(~0, qpacks[j][i], mask);
                }
            }
        }
        // lane 0,1,2,3 / 4,5,6,7 / ...  should have identical qpacks now

    #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (laneId % 4 == i) {
                output.x = qpacks[0][0 + i];
                output.y = qpacks[1][0 + i];
                output.z = qpacks[0][4 + i];
                output.w = qpacks[1][4 + i];
            }
        }
    }

    // loads act of [WARP_M, WARP_N] and stores to fpsum_warp
    // [WARP_M, WARP_N * 2] when fuse_glu
    template<bool fuse_glu>
    struct load_act_to_fpsum {
        using matrix_t = half_t[WARP_M][WARP_N + 8];
        static constexpr size_t SHMEM_SIZE = sizeof(matrix_t);

        __device__ __forceinline__
        void operator()(const half_t *input, int stride, int maxRows, int maxCols, fpsum_warp &out, void *shmem) {
            const int laneId = threadIdx.x % WARP_SIZE;

            matrix_t &mat = *reinterpret_cast<matrix_t *>(shmem);

            constexpr int PACK_SIZE = WARP_N / WARP_SIZE;
            using packed_input = std::array<half_t, PACK_SIZE>;
            using packed_raw_input = std::array<half2_t, PACK_SIZE>;

        #pragma unroll
            for (int row = 0; row < WARP_M; row++) {
                packed_input pack;
                // TODO: numCols not multiples of PACK_SIZE
                if constexpr (fuse_glu) {
                    packed_raw_input raw;
                    raw.fill(half2_t(0, 0));
                    bool pred = row < maxRows && laneId * PACK_SIZE * 2 < maxCols;
                    if (pred) {
                        raw = load(reinterpret_cast<const packed_raw_input *>(input + row * stride + laneId * PACK_SIZE * 2));
                    }
                #pragma unroll
                    for (int j = 0; j < PACK_SIZE; j++) {
                        pack[j] = raw[j].x * silu(raw[j].y);
                    }
                } else {
                    pack.fill(half_t(0));
                    bool pred = row < maxRows && laneId * PACK_SIZE < maxCols;
                    if (pred) {
                        pack = load(reinterpret_cast<const packed_input *>(input + row * stride + laneId * PACK_SIZE));
                    }
                }
                store<true>(reinterpret_cast<packed_input *>(&mat[row][laneId * PACK_SIZE]), pack);
            }
            __syncwarp();

            for (int m = 0; m < WARP_M_TILES; m++) {
                for (int n = 0; n < WARP_N_TILES; n++) {
                    const int row = m * INSN_M + laneId % 16;
                    const int col = n * INSN_N + laneId / 16 * 8;
                    uint4 tmp;
                    ldmatrix(&mat[row][col], tmp);
                    *reinterpret_cast<uint4 *>(&out[m * WARP_N_TILES + n]) = tmp;
                }
            }
            __syncwarp();
        }
    };

    

    /**
     * each warp quantizes a INSN_M * INSN_K (16 * 64) matrix
     * input is per-warp (in global memory)
     * output is per-thread (in regs)
     * output_scale is per-warp (in shared memory)
     * shmem must be at least INSN_M * INSN_K * sizeof(element) (16 * 64 * 0.5 = 512 Bytes)
     * default to quantize activation, if quantize weight, input should be column-majored and output should be transposed ({x, y, z, w} = {x, z, y, w})
     */
    __device__ __forceinline__
    static void quantize_w4a4_warp(const half_t *input, int stride, packed_act_t &output, half_t *output_scale, void *shmem) {
        const int laneId = threadIdx.x % WARP_SIZE;

        constexpr int QUANTIZE_BITWIDTH = 4;
        constexpr int QVALUE_MAX = 7;   // 4 bit => [-8, 7]

        // 1 lane = 1 pack
        // 1 warp = 32 lanes = 32 packs = 1 packwarp
        // a pack is {a0, ..., a7} in figure https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=ex2#mma-16864-a
        // PACK_SIZE * 4 = INSN_K / 2
        constexpr int PACK_SIZE = INSN_K / 8;  // = 8 for 4bit
        constexpr int NUM_PACKS_PER_ROW = INSN_K / PACK_SIZE;
        constexpr int NUM_ROWS_PER_PACKWARP = PACK_SIZE * WARP_SIZE / INSN_K;
        constexpr int NUM_PACKWARPS = INSN_M / NUM_ROWS_PER_PACKWARP;
        using packed_input = std::array<half_t, PACK_SIZE>;

        packed_input packs[NUM_PACKWARPS];

        // load
    #pragma unroll
        for (int i = 0; i < NUM_PACKWARPS; i++) {
            int rowId = i * NUM_ROWS_PER_PACKWARP + laneId / NUM_PACKS_PER_ROW;
            int colId = laneId % NUM_PACKS_PER_ROW * PACK_SIZE;
            packs[i] = load(reinterpret_cast<const packed_input *>(input + rowId * stride + colId));
        }

        // find max
        half_t maxvalue[NUM_PACKWARPS];
    #pragma unroll
        for (int i = 0; i < NUM_PACKWARPS; i++) {
            maxvalue[i] =  __habs(packs[i][0]);
    #pragma unroll
            for (int j = 1; j < PACK_SIZE; j++) {
                
                maxvalue[i] = __hmax(maxvalue[i], __habs(packs[i][j]));
            }
        }

        // warp reduce (max)
    #pragma unroll
        for (int mask = NUM_PACKS_PER_ROW / 2; mask > 0; mask /= 2) {
    #pragma unroll
            for (int i = 0; i < NUM_PACKWARPS; i++) {
                maxvalue[i] = __hmax(maxvalue[i], __shfl_xor_sync(~0, maxvalue[i], mask));
            }
        }

        // broadcast (max)
    #pragma unroll
        for (int i = 0; i < NUM_PACKWARPS; i++) {
            maxvalue[i] = __shfl_sync(~0, maxvalue[i], laneId / NUM_PACKS_PER_ROW * NUM_PACKS_PER_ROW);
        }

        // quantize
        using matrix_t = uint32_t[INSN_M][NUM_PACKS_PER_ROW];
        matrix_t &mat = *reinterpret_cast<matrix_t *>(shmem);
    #pragma unroll
        for (int i = 0; i < NUM_PACKWARPS; i++) {
            half_t scale  = maxvalue[i] / half_t(QVALUE_MAX);
            half_t rscale = half_t(QVALUE_MAX) / maxvalue[i];
            if (laneId % NUM_PACKS_PER_ROW == 0) {
                output_scale[i * NUM_ROWS_PER_PACKWARP + laneId / NUM_PACKS_PER_ROW] = scale;
            }

            uint32_t qpack = 0;
    // #pragma unroll
    //         for (int j = 0; j < PACK_SIZE; j++) {
    //             int intvalue = __half2int_rn(packs[i][j] / scale);
    //             intvalue = clamp(intvalue, -QVALUE_MAX, QVALUE_MAX);
    //             qpack |= (intvalue & QUANTIZE_BITMASK) << (QUANTIZE_BITWIDTH * j);
    //         }
    #pragma unroll
            for (int j = 0; j < PACK_SIZE; j += 2) {
                half2_t hval = __hmul2(half2_t(rscale, rscale), half2_t(packs[i][j], packs[i][j + 1]));
                qpack |= quantize_float2<QUANTIZE_BITWIDTH, false>(half22float2(hval)) << (j * QUANTIZE_BITWIDTH);
            }
            mat[i * NUM_ROWS_PER_PACKWARP + laneId / NUM_PACKS_PER_ROW][laneId % NUM_PACKS_PER_ROW] = qpack;
        }
        __syncwarp();
        
        // convert to imma format
        int row = laneId % 16;
        int col = laneId / 16 * 4;
        ldmatrix(&mat[row][col], output);

        __syncwarp();
    }

    // each thread block (1 warp) quantize WARP_M * WARP_K tile (32 * 64)
    struct quantize_w4a4_act_kernel {
        __device__ 
        void operator()(const half_t *input, packed_act_t *output, packed_ascale_t *oscales, int K) {
            const int laneId = threadIdx.x % WARP_SIZE;

            const int bm = blockIdx.x / (BLOCK_M / WARP_M);
            const int bk = blockIdx.y;
            const int warpId = blockIdx.x % (BLOCK_M / WARP_M);

            const int row = blockIdx.x * WARP_M;
            const int col = blockIdx.y * WARP_K;

            __shared__ alignas(128) half_t oscale_shmem[WARP_M];
            __shared__ alignas(128) uint8_t tmp_shmem[INSN_M * INSN_K / 2];

            for (int tileId = 0; tileId < WARP_M_TILES; tileId++) {
                packed_act_t tmpout;

                quantize_w4a4_warp(
                    input + (row + tileId * INSN_M) * K + col,
                    K,
                    tmpout,
                    oscale_shmem + tileId * INSN_M,
                    tmp_shmem
                );

                store(&output[(((bm * K / WARP_K + bk) * NUM_WARPS + warpId) * WARP_M_TILES + tileId) * WARP_SIZE + laneId], tmpout);
            }

            // if (threadIdx.x == 0) {
            //     printf("Block (%d, %d) => offset = %d\n", blockIdx.x, blockIdx.y, (bm * K / WARP_K + bk) * NUM_WARPS + warpId);
            // }
            pack_ascales(oscale_shmem, &oscales[((bm * K / WARP_K + bk) * NUM_WARPS + warpId) * ASCALES_NUM_PACKS * ASCALES_VALID_LANES]);
        }
    };
    
    // each thread block (1 warp) quantize WARP_N * WARP_K tile (128 * 64)
    struct quantize_w4a4_wgt_kernel {
        __device__ 
        void operator()(const half_t *input, packed_wgt_t *output, packed_wscale_t *oscales, int K) {
            const int laneId = threadIdx.x % WARP_SIZE;

            const int bn = blockIdx.x / (BLOCK_N / WARP_N);
            const int bk = blockIdx.y;

            const int col = blockIdx.x * WARP_N;
            const int row = blockIdx.y * WARP_K;

            __shared__ alignas(128) half_t oscale_shmem[WARP_N];
            __shared__ alignas(128) uint8_t tmp_shmem[INSN_M * INSN_K / 2];

            for (int tileId = 0; tileId < WARP_N_TILES; tileId++) {
                packed_wgt_t tmpout;

                quantize_w4a4_warp(
                    input + (col + tileId * INSN_N) * K + row,
                    K,
                    tmpout,
                    oscale_shmem + tileId * INSN_N,
                    tmp_shmem
                );

                std::swap(tmpout.y, tmpout.z);

                store(&output[((bn * K / WARP_K + bk) * WARP_N_TILES + tileId) * WARP_SIZE + laneId], tmpout);
            }

            pack_wscales(oscale_shmem, &oscales[(bn * K / WARP_K + bk) * WSCALES_NUM_PACKS * WSCALES_VALID_LANES]);
        }
    };

    template<bool ACT_UNSIGNED, typename T>
    __device__ __forceinline__
    static void compute(act_warp A, wgt_warp W, ascale_warp ascale, wscale_warp wscale, T &fpsum) {
        apply_scales<true>([&](int i, int j) {
            return mma<ACT_UNSIGNED>(A[i], W[j]);
        }, ascale, wscale, fpsum);
    }

    __device__ __forceinline__
    static void checkNan(fpsum_warp fpsum, const char *info = "") {
#if ENABLE_NAN_CHECK
        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;

        for (int i = 0; i < fpsum.size(); i++) {
            for (int j = 0; j < 4; j++) {
                bool abnormal = !isfinite((float)fpsum[i].data[j].x) || !isfinite((float)fpsum[i].data[j].y);
                if (abnormal) {
                    printf("abnormal value detected at block.x=%d block.y=%d warpId=%d laneId=%d fpsum_warp (%s) i=%d j=%d data.x=%f data.y=%f\n", 
                        blockIdx.x, blockIdx.y,
                        warpId, laneId,
                        info,
                        i, j,
                        (float)fpsum[i].data[j].x,
                        (float)fpsum[i].data[j].y
                    );
                    __trap();
                }
            }
        }
#endif
    }

    __device__ __forceinline__
    static void checkNan(packed_f32psum_t fpsum, const char *info = "") {
#if ENABLE_NAN_CHECK
        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;

        for (int j = 0; j < 8; j++) {
            bool abnormal = !isfinite(fpsum.data[j]);
            if (abnormal) {
                printf("abnormal value detected at bm=%d bn=%d warpId=%d laneId=%d packed_f32psum_t (%s) j=%d data=%f\n", 
                    blockIdx.x, blockIdx.y,
                    warpId, laneId,
                    info,
                    j,
                    fpsum.data[j]
                );
                __trap();
            }
        }
#endif
    }

    __device__ __forceinline__
    static void checkNan(packed_fpsum_t fpsum, const char *info = "") {
#if ENABLE_NAN_CHECK
        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;

        for (int j = 0; j < 4; j++) {
            bool abnormal = !isfinite((float)fpsum.data[j].x) || !isfinite((float)fpsum.data[j].y);
            if (abnormal) {
                printf("abnormal value detected at bm=%d bn=%d warpId=%d laneId=%d packed_fpsum_t (%s) j=%d data.x=%f data.y=%f\n", 
                    blockIdx.x, blockIdx.y,
                    warpId, laneId,
                    info,
                    j,
                    (float)fpsum.data[j].x,
                    (float)fpsum.data[j].y
                );
                __trap();
            }
        }
#endif
    }

    __device__ __forceinline__
    static void checkNan(float data, const char *info = "") {
#if ENABLE_NAN_CHECK
        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;

        bool abnormal = !isfinite(data);
        if (abnormal) {
            printf("abnormal value detected at bm=%d bn=%d warpId=%d laneId=%d packed_fpsum_t (%s) data=%f\n", 
                blockIdx.x, blockIdx.y,
                warpId, laneId,
                info,
                data
            );
            __trap();
        }
#endif
    }

    // out: [M / BLOCK_M, N / BLOCK_N, NUM_WARPS, 1, NUM_M_TILES, NUM_N_TILES, WARP_SIZE] of fpsum_warp
    template<typename Epilogue, bool ACT_UNSIGNED>
    __device__ __forceinline__
    static void gemm_w4a4_block(
        const BlockInfo binfo,
        const packed_act_t *act,
        const packed_wgt_t *wgt,
        const packed_ascale_t *ascales,
        const packed_wscale_t *wscales,
        // const packed_wscale_t *bias_ptr,
        // half_t *out,
        int M, int N, int K, 
        Epilogue::Arguments epilogueArgs,
        bool alwaysfalse)
    {
        constexpr int NUM_STAGES = 2;

        const int laneId = threadIdx.x % WARP_SIZE;
        const int warpId = threadIdx.x / WARP_SIZE;

#if 0
        fpsum_warp fpsum;
        GEMM_W4A4_Block<Config>()(act, wgt, ascales, wscales, K, fpsum, alwaysfalse);
#else
        act_warp A[NUM_STAGES];  // 8
        wgt_warp W[NUM_STAGES];  // 32
        ascale_warp ascale[NUM_STAGES];  // 1
        wscale_warp wscale[NUM_STAGES];  // 2
        fpsum_warp fpsum;   // 64

        // load_wscale<true>(wscales, wscale[0], true);
        // load_wscale<false>(wscales, wscale[1], true);
        // load_wscale<false>(wscales, wscale[2], true);

        for (int k = 0; k < NUM_STAGES - 1; k++) {
            load_act(act, k, K, A[k], true);
            load_wgt(wgt, k, K, W[k], true);
            load_ascale(ascales, k, M, ascale[k], true);
            load_wscale(wscales, k, N, wscale[k], true);
        }

        for (auto &pack : fpsum) {
    #if 1
            for (int i = 0; i < 4; i++) {
                pack.data[i].x = 0;
                pack.data[i].y = 0;
            }
    #else 
            for (int i = 0; i < 8; i++) {
                pack.data[i] = 0;
            }
    #endif
        }
        
        int dummy = 0;

        for (int k1 = 0; k1 < K / WARP_K; k1 += NUM_STAGES) {
    #pragma unroll
            for (int k2 = 0; k2 < NUM_STAGES; k2++) {
                int nextk = k1 + k2 + NUM_STAGES - 1;
                int idx = (k2 + NUM_STAGES - 1) % NUM_STAGES;
                bool pred = nextk < K / WARP_K;
                load_act(act, nextk, K, A[idx], pred);
                load_wgt(wgt, nextk, K, W[idx], pred);
                load_ascale(ascales, nextk, M, ascale[idx], pred);
                load_wscale(wscales, nextk, N, wscale[idx], pred);
                // load_wscale<false>(wscales, wscale[idx], pred);

                // __syncthreads();
                // if (alwaysfalse) {
                //     dummy = clock();
                // }

                compute<ACT_UNSIGNED>(A[k2], W[k2], ascale[k2], wscale[k2], fpsum);

                if (alwaysfalse) {
                    dummy = clock();
                }

                // asm volatile ("membar.cta;");
            }
        }

        unused_var(dummy, alwaysfalse);

#endif

    #if 0
        auto f16psum = packed_fp32_to_fp16(fpsum);
    #else
        auto f16psum = fpsum;
    #endif

        CHECK_NAN(f16psum, "f16psum");

        Epilogue()(binfo, f16psum, M, N, K, epilogueArgs);
    }

    template<bool FUSE_GELU, bool USE_UNSIGNED, bool USE_FP4>
    struct EpilogueQuantize {
        using oscales_t = typename std::conditional_t<USE_FP4, packed_amscale_t, packed_ascale_t>;

        struct Arguments {
            packed_act_t *qout;
            oscales_t *oscales;

            half_t shift_value;
            const packed_wscale_t *smooth_factor;
        };

        static constexpr int NUM_PACKS = INSN_K / INSN_N;
        static constexpr int NUM_GROUPS = WARP_N_TILES / NUM_PACKS;

        __device__ __forceinline__
        void apply_quantize(fpsum_warp fpsum, int M, int N, int K, packed_act_t *qout, oscales_t *oscales, half_t shift_value, const packed_wscale_t *smooth_factor) {
            const int laneId = threadIdx.x % WARP_SIZE;
            const int warpId = threadIdx.x / WARP_SIZE;

            __shared__ half_t oscale_shmem[NUM_WARPS][WARP_M];

            wscale_warp smooth;
            load_wscale(smooth_factor, 0, N, smooth, true);

        #pragma unroll
            for (int group = 0; group < NUM_GROUPS; group++) {
                amscale_warp omscale;

        #pragma unroll
                for (int i = 0; i < WARP_M_TILES; i++) {
                    packed_fpsum_t tmp[NUM_PACKS];

        #pragma unroll
                    for (int j = 0; j < NUM_PACKS; j++) {
                        half2_t ws1 = broadcast_wscale(smooth, (group * NUM_PACKS + j) * 4, laneId);
                        half2_t ws2 = broadcast_wscale(smooth, (group * NUM_PACKS + j) * 4 + 2, laneId);
        #pragma unroll
                        for (int k = 0; k < 4; k++) {
                            half2_t  src = fpsum[i * WARP_N_TILES + group * NUM_PACKS + j].data[k];
                            half2_t &dst = tmp[j].data[k];

                            // dst.x = gelu(src.x);
                            // dst.y = gelu(src.y);
                            if constexpr (FUSE_GELU) {
                                dst = gelu_half2(src);
                            } else {
                                dst = src;
                            }

                            dst += half2_t(shift_value, shift_value);
                            // dst = src;
                        }

                        tmp[j].data[0] = h2div(tmp[j].data[0], ws1);
                        tmp[j].data[1] = h2div(tmp[j].data[1], ws1);
                        tmp[j].data[2] = h2div(tmp[j].data[2], ws2);
                        tmp[j].data[3] = h2div(tmp[j].data[3], ws2);
                    }

                    packed_act_t qresult;
                    if constexpr (USE_FP4) {
                        quantize_w4a4_fp4_from_fpsum_warp(tmp, qresult, omscale[i / 2 / AMSCALES_PACK_SIZE].data[i / 2 % AMSCALES_PACK_SIZE], i % 2);
                    } else {
                        quantize_w4a4_from_fpsum_warp<USE_UNSIGNED>(tmp, qresult, &oscale_shmem[warpId][i * INSN_M]);
                    }
                    store(&qout[((group * NUM_WARPS + warpId) * WARP_M_TILES + i) * WARP_SIZE + laneId], qresult);
                }

                if constexpr (USE_FP4) {
                #pragma unroll
                    for (int k = 0; k < AMSCALES_NUM_PACKS; k++) {
                        store(&oscales[((group * NUM_WARPS + warpId) * AMSCALES_NUM_PACKS + k) * AMSCALES_VALID_LANES + laneId], omscale[k]);
                    }
                }
                if constexpr (!USE_FP4) {
                    __syncwarp();
                    pack_ascales(&oscale_shmem[warpId][0], &oscales[(group * NUM_WARPS + warpId) * ASCALES_NUM_PACKS * ASCALES_VALID_LANES]);
                    __syncwarp();
                }
                
            }
        }

        __device__ __forceinline__
        void operator()(const BlockInfo binfo, fpsum_warp fpsum, int M, int N, int K, Arguments args) {
            const int bm = binfo.bm;
            const int bn = binfo.bn;

            if constexpr (!USE_FP4 || FP4_AVAILABLE) {
                apply_quantize(
                    fpsum, M, N, K, 
                    args.qout + (bm * N / WARP_K + bn * NUM_GROUPS) * NUM_WARPS * WARP_M_TILES * WARP_SIZE,
                    args.oscales + (bm * N / WARP_K + bn * NUM_GROUPS) * NUM_WARPS * 
                        (USE_FP4 ? AMSCALES_NUM_PACKS * AMSCALES_VALID_LANES : ASCALES_NUM_PACKS * ASCALES_VALID_LANES),
                    args.shift_value,
                    args.smooth_factor + bn * WSCALES_NUM_PACKS * WSCALES_VALID_LANES
                );
            } else {
                trap_no_fp4();
            }
        }
    };
    // using EpilogueQuantizeFuseGelu = EpilogueQuantize<true>;

    template<int rank = 32>
    struct Lora {
        static_assert(rank % 16 == 0);

        static constexpr int LORA_RANK = rank;
        static constexpr int LORA_M_TILES = WARP_M / 16;
        static constexpr int LORA_R_TILES = LORA_RANK / 16;
        static constexpr int LORA_N_TILES = WARP_N / 16;

        static_assert(LORA_M_TILES == WARP_M_TILES);
        static_assert(LORA_N_TILES == WARP_N_TILES);
        
        // lora_down: [WARP_M, WARP_N] x [WARP_N, R] (row-wise) = [WARP_M, R]
        // lora up:   [WARP_M, R]      x [WARP_N, R] (col-wise) = [WARP_M, WARP_N]
        // we use fp32 for lora activation since there's no bf16 reduction in sm_89 :(

        using lora_act_warp   = std::array<packed_f32psum_t, LORA_M_TILES * LORA_R_TILES>;
        using lora_act16_warp = std::array<packed_fpsum_t, LORA_M_TILES * LORA_R_TILES>;
        using lora_wgt_warp = std::array<packed_fpsum_t, LORA_N_TILES * LORA_R_TILES>;

        using scale_t = std::array<float, LORA_R_TILES>;

        // lora_wgt:   [N / 16, LORA_R_TILES, WARP_SIZE] of packed_fpsum_t
        __device__ __forceinline__
        static lora_wgt_warp load_lora_wgt(const packed_fpsum_t *ptr) {
            const int laneId = threadIdx.x % WARP_SIZE;

            const packed_fpsum_t *ptr_lane = ptr + laneId;

            lora_wgt_warp result;
    #if 0
        #pragma unroll
            for (int n = 0; n < LORA_N_TILES; n++) {
        #pragma unroll
                for (int r = 0; r < LORA_R_TILES; r++) {
                    result[n * LORA_R_TILES + r] = load(ptr_lane + (n * LORA_R_TILES + r) * WARP_SIZE);
                }
            }
    #else
            unrolled_loop<LORA_N_TILES>([&]<int n>() {
                unrolled_loop<LORA_R_TILES>([&]<int r>() {
                    constexpr int offset = (n * LORA_R_TILES + r) * WARP_SIZE;
                    result[n * LORA_R_TILES + r] = load(ptr_lane + offset);
                });
            });
    #endif
            return result;
        }

        // lora_act: [M / BLOCK_M, NUM_WARPS, LORA_M_TILES, LORA_R_TILES, 8, WARP_SIZE] of float
        __device__ __forceinline__
        static lora_act16_warp load_lora_act(const float *ptr, scale_t scales) {
            const int laneId = threadIdx.x % WARP_SIZE;

            const float *ptrlane = ptr + laneId;

            lora_act16_warp result;
    #if 0
        #pragma unroll
            for (int i = 0; i < LORA_M_TILES * LORA_R_TILES; i++) {
                packed_f32psum_t tmp;
        #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int offset = i * 8 * WARP_SIZE + j * WARP_SIZE;
                    tmp.data[j] = ptrlane[offset];
                    // tmp.data[j] = ptr[i * 8 * WARP_SIZE + j * WARP_SIZE + laneId];
                }
                CHECK_NAN(tmp, "load_lora_act.tmp");
                result[i] = packed_fp32_to_fp16(tmp);
            }
    #else
            unrolled_loop<LORA_M_TILES>([&]<int m>() {
                unrolled_loop<LORA_R_TILES>([&]<int r>{
                    constexpr int i = m * LORA_R_TILES + r;
                    packed_f32psum_t tmp;
                    unrolled_loop<8>([&]<int j>() { 
                        constexpr int offset = i * 8 * WARP_SIZE + j * WARP_SIZE;
                        tmp.data[j] = ptrlane[offset] * scales[r];
                    });
                    CHECK_NAN(tmp, "load_lora_act.tmp");
                    result[i] = packed_fp32_to_fp16(tmp);
                });
            });
    #endif
            return result;
        }
        // no vector reduction in sm_89 :(
        __device__ __forceinline__
        static void reduce_lora_act(float *ptr, lora_act_warp val) {
            const int laneId = threadIdx.x % WARP_SIZE;

            float *ptrlane = ptr + laneId;

        // #pragma unroll
        //     for (int i = 0; i < LORA_M_TILES * LORA_R_TILES; i++) {
        // #pragma unroll
        //         for (int j = 0; j < 8; j++) {
        //             int offset = i * 8 * WARP_SIZE + j * WARP_SIZE;
        //             reduce_add(&ptrlane[offset], val[i].data[j]);
        //         }
        //     }

            unrolled_loop<LORA_M_TILES * LORA_R_TILES>([&]<int i>() {
                unrolled_loop<8>([&]<int j>() {
                    constexpr int offset = i * 8 * WARP_SIZE + j * WARP_SIZE;
                    reduce_add(&ptrlane[offset], val[i].data[j]);
                });
            });
        }

        // __device__ __forceinline__
        // static void reduce_lora_act(float *ptr, lora_act_warp val, int m) {
        //     const int laneId = threadIdx.x % WARP_SIZE;

        //     float *ptrlane = ptr + laneId + m * LORA_R_TILES * 8 * WARP_SIZE;

        //     unrolled_loop<LORA_R_TILES>([&]<int r>() {
        //         unrolled_loop<8>([&]<int j>() {
        //             constexpr int offset = r * 8 * WARP_SIZE + j * WARP_SIZE;
        //             reduce_add(&ptrlane[offset], val[m * LORA_R_TILES + r].data[j]);
        //         });
        //     });
        // }


        struct EpilogueLoraUp {
            struct Arguments {
                const float *lora_act;
                const packed_fpsum_t *lora_wgt_up;
                scale_t scales;
            };

            __device__ __forceinline__
            static void apply_lora_up(fpsum_warp &fpsum, int M, int N, int K, const float *act, const packed_fpsum_t *wgt, const scale_t scales, const BlockInfo binfo) {
                const int laneId = threadIdx.x % WARP_SIZE;
                const int warpId = threadIdx.x / WARP_SIZE;

                if constexpr (rank > 0) {
                    lora_act16_warp lora_act = load_lora_act(act + warpId * (LORA_M_TILES * LORA_R_TILES * 8 * WARP_SIZE), scales);
                    lora_wgt_warp lora_wgt   = load_lora_wgt(wgt);
                    for (int m = 0; m < LORA_M_TILES; m++) {
                        for (int n = 0; n < LORA_N_TILES; n++) {
                            packed_f32psum_t psum = packed_fp16_to_fp32(fpsum[m * WARP_N_TILES + n]);
                            for (int r = 0; r < LORA_R_TILES; r++) {
                                CHECK_NAN(lora_act[m * LORA_R_TILES + r], "lora_act");
                                CHECK_NAN(lora_wgt[n * LORA_R_TILES + r], "lora_wgt");
                                psum = mma_f16xf16_f32(lora_act[m * LORA_R_TILES + r], lora_wgt[n * LORA_R_TILES + r], psum);
                            }
                            fpsum[m * WARP_N_TILES + n] = packed_fp32_to_fp16(psum);
                        }
                    }
                }
            }

            __device__ __forceinline__
            void operator()(const BlockInfo binfo, fpsum_warp &fpsum, int M, int N, int K, Arguments args) {
                const int bm = binfo.bm;
                const int bn = binfo.bn;

                CHECK_NAN(fpsum, "fpsum");

                if constexpr (rank == 0) {
                    return;
                }

                apply_lora_up(
                    fpsum, M, N, K,
                    args.lora_act + bm * (NUM_WARPS * LORA_M_TILES * LORA_R_TILES * 8 *  WARP_SIZE),
                    args.lora_wgt_up + bn * (BLOCK_N / 16) * LORA_R_TILES * WARP_SIZE,
                    args.scales,
                    binfo   // for debug
                );

                CHECK_NAN(fpsum, "fpsum");
            }
        };

        struct EpilogueLoraDown {
            struct Arguments {
                const packed_fpsum_t *lora_wgt_down;
                float *lora_act;
            };

            __device__ __forceinline__
            static void apply_lora_down(fpsum_warp &fpsum, int M, int N, int K, float *act, const packed_fpsum_t *wgt) {
                const int laneId = threadIdx.x % WARP_SIZE;
                const int warpId = threadIdx.x / WARP_SIZE;

                if constexpr (rank > 0) {
                    lora_act_warp lora_act;
                    lora_act.fill(packed_f32psum_t::zeros());

                    lora_wgt_warp lora_wgt = load_lora_wgt(wgt);

                    // clock_t dummy = 0;

                #pragma unroll
                    for (int m = 0; m < LORA_M_TILES; m++) {
                #pragma unroll
                        for (int n = 0; n < LORA_N_TILES; n++) {
                #pragma unroll
                            for (int r = 0; r < LORA_R_TILES; r++) {
                                auto &psum = lora_act[m * LORA_R_TILES + r];

                                CHECK_NAN(fpsum[m * WARP_N_TILES + n], "apply_lora_down.fpsum");
                                CHECK_NAN(lora_wgt[n * LORA_R_TILES + r], "apply_lora_down.lora_wgt");

                                psum = mma_f16xf16_f32(fpsum[m * WARP_N_TILES + n], lora_wgt[n * LORA_R_TILES + r], psum);

                                CHECK_NAN(psum, "apply_lora_down.psum");
                            }
                        }
                        // reduce_lora_act(act + warpId * (LORA_M_TILES * LORA_R_TILES * 8 * WARP_SIZE), lora_act, m);

                        // if (alwaysfalse) {
                        //     dummy = clock();
                        // }
                    }

                    reduce_lora_act(act + warpId * (LORA_M_TILES * LORA_R_TILES * 8 * WARP_SIZE), lora_act);

                    // unused_var(dummy, alwaysfalse);
                }

            }

            __device__ __forceinline__
            void operator()(const BlockInfo binfo, fpsum_warp &fpsum, int M, int N, int K, Arguments args) {
                const int bm = binfo.bm;
                const int bn = binfo.bn;

                if constexpr (rank == 0) {
                    return;
                }

                apply_lora_down(
                    fpsum, M, N, K,
                    args.lora_act + bm * (NUM_WARPS * LORA_M_TILES * LORA_R_TILES * 8 *  WARP_SIZE),
                    args.lora_wgt_down + bn * (BLOCK_N / 16) * LORA_R_TILES * WARP_SIZE
                );
            }
        };

        template<bool fuse_glu, bool use_fp4>
        struct quantize_w4a4_fuse_lora_kernel {
            using oscales_t = typename std::conditional_t<use_fp4, packed_amscale_t, packed_ascale_t>;

            static constexpr size_t SHMEM_PER_WARP = ceilDiv<size_t>(load_act_to_fpsum<fuse_glu>::SHMEM_SIZE, 128) * 128;
            static constexpr size_t SHMEM_SIZE = SHMEM_PER_WARP * NUM_WARPS;

            struct Arguments {
                const half_t *input;
                const packed_wscale_t *smooth_factor;
                packed_act_t *output;
                oscales_t *oscales;
                const packed_fpsum_t *lora_wgt_down;
                float *lora_act;

                // aligned to BLOCK_M and BLOCK_N
                int M, N;   // N should be the actual K in the next GEMM (needs /2 if fuse_glu)
                // the actual M and N   (no need to /2 if fuse_glu)
                int actualM, actualN;
            };

            __device__ __forceinline__
            void operator()(Arguments args) 
            {
                const BlockInfo binfo = {
                    .bm = (int)blockIdx.x,
                    .bn = (int)blockIdx.y,
                    .numBlocksM = (int)gridDim.x,
                    .numBlocksN = (int)gridDim.y,
                };

                const int bm = binfo.bm;
                const int bn = binfo.bn;
                const int warpId = threadIdx.x / WARP_SIZE;

                const int m_offset = bm * BLOCK_M + warpId * WARP_M;
                const int n_offset = bn * BLOCK_N * (fuse_glu ? 2 : 1);

                extern __shared__ uint8_t shmem[];

                fpsum_warp fpsum;

                load_act_to_fpsum<fuse_glu>()(
                    args.input + m_offset * args.actualN + n_offset,
                    args.actualN,
                    args.actualM - m_offset,
                    args.actualN - n_offset,
                    fpsum,
                    shmem + warpId * SHMEM_PER_WARP
                    // args.smooth_factor ? args.smooth_factor + n_offset : nullptr
                );

                CHECK_NAN(fpsum, "fpsum");
                // for (int i = 0; i < 16; i++) {
                //     printf("bm=%d bn=%d warp=%d lane=%d fpsum[%d][0:1]=%f %f\n", 
                //         bm, bn, warpId, threadIdx.x % WARP_SIZE, i,
                //         (float)fpsum[i].data[0].x, (float)fpsum[i].data[0].y);
                // }

                EpilogueLoraDown()(binfo, fpsum, args.M, args.N, 0, typename EpilogueLoraDown::Arguments{
                    .lora_wgt_down = args.lora_wgt_down,
                    .lora_act = args.lora_act,
                });

                EpilogueQuantize<false, false, use_fp4>()(binfo, fpsum, args.M, args.N, 0, typename EpilogueQuantize<false, false, use_fp4>::Arguments{
                    .qout = args.output,
                    .oscales = args.oscales,
                    .shift_value = 0,
                    .smooth_factor = args.smooth_factor
                });

            }
        };
    };

    struct EpilogueGelu {
        struct Arguments { size_t unused; };

        // static constexpr float SHIFT_VALUE = 0.171875f;

        __device__ __forceinline__
        void operator()(const BlockInfo binfo, fpsum_warp &fpsum, int M, int N, int K, Arguments args) {
        #pragma unroll
            for (int i = 0; i < WARP_M_TILES; i++) {
        #pragma unroll
                for (int j = 0; j < WARP_N_TILES; j++) {
        #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        half2_t &data = fpsum[i * WARP_N_TILES + j].data[k];
                        data = gelu_half2(data);
                        // data = __hadd2(data, half2_t(SHIFT_VALUE, SHIFT_VALUE));
                    }
                }
            }
        }
    };

    // template<int PoolSize = 128>
    struct EpilogueQKVProj {
        struct Arguments {
            half_t *out;
            int actualM, actualN;

            half_t *pool_out;         // [M / PoolSize, N]
            const float *rotary_emb;        // [M, HEAD_DIM / 2, ROTARY_EMB_NUM_ELEMENTS]
            const half_t *rmsnorm_weight_q; // [HEAD_DIM]
            const half_t *rmsnorm_weight_k; // [HEAD_DIM]
            float epsilon;
        };

        static constexpr int HEAD_DIM = 128;
        static constexpr int NUM_HEADS_PER_WARP = WARP_N / HEAD_DIM;

        static constexpr int PoolSize = 128;
        static constexpr int NUM_WARPS_PER_POOL = PoolSize / WARP_M;
        static constexpr int NUM_POOLS_PER_BLOCK = BLOCK_M / PoolSize;

        static constexpr int ROTARY_EMB_NUM_ELEMENTS = 2;   // 1 for theta, 2 for {sin, cos} pair

        __device__ __forceinline__
        static void apply(fpsum_warp fpsum, half_t *out, int M, int N, int K, half_t *pool_out, const float *rotary_emb, const half_t *rmsnorm_weight, float epsilon) {
            const int laneId = threadIdx.x % WARP_SIZE;
            const int warpId = threadIdx.x / WARP_SIZE;
            
            __shared__ alignas(128) uint8_t shmem[NUM_WARPS][ceilDiv(unpack_fpsum::SHMEM_SIZE, 128) * 128];

            constexpr int PACK_SIZE = unpack_fpsum::PACK_SIZE;
            using pack_t = unpack_fpsum::pack_t;

            using pack_rope_t = std::array<float, PACK_SIZE / 2 * ROTARY_EMB_NUM_ELEMENTS>;
            constexpr int LANES_PER_HEAD = HEAD_DIM / PACK_SIZE;

            pack_t reduce_tmp;
            __shared__ alignas(128) pack_t pool[NUM_WARPS];

            // load rmsnorm scales
            pack_t rms;
            if (laneId < LANES_PER_HEAD) {
                rms = load(reinterpret_cast<const pack_t *>(&rmsnorm_weight[laneId * PACK_SIZE]));
            }
            if constexpr (LANES_PER_HEAD < WARP_SIZE) {
                for (int i = 0; i < PACK_SIZE; i++) {
                    rms[i] = __shfl_sync(~0, rms[i], laneId % LANES_PER_HEAD);
                }
            }

            const float *rotary_emb_base_addr = &rotary_emb[(warpId * WARP_M) * HEAD_DIM / 2 * ROTARY_EMB_NUM_ELEMENTS + laneId * PACK_SIZE / 2 * ROTARY_EMB_NUM_ELEMENTS];

            CHECK_NAN(fpsum, "fpsum");

            unpack_fpsum()(fpsum, out + warpId * WARP_M * N, N, INT_MAX, INT_MAX, shmem[warpId], [&](int rowId, pack_t &pack) ALWAYSINLINE {
                // load rope
                pack_rope_t rope; 
                if (laneId < LANES_PER_HEAD) {
                    // freq = load(reinterpret_cast<pack_freq_t *>(&freqs_cis[(warpId * WARP_M + rowId) * HEAD_DIM * 2 + laneId * PACK_SIZE * 2]));
                    rope = load(reinterpret_cast<const pack_rope_t *>(&rotary_emb_base_addr[rowId * HEAD_DIM / 2 * ROTARY_EMB_NUM_ELEMENTS]));
                }
                if constexpr (LANES_PER_HEAD < WARP_SIZE) {
                    for (int i = 0; i < rope.size(); i++) {
                        rope[i] = __shfl_sync(~0, rope[i], laneId % LANES_PER_HEAD);
                    }
                }

                // rmsnorm
                float sqrsum = 0.0f;
                for (int i = 0; i < PACK_SIZE; i++) {
                    sqrsum += float(pack[i]) * float(pack[i]);
                    CHECK_NAN(sqrsum, "sqrsum");
                }
            #pragma unroll
                for (int mask = LANES_PER_HEAD / 2; mask > 0; mask /= 2) {
                    sqrsum += __shfl_xor_sync(~0, sqrsum, mask);
                }
                sqrsum /= HEAD_DIM;
                float coef = cuda_frsqrt(sqrsum + epsilon);
                CHECK_NAN(coef, "coef");

                for (int i = 0; i < PACK_SIZE; i++) {
                    pack[i] *= coef * float(rms[i]);

                    CHECK_NAN(rms[i], "rms.wgt");
                    CHECK_NAN(pack[i], "rms.out");
                }

#if 1
                // rope
                for (int i = 0; i < PACK_SIZE; i += 2) {
                    float2 pack2 = half22float2(half2_t(pack[i], pack[i+1]));
                    
                    CHECK_NAN(freq[i].x, "rope.freq");
                    CHECK_NAN(freq[i].y, "rope.freq");
                    CHECK_NAN(freq[i+1].x, "rope.freq");
                    CHECK_NAN(freq[i+1].y, "rope.freq");

                    // half2_t tmp = __hmul2(freq[i], pack2);
                    // tmp = __hfma2(freq[i+1], pack2, tmp);
                    // pack[i] = tmp.x;
                    // pack[i+1] = tmp.y;

                    // printf("block.x=%d block.y=%d warpId=%d rowId=%d (%d) freqs = %f %f %f %f\n", 
                    //     blockIdx.x, blockIdx.y, warpId, rowId,
                    //     blockIdx.x * BLOCK_M + warpId * WARP_M + rowId,
                    //     (float)freq[i].x, (float)freq[i].y, (float)freq[i+1].x, (float)freq[i+1].y
                    // );
                    // __trap();

                    // half2_t tmp = __hmul2(half2_t(pack2.x, pack2.x), freq[i]);
                    // tmp = __hfma2(half2_t(pack2.y, pack2.y), freq[i+1], tmp);
                    // pack[i] = tmp.x;
                    // pack[i+1] = tmp.y;

                    float sin, cos;

                    if constexpr (ROTARY_EMB_NUM_ELEMENTS == 1) {
                        sin = cuda_sin(rope[i / 2]);
                        cos = cuda_cos(rope[i / 2]);
                    }
                    if constexpr (ROTARY_EMB_NUM_ELEMENTS == 2) {
                        sin = rope[i];
                        cos = rope[i+1];
                    }

                    // pack[i]   = pack2.x * freq[i].x   + pack2.y * freq[i].y;
                    // pack[i+1] = pack2.x * freq[i+1].x + pack2.y * freq[i+1].y;

                    pack[i]   = half_t(pack2.x * cos - pack2.y * sin);
                    pack[i+1] = half_t(pack2.x * sin + pack2.y * cos);

                    CHECK_NAN(pack[i], "rope.out");
                    CHECK_NAN(pack[i+1], "rope.out");
                }
#endif

                // mean pool
                for (int i = 0; i < PACK_SIZE; i++) {
                    reduce_tmp[i] += pack[i];
                }
            });

            if (!pool_out) {
                return;
            }

            store<true>(&pool[warpId], reduce_tmp);
            __syncthreads();

            if (warpId < NUM_POOLS_PER_BLOCK) {
                const int row = warpId * NUM_WARPS_PER_POOL;
                reduce_tmp = load<true>(&pool[row]);

                for (int i = 1; i < NUM_WARPS_PER_POOL; i++) {
                    pack_t pack = load<true>(&pool[row + i]);
                    for (int j = 0; j < PACK_SIZE; j++) {
                        reduce_tmp[j] += pack[j];
                    }
                }
                for (int j = 0; j < PACK_SIZE; j++) {
                    reduce_tmp[j] /= PoolSize;
                }
                
                store(reinterpret_cast<pack_t *>(pool_out + warpId * N), reduce_tmp);
            }
            __syncthreads();
        }

        __device__ __forceinline__
        void operator()(const BlockInfo binfo, fpsum_warp fpsum, int M, int N, int K, Arguments args) {
            const int bm = binfo.bm;
            const int bn = binfo.bn;

            assert(binfo.numBlocksN % 3 == 0);
            const bool is_q = bn < binfo.numBlocksN / 3;
            const bool is_k = !is_q && bn < binfo.numBlocksN / 3 * 2;

            assert(!args.pool_out || args.actualM == M);
            assert(args.actualN == N);

            if (is_q || is_k) {
                apply(
                    fpsum, 
                    args.out + bm * BLOCK_M * args.actualN + bn * BLOCK_N,
                    M, N, K, 
                    args.pool_out ? args.pool_out + bm * BLOCK_M / PoolSize * N : nullptr,
                    args.rotary_emb + bm * BLOCK_M * (HEAD_DIM / 2 * ROTARY_EMB_NUM_ELEMENTS),
                    is_q ? args.rmsnorm_weight_q : args.rmsnorm_weight_k,
                    args.epsilon
                );
            } else {
                EpilogueDefault()(binfo, fpsum, M, N, K, typename EpilogueDefault::Arguments{
                    .out = args.out,
                    .actualM = args.actualM,
                    .actualN = args.actualN,
                });
            }
        }
    };

    struct EpilogueLiteLA {
        __device__ __forceinline__
        static half2_t movmatrix(half2_t x) {
            asm volatile ("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(*reinterpret_cast<uint32_t *>(&x)) : "r"(*reinterpret_cast<uint32_t *>(&x)));
            return x;
        }
        
        
        __device__ __forceinline__
        static packed_f32psum_t mma_litela(packed_fpsum_t k, packed_fpsum_t v, packed_f32psum_t psum) {
            for (int i = 0; i < 4; i++) {
                k.data[i] = movmatrix(k.data[i]);
                v.data[i] = movmatrix(v.data[i]);
            }
            std::swap(v.data[1], v.data[2]);
            return mma_f16xf16_f32(v, k, psum);
        }

        static constexpr int LITELA_HEAD_DIM = 32;
        static constexpr int LITELA_K_TILES = LITELA_HEAD_DIM / 16;
        static constexpr int LITELA_V_TILES = LITELA_HEAD_DIM / 16;

        static constexpr int SHMEM_SIZE = NUM_WARPS * (LITELA_HEAD_DIM + 1) * (LITELA_HEAD_DIM + 8) * sizeof(float);

        // out_vk: [batch_size, num_heads, head_dim + 1, head_dim]
        __device__ __forceinline__
        static void apply_litela(const BlockInfo binfo, fpsum_warp fpsum, float *out_vk, int num_blocks_per_batch) {
            const int laneId = threadIdx.x % WARP_SIZE;
            const int warpId = threadIdx.x / WARP_SIZE;

            using vk_t = float[NUM_WARPS][LITELA_HEAD_DIM + 1][LITELA_HEAD_DIM + 8];
            extern __shared__ uint8_t shmem[];
            
            vk_t &shmem_vk = *reinterpret_cast<vk_t *>(shmem);

            static_assert(sizeof(vk_t) == SHMEM_SIZE);
            static_assert(WARP_N == BLOCK_N);
            assert(binfo.numBlocksN % 3 == 0);

            const int num_heads = binfo.numBlocksN / 3 * 2 * (WARP_N / (LITELA_HEAD_DIM * 2));
            const int batch_id = binfo.bm / num_blocks_per_batch;

            for (int head_id = 0; head_id < WARP_N / (LITELA_HEAD_DIM * 2); head_id++) {
                const int global_head_id = (binfo.bn - binfo.numBlocksN / 3) * (WARP_N / (LITELA_HEAD_DIM * 2)) + head_id;
                float *out_vk_current_head = out_vk + (batch_id * num_heads + global_head_id) * (LITELA_HEAD_DIM + 1) * LITELA_HEAD_DIM;

                for (int i = laneId; i < sizeof(shmem_vk) / sizeof(float) / NUM_WARPS; i += WARP_SIZE) {
                    *((&shmem_vk[warpId][0][0]) + i) = 0;
                }
                __syncwarp();

                for (int tile_v = 0; tile_v < LITELA_V_TILES; tile_v++) {
                    for (int tile_k = 0; tile_k < LITELA_K_TILES; tile_k++) {
                        packed_f32psum_t attn_sum = { 0 };
                        for (int i = 0; i < WARP_M_TILES; i++) {
                            packed_fpsum_t k = fpsum[i * WARP_N_TILES + head_id * (LITELA_HEAD_DIM * 2) / 16 + tile_k];
                            packed_fpsum_t v = fpsum[i * WARP_N_TILES + head_id * (LITELA_HEAD_DIM * 2) / 16 + LITELA_HEAD_DIM / 16 + tile_v];
                            for (int j = 0; j < 4; j++) {
                                k.data[j] = __hmax2(k.data[j], half2_t(0, 0));  // relu
                            }
                            attn_sum = mma_litela(k, v, attn_sum);
                        }

                        const int row = tile_v * 16 + laneId / 4;
                        const int col = tile_k * 16 + laneId % 4 * 2;

                        shmem_vk[warpId][row + 0][col + 0] = attn_sum.data[0];
                        shmem_vk[warpId][row + 0][col + 1] = attn_sum.data[1];
                        shmem_vk[warpId][row + 8][col + 0] = attn_sum.data[2];
                        shmem_vk[warpId][row + 8][col + 1] = attn_sum.data[3];
                        shmem_vk[warpId][row + 0][col + 8] = attn_sum.data[4];
                        shmem_vk[warpId][row + 0][col + 9] = attn_sum.data[5];
                        shmem_vk[warpId][row + 8][col + 8] = attn_sum.data[6];
                        shmem_vk[warpId][row + 8][col + 9] = attn_sum.data[7];
                    }
                }
                for (int tile_k = 0; tile_k < LITELA_K_TILES; tile_k++) {
                    packed_f32psum_t attn_sum = { 0 };
                    for (int i = 0; i < WARP_M_TILES; i++) {
                        packed_fpsum_t k = fpsum[i * WARP_N_TILES + head_id * (LITELA_HEAD_DIM * 2) / 16 + tile_k];
                        packed_fpsum_t v = {};
                        for (int j = 0; j < 4; j++) {
                            k.data[j] = __hmax2(k.data[j], half2_t(0, 0));  // relu
                        }
                    #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            v.data[i] = half2_t(1, 1);
                        }
                        // if (laneId < 4) {
                        //     v.data[0] = half2_t(1, 1);
                        //     v.data[2] = half2_t(1, 1);
                        // }
                        // if (laneId % 4 == 0) {
                        //     v.data[0] = half2_t(1, 0);
                        //     v.data[1] = half2_t(1, 0);
                        // }
                        attn_sum = mma_litela(k, v, attn_sum);
                    }
                    const int row = LITELA_HEAD_DIM + laneId / 4;
                    const int col = tile_k * 16 + laneId % 4 * 2;

                    if (laneId < 4) {
                        shmem_vk[warpId][row + 0][col + 0] = attn_sum.data[0];
                        shmem_vk[warpId][row + 0][col + 1] = attn_sum.data[1];
                        shmem_vk[warpId][row + 0][col + 8] = attn_sum.data[4];
                        shmem_vk[warpId][row + 0][col + 9] = attn_sum.data[5];
                    }
                }
                __syncthreads();

                for (int i = warpId; i < LITELA_HEAD_DIM + 1; i += NUM_WARPS) {
                    for (int j = laneId; j < LITELA_HEAD_DIM; j += WARP_SIZE) {
                        float sum = 0;
                        for (int k = 0; k < NUM_WARPS; k++) {
                            sum += shmem_vk[k][i][j];
                        }
                        reduce_add(&out_vk_current_head[i * LITELA_HEAD_DIM + j], sum);
                    }
                }
                __syncthreads();
            }
        }

        struct Arguments {
            half_t *out_q;
            float *out_vk;
            int num_blocks_per_batch;
            int actualM;
        };

        __device__ __forceinline__
        void operator()(const BlockInfo binfo, fpsum_warp fpsum, int M, int N, int K, Arguments args) {
            const int bm = binfo.bm;
            const int bn = binfo.bn;

            if (bn < binfo.numBlocksN / 3) {
                fpsum = apply_act(fpsum, [](half_t x) { return __hmax(x, 0); });    // relu
                return EpilogueDefault()(
                    binfo,
                    fpsum, 
                    M, N / 3, K, typename EpilogueDefault::Arguments{
                        .out = args.out_q,
                        .actualM = args.actualM,
                        .actualN = N / 3,
                    });
            }

            return apply_litela(binfo, fpsum, args.out_vk, args.num_blocks_per_batch);
        }

        // each thread block mults BlockSize*HEAD_DIM q and (HEAD_DIM+1)*HEAD_DIM vk, in-place writes back to q
        // q:   [batch_size, #blocks, block_size, #heads, HEAD_DIM]
        // vk:  [batch_size, #heads, HEAD_DIM+1, HEAD_DIM]
        struct vk_mul_q_kernel {
            // FIXME FIXME FIXME
            __device__
            void operator()(half_t *q, const float *vk, float eps, int num_tokens) {
                const int block_id = blockIdx.x;
                const int head_id  = blockIdx.y;
                const int batch_id = blockIdx.z;

                const int num_blocks = gridDim.x;
                const int num_heads = gridDim.y;
                const int block_size = blockDim.x;

                bool pred = block_id * block_size + threadIdx.x < num_tokens;

                half_t *localq = &q[(((batch_id * num_blocks + block_id) * block_size + threadIdx.x) * num_heads + head_id) * LITELA_HEAD_DIM];
                const float *localvk = &vk[(batch_id * num_heads + head_id) * (LITELA_HEAD_DIM + 1) * LITELA_HEAD_DIM];
                // half_t *localout = &out[(((batch_id * num_blocks + block_id) * block_size + threadIdx.x) * num_heads + head_id) * LITELA_HEAD_DIM];

                using packed_q = std::array<half_t, 8>;
                using packed_vk = std::array<float, 4>;

                half_t qblock[LITELA_HEAD_DIM];
                for (int i = 0; i < LITELA_HEAD_DIM; i += sizeof(packed_q) / sizeof(half_t)) {
                    if (pred) {
                        *reinterpret_cast<packed_q *>(&qblock[i]) = load(reinterpret_cast<const packed_q *>(&localq[i]));
                    }
                }

                float outblock[LITELA_HEAD_DIM + 1];
            #pragma unroll
                for (int j = 0; j < LITELA_HEAD_DIM + 1; j++) {
                    outblock[j] = 0;
            #pragma unroll
                    for (int i = 0; i < LITELA_HEAD_DIM; i += sizeof(packed_vk) / sizeof(float)) {
                        packed_vk vkpack = load(reinterpret_cast<const packed_vk *>(&localvk[j * LITELA_HEAD_DIM + i]));
            #pragma unroll
                        for (int k = 0; k < vkpack.size(); k++) {
                            outblock[j] += (float)qblock[i + k] * vkpack[k];
                        }
                    }
                }
                
                for (int i = 0; i < LITELA_HEAD_DIM; i += sizeof(packed_q) / sizeof(half_t)) {
                    packed_q opack;
                    for (int k = 0; k < opack.size(); k++) {
                        opack[k] = __fdividef(outblock[i + k], outblock[LITELA_HEAD_DIM] + eps);
                    }
                    if (pred) {
                        store(reinterpret_cast<packed_q *>(&localq[i]), opack);
                    }
                }
            }
        };
    };


    template<typename Epilogue, bool ACT_UNSIGNED>
    struct gemm_w4a4_kernel {
        __device__
        void operator()(
            const packed_act_t *act,
            const packed_wgt_t *wgt,
            const packed_ascale_t *ascales,
            const packed_wscale_t *wscales,
            int M, int N, int K, 
            Epilogue::Arguments epilogueArgs,
            bool swapBlockXY,
            bool alwaysfalse)
        {
            // printf("Device sizeof(args) = %d", (int)sizeof(epilogueArgs));

            BlockInfo binfo = {
                .bm = (int)blockIdx.x,
                .bn = (int)blockIdx.y,
                .numBlocksM = (int)gridDim.x,
                .numBlocksN = (int)gridDim.y,
            };

            if (swapBlockXY) {
                std::swap(binfo.bm, binfo.bn);
                std::swap(binfo.numBlocksM, binfo.numBlocksN);
            }

            const int bm = binfo.bm;
            const int bn = binfo.bn;

            // bool fusequant = !out;

            gemm_w4a4_block<Epilogue, ACT_UNSIGNED>(
                binfo,
                act + bm * (K / WARP_K) * NUM_WARPS * WARP_M_TILES * WARP_SIZE,
                wgt + bn * (K / WARP_K) * WARP_N_TILES * WARP_SIZE,
                ascales + bm * (K / WARP_K) * NUM_WARPS * ASCALES_NUM_PACKS * ASCALES_VALID_LANES,
                wscales + bn * (K / WARP_K) * WSCALES_NUM_PACKS * WSCALES_VALID_LANES,
                // bias ? bias + bn * WSCALES_NUM_PACKS * WSCALES_VALID_LANES : nullptr,
                // out + (bm * BLOCK_M * N) + bn * BLOCK_N,
                // out + (bm * N / BLOCK_N + bn) * NUM_WARPS * WARP_M_TILES * WARP_N_TILES * WARP_SIZE,
                M, N, K,
                epilogueArgs,
                alwaysfalse
            );
        }
    };
};

};  // namespace nunchaku::kernels