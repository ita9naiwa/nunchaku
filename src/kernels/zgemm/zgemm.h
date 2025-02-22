#pragma once

#include "common.h"
#include "Tensor.h"

namespace nunchaku::kernels {

void gemm_w4a4( 
        Tensor act,           // packed act [M, K / 2]
        Tensor wgt,           // packed act [N, K / 2]
        Tensor out,           // linear     [M, N]
        Tensor qout,          // packed act [M, N / 2]
        Tensor ascales,       // packed as  [K / 64, M]
        Tensor wscales,       // packed ws  [K / 64, N]
        Tensor oscales,       // packed as  [N / 64, M]
        Tensor poolout,       // linear     [M / PoolSize, N]
        Tensor lora_act_in,   // packed lora_act [M, R]
        Tensor lora_up,       // packed lora_wgt [N, R]
        Tensor lora_down,     // packed lora_wgt [N, R]
        Tensor lora_act_out,  // packed lora_act [M, R]
        Tensor norm_q,        // linear     [HEAD_DIM]
        Tensor norm_k,        // linear     [HEAD_DIM]
        Tensor rotary_emb,    // linear     [M, HEAD_DIM / 2, 2, 2]
        Tensor bias,          // packed ws  [N]
        Tensor smooth_factor, // packed ws  [N], for quantization of the next layer
        Tensor out_vk,        // linear     [B, num_heads, head_dim + 1, head_dim]
        Tensor out_linearattn,// linear     [B, (M), N / 3]
        bool act_unsigned,
        std::vector<float> lora_scales,  // [R / 16]
        bool fuse_silu,
        bool fp4,
        float alpha,
        Tensor wcscales
);
void linearattn_vk_mul_q(Tensor q, Tensor vk);

void quantize_w4a4_act_fuse_lora(Tensor input, Tensor output, Tensor oscales, Tensor lora_down, Tensor lora_act_out, Tensor smooth = {}, bool fuse_glu = false, bool fp4 = false);
void quantize_w4a4_act(Tensor input, Tensor output, Tensor oscales);
void quantize_w4a4_wgt(Tensor input, Tensor output, Tensor oscales);

void gemm_w8a8(Tensor act,      // [M, K]
               Tensor wgt,      // [N, K]
               Tensor out,      // [M, N]
               Tensor ascales,  // [1, M]
               Tensor wscales,  // [1, N]
               Tensor bias      // packed ws  [N]
               );

void quantize_w8a8_act(Tensor input, Tensor output, Tensor oscales, bool fuse_glu);

// void gemm_w8a8_fuse_litela(
//     Tensor act,      // [B, (M), K]
//     Tensor wgt,      // [N, K]
//     Tensor out_q,    // [B, (M), N / 3]
//     Tensor out_vk,   // [B, num_heads, head_dim + 1, head_dim]
//     Tensor ascales,  // [1, M]
//     Tensor wscales   // [1, N]
// );

};  // namespace nunchaku::kernels