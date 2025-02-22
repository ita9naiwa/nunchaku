#include "zgemm.h"
#include "gemm_w4a4_launch.cuh"

namespace nunchaku::kernels {

template<typename F>
static void invoke_launch(Tensor::ScalarType dtype, F &&launch) {
    if (dtype == Tensor::FP16) {
        launch.template operator()<GEMMConfig_W4A4_FP16>();
    } else if (dtype == Tensor::BF16) {
        launch.template operator()<GEMMConfig_W4A4_BF16>();
    } else {
        assert(false);
    }
}

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
) {
    Tensor::ScalarType dtype = Tensor::INVALID_SCALAR_TYPE;
    if (!fp4) {
        dtype = ascales.dtype();
    } else {
        for (auto tensor : {out, bias, lora_up, lora_down, poolout, wcscales}) {
            if (tensor.valid()) {
                assert(dtype == Tensor::INVALID_SCALAR_TYPE || dtype == tensor.dtype());
                dtype = tensor.dtype();
            }
        }
    }
    invoke_launch(dtype, [&]<typename Config>() {
        GEMM_W4A4_Launch<Config>::gemm_w4a4(
            act,           
            wgt,           
            out,           
            qout,          
            ascales,       
            wscales,       
            oscales,       
            poolout,       
            lora_act_in,   
            lora_up,       
            lora_down,     
            lora_act_out,  
            norm_q,        
            norm_k,        
            rotary_emb,    
            bias,          
            smooth_factor,
            out_vk,
            out_linearattn, 
            act_unsigned,
            lora_scales,
            fuse_silu,
            fp4,
            alpha,
            wcscales
        );
    });
}

void linearattn_vk_mul_q(Tensor q, Tensor vk) {
    invoke_launch(q.dtype(), [&]<typename Config>() {
        GEMM_W4A4_Launch<Config>::linearattn_vk_mul_q(q, vk);
    });
}

void quantize_w4a4_act_fuse_lora(Tensor input, Tensor output, Tensor oscales, Tensor lora_down, Tensor lora_act_out, Tensor smooth, bool fuse_glu, bool fp4) {
    invoke_launch(input.dtype(), [&]<typename Config>() {
        GEMM_W4A4_Launch<Config>::quantize_w4a4_act_fuse_lora(
            input, output, oscales, lora_down, lora_act_out, smooth, fuse_glu, fp4
        );
    });
}

void quantize_w4a4_act(Tensor input, Tensor output, Tensor oscales) {
    invoke_launch(input.dtype(), [&]<typename Config>() {
        GEMM_W4A4_Launch<Config>::quantize_w4a4_act(
            input, output, oscales
        );
    });
}
void quantize_w4a4_wgt(Tensor input, Tensor output, Tensor oscales) {
    invoke_launch(input.dtype(), [&]<typename Config>() {
        GEMM_W4A4_Launch<Config>::quantize_w4a4_wgt(
            input, output, oscales
        );
    });
}

};