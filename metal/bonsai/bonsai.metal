/*
Ported without arithmetic changes from GiorgioOppo/ds4
revision acf5c16bb6a01f8c4f027a0db265ae270d8907ab, metal/bonsai.metal.

MIT License

Copyright (c) 2026 The ds4.c authors
Copyright (c) 2023-2026 The ggml authors
Copyright (c) 2023 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include <metal_stdlib>
using namespace metal;

// Qwen3.5 dense/Bonsai text graph. States, reductions and activations are
// float32; ternary weights stay packed all the way to the dot product.
struct BonsaiArgs {
    uint n, rows, cols, type, row_bytes, pos, heads, kvheads;
    uint dim, rot, width, mode, groups;
    float eps, base;
};

static inline float bs_scalar(device const uchar *w, uint i, uint type) {
    if (type == 0u) return ((device const float *)w)[i];
    if (type == 1u) return float(((device const half *)w)[i]);
    return as_type<float>(uint(((device const ushort *)w)[i]) << 16); // BF16
}

static inline float bs_weight(device const uchar *w, uint i, uint type) {
    if (type == 0u || type == 1u || type == 30u) return bs_scalar(w, i, type);
    if (type == 8u) {
        device const uchar *b = w + (i / 32u) * 34u;
        return float(*(device const half *)b) * float(((device const char *)(b + 2))[i % 32u]);
    }
    if (type == 142u) {
        device const uchar *b = w + (i / 128u) * 34u;
        uint j = i % 128u;
        return float(*(device const half *)b) * float(int((b[2u + j / 4u] >> (2u * (j % 4u))) & 3u) - 1);
    }
    // PTQ1_0: 16 bytes x 5 trits, 8 bytes x 5 trits, 2 bytes x 4
    // trits, then fp16 scale. The byte product wraps before trit extraction.
    device const uchar *b = w + (i / 128u) * 28u;
    uint j = i % 128u, byte, trit;
    if (j < 80u) { byte = j % 16u; trit = j / 16u; }
    else if (j < 120u) { byte = 16u + (j - 80u) % 8u; trit = (j - 80u) / 8u; }
    else { byte = 24u + (j - 120u) % 2u; trit = (j - 120u) / 2u; }
    constexpr uint pow3[] = {1, 3, 9, 27, 81};
    const uint q = (uint(b[byte]) * pow3[trit]) & 255u;
    return float(*(device const half *)(b + 26u)) * float(int((q * 3u) >> 8) - 1);
}

// PTQ lane metadata is independent of the quantization block. Four entries
// cover the original lane, lane+32, lane+64 and lane+96 dot-product walk.
static inline uint bs_ptq_byte(uint j) {
    return j < 80u ? j % 16u : (j < 120u ? 16u + (j - 80u) % 8u : 24u + (j - 120u) % 2u);
}
static inline uint bs_ptq_power(uint j) {
    constexpr uint powers[] = {1u, 3u, 9u, 27u, 81u};
    return powers[j < 80u ? j / 16u : (j < 120u ? (j - 80u) / 8u : (j - 120u) / 2u)];
}
static inline float4 bs_ptq_weights4(device const uchar *p, uint4 bytes, uint4 powers) {
    const uint4 packed = uint4(p[bytes.x], p[bytes.y], p[bytes.z], p[bytes.w]);
    // The byte multiplication wraps BEFORE extraction, including qh's
    // four-trit tail. Applying modulo 243 or interpreting raw base 3 differs.
    const uint4 codes = (((packed * powers) & 255u) * 3u) >> 8u;
    return float(*(device const half *)(p + 26u)) * float4(int4(codes) - 1);
}

kernel void bonsai_embed(constant BonsaiArgs &a [[buffer(0)]],
                         device const uchar *w [[buffer(1)]], device float *out [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < a.cols) out[i] = bs_weight(w + ulong(a.pos) * a.row_bytes, i, a.type);
}

kernel void bonsai_mv(constant BonsaiArgs &a [[buffer(0)]],
                      device const uchar *w [[buffer(1)]], device const float *x [[buffer(2)]],
                      device float *out [[buffer(3)]], uint group [[threadgroup_position_in_grid]],
                      ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    uint row = group * 4u + sg;
    float sum = 0;
    if (row < a.rows) {
        device const uchar *wr = w + ulong(row) * a.row_bytes;
        if (a.type == 142u) {
            // One PQ2 block holds four successive iterations of the scalar
            // lane loop. Decode its scale once, keeping the original lane
            // assignment and addition order (including the final simd_sum).
            // This avoids repeating the generic type/scale decode per weight.
            const uint byte = 2u + lane / 4u, shift = (lane % 4u) * 2u;
            for (uint block = 0; block < a.cols / 128u; ++block) {
                device const uchar *p = wr + block * 34u;
                const float scale = float(*(device const half *)p);
                const uint4 packed = uint4(p[byte], p[byte + 8u], p[byte + 16u], p[byte + 24u]) >> shift;
                const float4 q = float4(int4(packed & 3u) - 1);
                const uint k = block * 128u + lane;
                sum += (scale * q.x) * x[k];
                sum += (scale * q.y) * x[k + 32u];
                sum += (scale * q.z) * x[k + 64u];
                sum += (scale * q.w) * x[k + 96u];
            }
        } else if (a.type == 143u) {
            const uint4 bytes = uint4(bs_ptq_byte(lane), bs_ptq_byte(lane + 32u),
                                     bs_ptq_byte(lane + 64u), bs_ptq_byte(lane + 96u));
            const uint4 powers = uint4(bs_ptq_power(lane), bs_ptq_power(lane + 32u),
                                      bs_ptq_power(lane + 64u), bs_ptq_power(lane + 96u));
#pragma unroll 4
            for (uint block = 0; block < a.cols / 128u; ++block) {
                const float4 weight = bs_ptq_weights4(wr + block * 28u, bytes, powers);
                const uint k = block * 128u + lane;
                sum += weight.x * x[k];
                sum += weight.y * x[k + 32u];
                sum += weight.z * x[k + 64u];
                sum += weight.w * x[k + 96u];
            }
        } else if (a.type == 30u) {
            // GDN alpha/beta have few BF16 output rows. Resolve the storage
            // type outside the long K loop, as in the typed dense kernels.
            // Keep the same lane walk, FP32 arithmetic and SIMD reduction.
            device const ushort *wb = (device const ushort *)wr;
            for (uint k = lane; k < a.cols; k += 32u)
                sum += as_type<float>(uint(wb[k]) << 16) * x[k];
        } else {
            for (uint k = lane; k < a.cols; k += 32u) sum += bs_weight(wr, k, a.type) * x[k];
        }
    }
    sum = simd_sum(sum);
    if (!lane && row < a.rows) out[row] = sum;
}

// Prefill uses four independent token accumulators per SIMD group. The
// coefficient decode is shared, but each token retains bonsai_mv's K walk,
// multiply/add sequence and final SIMD reduction. X is [n][cols], Y [n][rows].
// Grid: (ceil(rows/4), ceil(n/4), 1), 128 threads per threadgroup.
kernel void bonsai_mm(constant BonsaiArgs &a [[buffer(0)]],
                      device const uchar *w [[buffer(1)]], device const float *x [[buffer(2)]],
                      device float *out [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
                      ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint row = group.x * 4u + sg, first = group.y * 4u;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (row < a.rows) {
        device const uchar *wr = w + ulong(row) * a.row_bytes;
        if (a.type == 142u) {
            const uint byte = 2u + lane / 4u, shift = (lane % 4u) * 2u;
            for (uint block = 0; block < a.cols / 128u; ++block) {
                device const uchar *p = wr + block * 34u;
                const float scale = float(*(device const half *)p);
                const uint4 packed = uint4(p[byte], p[byte + 8u], p[byte + 16u], p[byte + 24u]) >> shift;
                const float4 q = float4(int4(packed & 3u) - 1);
                const uint k = block * 128u + lane;
#pragma unroll
                for (uint t = 0; t < 4u; ++t) {
                    if (first + t >= a.n) continue;
                    device const float *xt = x + ulong(first + t) * a.cols;
                    sums[t] += (scale * q.x) * xt[k];
                    sums[t] += (scale * q.y) * xt[k + 32u];
                    sums[t] += (scale * q.z) * xt[k + 64u];
                    sums[t] += (scale * q.w) * xt[k + 96u];
                }
            }
        } else if (a.type == 143u) {
            const uint4 bytes = uint4(bs_ptq_byte(lane), bs_ptq_byte(lane + 32u),
                                     bs_ptq_byte(lane + 64u), bs_ptq_byte(lane + 96u));
            const uint4 powers = uint4(bs_ptq_power(lane), bs_ptq_power(lane + 32u),
                                      bs_ptq_power(lane + 64u), bs_ptq_power(lane + 96u));
            for (uint block = 0; block < a.cols / 128u; ++block) {
                const float4 weight = bs_ptq_weights4(wr + block * 28u, bytes, powers);
                const uint k = block * 128u + lane;
#pragma unroll
                for (uint t = 0; t < 4u; ++t) {
                    if (first + t >= a.n) continue;
                    device const float *xt = x + ulong(first + t) * a.cols;
                    sums[t] += weight.x * xt[k];
                    sums[t] += weight.y * xt[k + 32u];
                    sums[t] += weight.z * xt[k + 64u];
                    sums[t] += weight.w * xt[k + 96u];
                }
            }
        } else if (a.type == 30u) {
            device const ushort *wb = (device const ushort *)wr;
            for (uint k = lane; k < a.cols; k += 32u) {
                const float weight = as_type<float>(uint(wb[k]) << 16);
#pragma unroll
                for (uint t = 0; t < 4u; ++t)
                    if (first + t < a.n) sums[t] += weight * x[ulong(first + t) * a.cols + k];
            }
        } else {
            for (uint k = lane; k < a.cols; k += 32u) {
                const float weight = bs_weight(wr, k, a.type);
#pragma unroll
                for (uint t = 0; t < 4u; ++t)
                    if (first + t < a.n) sums[t] += weight * x[ulong(first + t) * a.cols + k];
            }
        }
    }
#pragma unroll
    for (uint t = 0; t < 4u; ++t) {
        const float sum = simd_sum(sums[t]);
        if (!lane && row < a.rows && first + t < a.n)
            out[ulong(first + t) * a.rows + row] = sum;
    }
}

// Fused PTQ gate/up/SiLU decode: two row pairs per SIMD group share input
// loads and lane metadata. Both dot products retain bonsai_mv's four updates
// per 128-weight block and final simd_sum. Grid ceil(rows/8), 128 threads.
static inline uint bs_ptq_shift(uint j) {
    return 2u * (j < 80u ? j / 16u : (j < 120u ? (j - 80u) / 8u : (j - 120u) / 2u));
}
// Exact five-trit lookup for fused decode only. For byte b, field n is
// ((((b * 3^n) & 255) * 3) >> 8), packed at bit 2*n (n=0..4).
// Include all 256 bytes, not just canonical encodings. This 512-byte table
// replaces repeated integer multiplies; scales and dot products remain FP32.
constant ushort bs_ptq_lut[256]={
    0,0,256,512,64,320,576,128,384,640,16,272,528,80,336,592,
    144,400,656,32,32,288,544,96,352,608,160,416,672,4,260,516,
    68,324,580,132,388,644,20,276,276,532,84,340,596,148,404,660,
    36,292,548,100,356,612,164,420,676,8,264,520,520,72,328,584,
    136,392,648,24,280,536,88,344,600,152,408,664,40,296,552,552,
    104,360,616,168,424,680,1,257,513,65,321,577,129,385,641,17,
    273,529,81,81,337,593,145,401,657,33,289,545,97,353,609,161,
    417,673,5,261,517,69,325,325,581,133,389,645,21,277,533,85,
    341,597,149,405,661,37,293,549,101,357,357,613,165,421,677,9,
    265,521,73,329,585,137,393,649,25,281,537,89,345,601,601,153,
    409,665,41,297,553,105,361,617,169,425,681,2,258,514,66,322,
    578,130,130,386,642,18,274,530,82,338,594,146,402,658,34,290,
    546,98,354,610,162,162,418,674,6,262,518,70,326,582,134,390,
    646,22,278,534,86,342,598,150,406,406,662,38,294,550,102,358,
    614,166,422,678,10,266,522,74,330,586,138,394,650,650,26,282,
    538,90,346,602,154,410,666,42,298,554,106,362,618,170,426,682
};
static inline float4 bs_ptq_lut_weights4(device const uchar *p, uint4 bytes, uint4 powers) {
    const uint aa=bs_ptq_lut[p[bytes.x]];
    const uint bb=bs_ptq_lut[p[16u+(bytes.x&7u)]];
    const uint cc=bs_ptq_lut[p[24u+(bytes.x&1u)]];
    const uint4 packed=uint4(aa,aa,bytes.z<16u?aa:bb,bytes.w<24u?bb:cc);
    const uint4 codes=(packed>>powers)&3u;
    return float(*(device const half *)(p + 26u)) * float4(int4(codes) - 1);
}

kernel void bonsai_ptq_gate_up(constant BonsaiArgs &a [[buffer(0)]],
    device const uchar *gate [[buffer(1)]], device const uchar *up [[buffer(2)]],
    device const float *x [[buffer(3)]], device float *mid [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint NR=2u;
    const uint first=(group*4u+sg)*NR;
    const uint4 bytes=uint4(bs_ptq_byte(lane),bs_ptq_byte(lane+32u),
                           bs_ptq_byte(lane+64u),bs_ptq_byte(lane+96u));
    const uint4 powers=uint4(bs_ptq_shift(lane),bs_ptq_shift(lane+32u),
                            bs_ptq_shift(lane+64u),bs_ptq_shift(lane+96u));
    float sums_g[NR],sums_u[NR];
#pragma unroll
    for (uint r=0;r<NR;++r) {sums_g[r]=0.0f;sums_u[r]=0.0f;}
    for (uint block=0;block<a.cols/128u;++block) {
        const uint k=block*128u+lane;
        const float4 input=float4(x[k],x[k+32u],x[k+64u],x[k+96u]);
#pragma unroll
        for (uint r=0;r<NR;++r) {
            if (first+r>=a.rows) continue;
            const ulong offset=ulong(first+r)*a.row_bytes+block*28u;
            const float4 g=bs_ptq_lut_weights4(gate+offset,bytes,powers);
            const float4 u=bs_ptq_lut_weights4(up+offset,bytes,powers);
            sums_g[r]+=g.x*input.x;
            sums_g[r]+=g.y*input.y;
            sums_g[r]+=g.z*input.z;
            sums_g[r]+=g.w*input.w;
            sums_u[r]+=u.x*input.x;
            sums_u[r]+=u.y*input.y;
            sums_u[r]+=u.z*input.z;
            sums_u[r]+=u.w*input.w;
        }
    }
#pragma unroll
    for (uint r=0;r<NR;++r) {
        const float g=simd_sum(sums_g[r]),u=simd_sum(sums_u[r]);
        if (!lane && first+r<a.rows) mid[first+r]=(g/(1.0f+exp(-g)))*u;
    }
}

// Two rows per SIMD group; each token keeps the same four additions per
// block and final SIMD reduction. Full removes bounds checks only when the
// complete threadgroup output tile fits, before entering the K loop.
template<uint NT,bool Full>
inline void bonsai_ptq_mm_body(constant BonsaiArgs&a,
    device const uchar*w,device const float*x,device float*out,
    uint2 group,ushort lane,ushort sg) {
    const uint first_row=(group.x*4u+sg)*2u,first_token=group.y*NT;
    const uint4 bytes=uint4(bs_ptq_byte(lane),bs_ptq_byte(lane+32u),bs_ptq_byte(lane+64u),bs_ptq_byte(lane+96u));
    const uint4 powers=uint4(bs_ptq_power(lane),bs_ptq_power(lane+32u),bs_ptq_power(lane+64u),bs_ptq_power(lane+96u));
    float sums[2][NT];
#pragma unroll
    for(uint r=0;r<2;++r) {
#pragma unroll
        for(uint t=0;t<NT;++t)sums[r][t]=0.0f;
    }
    for(uint block=0;block<a.cols/128u;++block) {
        const uint k=block*128u+lane;
        float4 input[NT];
#pragma unroll
        for(uint t=0;t<NT;++t) {
            if(!Full && first_token+t>=a.n)continue;
            device const float*xt=x+ulong(first_token+t)*a.cols;
            input[t]=float4(xt[k],xt[k+32u],xt[k+64u],xt[k+96u]);
        }
#pragma unroll
        for(uint r=0;r<2;++r) {
            if(!Full && first_row+r>=a.rows)continue;
            const float4 weight=bs_ptq_weights4(w+ulong(first_row+r)*a.row_bytes+block*28u,bytes,powers);
#pragma unroll
            for(uint t=0;t<NT;++t) {
                if(!Full && first_token+t>=a.n)continue;
                sums[r][t]+=weight.x*input[t].x;
                sums[r][t]+=weight.y*input[t].y;
                sums[r][t]+=weight.z*input[t].z;
                sums[r][t]+=weight.w*input[t].w;
            }
        }
    }
#pragma unroll
    for(uint r=0;r<2;++r) {
#pragma unroll
        for(uint t=0;t<NT;++t) {
            const float total=simd_sum(sums[r][t]);
            if(!lane && (Full || (first_row+r<a.rows && first_token+t<a.n)))
                out[ulong(first_token+t)*a.rows+first_row+r]=total;
        }
    }
}

// Grid ceil(rows/8),ceil(n/4),128 threads. Retain the four-token path for
// small batches, including its original per-row/per-token bounds checks.
kernel void bonsai_ptq_mm(constant BonsaiArgs&a [[buffer(0)]],
    device const uchar*w [[buffer(1)]],device const float*x [[buffer(2)]],device float*out [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],ushort lane [[thread_index_in_simdgroup]],ushort sg [[simdgroup_index_in_threadgroup]]) {
    bonsai_ptq_mm_body<4,false>(a,w,x,out,group,lane,sg);
}

// Grid ceil(rows/8),ceil(n/8),128 threads. The last tile keeps bounded loads
// and stores even when only some rows or tokens are present.
kernel void bonsai_ptq_mm_8(constant BonsaiArgs&a [[buffer(0)]],
    device const uchar*w [[buffer(1)]],device const float*x [[buffer(2)]],device float*out [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],ushort lane [[thread_index_in_simdgroup]],ushort sg [[simdgroup_index_in_threadgroup]]) {
    if((group.x+1u)*8u<=a.rows && (group.y+1u)*8u<=a.n)
        bonsai_ptq_mm_body<8,true>(a,w,x,out,group,lane,sg);
    else bonsai_ptq_mm_body<8,false>(a,w,x,out,group,lane,sg);
}

// One output pair per SIMD group. Both projections retain their independent
// FP32 accumulators and the same SwiGLU expression as the separate dispatch.
template<uint NT,bool Full>
inline void bonsai_ptq_gate_up_batch_body(constant BonsaiArgs&a,
    device const uchar*gate,device const uchar*up,device const float*x,device float*mid,
    uint2 group,ushort lane,ushort sg) {
    const uint row=group.x*4u+sg,first_token=group.y*NT;
    const uint4 bytes=uint4(bs_ptq_byte(lane),bs_ptq_byte(lane+32u),bs_ptq_byte(lane+64u),bs_ptq_byte(lane+96u));
    const uint4 powers=uint4(bs_ptq_power(lane),bs_ptq_power(lane+32u),bs_ptq_power(lane+64u),bs_ptq_power(lane+96u));
    float sums_g[NT]={0.0f,0.0f,0.0f,0.0f},sums_u[NT]={0.0f,0.0f,0.0f,0.0f};
    if(Full || row<a.rows)for(uint block=0;block<a.cols/128u;++block) {
        const uint k=block*128u+lane;
        const ulong offset=ulong(row)*a.row_bytes+block*28u;
        const float4 g=bs_ptq_weights4(gate+offset,bytes,powers);
        const float4 u=bs_ptq_weights4(up+offset,bytes,powers);
#pragma unroll
        for(uint t=0;t<NT;++t) {
            if(!Full && first_token+t>=a.n)continue;
            device const float*xt=x+ulong(first_token+t)*a.cols;
            const float4 input=float4(xt[k],xt[k+32u],xt[k+64u],xt[k+96u]);
            sums_g[t]+=g.x*input.x;
            sums_g[t]+=g.y*input.y;
            sums_g[t]+=g.z*input.z;
            sums_g[t]+=g.w*input.w;
            sums_u[t]+=u.x*input.x;
            sums_u[t]+=u.y*input.y;
            sums_u[t]+=u.z*input.z;
            sums_u[t]+=u.w*input.w;
        }
    }
#pragma unroll
    for(uint t=0;t<NT;++t) {
        const float g=simd_sum(sums_g[t]),u=simd_sum(sums_u[t]);
        if(!lane && (Full || (row<a.rows && first_token+t<a.n)))
            mid[ulong(first_token+t)*a.rows+row]=(g/(1.0f+exp(-g)))*u;
    }
}

// Grid ceil(rows/4),ceil(n/4),128 threads.
kernel void bonsai_ptq_gate_up_batch(constant BonsaiArgs&a [[buffer(0)]],
    device const uchar*gate [[buffer(1)]],device const uchar*up [[buffer(2)]],
    device const float*x [[buffer(3)]],device float*mid [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],ushort lane [[thread_index_in_simdgroup]],ushort sg [[simdgroup_index_in_threadgroup]]) {
    bonsai_ptq_gate_up_batch_body<4,false>(a,gate,up,x,mid,group,lane,sg);
}

// Grid ceil(rows/4),ceil(n/8),128 threads; select the complete-tile path
// uniformly before the projection loop, with the checked path for tails.
kernel void bonsai_ptq_gate_up_batch_8(constant BonsaiArgs&a [[buffer(0)]],
    device const uchar*gate [[buffer(1)]],device const uchar*up [[buffer(2)]],
    device const float*x [[buffer(3)]],device float*mid [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],ushort lane [[thread_index_in_simdgroup]],ushort sg [[simdgroup_index_in_threadgroup]]) {
    if((group.x+1u)*4u<=a.rows && (group.y+1u)*8u<=a.n)
        bonsai_ptq_gate_up_batch_body<8,true>(a,gate,up,x,mid,group,lane,sg);
    else bonsai_ptq_gate_up_batch_body<8,false>(a,gate,up,x,mid,group,lane,sg);
}

kernel void bonsai_hadamard(constant BonsaiArgs &a [[buffer(0)]],
                            device const float *x [[buffer(1)]], device const int *signs [[buffer(2)]],
                            device float *out [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
    const uint block = group.x;
    const ulong row_offset = ulong(group.y) * a.n;
    threadgroup float v[1024];
    float4 values;
    for (uint slot = 0; slot < 4; ++slot) {
        uint dst = block * 1024u + tid + slot * 256u, src = dst;
        if (a.groups) {
            // Output head order: [rep][key head] -> [key head][rep].
            uint head = dst / a.dim, d = dst % a.dim, rep = a.heads / a.groups;
            src = ((head % rep) * a.groups + head / rep) * a.dim + d;
        }
        values[slot] = x[row_offset + src] * (a.mode ? 1.0f : float(signs[dst]));
    }

    // Strides 1..16 stay inside a SIMD group. Keep the exact p+q / p-q
    // orientation of each Sylvester butterfly, including its upper lane.
    for (uint stride = 1; stride < 32; stride *= 2) {
        const float4 other = simd_shuffle_xor(values, stride);
        values = (tid & stride) ? other - values : values + other;
    }
    for (uint slot = 0; slot < 4; ++slot) v[tid + slot * 256u] = values[slot];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Only strides 32, 64 and 128 cross SIMD groups. Each pair has one
    // writer, so one barrier per stage suffices before the next stage.
    for (uint stride = 32; stride < 256; stride *= 2) {
        for (uint pair = tid; pair < 512; pair += 256) {
            uint i = (pair / stride) * (2u * stride) + pair % stride;
            float p = v[i], q = v[i + stride];
            v[i] = p + q; v[i + stride] = p - q;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // The last two stages pair this thread's four registers. The complete
    // transform uses four threadgroup barriers instead of eleven without
    // reassociating any addition or subtraction.
    values = float4(v[tid], v[tid + 256u], v[tid + 512u], v[tid + 768u]);
    float p = values.x, q = values.y;
    values.x = p + q; values.y = p - q;
    p = values.z; q = values.w;
    values.z = p + q; values.w = p - q;
    p = values.x; q = values.z;
    values.x = p + q; values.z = p - q;
    p = values.y; q = values.w;
    values.y = p + q; values.w = p - q;
    for (uint slot = 0; slot < 4; ++slot) {
        uint i = block * 1024u + tid + slot * 256u;
        out[row_offset + i] = (values[slot] * (1.0f / 32.0f)) * (a.mode ? float(signs[i]) : 1.0f);
    }
}

// width is the input stride; output rows are contiguous. L2 deliberately
// uses max(sqrt(sum), epsilon), not rsqrt(sum + epsilon).
kernel void bonsai_norm(constant BonsaiArgs &a [[buffer(0)]],
                        device const float *x [[buffer(1)]], device const uchar *w [[buffer(2)]],
                        device float *out [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tmp[256];
    float sum = 0;
    for (uint i = tid; i < a.cols; i += 256) { float v = x[ulong(row) * a.width + i]; sum += v * v; }
    tmp[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reproduce the original 256-thread tree, rather than replacing it
    // with simd_sum (which may use a different association). SIMD group 0
    // performs stages 128, 64 and 32 from the same eight partials per lane;
    // shuffle-down then reproduces stages 16, 8, 4, 2 and 1 exactly.
    if (tid < 32) {
        const float p0 = tmp[tid] + tmp[tid + 128u];
        const float p1 = tmp[tid + 64u] + tmp[tid + 192u];
        const float p2 = tmp[tid + 32u] + tmp[tid + 160u];
        const float p3 = tmp[tid + 96u] + tmp[tid + 224u];
        float total = (p0 + p1) + (p2 + p3);
        total += simd_shuffle_down(total, 16);
        total += simd_shuffle_down(total, 8);
        total += simd_shuffle_down(total, 4);
        total += simd_shuffle_down(total, 2);
        total += simd_shuffle_down(total, 1);
        if (!tid) tmp[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = a.mode ? 1.0f / max(sqrt(tmp[0]), a.eps) : rsqrt(tmp[0] / float(a.cols) + a.eps);
    for (uint i = tid; i < a.cols; i += 256)
        out[ulong(row) * a.cols + i] = x[ulong(row) * a.width + i] * scale * (a.mode ? 1.0f : bs_scalar(w, i, a.type));
}

kernel void bonsai_element(constant BonsaiArgs &a [[buffer(0)]],
                           device const float *x [[buffer(1)]], device const float *y [[buffer(2)]],
                           device float *out [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.n) return;
    if (a.mode == 0) out[i] = x[i] + y[i];
    else if (a.mode == 1) out[i] = (x[i] / (1.0f + exp(-x[i]))) * y[i];
    else {
        // Full attention gates are interleaved with Q, one pair per head.
        ulong gi = ulong(i / a.dim) * (2ul * a.dim) + a.dim + i % a.dim;
        out[i] = x[i] / (1.0f + exp(-y[gi]));
    }
}

kernel void bonsai_conv(constant BonsaiArgs &a [[buffer(0)]],
                        device const float *x [[buffer(1)]], device const uchar *w [[buffer(2)]],
                        device float *history [[buffer(3)]], device float *out [[buffer(4)]],
                        uint c [[thread_position_in_grid]]) {
    if (c >= a.n) return;
    float sum = 0;
    for (uint j = 0; j + 1 < a.width; ++j) {
        sum += history[ulong(c) * (a.width - 1u) + j] * bs_scalar(w, c * a.width + j, a.type);
    }
    sum += x[c] * bs_scalar(w, c * a.width + a.width - 1u, a.type);
    for (uint j = 0; j + 2 < a.width; ++j)
        history[ulong(c) * (a.width - 1u) + j] = history[ulong(c) * (a.width - 1u) + j + 1u];
    if (a.width > 1) history[ulong(c) * (a.width - 1u) + a.width - 2u] = x[c];
    out[c] = sum / (1.0f + exp(-sum));
}

kernel void bonsai_gdn(constant BonsaiArgs &a [[buffer(0)]],
                       device const float *qkv [[buffer(1)]], device const float *alpha [[buffer(2)]],
                       device const float *beta [[buffer(3)]], device const uchar *A [[buffer(4)]],
                       device const uchar *dt [[buffer(5)]], device float *state [[buffer(6)]],
                       device float *out [[buffer(7)]], uint h [[threadgroup_position_in_grid]],
                       uint dv [[thread_index_in_threadgroup]]) {
    if (dv >= a.dim) return;
    const uint kh = h % a.kvheads;
    device const float *q = qkv + kh * a.dim, *k = qkv + (a.kvheads + kh) * a.dim;
    const float v = qkv[2u * a.kvheads * a.dim + h * a.dim + dv];
    const float ab = alpha[h] + bs_scalar(dt, h, a.mode);
    // Metal lacks log1p. Compensate rounding in 1+e; for very small e
    // the exact limiting value avoids cancellation to zero.
    const float e = exp(-abs(ab)), u = 1.0f + e;
    const float softplus = max(ab, 0.0f) + (u == 1.0f ? e : log(u) * (e / (u - 1.0f)));
    const float decay = exp(bs_scalar(A, h, a.type) * softplus);
    const float b = 1.0f / (1.0f + exp(-beta[h]));
    // Private Metal state is [head][key dimension][value dimension], so
    // neighboring value lanes access neighboring floats throughout the scan.
    float prediction = 0;
    for (uint dk = 0; dk < a.dim; ++dk) {
        const ulong idx = (ulong(h) * a.dim + dk) * a.dim + dv;
        float s = state[idx] * decay;
        state[idx] = s;
        prediction += s * k[dk];
    }
    const float delta = (v - prediction) * b;
    float result = 0;
    for (uint dk = 0; dk < a.dim; ++dk) {
        const ulong idx = (ulong(h) * a.dim + dk) * a.dim + dv;
        float s = state[idx] + delta * k[dk];
        state[idx] = s;
        result += s * q[dk];
    }
    out[h * a.dim + dv] = result * rsqrt(float(a.dim));
}

kernel void bonsai_rope(constant BonsaiArgs &a [[buffer(0)]], device float *x [[buffer(1)]],
                        uint2 index [[thread_position_in_grid]]) {
    const uint i = index.x;
    const ulong offset = ulong(index.y) * a.heads * a.dim;
    uint half_rot = a.rot / 2u;
    if (i >= a.heads * half_rot) return;
    uint h = i / half_rot, j = i % half_rot;
    float angle = float(a.pos + index.y) * pow(a.base, -2.0f * float(j) / float(a.rot));
    float c = cos(angle), s = sin(angle);
    float p = x[offset + h * a.dim + j], q = x[offset + h * a.dim + j + half_rot];
    x[offset + h * a.dim + j] = p * c - q * s;
    x[offset + h * a.dim + j + half_rot] = p * s + q * c;
}

// Qwen3.5 interleaved MRoPE [11,11,10]: the 32 rotary pairs take
// t,h,w,t,h,w,...,t,h. This is the Qwen/Prism IMRoPE section selection,
// retaining Bonsai's existing NeoX pair arithmetic and inverse frequencies.
// Args.n is the row count; positions are packed signed [row][3].
kernel void bonsai_mrope(constant BonsaiArgs &a [[buffer(0)]], device float *x [[buffer(1)]],
                         constant int *positions [[buffer(2)]], uint2 index [[thread_position_in_grid]]) {
    const uint i=index.x,half_rot=a.rot/2u;
    if(index.y>=a.n || i>=a.heads*half_rot)return;
    const uint h=i/half_rot,j=i%half_rot;
    const ulong offset=ulong(index.y)*a.heads*a.dim;
    const float angle=float(positions[ulong(index.y)*3u+j%3u])*pow(a.base,-2.0f*float(j)/float(a.rot));
    const float c=cos(angle),s=sin(angle);
    const float p=x[offset+h*a.dim+j],q=x[offset+h*a.dim+j+half_rot];
    x[offset+h*a.dim+j]=p*c-q*s;
    x[offset+h*a.dim+j+half_rot]=p*s+q*c;
}

kernel void bonsai_cache(constant BonsaiArgs &a [[buffer(0)]],
                         device const float *k [[buffer(1)]], device const float *v [[buffer(2)]],
                         device float *kc [[buffer(3)]], device float *vc [[buffer(4)]],
                         uint2 index [[thread_position_in_grid]]) {
    const uint i = index.x;
    if (i < a.n) {
        const ulong src = ulong(index.y) * a.n + i, dst = ulong(a.pos + index.y) * a.n + i;
        kc[dst] = k[src]; vc[dst] = v[src];
    }
}

kernel void bonsai_scores(constant BonsaiArgs &a [[buffer(0)]],
                          device const float *q [[buffer(1)]], device const float *kc [[buffer(2)]],
                          device float *scores [[buffer(3)]], uint2 i [[thread_position_in_grid]]) {
    if (i.x > a.pos || i.y >= a.heads) return;
    uint kh = i.y / (a.heads / a.kvheads);
    device const float *k = kc + (ulong(i.x) * a.kvheads + kh) * a.dim;
    float sum = 0;
    for (uint d = 0; d < a.dim; ++d) sum += q[i.y * a.dim + d] * k[d];
    scores[ulong(i.y) * a.width + i.x] = sum * rsqrt(float(a.dim));
}

kernel void bonsai_softmax(constant BonsaiArgs &a [[buffer(0)]], device float *scores [[buffer(1)]],
                           uint h [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tmp[256];
    device float *row = scores + ulong(h) * a.width;
    float mx = -INFINITY;
    for (uint t = tid; t <= a.pos; t += 256) mx = max(mx, row[t]);
    tmp[tid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = 128; d; d /= 2) { if (tid < d) tmp[tid] = max(tmp[tid], tmp[tid + d]); threadgroup_barrier(mem_flags::mem_threadgroup); }
    mx = tmp[0];
    float sum = 0;
    for (uint t = tid; t <= a.pos; t += 256) { float v = exp(row[t] - mx); row[t] = v; sum += v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tmp[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = 128; d; d /= 2) { if (tid < d) tmp[tid] += tmp[tid + d]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    sum = tmp[0];
    for (uint t = tid; t <= a.pos; t += 256) row[t] /= sum;
}

kernel void bonsai_attention(constant BonsaiArgs &a [[buffer(0)]],
                             device const float *scores [[buffer(1)]], device const float *vc [[buffer(2)]],
                             device float *out [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.heads * a.dim) return;
    uint h = i / a.dim, d = i % a.dim, kh = h / (a.heads / a.kvheads);
    float sum = 0;
    for (uint t = 0; t <= a.pos; ++t)
        sum += scores[ulong(h) * a.width + t] * vc[(ulong(t) * a.kvheads + kh) * a.dim + d];
    out[i] = sum;
}

#include <metal_simdgroup_matrix>

// Large PQ2 prefill projection: 64 output rows x 32 tokens x 32 K.
// Both operands and the matrix accumulators are FP32. Matrix multiplication
// changes the summation association relative to bonsai_mm/bonsai_mv.
// Each thread expands 16 adjacent PQ2 coefficients using one scale and four
// packed bytes; packed 8x8 threadgroup tiles follow the existing DeepSeek
// Metal matrix kernels. Activations use aligned contiguous float4 loads;
// complete output tiles are stored directly from the SIMD matrices. No
// persistent expanded weight matrix is needed.
// Requires type142, cols divisible by128, X[n][cols], Y[n][rows].
// Grid: (ceil(rows/64), ceil(n/32), 1); 128 threads per group.
kernel void bonsai_mm_pq2_tiled(constant BonsaiArgs &a [[buffer(0)]],
                               device const uchar *w [[buffer(1)]],
                               device const float *x [[buffer(2)]],
                               device float *out [[buffer(3)]],
                               uint2 group [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint BM=64u, BN=32u, BK=32u;
    threadgroup float weights[BM*BK];
    threadgroup float inputs[BN*BK];

    constexpr uint MR=BM/16u, NR=BN/16u;
    constexpr uint COEFFS=BM*BK/128u, ROW_THREADS=BK/COEFFS;
    constexpr uint INPUT_RUN=BK*BN/128u;
    const uint first_row=group.x*BM,first_token=group.y*BN;
    simdgroup_float8x8 wf[MR],xf[NR],acc[MR*NR];
#pragma unroll
    for(uint i=0;i<MR*NR;i++)acc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
    const uint row=tid/ROW_THREADS,start_k=(tid%ROW_THREADS)*COEFFS;
    const uint source_row=first_row+row;
    device const uchar *source_w=w;
    for(uint block_index=0;block_index<a.cols/128u;++block_index) {
        float scale=0.0f;
        uint4 packed=uint4(0x55555555u);
        if(source_row<a.rows) {
            device const uchar *block=source_w+ulong(source_row)*a.row_bytes+block_index*34u;
            const uint byte=2u+start_k/4u;
            scale=float(*(device const half*)block);
            // Packed rows need only two-byte alignment. Retain the four
            // 16-coefficient words across this block's four K32 passes.
            device const ushort *halves=(device const ushort *)(block+byte);
            packed=uint4(uint(halves[0])|(uint(halves[1])<<16u),
                         uint(halves[4])|(uint(halves[5])<<16u),
                         uint(halves[8])|(uint(halves[9])<<16u),
                         uint(halves[12])|(uint(halves[13])<<16u));
        }
#pragma unroll
        for(uint part=0;part<4u;++part) {
            const uint first_k=block_index*128u+part*BK;
            const uint word=packed[part];
            threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
            for(uint j=0;j<COEFFS;j++) {
                const uint k=start_k+j;
                const uint dst=64u*((k/8u)*(BM/8u)+row/8u)+(k%8u)*8u+row%8u;
                weights[dst]=source_row<a.rows
                    ?scale*float(int((word>>(2u*j))&3u)-1):0.0f;
            }
            const uint token=tid/(BK/INPUT_RUN),input_k=(tid%(BK/INPUT_RUN))*INPUT_RUN;
#pragma unroll
            for(uint j=0;j<INPUT_RUN;j+=4u) {
                const uint k=input_k+j;
                float4 values=0.0f;
                if(first_token+token<a.n&&first_k+k+3u<a.cols)
                    values=*(device const float4*)(x+ulong(first_token+token)*a.cols+first_k+k);
                const uint dst=64u*((k/8u)*(BN/8u)+token/8u)+(token%8u)*8u+k%8u;
                *(threadgroup float4*)(inputs+dst)=values;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
            for(uint k=0;k<BK;k+=8u) {
#pragma unroll
                for(uint i=0;i<MR;i++)
                    simdgroup_load(wf[i],weights+64u*((k/8u)*(BM/8u)+(sg%2u)*MR+i),8,0,false);
#pragma unroll
                for(uint j=0;j<NR;j++)
                    simdgroup_load(xf[j],inputs+64u*((k/8u)*(BN/8u)+(sg/2u)*NR+j),8,0,false);
#pragma unroll
                for(uint j=0;j<NR;j++) {
#pragma unroll
                    for(uint i=0;i<MR;i++)simdgroup_multiply_accumulate(acc[j*MR+i],xf[j],wf[i],acc[j*MR+i]);
                }
            }
        }
    }
    // Uniform tile check: every thread takes the same output path.
    if(first_row+BM<=a.rows && first_token+BN<=a.n) {
        device float *dst=out+ulong(first_token+(sg/2u)*(BN/2u))*a.rows+first_row+(sg%2u)*(BM/2u);
#pragma unroll
        for(uint j=0;j<NR;j++) {
#pragma unroll
            for(uint i=0;i<MR;i++)simdgroup_store(acc[j*MR+i],dst+ulong(j*8u)*a.rows+i*8u,a.rows,0,false);
        }
    } else {
        // Reuse the weight scratch for bounded tails. Each token-half is
        // staged independently, so no extra output allocation is needed.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint part=0;part<2u;part++) {
            if(sg/2u==part) {
#pragma unroll
                for(uint j=0;j<NR;j++) {
#pragma unroll
                    for(uint i=0;i<MR;i++)simdgroup_store(acc[j*MR+i],weights+j*8u*BM+(sg%2u)*(BM/2u)+i*8u,BM,0,false);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
            for(uint i=tid;i<BM*(BN/2u);i+=128u) {
                uint t=part*(BN/2u)+i/BM,r=i%BM;
                if(first_token+t<a.n&&first_row+r<a.rows)out[ulong(first_token+t)*a.rows+first_row+r]=weights[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

/* The row/stripe mapping is adapted from PrismML-Eng/llama.cpp,
 * commit 9a9394a895b96003ca842a6041cb28ac49a108f7,
 * ggml/src/ggml-metal/kernels/mul_mv.metal.
 * The centered coefficient algebra below avoids the separate sum(y) correction.
 *
 * MIT License
 *
 * Copyright (c) 2023-2026 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// Every byte packs four codes t0..t3, each decoded as ti-1. Centering
// c = byte-85 makes the all-zero byte 0x55 exactly zero. The prefix values
// g1=floor((c+21)/64), g2=floor((c+5)/16), g3=floor((c+1)/4) obey
//   (t0-1,t1-1,t2-1,t3-1) = (c-4*g3,g3-4*g2,g2-4*g1,g1).
// Dotting them with (x0,x1,x2,x3) uses four activation coefficients and
// needs no subtraction of sum(x). Each FMA that forms a floor argument
// is exact: c is an integer in [-85,170] and all constants are dyadic.
// Accumulation remains FP32, with a different association from bonsai_mv.
template<bool FullRows>
inline void bonsai_pq2_mv_body(constant BonsaiArgs &a,
                          device const uchar *weights,
                          device const float *x, device float *out,
                          uint group,
                          ushort lane,
                          ushort sg) {
    constexpr uint NR = 8;
    const uint first_row = (group * 2u + sg) * NR;
    const uint stripe = lane / 8u, first_col = (lane % 8u) * 16u;
    float sums[NR];
#pragma unroll
    for (uint row = 0; row < NR; ++row) sums[row] = 0.0f;

    for (uint block = stripe; block < a.cols / 128u; block += 4u) {
        float coeff[16];
        device const packed_float4 *input =
            (device const packed_float4 *)(x + block * 128u + first_col);
#pragma unroll
        for (uint j = 0; j < 4; ++j) {
            const float4 v = float4(input[j]);
            coeff[4u*j] = v.w - 4.0f * v.z;
            coeff[4u*j+1u] = v.z - 4.0f * v.y;
            coeff[4u*j+2u] = v.y - 4.0f * v.x;
            coeff[4u*j+3u] = v.x;
        }
#pragma unroll
        for (uint row = 0; row < NR; ++row) {
            if (!FullRows && first_row + row >= a.rows) continue;
            device const uchar *p = weights + ulong(first_row + row) * a.row_bytes + block * 34u;
            float partial = 0.0f;
#pragma unroll
            for (uint j = 0; j < 4; ++j) {
                const float c = float(p[2u + first_col / 4u + j]) - 85.0f;
                partial += floor(fma(c, 1.0f/64.0f, 21.0f/64.0f)) * coeff[4u*j];
                partial += floor(fma(c, 1.0f/16.0f, 5.0f/16.0f)) * coeff[4u*j+1u];
                partial += floor(fma(c, 1.0f/4.0f, 1.0f/4.0f)) * coeff[4u*j+2u];
                partial += c * coeff[4u*j+3u];
            }
            sums[row] += float(*(device const half *)p) * partial;
        }
    }
#pragma unroll
    for (uint row = 0; row < NR; ++row) {
        const float total = simd_sum(sums[row]);
        if (!lane && (FullRows || first_row + row < a.rows)) out[first_row + row] = total;
    }
}

kernel void bonsai_pq2_mv(constant BonsaiArgs &a [[buffer(0)]],
                          device const uchar *weights [[buffer(1)]],
                          device const float *x [[buffer(2)]], device float *out [[buffer(3)]],
                          uint group [[threadgroup_position_in_grid]],
                          ushort lane [[thread_index_in_simdgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]]) {
    bonsai_pq2_mv_body<false>(a,weights,x,out,group,lane,sg);
}

// The host selects this entrypoint only for complete 16-row threadgroups.
kernel void bonsai_pq2_mv_full(constant BonsaiArgs &a [[buffer(0)]],
                          device const uchar *weights [[buffer(1)]],
                          device const float *x [[buffer(2)]], device float *out [[buffer(3)]],
                          uint group [[threadgroup_position_in_grid]],
                          ushort lane [[thread_index_in_simdgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]]) {
    bonsai_pq2_mv_body<true>(a,weights,x,out,group,lane,sg);
}

// Args: n=T, cols=C, width=conv_width, type=weight type. One thread/channel.
// History stays [C][width-1], oldest first. No channels share writable state.
kernel void bonsai_conv_batch(constant BonsaiArgs &a [[buffer(0)]],
                              device const float *x [[buffer(1)]],
                              device const uchar *w [[buffer(2)]],
                              device float *history [[buffer(3)]],
                              device float *out [[buffer(4)]],
                              uint c [[thread_position_in_grid]]) {
    if (c >= a.cols) return;
    for (uint token=0; token<a.n; ++token) {
        float sum=0;
        for (uint j=0; j+1<a.width; ++j)
            sum += history[ulong(c)*(a.width-1u)+j] * bs_scalar(w,c*a.width+j,a.type);
        const ulong index=ulong(token)*a.cols+c;
        sum += x[index] * bs_scalar(w,c*a.width+a.width-1u,a.type);
        for (uint j=0; j+2<a.width; ++j)
            history[ulong(c)*(a.width-1u)+j]=history[ulong(c)*(a.width-1u)+j+1u];
        if (a.width>1) history[ulong(c)*(a.width-1u)+a.width-2u]=x[index];
        out[index]=sum/(1.0f+exp(-sum));
    }
}

// Args: n=T, cols=D, heads=2*Hk, width=C, eps=L2 epsilon.
// Grid (2*Hk,T), exactly 256 threads. Normalize Q/K in place, skip the V tail.
// This is the frozen bonsai_norm tree and L2 expression, with only addressing
// specialized; it does NOT use Qwen's rsqrt(sum+epsilon) or move Q scaling.
kernel void bonsai_l2_batch(constant BonsaiArgs &a [[buffer(0)]],
                            device float *qkv [[buffer(1)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
    const ulong base=ulong(group.y)*a.width+ulong(group.x)*a.cols;
    threadgroup float tmp[256];
    float sum=0;
    for (uint i=tid; i<a.cols; i+=256) { float v=qkv[base+i]; sum += v*v; }
    tmp[tid]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid<32) {
        const float p0=tmp[tid]+tmp[tid+128u];
        const float p1=tmp[tid+64u]+tmp[tid+192u];
        const float p2=tmp[tid+32u]+tmp[tid+160u];
        const float p3=tmp[tid+96u]+tmp[tid+224u];
        float total=(p0+p1)+(p2+p3);
        total += simd_shuffle_down(total,16);
        total += simd_shuffle_down(total,8);
        total += simd_shuffle_down(total,4);
        total += simd_shuffle_down(total,2);
        total += simd_shuffle_down(total,1);
        if (!tid) tmp[0]=total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale=1.0f/max(sqrt(tmp[0]),a.eps);
    for (uint i=tid; i<a.cols; i+=256) qkv[base+i]=qkv[base+i]*scale*1.0f;
}

// Args: n=T, dim=D, heads=Hv, kvheads=Hk, type=A type, mode=dt type.
// One group/head, D threads. Each thread owns one dv column of [Hv][Dk][Dv]
// for every token, so sequential updates need no cross-thread synchronization.
// This generic variant intentionally keeps state in device memory, matching
// both dk passes and all accumulation orders of the frozen single-token GDN.
kernel void bonsai_gdn_batch(constant BonsaiArgs &a [[buffer(0)]],
                             device const float *qkv [[buffer(1)]],
                             device const float *alpha [[buffer(2)]],
                             device const float *beta [[buffer(3)]],
                             device const uchar *A [[buffer(4)]],
                             device const uchar *dt [[buffer(5)]],
                             device float *state [[buffer(6)]],
                             device float *out [[buffer(7)]],
                             uint h [[threadgroup_position_in_grid]],
                             uint dv [[thread_index_in_threadgroup]]) {
    if (dv>=a.dim) return;
    const uint kh=h%a.kvheads;
    const uint channels=(2u*a.kvheads+a.heads)*a.dim;
    for (uint token=0; token<a.n; ++token) {
        device const float *row=qkv+ulong(token)*channels;
        device const float *q=row+kh*a.dim,*k=row+(a.kvheads+kh)*a.dim;
        const float v=row[2u*a.kvheads*a.dim+h*a.dim+dv];
        const float ab=alpha[ulong(token)*a.heads+h]+bs_scalar(dt,h,a.mode);
        const float e=exp(-abs(ab)),u=1.0f+e;
        const float softplus=max(ab,0.0f)+(u==1.0f?e:log(u)*(e/(u-1.0f)));
        const float decay=exp(bs_scalar(A,h,a.type)*softplus);
        const float b=1.0f/(1.0f+exp(-beta[ulong(token)*a.heads+h]));
        float prediction=0;
        for (uint dk=0; dk<a.dim; ++dk) {
            const ulong idx=(ulong(h)*a.dim+dk)*a.dim+dv;
            float s=state[idx]*decay;
            state[idx]=s;
            prediction += s*k[dk];
        }
        const float delta=(v-prediction)*b;
        float result=0;
        for (uint dk=0; dk<a.dim; ++dk) {
            const ulong idx=(ulong(h)*a.dim+dk)*a.dim+dv;
            float s=state[idx]+delta*k[dk];
            state[idx]=s;
            result += s*q[dk];
        }
        out[(ulong(token)*a.heads+h)*a.dim+dv]=result*rsqrt(float(a.dim));
    }
}

// D128 specialization: keep each value lane's state column private across
// both dk passes and all tokens. Fully unrolled fixed-size dk loops let Metal
// scalarize the array; state is read and written only at scan boundaries.
// Preserve the generic kernel's FP32 expressions and ascending dk order.
// The host selects these entrypoints only for dim=128, 128 threads/head.
template<bool Batch>
inline void bonsai_gdn_128_body(constant BonsaiArgs &a,
    device const float *qkv, device const float *alpha, device const float *beta,
    device const uchar *A, device const uchar *dt,
    device float *state, device float *out, uint h, uint dv) {
    if (dv>=128u) return;
    const uint kh=h%a.kvheads;
    const uint channels=(2u*a.kvheads+a.heads)*128u;
    const ulong state_base=ulong(h)*128u*128u+dv;
    float column[128];
#pragma unroll
    for (uint dk=0; dk<128u; ++dk) column[dk]=state[state_base+dk*128u];
    const uint count=Batch?a.n:1u;
    for (uint token=0; token<count; ++token) {
        device const float *row=qkv+ulong(token)*channels;
        device const float *q=row+kh*128u,*k=row+(a.kvheads+kh)*128u;
        const float v=row[2u*a.kvheads*128u+h*128u+dv];
        const float ab=alpha[ulong(token)*a.heads+h]+bs_scalar(dt,h,a.mode);
        const float e=exp(-abs(ab)),u=1.0f+e;
        const float softplus=max(ab,0.0f)+(u==1.0f?e:log(u)*(e/(u-1.0f)));
        const float decay=exp(bs_scalar(A,h,a.type)*softplus);
        const float b=1.0f/(1.0f+exp(-beta[ulong(token)*a.heads+h]));
        float prediction=0;
#pragma unroll
        for (uint dk=0; dk<128u; ++dk) {
            float s=column[dk]*decay;
            column[dk]=s;
            prediction += s*k[dk];
        }
        const float delta=(v-prediction)*b;
        float result=0;
#pragma unroll
        for (uint dk=0; dk<128u; ++dk) {
            float s=column[dk]+delta*k[dk];
            column[dk]=s;
            result += s*q[dk];
        }
        out[(ulong(token)*a.heads+h)*128u+dv]=result*rsqrt(float(a.dim));
    }
#pragma unroll
    for (uint dk=0; dk<128u; ++dk) state[state_base+dk*128u]=column[dk];
}

kernel void bonsai_gdn_128(constant BonsaiArgs &a [[buffer(0)]],
    device const float *qkv [[buffer(1)]], device const float *alpha [[buffer(2)]],
    device const float *beta [[buffer(3)]], device const uchar *A [[buffer(4)]],
    device const uchar *dt [[buffer(5)]], device float *state [[buffer(6)]],
    device float *out [[buffer(7)]], uint h [[threadgroup_position_in_grid]],
    uint dv [[thread_index_in_threadgroup]]) {
    bonsai_gdn_128_body<false>(a,qkv,alpha,beta,A,dt,state,out,h,dv);
}

kernel void bonsai_gdn_batch_128(constant BonsaiArgs &a [[buffer(0)]],
    device const float *qkv [[buffer(1)]], device const float *alpha [[buffer(2)]],
    device const float *beta [[buffer(3)]], device const uchar *A [[buffer(4)]],
    device const uchar *dt [[buffer(5)]], device float *state [[buffer(6)]],
    device float *out [[buffer(7)]], uint h [[threadgroup_position_in_grid]],
    uint dv [[thread_index_in_threadgroup]]) {
    bonsai_gdn_128_body<true>(a,qkv,alpha,beta,A,dt,state,out,h,dv);
}

// Decode gate/up: retain bonsai_pq2_mv's centered coefficients, block/lane
// walk, scalar accumulation and SIMD reduction for both projections. Four
// output pairs give eight independent accumulators per lane, the
// same number of row accumulators as the NR8 standalone kernel. The activation follows the
// same expression as bonsai_element(mode=1); only mid is materialized.
// Both matrices consume the same prepared input and have the same layout.
// Grid: ceil(rows/8), 64 threads (two SIMD groups).
template<bool FullRows>
inline void bonsai_pq2_gate_up_body(constant BonsaiArgs &a,
    device const uchar *gate_w, device const uchar *up_w,
    device const float *x, device float *mid,
    uint group, ushort lane,
    ushort sg) {
    constexpr uint NR=4;
    const uint first_row=(group*2u+sg)*NR;
    const uint stripe=lane/8u, first_col=(lane%8u)*16u;
    float sums_g[NR],sums_u[NR];
#pragma unroll
    for(uint r=0;r<NR;++r) {sums_g[r]=0.0f;sums_u[r]=0.0f;}
    for(uint block=stripe;block<a.cols/128u;block+=4u) {
        float coeff[16];
        device const packed_float4 *input=(device const packed_float4 *)(x+block*128u+first_col);
#pragma unroll
        for(uint j=0;j<4;++j) {
            const float4 v=float4(input[j]);
            coeff[4u*j]=v.w-4.0f*v.z;
            coeff[4u*j+1u]=v.z-4.0f*v.y;
            coeff[4u*j+2u]=v.y-4.0f*v.x;
            coeff[4u*j+3u]=v.x;
        }
#pragma unroll
        for(uint r=0;r<NR;++r) {
            if(!FullRows && first_row+r>=a.rows)continue;
            ulong offset=ulong(first_row+r)*a.row_bytes+block*34u;
            device const uchar *g=gate_w+offset,*u=up_w+offset;
            float partial_g=0.0f,partial_u=0.0f;
#pragma unroll
            for(uint j=0;j<4;++j) {
                const float cg=float(g[2u+first_col/4u+j])-85.0f;
                const float cu=float(u[2u+first_col/4u+j])-85.0f;
                partial_g+=floor(fma(cg,1.0f/64.0f,21.0f/64.0f))*coeff[4u*j];
                partial_g+=floor(fma(cg,1.0f/16.0f,5.0f/16.0f))*coeff[4u*j+1u];
                partial_g+=floor(fma(cg,1.0f/4.0f,1.0f/4.0f))*coeff[4u*j+2u];
                partial_g+=cg*coeff[4u*j+3u];
                partial_u+=floor(fma(cu,1.0f/64.0f,21.0f/64.0f))*coeff[4u*j];
                partial_u+=floor(fma(cu,1.0f/16.0f,5.0f/16.0f))*coeff[4u*j+1u];
                partial_u+=floor(fma(cu,1.0f/4.0f,1.0f/4.0f))*coeff[4u*j+2u];
                partial_u+=cu*coeff[4u*j+3u];
            }
            sums_g[r]+=float(*(device const half *)g)*partial_g;
            sums_u[r]+=float(*(device const half *)u)*partial_u;
        }
    }
#pragma unroll
    for(uint r=0;r<NR;++r) {
        const float g=simd_sum(sums_g[r]),u=simd_sum(sums_u[r]);
        if(!lane && (FullRows || first_row+r<a.rows)) {
            mid[first_row+r]=(g/(1.0f+exp(-g)))*u;
        }
    }
}

kernel void bonsai_pq2_gate_up(constant BonsaiArgs &a [[buffer(0)]],
    device const uchar *gate_w [[buffer(1)]], device const uchar *up_w [[buffer(2)]],
    device const float *x [[buffer(3)]], device float *mid [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    bonsai_pq2_gate_up_body<false>(a,gate_w,up_w,x,mid,group,lane,sg);
}

// The host selects this entrypoint only for complete 8-row threadgroups.
kernel void bonsai_pq2_gate_up_full(constant BonsaiArgs &a [[buffer(0)]],
    device const uchar *gate_w [[buffer(1)]], device const uchar *up_w [[buffer(2)]],
    device const float *x [[buffer(3)]], device float *mid [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    bonsai_pq2_gate_up_body<true>(a,gate_w,up_w,x,mid,group,lane,sg);
}


// GDN alpha/beta decode: two unrotated BF16 projections share activation
// loads and expose independent sum chains. Keep bonsai_mv's k=lane,k+=32
// walk and simd_sum for each output. Grid: ceil(rows/4), 128 threads.
kernel void bonsai_bf16_pair(constant BonsaiArgs &a [[buffer(0)]],
    device const ushort *wa [[buffer(1)]],device const ushort *wb [[buffer(2)]],
    device const float *x [[buffer(3)]],device float *ya [[buffer(4)]],device float *yb [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint row=group*4+sg;float sa=0.0f,sb=0.0f;
    if(row<a.rows) {
        for(uint k=lane;k<a.cols;k+=32u) {
            const float input=x[k];
            sa+=as_type<float>(uint(wa[ulong(row)*a.cols+k])<<16)*input;
            sb+=as_type<float>(uint(wb[ulong(row)*a.cols+k])<<16)*input;
        }
    }
    sa=simd_sum(sa);sb=simd_sum(sb);
    if(!lane&&row<a.rows) {ya[row]=sa;yb[row]=sb;}
}

// Prefill gate/up/SiLU: the physical 64-row weight tile holds 32 gate rows
// followed by the matching 32 up rows. Each SIMD group retains the existing
// tiled kernel's 4x2 FP32 matrix accumulators and exact K walk; the two
// projections share activation loads without doubling accumulator pressure.
// The final weight scratch holds one 16-token half of both projections.
// Requires matching type142 shapes/prepared inputs. Grid: ceil(rows/32),
// ceil(n/32), 1; 128 threads. Declared threadgroup storage: 12 KiB.
kernel void bonsai_mm_pq2_gate_up_tiled(constant BonsaiArgs &a [[buffer(0)]],
    device const uchar *gate_w [[buffer(1)]], device const uchar *up_w [[buffer(2)]],
    device const float *x [[buffer(3)]], device float *mid [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint BM=64u, BN=32u, BK=32u, PAIRS=BM/2u;
    threadgroup float weights[BM*BK];
    threadgroup float inputs[BN*BK];
    constexpr uint MR=BM/16u, NR=BN/16u;
    constexpr uint COEFFS=BM*BK/128u, ROW_THREADS=BK/COEFFS;
    constexpr uint INPUT_RUN=BK*BN/128u;
    const uint first_row=group.x*PAIRS,first_token=group.y*BN;
    simdgroup_float8x8 wf[MR],xf[NR],acc[MR*NR];
#pragma unroll
    for(uint i=0;i<MR*NR;i++)acc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
    const uint row=tid/ROW_THREADS,start_k=(tid%ROW_THREADS)*COEFFS;
    const uint pair_row=row%PAIRS;
    const uint source_row=first_row+pair_row;
    device const uchar *source_w=row<PAIRS?gate_w:up_w;
    for(uint block_index=0;block_index<a.cols/128u;++block_index) {
        float scale=0.0f;
        uint4 packed=uint4(0x55555555u);
        if(source_row<a.rows) {
            device const uchar *block=source_w+ulong(source_row)*a.row_bytes+block_index*34u;
            const uint byte=2u+start_k/4u;
            scale=float(*(device const half*)block);
            // Packed rows need only two-byte alignment. Retain the four
            // 16-coefficient words across this block's four K32 passes.
            device const ushort *halves=(device const ushort *)(block+byte);
            packed=uint4(uint(halves[0])|(uint(halves[1])<<16u),
                         uint(halves[4])|(uint(halves[5])<<16u),
                         uint(halves[8])|(uint(halves[9])<<16u),
                         uint(halves[12])|(uint(halves[13])<<16u));
        }
#pragma unroll
        for(uint part=0;part<4u;++part) {
            const uint first_k=block_index*128u+part*BK;
            const uint word=packed[part];
            threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
            for(uint j=0;j<COEFFS;j++) {
                const uint k=start_k+j;
                const uint dst=64u*((k/8u)*(BM/8u)+row/8u)+(k%8u)*8u+row%8u;
                weights[dst]=source_row<a.rows
                    ?scale*float(int((word>>(2u*j))&3u)-1):0.0f;
            }
            const uint token=tid/(BK/INPUT_RUN),input_k=(tid%(BK/INPUT_RUN))*INPUT_RUN;
#pragma unroll
            for(uint j=0;j<INPUT_RUN;j+=4u) {
                const uint k=input_k+j;
                float4 values=0.0f;
                if(first_token+token<a.n&&first_k+k+3u<a.cols)
                    values=*(device const float4*)(x+ulong(first_token+token)*a.cols+first_k+k);
                const uint dst=64u*((k/8u)*(BN/8u)+token/8u)+(token%8u)*8u+k%8u;
                *(threadgroup float4*)(inputs+dst)=values;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
            for(uint k=0;k<BK;k+=8u) {
#pragma unroll
                for(uint i=0;i<MR;i++)
                    simdgroup_load(wf[i],weights+64u*((k/8u)*(BM/8u)+(sg%2u)*MR+i),8,0,false);
#pragma unroll
                for(uint j=0;j<NR;j++)
                    simdgroup_load(xf[j],inputs+64u*((k/8u)*(BN/8u)+(sg/2u)*NR+j),8,0,false);
#pragma unroll
                for(uint j=0;j<NR;j++) {
#pragma unroll
                    for(uint i=0;i<MR;i++)simdgroup_multiply_accumulate(acc[j*MR+i],xf[j],wf[i],acc[j*MR+i]);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint part=0;part<2u;part++) {
        if(sg/2u==part) {
#pragma unroll
            for(uint j=0;j<NR;j++) {
#pragma unroll
                for(uint i=0;i<MR;i++)
                    simdgroup_store(acc[j*MR+i],weights+j*8u*BM+(sg%2u)*PAIRS+i*8u,BM,0,false);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=tid;i<PAIRS*(BN/2u);i+=128u) {
            const uint t=part*(BN/2u)+i/PAIRS,r=i%PAIRS;
            if(first_token+t<a.n&&first_row+r<a.rows) {
                const uint src=(i/PAIRS)*BM+r;
                const float g=weights[src],u=weights[src+PAIRS];
                mid[ulong(first_token+t)*a.rows+first_row+r]=(g/(1.0f+exp(-g)))*u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Exact attention over independent causal queries. Args.n is the query
// count, pos is the absolute first query position, and width is the cache
// capacity. Scores are [query][head][width]; Q/output are [query][head][dim].
// Preserve every scalar dot product, softmax reduction and value walk from
// the single-query kernels. Only independent queries share each dispatch.
// Grid: ceil((pos+n)/256), heads, n; 256 threads per group.
kernel void bonsai_scores_batch(constant BonsaiArgs &a [[buffer(0)]],
    device const float *q [[buffer(1)]], device const float *kc [[buffer(2)]],
    device float *scores [[buffer(3)]], uint3 i [[thread_position_in_grid]]) {
    if(i.z>=a.n||i.y>=a.heads||i.x>a.pos+i.z)return;
    const uint kh=i.y/(a.heads/a.kvheads);
    device const float *k=kc+(ulong(i.x)*a.kvheads+kh)*a.dim;
    device const float *query=q+(ulong(i.z)*a.heads+i.y)*a.dim;
    float sum=0;
    for(uint d=0;d<a.dim;++d)sum+=query[d]*k[d];
    scores[(ulong(i.z)*a.heads+i.y)*a.width+i.x]=sum*rsqrt(float(a.dim));
}

// Grid: heads, n, 1; 256 threads per group. This is the unchanged 256-lane
// max/sum tree, including the ascending stride-256 per-lane token walk.
kernel void bonsai_softmax_batch(constant BonsaiArgs &a [[buffer(0)]],
    device float *scores [[buffer(1)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if(group.x>=a.heads||group.y>=a.n)return;
    threadgroup float tmp[256];
    const uint pos=a.pos+group.y;
    device float *row=scores+(ulong(group.y)*a.heads+group.x)*a.width;
    float mx=-INFINITY;
    for(ulong t=tid;t<=pos;t+=256)mx=max(mx,row[t]);
    tmp[tid]=mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=128;d;d/=2) {if(tid<d)tmp[tid]=max(tmp[tid],tmp[tid+d]);threadgroup_barrier(mem_flags::mem_threadgroup);}
    mx=tmp[0];
    float sum=0;
    for(ulong t=tid;t<=pos;t+=256) {float v=exp(row[t]-mx);row[t]=v;sum+=v;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tmp[tid]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=128;d;d/=2) {if(tid<d)tmp[tid]+=tmp[tid+d];threadgroup_barrier(mem_flags::mem_threadgroup);}
    sum=tmp[0];
    for(ulong t=tid;t<=pos;t+=256)row[t]/=sum;
}

// Grid: ceil(heads*dim/256), n, 1; 256 threads per group.
kernel void bonsai_attention_batch(constant BonsaiArgs &a [[buffer(0)]],
    device const float *scores [[buffer(1)]], device const float *vc [[buffer(2)]],
    device float *out [[buffer(3)]], uint2 i [[thread_position_in_grid]]) {
    if(i.y>=a.n||i.x>=a.heads*a.dim)return;
    const uint h=i.x/a.dim,d=i.x%a.dim,kh=h/(a.heads/a.kvheads);
    const uint pos=a.pos+i.y;
    float sum=0;
    for(uint t=0;t<=pos;++t)
        sum+=scores[(ulong(i.y)*a.heads+h)*a.width+t]*vc[(ulong(t)*a.kvheads+kh)*a.dim+d];
    out[ulong(i.y)*a.heads*a.dim+i.x]=sum;
}

// Four-token BF16 alpha/beta pair. Both matrices have identical unrotated
// shapes and compact BF16 rows. Share each activation load between the two
// projections, retaining bonsai_mm's k=lane,k+=32 FP32 accumulation and
// simd_sum independently for all eight sums. Args.n is the token count.
// Grid: ceil(rows/4), ceil(n/4), 1; 128 threads per threadgroup.
kernel void bonsai_bf16_pair_batch(constant BonsaiArgs &a [[buffer(0)]],
    device const ushort *wa [[buffer(1)]], device const ushort *wb [[buffer(2)]],
    device const float *x [[buffer(3)]], device float *ya [[buffer(4)]],
    device float *yb [[buffer(5)]], uint2 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint row=group.x*4u+sg,first=group.y*4u;
    float sa[4]={0.0f,0.0f,0.0f,0.0f},sb[4]={0.0f,0.0f,0.0f,0.0f};
    if(row<a.rows) {
        const ulong offset=ulong(row)*a.cols;
        for(uint k=lane;k<a.cols;k+=32u) {
            const float alpha=as_type<float>(uint(wa[offset+k])<<16);
            const float beta=as_type<float>(uint(wb[offset+k])<<16);
#pragma unroll
            for(uint t=0;t<4u;++t) {
                if(first+t>=a.n)continue;
                const float input=x[ulong(first+t)*a.cols+k];
                sa[t]+=alpha*input;
                sb[t]+=beta*input;
            }
        }
    }
#pragma unroll
    for(uint t=0;t<4u;++t) {
        const float alpha=simd_sum(sa[t]),beta=simd_sum(sb[t]);
        if(!lane&&row<a.rows&&first+t<a.n) {
            const ulong output=ulong(first+t)*a.rows+row;
            ya[output]=alpha;yb[output]=beta;
        }
    }
}
