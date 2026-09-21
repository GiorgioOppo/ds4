// GLM 5.3 operations. Pooling ported from GiorgioOppo/ds4
// acf5c16bb6a01f8c4f027a0db265ae270d8907ab (MIT).
// Concatenate after glm52_quant, glm53_bf16, and the shared dense kernels.
struct ds4_metal_args_glm53_indexer_pool_update {
    uint n_tokens, pos0, cache_cap, head_dim, pool_size, cache_f16;
    float eps;
};
static inline float glm53_pool_bf16_to_f32(ushort value) {
    return as_type<float>((uint)value << 16);
}

kernel void kernel_glm53_indexer_pool_update(
        constant ds4_metal_args_glm53_indexer_pool_update &args,
        device const char   *raw_k,
        device const char   *gate,
        device const float  *norm_weight,
        device const float  *norm_bias,
        device const ushort *ape,
        device       char   *pool_cache,
        device       float  *tail_k,
        device       float  *tail_gate,
        threadgroup  float  *shared [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    if (args.head_dim == 0u || args.pool_size == 0u ||
        tid >= args.head_dim || args.n_tokens == 0u) return;

    const uint pool = args.pos0 / args.pool_size + tgpig.x;
    const uint pool_start = pool * args.pool_size;
    const uint input_end = args.pos0 + args.n_tokens;
    if (pool_start >= input_end || pool_start + args.pool_size <= args.pos0) return;

    threadgroup float *rows = shared;
    threadgroup float *mean = rows + args.pool_size * args.head_dim;
    threadgroup float *inv = mean + args.pool_size;
    const bool complete = pool_start + args.pool_size <= input_end;

    for (uint r = 0; r < args.pool_size; r++) {
        const uint pos = pool_start + r;
        float k_value = 0.0f;
        float gate_value = 0.0f;
        if (pos >= args.pos0 && pos < input_end) {
            const uint src_row = pos - args.pos0;
            k_value = ((device const float *)raw_k)[
                (uint64_t)src_row * args.head_dim + tid];
            gate_value = ((device const float *)gate)[
                (uint64_t)src_row * args.head_dim + tid];
            if (!complete) {
                tail_k[(uint64_t)r * args.head_dim + tid] = k_value;
                tail_gate[(uint64_t)r * args.head_dim + tid] = gate_value;
            }
        } else {
            k_value = tail_k[(uint64_t)r * args.head_dim + tid];
            gate_value = tail_gate[(uint64_t)r * args.head_dim + tid];
        }
        rows[(uint64_t)r * args.head_dim + tid] = k_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!complete || pool >= (args.cache_cap + args.pool_size - 1u) / args.pool_size) {
        return;
    }

    if (tid < args.pool_size) {
        const uint r = tid;
        float sum = 0.0f;
        for (uint d = 0; d < args.head_dim; d++) {
            sum += rows[(uint64_t)r * args.head_dim + d];
        }
        const float m = sum / (float)args.head_dim;
        float ss = 0.0f;
        for (uint d = 0; d < args.head_dim; d++) {
            const float delta = rows[(uint64_t)r * args.head_dim + d] - m;
            ss += delta * delta;
        }
        mean[r] = m;
        inv[r] = rsqrt(ss / (float)args.head_dim + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float max_logit = -INFINITY;
    float logits[4];
    for (uint r = 0; r < args.pool_size; r++) {
        const uint pos = pool_start + r;
        float gate_value;
        if (pos >= args.pos0) {
            const uint src_row = pos - args.pos0;
            gate_value = ((device const float *)gate)[
                (uint64_t)src_row * args.head_dim + tid];
        } else {
            gate_value = tail_gate[(uint64_t)r * args.head_dim + tid];
        }
        logits[r] = gate_value +
            glm53_pool_bf16_to_f32(ape[(uint64_t)r * args.head_dim + tid]);
        max_logit = max(max_logit, logits[r]);
    }

    float denom = 0.0f;
    for (uint r = 0; r < args.pool_size; r++) {
        logits[r] = exp(logits[r] - max_logit);
        denom += logits[r];
    }
    float pooled = 0.0f;
    for (uint r = 0; r < args.pool_size; r++) {
        const float normalized =
            (rows[(uint64_t)r * args.head_dim + tid] - mean[r]) * inv[r] *
            norm_weight[tid] + norm_bias[tid];
        pooled += (logits[r] / denom) * normalized;
    }

    const uint64_t dst_index = (uint64_t)pool * args.head_dim + tid;
    if (args.cache_f16 != 0u) {
        ((device half *)pool_cache)[dst_index] = (half)pooled;
    } else {
        ((device float *)pool_cache)[dst_index] = pooled;
    }
}


struct glm53_projection_args { uint type, input, output, rows, heads, row_bytes; };
static inline float glm53_dot_lane(uint type, device const uchar *w,
                                  device const float *x, uint n, uint lane) {
    float sum = 0;
    if (type == 0u || type == 1u || type == 30u) {
        for (uint k = lane; k < n; k += 32u) {
            const float v = type == 0u ? ((device const float *)w)[k] :
                type == 1u ? float(((device const half *)w)[k]) :
                glm53_bf16_to_f32(((device const ushort *)w)[k]);
            sum = fma(v, x[k], sum);
        }
    } else {
        for (uint k = lane; k < n / 32u; k += 32u)
            sum += glm52_dot_kquant_group(type, w, x, k);
    }
    return simd_sum(sum);
}
kernel void kernel_glm53_grouped_projection(
    constant glm53_projection_args &a, device const uchar *w,
    device const float *x, device float *out,
    uint3 g [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint row = g.x * 4u + sg;
    if (row >= a.output || g.y >= a.rows || g.z >= a.heads) return;
    float sum = glm53_dot_lane(a.type, w + ((ulong)g.z * a.output + row) * a.row_bytes,
        x + ((ulong)g.y * a.heads + g.z) * a.input, a.input, lane);
    if (lane == 0u) out[((ulong)g.y * a.heads + g.z) * a.output + row] = sum;
}

static inline float glm53_weight_value(device const uchar *row, uint type, uint k) {
    if (type == 0u) return ((device const float *)row)[k];
    if (type == 1u) return float(((device const half *)row)[k]);
    if (type == 30u) return glm53_bf16_to_f32(((device const ushort *)row)[k]);
    if (type == 8u) {
        device const uchar *b = row + (k / 32u) * 34u;
        return glm52_half_at(b) * float(((device const char *)(b + 2))[k % 32u]);
    }
    device const uchar *b = row + (k / 256u) * 144u;
    const uint j = (k % 256u) / 32u;
    const uchar2 sm = glm52_scale_min_k4(j, b + 4);
    const uchar packed = b[16u + (j / 2u) * 32u + k % 32u];
    const uint q = j % 2u == 0u ? packed & 15u : packed >> 4u;
    return glm52_half_at(b) * float(sm.x) * float(q) - glm52_half_at(b + 2) * float(sm.y);
}
kernel void kernel_glm53_embedding_hc(constant glm53_projection_args &a,
    device const uchar *w, device const int *ids, device float *out,
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.input || gid.y >= a.rows) return;
    const float v = glm53_weight_value(w + (ulong)ids[gid.y] * a.row_bytes, a.type, gid.x);
    for (uint h = 0; h < 4; h++) out[((ulong)gid.y * 4 + h) * a.input + gid.x] = v;
}

kernel void kernel_glm53_route(constant uint &rows, device const float *logits,
    device const float *bias, device uint *ids, device float *weights,
    uint token [[thread_position_in_grid]]) {
    if (token >= rows) return;
    float probabilities[288], scores[288];
    for (uint i=0; i<288; i++) {
        probabilities[i] = 1.0f / (1.0f + exp(-logits[token*288+i]));
        scores[i] = probabilities[i] + bias[i];
    }
    float sum=0;
    for (uint slot=0; slot<8; slot++) {
        uint best=0; float score=-INFINITY;
        for (uint i=0; i<288; i++) if (scores[i]>score) { score=scores[i]; best=i; }
        ids[token*8+slot]=best; weights[token*8+slot]=probabilities[best];
        sum+=probabilities[best]; scores[best]=-INFINITY;
    }
    for (uint slot=0; slot<8; slot++) weights[token*8+slot]*=2.5f/max(sum,1e-20f);
}

// Applications of one expert are gathered to make a real batched projection;
// scatter is collision-free within an expert, with dispatches ordered by graph.
struct glm53_rows_args { uint width, rows; float scale; };
kernel void kernel_glm53_gather_rows(constant glm53_rows_args &a,
    device const float *src, device const uint *ids, device float *out,
    uint2 gid [[thread_position_in_grid]]) {
    if(gid.x<a.width && gid.y<a.rows) out[(ulong)gid.y*a.width+gid.x]=src[(ulong)ids[gid.y]*a.width+gid.x];
}
kernel void kernel_glm53_scatter_add(constant glm53_rows_args &a,
    device const float *src, device const uint *ids, device float *out,
    uint2 gid [[thread_position_in_grid]]) {
    if(gid.x<a.width && gid.y<a.rows) out[(ulong)ids[gid.y]*a.width+gid.x]+=src[(ulong)gid.y*a.width+gid.x];
}
kernel void kernel_glm53_swiglu_route(constant glm53_rows_args &a,
    device const float *gate, device const float *up, device const float *routes,
    device float *out, uint2 gid [[thread_position_in_grid]]) {
    if(gid.x>=a.width || gid.y>=a.rows) return;
    const ulong i=(ulong)gid.y*a.width+gid.x;
    const float g=min(gate[i],a.scale), u=clamp(up[i],-a.scale,a.scale);
    out[i]=(g/(1.0f+exp(-g)))*u*routes[gid.y];
}

struct glm53_index_args { uint visible, selected, heads; float scale; };
kernel void kernel_glm53_index_scores(constant glm53_index_args &a,
    device const float *q, device const float *weights, device const half *cache,
    device float *scores, uint pool [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if(pool>=a.visible) return;
    float score=0;
    for(uint h=0;h<a.heads;h++) {
        float dot=0;
        for(uint d=lane;d<128;d+=32) dot=fma(q[h*128+d],float(cache[(ulong)pool*128+d]),dot);
        dot=simd_sum(dot);
        score+=max(dot*a.scale,0.0f)*weights[h];
    }
    if(lane==0) scores[pool]=score;
}
kernel void kernel_glm53_select_raw(constant glm53_index_args &a,
    device const uint *pools, device uint *out, uint i [[thread_position_in_grid]]) {
    if(i>=a.selected) return;
    uint v=i;
    if(a.visible>2048u) {
        if(i<2048u) v=pools[i/4u]*4u+i%4u;
        else v=a.visible-a.visible%4u+(i-2048u);
    }
    out[i]=v<a.visible ? v:0xffffffffu;
}

// Absorbed MLA: every SIMD group scores one selected latent row, then the
// threadgroup accumulates the stable softmax in the 512-dimensional cache.
kernel void kernel_glm53_attention(constant glm53_index_args &a,
    device const float *q, device const half *cache, device const uint *selected,
    device float *out, threadgroup float *shared [[threadgroup(0)]],
    uint head [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float *scores=shared, *red=shared+a.selected;
    for(uint s=sg;s<a.selected;s+=4u) {
        const uint row=selected[s]; float dot=0;
        if(row<a.visible) for(uint d=lane;d<512;d+=32)
            dot=fma(q[head*512+d],float(cache[(ulong)row*512+d]),dot);
        dot=simd_sum(dot);
        if(lane==0) scores[s]=row<a.visible ? dot*a.scale:-INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m=-INFINITY;
    for(uint s=tid;s<a.selected;s+=128) m=max(m,scores[s]);
    m=simd_max(m); if(lane==0) red[sg]=m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m=max(max(red[0],red[1]),max(red[2],red[3]));
    float denom=0;
    for(uint s=tid;s<a.selected;s+=128) { float v=exp(scores[s]-m); scores[s]=v; denom+=v; }
    denom=simd_sum(denom); if(lane==0) red[sg]=denom;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    denom=max(red[0]+red[1]+red[2]+red[3],1e-20f);
    for(uint d=tid;d<512;d+=128) {
        float v=0;
        for(uint s=0;s<a.selected;s++) if(selected[s]<a.visible)
            v=fma(scores[s],float(cache[(ulong)selected[s]*512+d]),v);
        out[head*512+d]=v/denom;
    }
}

kernel void kernel_glm53_store_half(constant uint &count, device const float *src,
    device half *dst, uint i [[thread_position_in_grid]]) {
    if(i<count) dst[i]=half(src[i]);
}

kernel void kernel_glm53_scatter_contribution(constant glm53_rows_args &a,
    device const float *src, device const uint *slots, device float *out,
    uint2 gid [[thread_position_in_grid]]) {
    if(gid.x<a.width && gid.y<a.rows) out[(ulong)slots[gid.y]*a.width+gid.x]=src[(ulong)gid.y*a.width+gid.x];
}
kernel void kernel_glm53_combine_experts(constant glm53_rows_args &a,
    device const float *contributions, device float *shared_out,
    uint2 gid [[thread_position_in_grid]]) {
    if(gid.x>=a.width || gid.y>=a.rows) return;
    float sum=0;
    for(uint k=0;k<8;k++) sum+=contributions[((ulong)gid.y*8+k)*a.width+gid.x];
    shared_out[(ulong)gid.y*a.width+gid.x]+=sum;
}
