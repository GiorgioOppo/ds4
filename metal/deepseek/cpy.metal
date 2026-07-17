struct ds4_metal_args_cpy {
    int64_t  nk0;
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
};

// Typed copy/conversion between graph tensors. DS4 uses this for layout
// materialization and F32/F16 conversions at graph boundaries such as KV/cache
// packing and compressor pooling.
template<typename T0, typename T1>
kernel void kernel_cpy_t_t(
        constant ds4_metal_args_cpy & args,
        device  const char * src0,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tiitg/ntg[0];
    const int iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);

    device T1 * dst_data = (device T1 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = iw0*ntg[0] + tiitg%ntg[0]; i00 < args.ne00; ) {
        device const T0 * src = (device T0 *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);
        dst_data[i00] = (T1) src[0];
        break;
    }
}

typedef decltype(kernel_cpy_t_t<float, float>) kernel_cpy_t;
// Host-visible copy/conversion variants used by the DS4 graph.
template [[host_name("kernel_cpy_f32_f32")]] kernel kernel_cpy_t kernel_cpy_t_t<float, float>;
template [[host_name("kernel_cpy_f32_f16")]] kernel kernel_cpy_t kernel_cpy_t_t<float, half>;
template [[host_name("kernel_cpy_f16_f32")]] kernel kernel_cpy_t kernel_cpy_t_t<half, float>;

// Fast contiguous conversions used by decode staging.  Unlike the generic
// tensor copy above these kernels do not reconstruct four-dimensional tensor
// indices for every scalar.  packed_*4 keeps scalar alignment, so logical
// tensor views only need their normal float/half alignment.  The scalar tail
// preserves the generic conversion for lengths not divisible by four.
kernel void kernel_cpy_contig_f32_f16_4(
        constant uint & n,
        device const packed_float4 * src,
        device       packed_half4  * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint i = gid * 4u;
    if (i >= n) return;

    const uint remaining = n - i;
    if (remaining >= 4u) {
        dst[gid] = packed_half4(half4(float4(src[gid])));
        return;
    }

    device const float * src_scalar = (device const float *)src;
    device       half  * dst_scalar = (device       half  *)dst;
    for (uint lane = 0; lane < remaining; ++lane) {
        dst_scalar[i + lane] = half(src_scalar[i + lane]);
    }
}

kernel void kernel_cpy_contig_f16_f32_4(
        constant uint & n,
        device const packed_half4  * src,
        device       packed_float4 * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint i = gid * 4u;
    if (i >= n) return;

    const uint remaining = n - i;
    if (remaining >= 4u) {
        dst[gid] = packed_float4(float4(half4(src[gid])));
        return;
    }

    device const half  * src_scalar = (device const half  *)src;
    device       float * dst_scalar = (device       float *)dst;
    for (uint lane = 0; lane < remaining; ++lane) {
        dst_scalar[i + lane] = float(src_scalar[i + lane]);
    }
}

// Bitwise F16 transport for cache staging.  ushort is intentional: converting
// through half would be allowed to canonicalize NaN payloads.
kernel void kernel_cpy_contig_f16_f16_bits_4(
        constant uint & n,
        device const packed_ushort4 * src,
        device       packed_ushort4 * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint i = gid * 4u;
    if (i >= n) return;

    const uint remaining = n - i;
    if (remaining >= 4u) {
        dst[gid] = src[gid];
        return;
    }

    device const ushort * src_scalar = (device const ushort *)src;
    device       ushort * dst_scalar = (device       ushort *)dst;
    for (uint lane = 0; lane < remaining; ++lane) {
        dst_scalar[i + lane] = src_scalar[i + lane];
    }
}

struct ds4_metal_args_flash_kv_stage_f32 {
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_raw;
    uint32_t n_comp;
    uint32_t pad_rows;
    uint32_t fuse_pad;
};

// Gather the chronological raw F32 ring and the compressed F32 cache into the
// contiguous F16 buffer consumed by decode FlashAttention.  Both spans share a
// single dispatch.  When the final 32-row block is partial, the same dispatch
// also builds the historical K/V/mask pad layout, preserving the exact F16
// rounding and mask bits of the standalone copy + pad graph.
//
// This specialization is deliberately fixed to DeepSeek V4's 512-wide latent
// row: four elements per vector, 128 vectors per row.  The Swift caller keeps a
// shape-checked fallback for every other layout.
kernel void kernel_dsv4_flash_kv_stage_f32(
        constant ds4_metal_args_flash_kv_stage_f32 & args,
        device const char * raw_src,
        device const char * comp_src,
        device       char * dst,
        device const char * mask_src,
        device       char * pad_dst,
        uint gid [[thread_position_in_grid]]) {
    constexpr uint row_vecs = 128u;
    const uint raw_vecs = args.n_raw * row_vecs;
    const uint n_keys = args.n_raw + args.n_comp;
    const uint total_vecs = n_keys * row_vecs;

    if (gid < raw_vecs) {
        const uint logical_row = gid >> 7;
        const uint col = gid & 127u;
        uint physical_row = args.raw_start + logical_row;
        if (physical_row >= args.raw_cap) physical_row -= args.raw_cap;

        device const packed_float4 * raw =
            (device const packed_float4 *)raw_src;
        device packed_half4 * dst_half = (device packed_half4 *)dst;
        dst_half[gid] = packed_half4(half4(float4(
            raw[physical_row * row_vecs + col])));
        return;
    }

    if (gid < total_vecs) {
        device const packed_float4 * comp =
            (device const packed_float4 *)comp_src;
        device packed_half4 * dst_half = (device packed_half4 *)dst;
        dst_half[gid] = packed_half4(half4(float4(comp[gid - raw_vecs])));
        return;
    }

    if (args.fuse_pad == 0u || args.pad_rows == 0u) return;

    const uint pad_vecs = args.pad_rows * row_vecs;
    uint pad_gid = gid - total_vecs;
    if (pad_gid < pad_vecs) {
        const uint row = pad_gid / row_vecs;
        const uint col = pad_gid - row * row_vecs;
        const uint valid_rows = n_keys % args.pad_rows;
        device packed_half4 * pad_half = (device packed_half4 *)pad_dst;

        packed_half4 value = packed_half4(half4(0.0h));
        if (row < valid_rows) {
            const uint logical_row = n_keys - valid_rows + row;
            if (logical_row < args.n_raw) {
                uint physical_row = args.raw_start + logical_row;
                if (physical_row >= args.raw_cap) physical_row -= args.raw_cap;
                device const packed_float4 * raw =
                    (device const packed_float4 *)raw_src;
                value = packed_half4(half4(float4(
                    raw[physical_row * row_vecs + col])));
            } else {
                device const packed_float4 * comp =
                    (device const packed_float4 *)comp_src;
                value = packed_half4(half4(float4(
                    comp[(logical_row - args.n_raw) * row_vecs + col])));
            }
        }

        // The current FlashAttention specialization expects distinct K and V
        // pad planes even though MLA uses the same latent values for both.
        pad_half[pad_gid] = value;
        pad_half[pad_vecs + pad_gid] = value;
        return;
    }

    pad_gid -= pad_vecs;
    if (pad_gid < args.pad_rows) {
        const uint valid_rows = n_keys % args.pad_rows;
        device const ushort * mask_bits = (device const ushort *)mask_src;
        device ushort * pad_mask_bits =
            (device ushort *)pad_dst + 2u * pad_vecs * 4u;
        pad_mask_bits[pad_gid] = pad_gid < valid_rows
            ? mask_bits[n_keys - valid_rows + pad_gid]
            : 0xfbffu; // -MAXHALF, exactly as kernel_flash_attn_ext_pad.
    }
}

// Materialize a chronological raw-KV window from the circular F32 cache while
// converting it to the F16 layout consumed by decode FlashAttention. The old
// host path split a wrapped window into two cpy dispatches for every layer and
// token after the 128-row boundary. One 2D dispatch keeps every row coalesced
// and removes that extra command without changing storage or rounding.
struct ds4_metal_args_raw_ring_cpy {
    uint32_t n_rows;
    uint32_t row_width;
    uint32_t raw_cap;
    uint32_t raw_start;
};

kernel void kernel_dsv4_raw_ring_cpy_f32_f16(
        constant ds4_metal_args_raw_ring_cpy & args,
        device const float * src,
        device       half  * dst,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.row_width || gid.y >= args.n_rows) {
        return;
    }
    // raw_start and gid.y are both below raw_cap, so one subtraction is enough
    // and avoids a modulo/division in every GPU lane.
    uint physical_row = args.raw_start + gid.y;
    if (physical_row >= args.raw_cap) {
        physical_row -= args.raw_cap;
    }
    dst[gid.y * args.row_width + gid.x] = half(src[physical_row * args.row_width + gid.x]);
}
