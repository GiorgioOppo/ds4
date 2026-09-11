struct ds4_metal_args_argsort {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
};

struct ds4_metal_args_argsort_merge {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
    int32_t  len;
};

typedef void (argsort_t)(
        constant   ds4_metal_args_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

// Sort one float row into an index row. DS4 only exports the descending
// instance because router and indexer selection both need top-k order.
template<ds4_sort_order order>
kernel void kernel_argsort_f32_i32(
        constant   ds4_metal_args_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    // bitonic sort
    const int col = tpitg[0];
    const int ib  = tgpig[0] / args.ne01;

    const int i00 = ib*ntg.x;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    device const float * src0_row = (device const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);

    // initialize indices
    shmem_i32[col] = i00 + col;

    // Stage this block's score slice in threadgroup memory (indices stay in
    // [i00, i00+ntg.x), so shmem_f32[idx - i00] replaces the device gather).
    // The host allocates ntg.x extra floats after the index array.  Values and
    // the comparison network are unchanged, so the permutation is identical.
    threadgroup float * shmem_f32 = (threadgroup float *) (shmem_i32 + ntg.x);
    if (i00 + col < args.ne00) {
        shmem_f32[col] = src0_row[i00 + col];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= ntg.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (shmem_i32[col] >= args.ne00 ||
                       (shmem_i32[ixj] <  args.ne00 && (order == DS4_SORT_ORDER_ASC ?
                            shmem_f32[shmem_i32[col] - i00] > shmem_f32[shmem_i32[ixj] - i00] :
                            shmem_f32[shmem_i32[col] - i00] < shmem_f32[shmem_i32[ixj] - i00]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                } else {
                    if (shmem_i32[ixj] >= args.ne00 ||
                       (shmem_i32[col] <  args.ne00 && (order == DS4_SORT_ORDER_ASC ?
                            shmem_f32[shmem_i32[col] - i00] < shmem_f32[shmem_i32[ixj] - i00] :
                            shmem_f32[shmem_i32[col] - i00] > shmem_f32[shmem_i32[ixj] - i00]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int64_t i0 = ib*args.top_k;

    // copy the result to dst without the padding
    if (i0 + col < args.ne0 && col < args.top_k) {
        dst += i0 + args.ne0*i01 + args.ne0*args.ne1*i02 + args.ne0*args.ne1*args.ne2*i03;

        dst[col] = shmem_i32[col];
    }
}

// Host-visible sort variant used by DS4 top-k selection.
template [[host_name("kernel_argsort_f32_i32_desc")]] kernel argsort_t kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC>;

typedef void (argsort_merge_t)(
        constant   ds4_metal_args_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

// Merges sorted index runs produced by kernel_argsort_f32_i32. In the DS4 graph
// this finishes top-k over router or compressed-attention score rows.
template<ds4_sort_order order>
kernel void kernel_argsort_merge_f32_i32(
        constant   ds4_metal_args_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {

    const int im  = tgpig[0] / args.ne01;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    const int start = im * (2 * args.len);

    const int len0 = MIN(args.len, MAX(0, args.ne0 - (int)(start)));
    const int len1 = MIN(args.len, MAX(0, args.ne0 - (int)(start + args.len)));

    const int total = len0 + len1;

    device const int32_t * tmp0 = tmp + start
        + i01*args.ne0
        + i02*args.ne0*args.ne01
        + i03*args.ne0*args.ne01*args.ne02;

    device const int32_t * tmp1 = tmp0 + args.len;

    dst += start
        + i01*args.top_k
        + i02*args.top_k*args.ne01
        + i03*args.top_k*args.ne01*args.ne02;

    device const float * src0_row = (device const float *)(src0
        + args.nb01*i01
        + args.nb02*i02
        + args.nb03*i03);

    if (total == 0) {
        return;
    }

    const int chunk = (total + ntg.x - 1) / ntg.x;

    const int k0 = tpitg.x * chunk;
    const int k1 = MIN(MIN(k0 + chunk, total), args.top_k);

    if (k0 >= args.top_k) {
        return;
    }

    if (k0 >= total) {
        return;
    }

    int low  = k0 > len1 ? k0 - len1 : 0;
    int high = MIN(k0, len0);

    // binary-search partition (i, j) such that i + j = k
    while (low < high) {
        const int mid = (low + high) >> 1;

        const int32_t idx0 = tmp0[mid];
        const int32_t idx1 = tmp1[k0 - mid - 1];

        const float val0 = src0_row[idx0];
        const float val1 = src0_row[idx1];

        bool take_left;
        if (order == DS4_SORT_ORDER_ASC) {
            take_left = (val0 <= val1);
        } else {
            take_left = (val0 >= val1);
        }

        if (take_left) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    int i = low;
    int j = k0 - i;

    // keep the merge fronts into registers
    int32_t idx0 = 0;
    float   val0 = 0.0f;
    if (i < len0) {
        idx0 = tmp0[i];
        val0 = src0_row[idx0];
    }

    int32_t idx1 = 0;
    float   val1 = 0.0f;
    if (j < len1) {
        idx1 = tmp1[j];
        val1 = src0_row[idx1];
    }

    for (int k = k0; k < k1; ++k) {
        int32_t out_idx;

        if (i >= len0) {
            while (k < k1) {
                dst[k++] = tmp1[j++];
            }
            break;
        } else if (j >= len1) {
            while (k < k1) {
                dst[k++] = tmp0[i++];
            }
            break;
        } else {
            bool take_left;

            if (order == DS4_SORT_ORDER_ASC) {
                take_left = (val0 <= val1);
            } else {
                take_left = (val0 >= val1);
            }

            if (take_left) {
                out_idx = idx0;
                ++i;
                if (i < len0) {
                    idx0 = tmp0[i];
                    val0 = src0_row[idx0];
                }
            } else {
                out_idx = idx1;
                ++j;
                if (j < len1) {
                    idx1 = tmp1[j];
                    val1 = src0_row[idx1];
                }
            }
        }

        dst[k] = out_idx;
    }
}

// Host-visible merge variant used by DS4 top-k selection.
template [[host_name("kernel_argsort_merge_f32_i32_desc")]] kernel argsort_merge_t kernel_argsort_merge_f32_i32<DS4_SORT_ORDER_DESC>;

// Each input row contains equal-sized sorted runs, except its final run.
// Output keeps only the first min(K, len0 + len1) elements of each pair.
// Leaf sorting is unchanged and ties still select the left run first. Thus
// finite values, infinities and signed zero retain the legacy permutation;
// NaN ordering, which was never total in the legacy comparator, is unspecified.
struct ds4_metal_args_argsort_compact_merge {
    uint64_t score_stride;
    uint32_t n_rows;
    uint32_t input_width;
    uint32_t run_length;
    uint32_t output_width;
    uint32_t output_run_length;
    uint32_t pad;
};

kernel void kernel_argsort_merge_f32_i32_desc_compact(
        constant ds4_metal_args_argsort_compact_merge & args,
        device const char * src0,
        device const int32_t * tmp,
        device int32_t * dst,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint im = tgpig.x / args.n_rows;
    const uint row = tgpig.x % args.n_rows;
    const uint64_t start = (uint64_t)im * (2ull * args.run_length);
    if (start >= args.input_width) return;
    const uint len0 = min((uint64_t)args.run_length, (uint64_t)args.input_width - start);
    const uint len1 = start + len0 < args.input_width
        ? min((uint64_t)args.run_length, (uint64_t)args.input_width - start - len0) : 0u;
    const uint kept = min(args.output_run_length, len0 + len1);
    device const int32_t * tmp0 = tmp + (uint64_t)row * args.input_width + start;
    device const int32_t * tmp1 = tmp0 + len0;
    dst += (uint64_t)row * args.output_width + (uint64_t)im * args.output_run_length;
    device const float * scores = (device const float *)(src0 + (uint64_t)row * args.score_stride);

    // Distribute retained outputs, not the discarded tail, across threads.
    const uint chunk = (kept + ntg.x - 1u) / ntg.x;
    const uint k0 = tpitg.x * chunk;
    const uint k1 = min(k0 + chunk, kept);
    if (k0 >= kept) return;
    uint low = k0 > len1 ? k0 - len1 : 0u;
    uint high = min(k0, len0);
    while (low < high) {
        const uint mid = (low + high) >> 1u;
        const float val0 = scores[tmp0[mid]];
        const float val1 = scores[tmp1[k0 - mid - 1u]];
        if (val0 >= val1) low = mid + 1u;
        else high = mid;
    }
    uint i = low, j = k0 - i;
    int32_t idx0 = 0, idx1 = 0;
    float val0 = 0.0f, val1 = 0.0f;
    if (i < len0) { idx0 = tmp0[i]; val0 = scores[idx0]; }
    if (j < len1) { idx1 = tmp1[j]; val1 = scores[idx1]; }
    for (uint k = k0; k < k1; ++k) {
        if (i >= len0) {
            while (k < k1) dst[k++] = tmp1[j++];
            break;
        }
        if (j >= len1) {
            while (k < k1) dst[k++] = tmp0[i++];
            break;
        }
        if (val0 >= val1) {
            dst[k] = idx0;
            if (++i < len0) { idx0 = tmp0[i]; val0 = scores[idx0]; }
        } else {
            dst[k] = idx1;
            if (++j < len1) { idx1 = tmp1[j]; val1 = scores[idx1]; }
        }
    }
}
