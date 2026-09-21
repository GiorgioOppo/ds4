// Swift graph adapters for V4.1. Arithmetic for index scoring follows
// kernel_glm_indexer_score_one_direct in ds4 acf5c16 (MIT, see dsv41.metal).
kernel void kernel_dsv41_swift_index_scores(constant uint4 &a,
    device const float *q, device const float *weights, device const float *keys,
    device float *scores, threadgroup float *shared [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    uint row=group.x, token=group.y;
    if(row>=a.x || token>=a.y)return;
    if(row >= (a.z+token+1u)/a.w) { if(!tid)scores[ulong(token)*a.x+row]=-INFINITY; return; }
    if(tid<128)shared[tid]=keys[ulong(row)*128u+tid];
    float acc=0;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint h0=0;h0<32;h0+=4) {
        uint h=h0+sg;
        float dotv=dot(((device const float4*)(q+(ulong(token)*32u+h)*128u))[lane],((threadgroup const float4*)shared)[lane]);
        dotv=simd_sum(dotv);
        if(!lane)shared[128u+sg]=max(dotv*(1.0f/64.0f),0.0f)*weights[ulong(token)*32u+h];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(!tid){acc+=shared[128];acc+=shared[129];acc+=shared[130];acc+=shared[131];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(!tid)scores[ulong(token)*a.x+row]=acc;
}
// Sort a bounded selected set by descending score, then ascending row id,
// matching argsort's deterministic tie rule. Each thread owns one rank.
kernel void kernel_dsv41_swift_sort_selected(constant uint &count,
    device const float *scores, device const int *ids, device int *out,
    uint i [[thread_position_in_grid]]) {
    if(i>=count)return;
    int id=ids[i];float score=scores[id];uint rank=0;
    for(uint j=0;j<count;j++){int other=ids[j];float s=scores[other];rank+=(s>score || (s==score && other<id));}
    out[rank]=id;
}
kernel void kernel_dsv41_swift_mask(constant uint2 &a, device const int *ids,
    device float *out, uint i [[thread_position_in_grid]]) {
    if(i>=a.x)return;float value=-INFINITY;
    for(uint j=0;j<a.y;j++)if(ids[j]==int(i)){value=0.0f;break;}
    out[i]=value;
}
kernel void kernel_dsv41_swift_gather(constant uint2 &a, device const float *source,
    device const int *ids, device float *out, uint2 i [[thread_position_in_grid]]) {
    if(i.x<a.x && i.y<a.y)out[ulong(i.y)*a.x+i.x]=source[ulong(ids[i.y])*a.x+i.x];
}
// Bounded streamed-expert rows gather/scatter. `ids` are route-slot positions,
// so output reduction retains the original six-slot order independently of I/O.
kernel void kernel_dsv41_swift_route_gather(constant uint2 &a, device const float *source,
    device const int *ids, device const float *weights, device float *out, device float *route,
    uint2 i [[thread_position_in_grid]]) {
    if(i.x>=a.x || i.y>=a.y)return;uint slot=uint(ids[i.y]);
    out[ulong(i.y)*a.x+i.x]=source[ulong(slot/6u)*a.x+i.x];
    if(!i.x)route[i.y]=weights[slot];
}
kernel void kernel_dsv41_swift_route_scatter(constant uint2 &a, device const float *source,
    device const int *ids, device float *out, uint2 i [[thread_position_in_grid]]) {
    if(i.x<a.x && i.y<a.y)out[ulong(ids[i.y])*a.x+i.x]=source[ulong(i.y)*a.x+i.x];
}
// Q4_0 remains packed. Used when the checkpoint chooses it for a dense
// projection; the common Q8/Q4_K paths use the optimized existing GEMM/GEMV.
kernel void kernel_dsv41_swift_q4_0(constant uint4 &a, device const uchar *weight,
    device const float *input, device float *output, uint2 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    uint row=group.x*4u+sg, token=group.y;float sum=0;
    if(row<a.y && token<a.z)for(uint k=lane;k<a.x;k+=32u){
        device const uchar *block=weight+ulong(row)*a.w+(k/32u)*18u;
        uint j=k%32u;int q=int((block[2u+j%16u]>>(j<16u?0u:4u))&15u)-8;
        sum+=(float(*(device const half*)block)*float(q))*input[ulong(token)*a.x+k];
    }
    sum=simd_sum(sum);if(!lane && row<a.y && token<a.z)output[ulong(token)*a.y+row]=sum;
}
// Extended K-quants use the existing quantized SIMD row helpers.
kernel void kernel_dsv41_swift_kquant(constant uint4 &a, device const uchar *weight,
    device const float *input, device float *output, uint2 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    uint row=group.x*4u+sg,token=group.y;float sum=0;
    if(row<a.y) {
        uint bytes=glm52_kquant_row_bytes(a.w,a.x);
        for(uint g=lane;g<a.x/32u;g+=32u)sum+=glm52_dot_kquant_group(a.w,weight+ulong(row)*bytes,input+ulong(token)*a.x,g);
    }
    sum=simd_sum(sum);if(!lane && row<a.y)output[ulong(token)*a.y+row]=sum;
}
kernel void kernel_dsv41_swift_mxfp4(constant uint4 &a, device const uchar *weight,
    device const float *input, device float *output, uint2 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr float values[16]={0,.5,1,1.5,2,3,4,6,-0.0,-.5,-1,-1.5,-2,-3,-4,-6};
    uint row=group.x*4u+sg,token=group.y;float sum=0;
    if(row<a.y)for(uint k=lane;k<a.x;k+=32u){
        device const uchar *b=weight+ulong(row)*(a.x/32u)*17u+(k/32u)*17u;
        uint j=k%32u,q=(b[1u+j%16u]>>(j<16u?0u:4u))&15u;
        float scale=as_type<float>(b[0]==0?0x00400000u:uint(b[0])<<23);
        sum+=(scale*values[q])*input[ulong(token)*a.x+k];
    }
    sum=simd_sum(sum);if(!lane && row<a.y)output[ulong(token)*a.y+row]=sum;
}
