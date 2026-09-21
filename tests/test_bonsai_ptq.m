/* PTQ1_0 exact decode/prefill oracle and resident GPU microbenchmark.
 * clang -O2 -Wall -Wextra -fobjc-arc tests/test_bonsai_ptq.m \
 *   -framework Foundation -framework Metal -o /tmp/test_bonsai_ptq
 * MTL_DEBUG_LAYER=1 /tmp/test_bonsai_ptq [--shader metal/bonsai.metal]
 * /tmp/test_bonsai_ptq [--bench] [--bench-fused] [--shader metal/bonsai.metal]
 * Run timing separately from Metal validation. No model or runtime linkage.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t n,rows,cols,type,row_bytes,pos,heads,kvheads,dim,rot,width,mode,groups;
    float eps,base;
} Args;
_Static_assert(sizeof(Args)==60,"Bonsai argument ABI");
static const NSUInteger pad=64;

/* Frozen bfb488f kernels AND weight decoder: do not regenerate from production.
 * Keeping the complete generic branches also freezes the multiply/add order.
 * The oracle never calls production bs_weight or another production helper. */
static NSString *frozen_source =
@"static inline float ptq_reference_bs_scalar(device const uchar *w, uint i, uint type) {\n"
"    if (type == 0u) return ((device const float *)w)[i];\n"
"    if (type == 1u) return float(((device const half *)w)[i]);\n"
"    return as_type<float>(uint(((device const ushort *)w)[i]) << 16); // BF16\n"
"}\n"
"\n"
"static inline float ptq_reference_bs_weight(device const uchar *w, uint i, uint type) {\n"
"    if (type == 0u || type == 1u || type == 30u) return ptq_reference_bs_scalar(w, i, type);\n"
"    if (type == 8u) {\n"
"        device const uchar *b = w + (i / 32u) * 34u;\n"
"        return float(*(device const half *)b) * float(((device const char *)(b + 2))[i % 32u]);\n"
"    }\n"
"    if (type == 142u) {\n"
"        device const uchar *b = w + (i / 128u) * 34u;\n"
"        uint j = i % 128u;\n"
"        return float(*(device const half *)b) * float(int((b[2u + j / 4u] >> (2u * (j % 4u))) & 3u) - 1);\n"
"    }\n"
"    // PTQ1_0: 16 bytes x 5 trits, 8 bytes x 5 trits, 2 bytes x 4\n"
"    // trits, then fp16 scale. The byte product wraps before trit extraction.\n"
"    device const uchar *b = w + (i / 128u) * 28u;\n"
"    uint j = i % 128u, byte, trit;\n"
"    if (j < 80u) { byte = j % 16u; trit = j / 16u; }\n"
"    else if (j < 120u) { byte = 16u + (j - 80u) % 8u; trit = (j - 80u) / 8u; }\n"
"    else { byte = 24u + (j - 120u) % 2u; trit = (j - 120u) / 2u; }\n"
"    constexpr uint pow3[] = {1, 3, 9, 27, 81};\n"
"    const uint q = (uint(b[byte]) * pow3[trit]) & 255u;\n"
"    return float(*(device const half *)(b + 26u)) * float(int((q * 3u) >> 8) - 1);\n"
"}\n"
"\n"
"kernel void ptq_reference_bonsai_mv(constant BonsaiArgs &a [[buffer(0)]],\n"
"                      device const uchar *w [[buffer(1)]], device const float *x [[buffer(2)]],\n"
"                      device float *out [[buffer(3)]], uint group [[threadgroup_position_in_grid]],\n"
"                      ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {\n"
"    uint row = group * 4u + sg;\n"
"    float sum = 0;\n"
"    if (row < a.rows) {\n"
"        device const uchar *wr = w + ulong(row) * a.row_bytes;\n"
"        if (a.type == 142u) {\n"
"            // One PQ2 block holds four successive iterations of the scalar\n"
"            // lane loop. Decode its scale once, keeping the original lane\n"
"            // assignment and addition order (including the final simd_sum).\n"
"            // This avoids repeating the generic type/scale decode per weight.\n"
"            const uint byte = 2u + lane / 4u, shift = (lane % 4u) * 2u;\n"
"            for (uint block = 0; block < a.cols / 128u; ++block) {\n"
"                device const uchar *p = wr + block * 34u;\n"
"                const float scale = float(*(device const half *)p);\n"
"                const uint4 packed = uint4(p[byte], p[byte + 8u], p[byte + 16u], p[byte + 24u]) >> shift;\n"
"                const float4 q = float4(int4(packed & 3u) - 1);\n"
"                const uint k = block * 128u + lane;\n"
"                sum += (scale * q.x) * x[k];\n"
"                sum += (scale * q.y) * x[k + 32u];\n"
"                sum += (scale * q.z) * x[k + 64u];\n"
"                sum += (scale * q.w) * x[k + 96u];\n"
"            }\n"
"        } else if (a.type == 30u) {\n"
"            // GDN alpha/beta have few BF16 output rows. Resolve the storage\n"
"            // type outside the long K loop, as in the typed dense kernels.\n"
"            // Keep the same lane walk, FP32 arithmetic and SIMD reduction.\n"
"            device const ushort *wb = (device const ushort *)wr;\n"
"            for (uint k = lane; k < a.cols; k += 32u)\n"
"                sum += as_type<float>(uint(wb[k]) << 16) * x[k];\n"
"        } else {\n"
"            for (uint k = lane; k < a.cols; k += 32u) sum += ptq_reference_bs_weight(wr, k, a.type) * x[k];\n"
"        }\n"
"    }\n"
"    sum = simd_sum(sum);\n"
"    if (!lane && row < a.rows) out[row] = sum;\n"
"}\n"
"\n"
"kernel void ptq_reference_bonsai_mm(constant BonsaiArgs &a [[buffer(0)]],\n"
"                      device const uchar *w [[buffer(1)]], device const float *x [[buffer(2)]],\n"
"                      device float *out [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],\n"
"                      ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {\n"
"    const uint row = group.x * 4u + sg, first = group.y * 4u;\n"
"    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};\n"
"    if (row < a.rows) {\n"
"        device const uchar *wr = w + ulong(row) * a.row_bytes;\n"
"        if (a.type == 142u) {\n"
"            const uint byte = 2u + lane / 4u, shift = (lane % 4u) * 2u;\n"
"            for (uint block = 0; block < a.cols / 128u; ++block) {\n"
"                device const uchar *p = wr + block * 34u;\n"
"                const float scale = float(*(device const half *)p);\n"
"                const uint4 packed = uint4(p[byte], p[byte + 8u], p[byte + 16u], p[byte + 24u]) >> shift;\n"
"                const float4 q = float4(int4(packed & 3u) - 1);\n"
"                const uint k = block * 128u + lane;\n"
"#pragma unroll\n"
"                for (uint t = 0; t < 4u; ++t) {\n"
"                    if (first + t >= a.n) continue;\n"
"                    device const float *xt = x + ulong(first + t) * a.cols;\n"
"                    sums[t] += (scale * q.x) * xt[k];\n"
"                    sums[t] += (scale * q.y) * xt[k + 32u];\n"
"                    sums[t] += (scale * q.z) * xt[k + 64u];\n"
"                    sums[t] += (scale * q.w) * xt[k + 96u];\n"
"                }\n"
"            }\n"
"        } else if (a.type == 30u) {\n"
"            device const ushort *wb = (device const ushort *)wr;\n"
"            for (uint k = lane; k < a.cols; k += 32u) {\n"
"                const float weight = as_type<float>(uint(wb[k]) << 16);\n"
"#pragma unroll\n"
"                for (uint t = 0; t < 4u; ++t)\n"
"                    if (first + t < a.n) sums[t] += weight * x[ulong(first + t) * a.cols + k];\n"
"            }\n"
"        } else {\n"
"            for (uint k = lane; k < a.cols; k += 32u) {\n"
"                const float weight = ptq_reference_bs_weight(wr, k, a.type);\n"
"#pragma unroll\n"
"                for (uint t = 0; t < 4u; ++t)\n"
"                    if (first + t < a.n) sums[t] += weight * x[ulong(first + t) * a.cols + k];\n"
"            }\n"
"        }\n"
"    }\n"
"#pragma unroll\n"
"    for (uint t = 0; t < 4u; ++t) {\n"
"        const float sum = simd_sum(sums[t]);\n"
"        if (!lane && row < a.rows && first + t < a.n)\n"
"            out[ulong(first + t) * a.rows + row] = sum;\n"
"    }\n"
"}\n";

/* The activation/store boundary is also frozen from bfb488f. */
static NSString *frozen_element =
@"kernel void ptq_reference_bonsai_element(constant BonsaiArgs &a [[buffer(0)]],\n"
"                           device const float *x [[buffer(1)]], device const float *y [[buffer(2)]],\n"
"                           device float *out [[buffer(3)]], uint i [[thread_position_in_grid]]) {\n"
"    if (i >= a.n) return;\n"
"    if (a.mode == 0) out[i] = x[i] + y[i];\n"
"    else if (a.mode == 1) out[i] = (x[i] / (1.0f + exp(-x[i]))) * y[i];\n"
"    else {\n"
"        ulong gi = ulong(i / a.dim) * (2ul * a.dim) + a.dim + i % a.dim;\n"
"        out[i] = x[i] / (1.0f + exp(-y[gi]));\n"
"    }\n"
"}\n";

typedef struct { uint32_t rows,tokens; bool serial; } Dispatch;
static void need(bool ok,const char *message) {
    if (!ok) { fprintf(stderr,"FAIL PTQ: %s\n",message); exit(1); }
}
static void *data(id<MTLBuffer>b) { return (uint8_t *)b.contents+pad; }
static id<MTLBuffer> buffer(id<MTLDevice>dev,size_t bytes) {
    id<MTLBuffer>b=[dev newBufferWithLength:bytes+2*pad options:MTLResourceStorageModeShared];
    need(b!=nil,"guarded allocation"); memset(b.contents,0xcd,b.length); return b;
}
static void guards(id<MTLBuffer>b) {
    const uint8_t *p=b.contents;
    for (NSUInteger i=0;i<pad;i++)
        need(p[i]==0xcd && p[b.length-1-i]==0xcd,"buffer guard corruption");
}
static uint32_t mix(uint32_t x) {
    x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; return x^(x>>16);
}
static void fill(id<MTLBuffer>w,id<MTLBuffer>x,Args a,unsigned pattern,bool basis) {
    static const uint16_t scales[]={0,0x8000,1,0x03ff,0x0400,0x2400,0xa800,0x3555,0x3c00,0xbc00,0x7bff};
    uint8_t *wb=data(w);
    for (uint32_t r=0;r<a.rows;r++) for (uint32_t b=0;b<a.cols/128;b++) {
        uint8_t *p=wb+(size_t)r*a.row_bytes+b*28u;
        for (uint32_t j=0;j<26;j++)
            p[j]=basis?(uint8_t)(r+j*73u):(uint8_t)mix(r*131u+b*97u+j*17u);
        uint16_t scale=basis?0x3c00:scales[(r+b)%11];
        p[26]=(uint8_t)scale; p[27]=(uint8_t)(scale>>8);
    }
    float *v=data(x);
    for (uint32_t t=0;t<a.n;t++) for (uint32_t k=0;k<a.cols;k++) {
        const uint32_t h=mix(k+t*0x10001u+0x42u);
        const int q=(int)(h&1023u)-511;
        v[(size_t)t*a.cols+k]=basis?(t==k?1.0f:0.0f):
            pattern==2?(k&1u?-0.0f:0.0f):
            ldexpf((float)q,pattern==1?(int)((h>>10)%19u)-14:-12);
    }
}
static uint32_t groups(uint32_t count,uint32_t tile) {
    // A deliberately empty shape still launches one guarded group in this
    // kernel oracle, so zero rows/tokens must leave the output untouched.
    return count?(count+tile-1)/tile:1;
}
static void encode(id<MTLComputeCommandEncoder>enc,id<MTLComputePipelineState>pipeline,
                   Dispatch d,Args a,id<MTLBuffer>w,id<MTLBuffer>x,id<MTLBuffer>y) {
    [enc setComputePipelineState:pipeline]; [enc setBytes:&a length:sizeof(a) atIndex:0];
    [enc setBuffer:w offset:pad atIndex:1];
    const uint32_t dispatches=d.serial?a.n:1;
    for (uint32_t t=0;t<dispatches;t++) {
        [enc setBuffer:x offset:pad+(d.serial?(NSUInteger)t*a.cols*4u:0) atIndex:2];
        [enc setBuffer:y offset:pad+(d.serial?(NSUInteger)t*a.rows*4u:0) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(groups(a.rows,d.rows),
            d.serial?1:groups(a.n,d.tokens),1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    }
}
static double complete(id<MTLCommandBuffer>cb,id<MTLComputeCommandEncoder>enc,unsigned repeats) {
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status!=MTLCommandBufferStatusCompleted)
        fprintf(stderr,"%s\n",cb.error.localizedDescription.UTF8String);
    need(cb.status==MTLCommandBufferStatusCompleted,"GPU completion");
    return (cb.GPUEndTime-cb.GPUStartTime)*1e6/repeats;
}
static double run(id<MTLCommandQueue>queue,id<MTLComputePipelineState>pipeline,
                  Dispatch d,Args a,id<MTLBuffer>w,id<MTLBuffer>x,id<MTLBuffer>y,
                  uint32_t repeats) {
    id<MTLCommandBuffer>cb=[queue commandBuffer];
    id<MTLComputeCommandEncoder>enc=[cb computeCommandEncoder];
    need(cb && enc,"command encoding");
    for (uint32_t r=0;r<repeats;r++) encode(enc,pipeline,d,a,w,x,y);
    return complete(cb,enc,repeats);
}
static void same(const float *ref,const float *out,Args a,const char *label) {
    for (size_t i=0;i<(size_t)a.rows*a.n;i++) {
        if (!isfinite(ref[i]) || !isfinite(out[i]) || memcmp(ref+i,out+i,4)) {
            uint32_t rb,yb; memcpy(&rb,ref+i,4); memcpy(&yb,out+i,4);
            fprintf(stderr,"%s M%u K%u T%u row%zu token%zu got %.9g [%08x] expected %.9g [%08x]\n",
                label,a.rows,a.cols,a.n,i%a.rows,i/a.rows,out[i],yb,ref[i],rb);
            need(false,"bitwise output mismatch");
        }
    }
}
static void basis_check(const float *out,Args a) {
    /* Enumerate storage groups in the inverse direction of the shader's
     * element-to-byte mapping. Every byte position sees all 256 raw codes,
     * including noncanonical 243..255, and every input position is isolated. */
    const unsigned lanes[]={16,8,2},digits[]={5,5,4},byte_start[]={0,16,24},start[]={0,80,120};
    for (uint32_t r=0;r<a.rows;r++) for (unsigned group=0;group<3;group++)
        for (unsigned lane=0;lane<lanes[group];lane++) {
            uint8_t code=(uint8_t)(r+(byte_start[group]+lane)*73u);
            for (unsigned digit=0;digit<digits[group];digit++) {
                const unsigned k=start[group]+digit*lanes[group]+lane;
                const float expected=(float)((int)((unsigned)code*3u/256u)-1);
                const float actual=out[(size_t)k*a.rows+r];
                if (memcmp(&actual,&expected,4)) {
                    fprintf(stderr,"basis raw-row%u storage-byte%u digit%u element%u got %.9g expected %.9g\n",
                        r,byte_start[group]+lane,digit,k,actual,expected);
                    need(false,"independent exhaustive PTQ basis mapping");
                }
                code=(uint8_t)(code*3u); /* Wrap BEFORE extracting next trit. */
            }
        }
}
static int compare_double(const void*a,const void*b) {
    const double x=*(const double*)a,y=*(const double*)b; return (x>y)-(x<y);
}
static void benchmark(id<MTLCommandQueue>queue,NSArray *pipelines,const Dispatch *dispatch,
                      unsigned baseline,unsigned candidate,Args a,id<MTLBuffer>w,id<MTLBuffer>x,id<MTLBuffer>out) {
    const unsigned arms[]={baseline,candidate};
    const uint32_t repeats=a.n==1?12:a.n<16?3:1;
    double times[2][8];
    // Eight samples per arm in balanced ABBA order; all buffers are resident.
    for (unsigned trial=0;trial<8;trial++) for (unsigned j=0;j<2;j++) {
        const unsigned arm=trial&1u?1u-j:j,which=arms[arm];
        times[arm][trial]=run(queue,pipelines[which],dispatch[which],a,w,x,out,repeats);
        need(times[arm][trial]>0,"GPU timestamps unavailable");
    }
    for (unsigned arm=0;arm<2;arm++) qsort(times[arm],8,sizeof(double),compare_double);
    const double ref=(times[0][3]+times[0][4])*.5,test=(times[1][3]+times[1][4])*.5;
    id<MTLComputePipelineState>pipeline=pipelines[candidate],reference=pipelines[baseline];
    printf("BENCH %s M=%u K=%u T=%u baseline=%s baseline_us=%.3f production_us=%.3f speedup=%.4f "
           "baseline_range=%.3f..%.3f production_range=%.3f..%.3f exact=1\n",
        pipeline.label.UTF8String,a.rows,a.cols,a.n,reference.label.UTF8String,ref,test,ref/test,
        times[0][0],times[0][7],times[1][0],times[1][7]);
    fflush(stdout);
}
static void shape(id<MTLDevice>dev,id<MTLCommandQueue>queue,NSArray *pipelines,
                  const Dispatch *dispatch,uint32_t rows,uint32_t cols,uint32_t tokens,
                  unsigned pattern,bool basis,bool timing) {
    Args a={.rows=rows,.cols=cols,.n=tokens,.type=143,.row_bytes=cols/128*28};
    const size_t wbytes=(size_t)rows*a.row_bytes,xbytes=(size_t)tokens*cols*4,ybytes=(size_t)tokens*rows*4;
    id<MTLBuffer>w=buffer(dev,wbytes),x=buffer(dev,xbytes),ref=buffer(dev,ybytes),out=buffer(dev,ybytes);
    fill(w,x,a,pattern,basis);
    NSData *saved_w=[NSData dataWithBytes:data(w) length:wbytes],*saved_x=[NSData dataWithBytes:data(x) length:xbytes];
    run(queue,pipelines[0],dispatch[0],a,w,x,ref,1);
    if (basis) basis_check(data(ref),a);
    for (NSUInteger variant=1;variant<pipelines.count;variant++) {
        memset(data(out),0xa5,ybytes);
        run(queue,pipelines[variant],dispatch[variant],a,w,x,out,1);
        id<MTLComputePipelineState>pipeline=pipelines[variant];
        same(data(ref),data(out),a,pipeline.label.UTF8String);
        if (basis) basis_check(data(out),a);
        guards(out);
    }
    if (timing) {
        benchmark(queue,pipelines,dispatch,a.n==1?0:1,a.n==1?2:3,a,w,x,out);
        if (a.n>1) for (unsigned v=4;v<pipelines.count;v++)
            benchmark(queue,pipelines,dispatch,1,v,a,w,x,out);
        if (a.n>1) benchmark(queue,pipelines,dispatch,3,4,a,w,x,out);
        if (a.n>=8) benchmark(queue,pipelines,dispatch,4,5,a,w,x,out);
        same(data(ref),data(out),a,"post-benchmark");
    } else {
        printf("PASS PTQ M=%u K=%u T=%u pattern=%u%s exact=1\n",rows,cols,tokens,pattern,basis?" exhaustive-basis":"");
        fflush(stdout);
    }
    need(!wbytes || !memcmp(saved_w.bytes,data(w),wbytes),"weights modified");
    need(!xbytes || !memcmp(saved_x.bytes,data(x),xbytes),"inputs modified");
    guards(w); guards(x); guards(ref); guards(out);
}

static void gate_weights(id<MTLBuffer>w,Args a,unsigned seed,bool gate) {
    const uint16_t scales[]={0x0001,0x03ff,0x0400,0x2400,0xac00,0x3555,0x3800,0xbc00};
    for (uint32_t r=0;r<a.rows;r++) for (uint32_t block=0;block<a.cols/128;block++) {
        uint8_t *p=(uint8_t *)data(w)+(size_t)r*a.row_bytes+block*28u;
        for (unsigned j=0;j<26;j++) p[j]=(uint8_t)mix(r*1777u+block*91u+j*73u+seed);
        uint16_t scale=scales[mix(r*37u+block+seed)%8u];
        // Different zero rows and independent scales detect gate/up aliasing:
        // 0x80 encodes five zero coefficients, while up uses negative-zero scale.
        if (gate && r==0) { memset(p,0x80,26); scale=0x3c00; }
        if (!gate && r+1==a.rows) scale=0x8000;
        p[26]=(uint8_t)scale; p[27]=(uint8_t)(scale>>8);
    }
}
static double gate_run(id<MTLCommandQueue>queue,NSArray *projections,const Dispatch *dispatch,
                       NSArray *fusion,Args a,NSArray<id<MTLBuffer>> *b,
                       unsigned variant,unsigned repeats) {
    id<MTLCommandBuffer>cb=[queue commandBuffer];
    id<MTLComputeCommandEncoder>enc=[cb computeCommandEncoder];
    need(cb && enc,"gate/up command encoding");
    for (unsigned repeat=0;repeat<repeats;repeat++) {
        if (variant==0) {
            const unsigned p=a.n==1?0:1;
            encode(enc,projections[p],dispatch[p],a,b[0],b[2],b[3]);
            encode(enc,projections[p],dispatch[p],a,b[1],b[2],b[4]);
            const Args element={.n=a.n*a.rows,.mode=1};
            [enc setComputePipelineState:fusion[0]];
            [enc setBytes:&element length:sizeof(element) atIndex:0];
            for (NSUInteger i=0;i<3;i++) [enc setBuffer:b[i+3] offset:pad atIndex:i+1];
            [enc dispatchThreadgroups:MTLSizeMake(groups(element.n,256),1,1)
                threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        } else {
            [enc setComputePipelineState:fusion[variant]];
            [enc setBytes:&a length:sizeof(a) atIndex:0];
            [enc setBuffer:b[0] offset:pad atIndex:1];
            [enc setBuffer:b[1] offset:pad atIndex:2];
            const bool serial=variant==1;
            const uint32_t token_tile=variant==3?8:4;
            for (uint32_t t=0;t<(serial?a.n:1);t++) {
                [enc setBuffer:b[2] offset:pad+(serial?(NSUInteger)t*a.cols*4u:0) atIndex:3];
                [enc setBuffer:b[5] offset:pad+(serial?(NSUInteger)t*a.rows*4u:0) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(groups(a.rows,serial?8:4),
                    serial?1:groups(a.n,token_tile),1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            }
        }
    }
    return complete(cb,enc,repeats);
}
static void gate_shape(id<MTLDevice>dev,id<MTLCommandQueue>queue,NSArray *projections,
                       const Dispatch *dispatch,NSArray *fusion,uint32_t rows,uint32_t cols,
                       uint32_t tokens,unsigned pattern,bool timing) {
    const Args a={.rows=rows,.cols=cols,.n=tokens,.type=143,.row_bytes=cols/128*28};
    const size_t wbytes=(size_t)rows*a.row_bytes,xbytes=(size_t)tokens*cols*4,ybytes=(size_t)tokens*rows*4;
    NSArray<id<MTLBuffer>> *b=@[buffer(dev,wbytes),buffer(dev,wbytes),buffer(dev,xbytes),
        buffer(dev,ybytes),buffer(dev,ybytes),buffer(dev,ybytes)];
    const bool basis=pattern==3 || pattern==4;
    fill(b[0],b[2],a,pattern,basis);
    if (basis) {
        need(rows==257 && cols==128 && tokens==128,"fused exhaustive basis shape");
        // Every byte position sees all 256 raw values. One projection varies;
        // raw255/scale1 makes the other exactly +1 at every one-hot input.
        // Swap gate/up to exercise both the nonlinear and multiplicative side.
        for (uint32_t r=0;r<rows;r++) for (unsigned side=0;side<2;side++) {
            uint8_t *p=(uint8_t *)data(b[side])+(size_t)r*a.row_bytes;
            for (unsigned j=0;j<26;j++)
                p[j]=side==(pattern==3?0u:1u)?(uint8_t)(r+j*73u):255;
            p[26]=0; p[27]=0x3c;
        }
    } else {
        gate_weights(b[0],a,19,true); gate_weights(b[1],a,113,false);
        if (tokens>2) memset((float *)data(b[2])+cols,0,cols*4u);
    }
    NSArray<NSData *> *saved=@[[NSData dataWithBytes:data(b[0]) length:wbytes],
        [NSData dataWithBytes:data(b[1]) length:wbytes],[NSData dataWithBytes:data(b[2]) length:xbytes]];
    gate_run(queue,projections,dispatch,fusion,a,b,0,1);
    NSData *reference=[NSData dataWithBytes:data(b[5]) length:ybytes];
    for (unsigned variant=1;variant<fusion.count;variant++) {
        memset(data(b[5]),0xa5,ybytes);
        gate_run(queue,projections,dispatch,fusion,a,b,variant,1);
        id<MTLComputePipelineState>p=fusion[variant];
        same(reference.bytes,data(b[5]),a,p.label.UTF8String);
        for (id<MTLBuffer>v in b) guards(v);
    }
    if (timing) {
      // Keep the frozen pair+element baseline and isolate the 4-to-8 change.
      for (unsigned pairing=0;pairing<(tokens>=8?2u:1u);pairing++) {
        const unsigned baseline=pairing?2:0,candidate=tokens==1?1:tokens>=8?3:2;
        const unsigned repeats=tokens==1?12:tokens<16?3:1;
        double times[2][8];
        for (unsigned trial=0;trial<8;trial++) for (unsigned j=0;j<2;j++) {
            const unsigned arm=trial&1u?1u-j:j;
            times[arm][trial]=gate_run(queue,projections,dispatch,fusion,a,b,arm?candidate:baseline,repeats);
            need(times[arm][trial]>0,"gate/up GPU timestamps unavailable");
        }
        for (unsigned arm=0;arm<2;arm++) qsort(times[arm],8,sizeof(double),compare_double);
        const double base=(times[0][3]+times[0][4])*.5,fused=(times[1][3]+times[1][4])*.5;
        id<MTLComputePipelineState>p=fusion[candidate];
        printf("BENCH %s M=%u K=%u T=%u baseline=%s baseline_us=%.3f fused_us=%.3f speedup=%.4f "
               "baseline_range=%.3f..%.3f fused_range=%.3f..%.3f exact=1\n",
            p.label.UTF8String,rows,cols,tokens,pairing?"bonsai_ptq_gate_up_batch":"frozen_pair_element",base,fused,base/fused,
            times[0][0],times[0][7],times[1][0],times[1][7]);
        same(reference.bytes,data(b[5]),a,"gate/up post-benchmark");
      }
    } else printf("PASS PTQ gate/up M=%u K=%u T=%u pattern=%u%s separate/fused exact=1\n",
        rows,cols,tokens,pattern,basis?(pattern==3?" exhaustive-gate-basis":" exhaustive-up-basis"):"");
    fflush(stdout);
    for (unsigned i=0;i<3;i++) need(!saved[i].length ||
        !memcmp(saved[i].bytes,data(b[i]),saved[i].length),"gate/up input modified");
    for (id<MTLBuffer>v in b) guards(v);
}
static id<MTLComputePipelineState> pipeline(id<MTLDevice>dev,id<MTLLibrary>library,NSString *name) {
    id<MTLFunction>fn=[library newFunctionWithName:name];
    if (!fn) fprintf(stderr,"Missing kernel: %s\n",name.UTF8String);
    need(fn!=nil,"kernel entry point");
    MTLComputePipelineDescriptor *descriptor=[MTLComputePipelineDescriptor new];
    descriptor.computeFunction=fn; descriptor.label=name;
    NSError *error=nil;
    id<MTLComputePipelineState>p=[dev newComputePipelineStateWithDescriptor:descriptor
        options:MTLPipelineOptionNone reflection:NULL error:&error];
    if (!p) fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);
    need(p && p.threadExecutionWidth==32 && p.maxTotalThreadsPerThreadgroup>=128,"pipeline geometry");
    return p;
}

int main(int argc,char **argv) { @autoreleasepool {
    bool timing=false,fused_timing=false;
    NSString *path=@"metal/bonsai.metal",*mv=@"bonsai_mv",*tiled=nil;
    Dispatch dispatch[7]={{4,1,true},{4,4,false},{4,1,true},{4,4,false},{8,4,false},{8,8,false},{0,0,false}};
    for (int i=1;i<argc;i++) {
        if (!strcmp(argv[i],"--bench")) timing=true;
        else if (!strcmp(argv[i],"--bench-fused")) fused_timing=true;
        else if (!strcmp(argv[i],"--shader") && i+1<argc) path=[NSString stringWithUTF8String:argv[++i]];
        else if (!strcmp(argv[i],"--mv") && i+2<argc) {
            mv=[NSString stringWithUTF8String:argv[++i]];
            dispatch[2].rows=(uint32_t)strtoul(argv[++i],NULL,10);
            need(dispatch[2].rows>0 && dispatch[2].rows<=1024,"prototype matvec row tile");
        }
        else if (!strcmp(argv[i],"--tiled") && i+3<argc) {
            // Optional prototype: same buffers/Args, static threadgroup memory,
            // 128 threads, caller supplies the output tile. Exactness remains required.
            tiled=[NSString stringWithUTF8String:argv[++i]];
            dispatch[6].rows=(uint32_t)strtoul(argv[++i],NULL,10);
            dispatch[6].tokens=(uint32_t)strtoul(argv[++i],NULL,10);
            need(dispatch[6].rows>0 && dispatch[6].rows<=1024 &&
                 dispatch[6].tokens>0 && dispatch[6].tokens<=1024,"prototype tile bounds");
        } else need(false,"usage: test_bonsai_ptq [--bench] [--bench-fused] [--shader PATH] [--mv KERNEL ROW_TILE] [--tiled KERNEL ROW_TILE TOKEN_TILE]");
    }
    id<MTLDevice>dev=MTLCreateSystemDefaultDevice(); need(dev!=nil,"Metal device");
    id<MTLCommandQueue>queue=[dev newCommandQueue]; need(queue!=nil,"command queue");
    NSError *error=nil;
    NSString *source=[NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
    need(source!=nil,"production shader file");
    source=[source stringByAppendingFormat:@"\n%@\n%@",frozen_source,frozen_element];
    MTLCompileOptions *options=[MTLCompileOptions new];
    // Match ds4_bonsai_metal.m's production math mode.
    if (@available(macOS 15.0,*)) options.mathMode=MTLMathModeSafe;
    else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled=NO;
#pragma clang diagnostic pop
    }
    id<MTLLibrary>library=[dev newLibraryWithSource:source options:options error:&error];
    if (!library) fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);
    need(library!=nil,"Metal library");
    NSMutableArray *names=[@[@"ptq_reference_bonsai_mv",@"ptq_reference_bonsai_mm",mv,@"bonsai_mm",@"bonsai_ptq_mm",@"bonsai_ptq_mm_8"] mutableCopy];
    if (tiled) [names addObject:tiled];
    NSMutableArray *pipelines=[NSMutableArray array];
    for (NSString *name in names) [pipelines addObject:pipeline(dev,library,name)];
    NSArray *fusion=@[pipeline(dev,library,@"ptq_reference_bonsai_element"),
        pipeline(dev,library,@"bonsai_ptq_gate_up"),pipeline(dev,library,@"bonsai_ptq_gate_up_batch"),
        pipeline(dev,library,@"bonsai_ptq_gate_up_batch_8")];
    id<MTLComputePipelineState>element=fusion[0];
    need(element.maxTotalThreadsPerThreadgroup>=256,"element pipeline geometry");
    fprintf(stderr,"PTQ oracle: %s; frozen bfb488f; safe math; %s.\n",dev.name.UTF8String,
        timing||fused_timing?"balanced ABBA GPU medians (no validation)":"bitwise comparisons and memory guards");
    @autoreleasepool { shape(dev,queue,pipelines,dispatch,257,128,128,0,true,false); }
    for (unsigned phase=3;phase<=4;phase++) { @autoreleasepool {
        gate_shape(dev,queue,pipelines,dispatch,fusion,257,128,128,phase,false);
    }}
    const uint32_t counts[]={1,2,3,4,5,7,8,9,15,16,17,31,32,33,63,64,65,127,128,129};
    const uint32_t rows[]={1,2,3,4,5,7,8,9,15,16,17,31,32,33,63,64,65};
    for (unsigned i=0;i<sizeof(counts)/sizeof(counts[0]);i++) { @autoreleasepool {
        const uint32_t r=rows[i%(sizeof(rows)/sizeof(rows[0]))],k=128u*(1u+i%5u);
        shape(dev,queue,pipelines,dispatch,r,k,counts[i],i%3u,false,false);
        gate_shape(dev,queue,pipelines,dispatch,fusion,r,k,counts[i],i%3u,false);
    }}
    const uint32_t big[][2]={{16384,3072},{3072,16384}},big_counts[]={1,5,33};
    for (unsigned i=0;i<2;i++) for (unsigned t=0;t<3;t++) { @autoreleasepool {
        shape(dev,queue,pipelines,dispatch,big[i][0],big[i][1],big_counts[t],t%2u,false,timing);
        gate_shape(dev,queue,pipelines,dispatch,fusion,big[i][0],big[i][1],big_counts[t],t%2u,fused_timing);
    }}
    const uint32_t checkpoint[][2]={{17408,5120},{5120,17408}},checkpoint_counts[]={1,5,8,17};
    for (unsigned i=0;i<2;i++) for (unsigned t=0;t<4;t++) { @autoreleasepool {
        shape(dev,queue,pipelines,dispatch,checkpoint[i][0],checkpoint[i][1],checkpoint_counts[t],0,false,timing);
        gate_shape(dev,queue,pipelines,dispatch,fusion,checkpoint[i][0],checkpoint[i][1],checkpoint_counts[t],0,fused_timing);
    }}
    // Large-K row/count tails retain complete 128-element PTQ blocks.
    @autoreleasepool { shape(dev,queue,pipelines,dispatch,65,3072,17,1,false,false); }
    @autoreleasepool { shape(dev,queue,pipelines,dispatch,7,16384,3,1,false,false); }
    const uint32_t empty[][2]={{0,0},{0,1},{0,5},{7,0}};
    for (unsigned i=0;i<4;i++) { @autoreleasepool {
        shape(dev,queue,pipelines,dispatch,empty[i][0],128,empty[i][1],0,false,false);
        gate_shape(dev,queue,pipelines,dispatch,fusion,empty[i][0],128,empty[i][1],0,false);
    }}
    puts("PASS PTQ1_0: exhaustive byte/basis mapping, frozen MV/MM/SwiGLU bit parity, all production fusions, tails, readonly inputs and guards");
    return 0;
} }
