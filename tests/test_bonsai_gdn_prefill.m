/* Causal Bonsai prefill/decode GDN against frozen single-token kernels; no model.
 * make tests/test_bonsai_gdn_prefill
 * MTL_SHADER_VALIDATION=1 ./tests/test_bonsai_gdn_prefill
 * Optional --bench measures only synthetic GPU work, with validation disabled.
 * Generic and D128 register scans must match every intermediate, state/history,
 * continued chunk and prefill-to-decode transition bitwise. Production source
 * is embedded by the normal Makefile dependency.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../metal/bonsai.metal.inc"

// Frozen single-token kernels; never regenerate these from production.
static NSString *frozen_source=
@"static inline float frozen_bs_scalar(device const uchar *w, uint i, uint type) {\n"
"    if (type == 0u) return ((device const float *)w)[i];\n"
"    if (type == 1u) return float(((device const half *)w)[i]);\n"
"    return as_type<float>(uint(((device const ushort *)w)[i]) << 16); // BF16\n"
"}\n"
"\n"
"kernel void frozen_bonsai_conv(constant BonsaiArgs &a [[buffer(0)]],\n"
"                        device const float *x [[buffer(1)]], device const uchar *w [[buffer(2)]],\n"
"                        device float *history [[buffer(3)]], device float *out [[buffer(4)]],\n"
"                        uint c [[thread_position_in_grid]]) {\n"
"    if (c >= a.n) return;\n"
"    float sum = 0;\n"
"    for (uint j = 0; j + 1 < a.width; ++j) {\n"
"        sum += history[ulong(c) * (a.width - 1u) + j] * frozen_bs_scalar(w, c * a.width + j, a.type);\n"
"    }\n"
"    sum += x[c] * frozen_bs_scalar(w, c * a.width + a.width - 1u, a.type);\n"
"    for (uint j = 0; j + 2 < a.width; ++j)\n"
"        history[ulong(c) * (a.width - 1u) + j] = history[ulong(c) * (a.width - 1u) + j + 1u];\n"
"    if (a.width > 1) history[ulong(c) * (a.width - 1u) + a.width - 2u] = x[c];\n"
"    out[c] = sum / (1.0f + exp(-sum));\n"
"}\n"
"\n"
"kernel void frozen_bonsai_norm(constant BonsaiArgs &a [[buffer(0)]],\n"
"                        device const float *x [[buffer(1)]], device const uchar *w [[buffer(2)]],\n"
"                        device float *out [[buffer(3)]], uint row [[threadgroup_position_in_grid]],\n"
"                        uint tid [[thread_index_in_threadgroup]]) {\n"
"    threadgroup float tmp[256];\n"
"    float sum = 0;\n"
"    for (uint i = tid; i < a.cols; i += 256) { float v = x[ulong(row) * a.width + i]; sum += v * v; }\n"
"    tmp[tid] = sum;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    // Reproduce the original 256-thread tree, rather than replacing it\n"
"    // with simd_sum (which may use a different association). SIMD group 0\n"
"    // performs stages 128, 64 and 32 from the same eight partials per lane;\n"
"    // shuffle-down then reproduces stages 16, 8, 4, 2 and 1 exactly.\n"
"    if (tid < 32) {\n"
"        const float p0 = tmp[tid] + tmp[tid + 128u];\n"
"        const float p1 = tmp[tid + 64u] + tmp[tid + 192u];\n"
"        const float p2 = tmp[tid + 32u] + tmp[tid + 160u];\n"
"        const float p3 = tmp[tid + 96u] + tmp[tid + 224u];\n"
"        float total = (p0 + p1) + (p2 + p3);\n"
"        total += simd_shuffle_down(total, 16);\n"
"        total += simd_shuffle_down(total, 8);\n"
"        total += simd_shuffle_down(total, 4);\n"
"        total += simd_shuffle_down(total, 2);\n"
"        total += simd_shuffle_down(total, 1);\n"
"        if (!tid) tmp[0] = total;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    const float scale = a.mode ? 1.0f / max(sqrt(tmp[0]), a.eps) : rsqrt(tmp[0] / float(a.cols) + a.eps);\n"
"    for (uint i = tid; i < a.cols; i += 256)\n"
"        out[ulong(row) * a.cols + i] = x[ulong(row) * a.width + i] * scale * (a.mode ? 1.0f : frozen_bs_scalar(w, i, a.type));\n"
"}\n"
"\n"
"kernel void frozen_bonsai_gdn(constant BonsaiArgs &a [[buffer(0)]],\n"
"                       device const float *qkv [[buffer(1)]], device const float *alpha [[buffer(2)]],\n"
"                       device const float *beta [[buffer(3)]], device const uchar *A [[buffer(4)]],\n"
"                       device const uchar *dt [[buffer(5)]], device float *state [[buffer(6)]],\n"
"                       device float *out [[buffer(7)]], uint h [[threadgroup_position_in_grid]],\n"
"                       uint dv [[thread_index_in_threadgroup]]) {\n"
"    if (dv >= a.dim) return;\n"
"    const uint kh = h % a.kvheads;\n"
"    device const float *q = qkv + kh * a.dim, *k = qkv + (a.kvheads + kh) * a.dim;\n"
"    const float v = qkv[2u * a.kvheads * a.dim + h * a.dim + dv];\n"
"    const float ab = alpha[h] + frozen_bs_scalar(dt, h, a.mode);\n"
"    // Metal lacks log1p. Compensate rounding in 1+e; for very small e\n"
"    // the exact limiting value avoids cancellation to zero.\n"
"    const float e = exp(-abs(ab)), u = 1.0f + e;\n"
"    const float softplus = max(ab, 0.0f) + (u == 1.0f ? e : log(u) * (e / (u - 1.0f)));\n"
"    const float decay = exp(frozen_bs_scalar(A, h, a.type) * softplus);\n"
"    const float b = 1.0f / (1.0f + exp(-beta[h]));\n"
"    // Private Metal state is [head][key dimension][value dimension], so\n"
"    // neighboring value lanes access neighboring floats throughout the scan.\n"
"    float prediction = 0;\n"
"    for (uint dk = 0; dk < a.dim; ++dk) {\n"
"        const ulong idx = (ulong(h) * a.dim + dk) * a.dim + dv;\n"
"        float s = state[idx] * decay;\n"
"        state[idx] = s;\n"
"        prediction += s * k[dk];\n"
"    }\n"
"    const float delta = (v - prediction) * b;\n"
"    float result = 0;\n"
"    for (uint dk = 0; dk < a.dim; ++dk) {\n"
"        const ulong idx = (ulong(h) * a.dim + dk) * a.dim + dv;\n"
"        float s = state[idx] + delta * k[dk];\n"
"        state[idx] = s;\n"
"        result += s * q[dk];\n"
"    }\n"
"    out[h * a.dim + dv] = result * rsqrt(float(a.dim));\n"
"}\n";

typedef struct {
    uint32_t n,rows,cols,type,row_bytes,pos,heads,kvheads,dim,rot,width,mode,groups;
    float eps,base;
} Args;
_Static_assert(sizeof(Args)==60,"Metal args layout");
static const NSUInteger guard=64;
enum { Frozen, GenericBatch, RegisterBatch, RegisterDecode, VariantCount };
static void need(bool ok,const char *msg) {
    if (!ok) { fprintf(stderr,"FAIL %s\n",msg); exit(1); }
}
static id<MTLBuffer> alloc_buffer(id<MTLDevice>dev,size_t count) {
    id<MTLBuffer>b=[dev newBufferWithLength:count*4+2*guard options:MTLResourceStorageModeShared];
    need(b!=nil,"allocate"); memset(b.contents,0xa5,b.length); return b;
}
static void *data(id<MTLBuffer>b) { return (char*)b.contents+guard; }
static void check_guard(id<MTLBuffer>b) {
    const unsigned char*p=b.contents;
    for (NSUInteger i=0;i<guard;++i) need(p[i]==0xa5 && p[b.length-i-1]==0xa5,"buffer canary");
}
static void same(id<MTLBuffer>a,id<MTLBuffer>b,const char*label) {
    need(a.length==b.length,"comparison size");
    const float*x=data(a),*y=data(b); const size_t n=(a.length-2*guard)/4;
    for(size_t i=0;i<n;++i) if (!isfinite(x[i]) || !isfinite(y[i]) || memcmp(x+i,y+i,4)) {
        fprintf(stderr,"%s i=%zu reference=%.9g candidate=%.9g\n",label,i,x[i],y[i]); need(false,"bitwise parity");
    }
}
static void scalar_store(id<MTLBuffer>b,uint32_t type,uint32_t i,float value) {
    if(type==0) ((float*)data(b))[i]=value;
    else if(type==1) ((_Float16*)data(b))[i]=(_Float16)value;
    else { uint32_t bits; memcpy(&bits,&value,4); ((uint16_t*)data(b))[i]=(uint16_t)(bits>>16); }
}

@interface Fixture:NSObject {
@public
    uint32_t T,D,H,K,W,C,type,atype,dtype;
    id<MTLBuffer>x,w,alpha,beta,A,dt,initialHistory,initialState;
    id<MTLBuffer>history[VariantCount],state[VariantCount],conv[VariantCount],out[VariantCount];
}
@end
@implementation Fixture
@end

static Fixture *fixture(id<MTLDevice>dev,uint32_t T,uint32_t D,uint32_t K,uint32_t H,
                       uint32_t W,uint32_t type,uint32_t atype,uint32_t dtype) {
    Fixture*f=[Fixture new]; f->T=T;f->D=D;f->K=K;f->H=H;f->W=W;f->C=(2*K+H)*D;
    f->type=type;f->atype=atype;f->dtype=dtype;
    f->x=alloc_buffer(dev,(size_t)T*f->C); f->w=alloc_buffer(dev,(size_t)f->C*W);
    f->alpha=alloc_buffer(dev,(size_t)T*H); f->beta=alloc_buffer(dev,(size_t)T*H);
    f->A=alloc_buffer(dev,H); f->dt=alloc_buffer(dev,H);
    f->initialHistory=alloc_buffer(dev,(size_t)f->C*(W-1));
    f->initialState=alloc_buffer(dev,(size_t)H*D*D);
    for(uint32_t t=0;t<T;++t) for(uint32_t c=0;c<f->C;++c)
        ((float*)data(f->x))[(size_t)t*f->C+c]=(c/D==0?0:sinf((float)(c+31*t)*.051f)*.7f);
    for(uint32_t c=0;c<f->C;++c) for(uint32_t j=0;j<W;++j)
        scalar_store(f->w,type,c*W+j,cosf((float)(c+7*j)*.061f)*.25f);
    for(uint32_t h=0;h<H;++h) {
        scalar_store(f->A,atype,h,-.4f-.03125f*(h%7));
        scalar_store(f->dt,dtype,h,((int)(h%5)-2)*.125f);
    }
    for(uint32_t t=0;t<T;++t) for(uint32_t h=0;h<H;++h) {
        uint32_t mode=(t+h)%7;
        ((float*)data(f->alpha))[(size_t)t*H+h]=mode==0?-100:mode==1?100:sinf((float)(t+11*h))*.75f;
        ((float*)data(f->beta))[(size_t)t*H+h]=mode==0?-100:mode==1?100:cosf((float)(t+7*h))*.75f;
    }
    for(size_t i=0;i<(size_t)f->C*(W-1);++i)
        ((float*)data(f->initialHistory))[i]=(i/(W-1)/D==0?0:sinf((float)(i%100003)*.021f)*.3f);
    for(size_t i=0;i<(size_t)H*D*D;++i)
        ((float*)data(f->initialState))[i]=sinf((float)(i%100003)*.017f)*.04f;
    for(uint32_t v=0;v<(D==128?VariantCount:2);++v) {
        f->history[v]=alloc_buffer(dev,(size_t)f->C*(W-1));f->state[v]=alloc_buffer(dev,(size_t)H*D*D);
        f->conv[v]=alloc_buffer(dev,(size_t)T*f->C);f->out[v]=alloc_buffer(dev,(size_t)T*H*D);
    }
    return f;
}
static void reset(Fixture*f,uint32_t v) {
    memcpy(data(f->history[v]),data(f->initialHistory),f->initialHistory.length-2*guard);
    memcpy(data(f->state[v]),data(f->initialState),f->initialState.length-2*guard);
    float*c=data(f->conv[v]),*o=data(f->out[v]);
    for(size_t i=0;i<(f->conv[v].length-2*guard)/4;++i)c[i]=NAN;
    for(size_t i=0;i<(f->out[v].length-2*guard)/4;++i)o[i]=NAN;
}
static void dispatch(id<MTLComputeCommandEncoder>enc,NSDictionary*ps,NSString*name,Args a,
                     NSArray<id<MTLBuffer>>*buffers,const NSUInteger*offsets,MTLSize groups,NSUInteger nth) {
    id<MTLComputePipelineState>p=ps[name];
    need(p && p.threadExecutionWidth==32 && p.maxTotalThreadsPerThreadgroup>=nth,"pipeline threads");
    [enc setComputePipelineState:p];[enc setBytes:&a length:sizeof(a) atIndex:0];
    for(NSUInteger i=0;i<buffers.count;++i) [enc setBuffer:buffers[i] offset:guard+(offsets?offsets[i]:0) atIndex:i+1];
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(nth,1,1)];
}
static void encode(id<MTLComputeCommandEncoder>enc,NSDictionary*ps,Fixture*f,uint32_t variant,
                   uint32_t begin,uint32_t count,uint32_t phase,bool decode) {
    const uint32_t C=f->C,D=f->D,H=f->H,K=f->K;
    if(variant!=Frozen && !decode) {
        const NSUInteger co[]={(NSUInteger)begin*C*4,0,0,(NSUInteger)begin*C*4};
        dispatch(enc,ps,@"bonsai_conv_batch",(Args){.n=count,.cols=C,.width=f->W,.type=f->type},
                 @[f->x,f->w,f->history[variant],f->conv[variant]],co,MTLSizeMake((C+255u)/256u,1,1),256);
        if(phase<2)return;
        const NSUInteger no[]={(NSUInteger)begin*C*4};
        dispatch(enc,ps,@"bonsai_l2_batch",(Args){.n=count,.cols=D,.heads=2*K,.width=C,.eps=1e-6f},
                 @[f->conv[variant]],no,MTLSizeMake(2*K,count,1),256);
        if(phase<3)return;
        const NSUInteger so[]={(NSUInteger)begin*C*4,(NSUInteger)begin*H*4,(NSUInteger)begin*H*4,0,0,0,(NSUInteger)begin*H*D*4};
        NSString*scan=variant==GenericBatch?@"bonsai_gdn_batch":@"bonsai_gdn_batch_128";
        dispatch(enc,ps,scan,(Args){.n=count,.dim=D,.heads=H,.kvheads=K,.type=f->atype,.mode=f->dtype},
                 @[f->conv[variant],f->alpha,f->beta,f->A,f->dt,f->state[variant],f->out[variant]],so,MTLSizeMake(H,1,1),D);
    } else for(uint32_t row=begin;row<begin+count;++row) {
        const NSUInteger co[]={(NSUInteger)row*C*4,0,0,(NSUInteger)row*C*4};
        dispatch(enc,ps,@"frozen_bonsai_conv",(Args){.n=C,.width=f->W,.type=f->type},
                 @[f->x,f->w,f->history[variant],f->conv[variant]],co,MTLSizeMake((C+255u)/256u,1,1),256);
        if(phase<2)continue;
        const NSUInteger no[]={(NSUInteger)row*C*4,(NSUInteger)row*C*4,(NSUInteger)row*C*4};
        dispatch(enc,ps,@"frozen_bonsai_norm",(Args){.cols=D,.width=D,.mode=1,.eps=1e-6f},
                 @[f->conv[variant],f->conv[variant],f->conv[variant]],no,MTLSizeMake(2*K,1,1),256);
        if(phase<3)continue;
        const NSUInteger so[]={(NSUInteger)row*C*4,(NSUInteger)row*H*4,(NSUInteger)row*H*4,0,0,0,(NSUInteger)row*H*D*4};
        NSString*scan=variant==Frozen?@"frozen_bonsai_gdn":@"bonsai_gdn_128";
        dispatch(enc,ps,scan,(Args){.dim=D,.heads=H,.kvheads=K,.type=f->atype,.mode=f->dtype},
                 @[f->conv[variant],f->alpha,f->beta,f->A,f->dt,f->state[variant],f->out[variant]],so,MTLSizeMake(H,1,1),D);
    }
}
static double run(id<MTLCommandQueue>q,NSDictionary*ps,Fixture*f,uint32_t variant,uint32_t phase,
                  bool chunks,uint32_t repeats) {
    id<MTLCommandBuffer>cb=[q commandBuffer];id<MTLComputeCommandEncoder>enc=[cb computeCommandEncoder];
    need(cb && enc,"command creation");
    for(uint32_t repeat=0;repeat<repeats;++repeat) {
        if(chunks) {
            const uint32_t sizes[]={1,3,2,7,5,14}; uint32_t begin=0,i=0;
            while(begin<f->T) { uint32_t n=MIN(sizes[i++%6],f->T-begin);encode(enc,ps,f,variant,begin,n,phase,variant==RegisterDecode);begin+=n; }
        } else encode(enc,ps,f,variant,0,f->T,phase,variant==RegisterDecode);
    }
    [enc endEncoding];[cb commit];[cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted)fprintf(stderr,"%s\n",cb.error.localizedDescription.UTF8String);
    need(cb.status==MTLCommandBufferStatusCompleted,"GPU command completion");
    return (cb.GPUEndTime-cb.GPUStartTime)*1e6/repeats;
}
static void same_result(Fixture*f,uint32_t variant,uint32_t phase) {
    same(f->history[Frozen],f->history[variant],"history");
    same(f->conv[Frozen],f->conv[variant],phase==1?"raw conv":"L2 conv");
    same(f->state[Frozen],f->state[variant],"state");
    if(phase==3)same(f->out[Frozen],f->out[variant],"GDN output");
}
// Complete prefill and the following decode in separate command buffers, so
// passing the test requires materializing all state/history at the boundary.
static void check_decode_continuation(id<MTLCommandQueue>q,NSDictionary*ps,Fixture*f) {
    if(f->D!=128 || f->T<2)return;
    reset(f,RegisterBatch);
    for(uint32_t part=0;part<2;++part) {
        id<MTLCommandBuffer>cb=[q commandBuffer];id<MTLComputeCommandEncoder>enc=[cb computeCommandEncoder];
        need(cb && enc,"continuation command creation");
        encode(enc,ps,f,RegisterBatch,part?f->T-1:0,part?1:f->T-1,3,part!=0);
        [enc endEncoding];[cb commit];[cb waitUntilCompleted];
        if(cb.status!=MTLCommandBufferStatusCompleted)fprintf(stderr,"%s\n",cb.error.localizedDescription.UTF8String);
        need(cb.status==MTLCommandBufferStatusCompleted,"continuation GPU completion");
    }
    same_result(f,RegisterBatch,3);
}
static void check(id<MTLCommandQueue>q,NSDictionary*ps,Fixture*f) {
    const uint32_t variants=f->D==128?VariantCount:2;
    for(uint32_t phase=1;phase<=3;++phase) {
        reset(f,Frozen);run(q,ps,f,Frozen,phase,false,1);
        for(uint32_t v=1;v<variants;++v) {
            reset(f,v);run(q,ps,f,v,phase,false,1);same_result(f,v,phase);
        }
    }
    for(uint32_t v=1;v<variants;++v) {
        reset(f,v);run(q,ps,f,v,3,true,1);same_result(f,v,3);
    }
    check_decode_continuation(q,ps,f);
    for(id<MTLBuffer>b in @[f->x,f->w,f->alpha,f->beta,f->A,f->dt,f->initialState,f->initialHistory])check_guard(b);
    for(uint32_t v=0;v<variants;++v)
        for(id<MTLBuffer>b in @[f->history[v],f->state[v],f->conv[v],f->out[v]])check_guard(b);
    printf("PASS T%u D%u Hk%u Hv%u W%u types%u/%u/%u variants%u intermediates/state/chunks/continuation bitexact\n",
           f->T,f->D,f->K,f->H,f->W,f->type,f->atype,f->dtype,variants);fflush(stdout);
}
static int double_cmp(const void*a,const void*b) { double x=*(const double*)a,y=*(const double*)b;return(x>y)-(x<y); }
static void bench(id<MTLCommandQueue>q,NSDictionary*ps,Fixture*f) {
    const uint32_t variants=f->D==128?VariantCount:2;
    double times[VariantCount][9];
    for(uint32_t trial=0;trial<9;++trial)for(uint32_t order=0;order<variants;++order) {
        const uint32_t v=(trial+order)%variants;reset(f,v);times[v][trial]=run(q,ps,f,v,3,false,3);
    }
    for(uint32_t v=0;v<variants;++v)qsort(times[v],9,sizeof(double),double_cmp);
    printf("BENCH T%u D%u Hk%u Hv%u serial_us=%.3f batch_us=%.3f batch_speedup=%.3f",
           f->T,f->D,f->K,f->H,times[Frozen][4],times[GenericBatch][4],times[Frozen][4]/times[GenericBatch][4]);
    if(variants==VariantCount)printf(" register_batch_us=%.3f register_decode_us=%.3f register_batch_speedup=%.3f register_decode_speedup=%.3f",
        times[RegisterBatch][4],times[RegisterDecode][4],times[GenericBatch][4]/times[RegisterBatch][4],times[Frozen][4]/times[RegisterDecode][4]);
    putchar('\n');fflush(stdout);
}
int main(int argc,char**argv) { @autoreleasepool {
    const bool timing=argc==2 && !strcmp(argv[1],"--bench");
    need(argc==1 || timing,"usage test_bonsai_gdn_prefill [--bench]");
    id<MTLDevice>dev=MTLCreateSystemDefaultDevice();need(dev!=nil,"Metal device");
    id<MTLCommandQueue>q=[dev newCommandQueue];need(q!=nil,"queue");
    NSError*err=nil;
    NSString*source=[[NSString alloc] initWithBytes:metal_bonsai_metal
        length:metal_bonsai_metal_len encoding:NSUTF8StringEncoding];
    need(source!=nil,"embedded production shader source");
    source=[source stringByAppendingString:frozen_source];
    MTLCompileOptions*options=[MTLCompileOptions new];
    if(@available(macOS 15.0,*))options.mathMode=MTLMathModeSafe;
    else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled=NO;
#pragma clang diagnostic pop
    }
    id<MTLLibrary>lib=[dev newLibraryWithSource:source options:options error:&err];
    if(!lib)fprintf(stderr,"%s\n",err.localizedDescription.UTF8String);need(lib!=nil,"compile Metal");
    NSMutableDictionary*ps=[NSMutableDictionary dictionary];
    for(NSString*name in @[@"frozen_bonsai_conv",@"frozen_bonsai_norm",@"frozen_bonsai_gdn",@"bonsai_conv_batch",@"bonsai_l2_batch",@"bonsai_gdn_batch",@"bonsai_gdn_128",@"bonsai_gdn_batch_128"]) {
        id<MTLComputePipelineState>p=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];
        need(p!=nil,"pipeline");ps[name]=p;
    }
    const uint32_t counts[]={1,2,3,4,5,15,16,31,32,128,129};
    size_t fixtures=0;
    for(size_t i=0;i<sizeof(counts)/sizeof(counts[0]);++i) {@autoreleasepool {
        check(q,ps,fixture(dev,counts[i],32,2,6,4,0,0,0));
        // Uneven head sharing, width=1, and all scalar storage formats.
        check(q,ps,fixture(dev,counts[i],128,2,3,1,30,1,30));fixtures+=2;
    }}
    const uint32_t dims[]={8,31,64,127,128,129,256};
    for(size_t i=0;i<sizeof(dims)/sizeof(dims[0]);++i) {@autoreleasepool {
        check(q,ps,fixture(dev,5,dims[i],1,2,2,0,0,0));
        check(q,ps,fixture(dev,7,dims[i],2,2,7,1,30,1));fixtures+=2;
    }}
    const uint32_t realCounts[]={1,4,16,32,65,128,129};
    for(size_t i=0;i<sizeof(realCounts)/sizeof(realCounts[0]);++i) {@autoreleasepool {
        Fixture*f=fixture(dev,realCounts[i],128,16,48,4,0,0,0);check(q,ps,f);if(timing)bench(q,ps,f);++fixtures;
    }}
    printf("PASS %zu fixtures; generic/register prefill/decode, three stages, continued chunks and tails; no model used\n",fixtures);return 0;
}}
