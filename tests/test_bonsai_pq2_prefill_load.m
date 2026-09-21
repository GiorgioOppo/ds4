/* Exact PQ2 prefill load hoisting versus frozen pre-hoist kernels.
 * No model needed. Metal validation checks memory safety; --bench measures
 * alternating GPU medians and must run separately without validation.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../metal/bonsai.metal.inc"

// Frozen pre-hoist tiled and fused kernels. Never regenerate this oracle
// from production changes; their exact arithmetic and K walk are the contract.
static NSString *frozen_source=
@"kernel void frozen_bonsai_mm_pq2_tiled(constant BonsaiArgs &a [[buffer(0)]],\n"
"                               device const uchar *w [[buffer(1)]],\n"
"                               device const float *x [[buffer(2)]],\n"
"                               device float *out [[buffer(3)]],\n"
"                               uint2 group [[threadgroup_position_in_grid]],\n"
"                               uint tid [[thread_index_in_threadgroup]],\n"
"                               uint sg [[simdgroup_index_in_threadgroup]]) {\n"
"    constexpr uint BM=64u, BN=32u, BK=32u;\n"
"    threadgroup float weights[BM*BK];\n"
"    threadgroup float inputs[BN*BK];\n"
"\n"
"    constexpr uint MR=BM/16u, NR=BN/16u;\n"
"    constexpr uint COEFFS=BM*BK/128u, ROW_THREADS=BK/COEFFS;\n"
"    constexpr uint INPUT_RUN=BK*BN/128u;\n"
"    const uint first_row=group.x*BM,first_token=group.y*BN;\n"
"    simdgroup_float8x8 wf[MR],xf[NR],acc[MR*NR];\n"
"#pragma unroll\n"
"    for(uint i=0;i<MR*NR;i++)acc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);\n"
"    for(uint first_k=0;first_k<a.cols;first_k+=BK) {\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        const uint row=tid/ROW_THREADS,start_k=(tid%ROW_THREADS)*COEFFS;\n"
"        float scale=0.0f;\n"
"        uint4 bytes0=uint4(1u);\n"
"        if(first_row+row<a.rows) {\n"
"            device const uchar *block=w+ulong(first_row+row)*a.row_bytes+(first_k/128u)*34u;\n"
"            const uint byte=2u+((first_k%128u)+start_k)/4u;\n"
"            scale=float(*(device const half*)block);\n"
"            bytes0=uint4(block[byte],block[byte+1u],block[byte+2u],block[byte+3u]);\n"
"        }\n"
"#pragma unroll\n"
"        for(uint j=0;j<COEFFS;j++) {\n"
"            const uint k=start_k+j;\n"
"            const uint byte=bytes0[j/4u];\n"
"            const uint dst=64u*((k/8u)*(BM/8u)+row/8u)+(k%8u)*8u+row%8u;\n"
"            weights[dst]=first_row+row<a.rows&&first_k+k<a.cols\n"
"                ?scale*float(int((byte>>(2u*(j%4u)))&3u)-1):0.0f;\n"
"        }\n"
"        // Each thread loads a contiguous activation segment, then copies whole\n"
"        // float4 vectors to the row-major rows of the packed 8x8 input tiles.\n"
"        const uint token=tid/(BK/INPUT_RUN),input_k=(tid%(BK/INPUT_RUN))*INPUT_RUN;\n"
"#pragma unroll\n"
"        for(uint j=0;j<INPUT_RUN;j+=4u) {\n"
"            const uint k=input_k+j;\n"
"            float4 values=0.0f;\n"
"            if(first_token+token<a.n && first_k+k+3u<a.cols)\n"
"                values=*(device const float4*)(x+ulong(first_token+token)*a.cols+first_k+k);\n"
"            const uint dst=64u*((k/8u)*(BN/8u)+token/8u)+(token%8u)*8u+k%8u;\n"
"            *(threadgroup float4*)(inputs+dst)=values;\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"#pragma unroll\n"
"        for(uint k=0;k<BK;k+=8u) {\n"
"#pragma unroll\n"
"            for(uint i=0;i<MR;i++)\n"
"                simdgroup_load(wf[i],weights+64u*((k/8u)*(BM/8u)+(sg%2u)*MR+i),8,0,false);\n"
"#pragma unroll\n"
"            for(uint j=0;j<NR;j++)\n"
"                simdgroup_load(xf[j],inputs+64u*((k/8u)*(BN/8u)+(sg/2u)*NR+j),8,0,false);\n"
"#pragma unroll\n"
"            for(uint j=0;j<NR;j++) {\n"
"#pragma unroll\n"
"                for(uint i=0;i<MR;i++)simdgroup_multiply_accumulate(acc[j*MR+i],xf[j],wf[i],acc[j*MR+i]);\n"
"            }\n"
"        }\n"
"    }\n"
"    // Uniform tile check: every thread takes the same output path.\n"
"    if(first_row+BM<=a.rows && first_token+BN<=a.n) {\n"
"        device float *dst=out+ulong(first_token+(sg/2u)*(BN/2u))*a.rows+first_row+(sg%2u)*(BM/2u);\n"
"#pragma unroll\n"
"        for(uint j=0;j<NR;j++) {\n"
"#pragma unroll\n"
"            for(uint i=0;i<MR;i++)simdgroup_store(acc[j*MR+i],dst+ulong(j*8u)*a.rows+i*8u,a.rows,0,false);\n"
"        }\n"
"    } else {\n"
"        // Reuse the weight scratch for bounded tails. Each token-half is\n"
"        // staged independently, so no extra output allocation is needed.\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for(uint part=0;part<2u;part++) {\n"
"            if(sg/2u==part) {\n"
"#pragma unroll\n"
"                for(uint j=0;j<NR;j++) {\n"
"#pragma unroll\n"
"                    for(uint i=0;i<MR;i++)simdgroup_store(acc[j*MR+i],weights+j*8u*BM+(sg%2u)*(BM/2u)+i*8u,BM,0,false);\n"
"                }\n"
"            }\n"
"            threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"#pragma unroll\n"
"            for(uint i=tid;i<BM*(BN/2u);i+=128u) {\n"
"                uint t=part*(BN/2u)+i/BM,r=i%BM;\n"
"                if(first_token+t<a.n&&first_row+r<a.rows)out[ulong(first_token+t)*a.rows+first_row+r]=weights[i];\n"
"            }\n"
"            threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"kernel void frozen_bonsai_mm_pq2_gate_up_tiled(constant BonsaiArgs &a [[buffer(0)]],\n"
"    device const uchar *gate_w [[buffer(1)]], device const uchar *up_w [[buffer(2)]],\n"
"    device const float *x [[buffer(3)]], device float *mid [[buffer(4)]],\n"
"    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],\n"
"    uint sg [[simdgroup_index_in_threadgroup]]) {\n"
"    constexpr uint BM=64u, BN=32u, BK=32u, PAIRS=BM/2u;\n"
"    threadgroup float weights[BM*BK];\n"
"    threadgroup float inputs[BN*BK];\n"
"    constexpr uint MR=BM/16u, NR=BN/16u;\n"
"    constexpr uint COEFFS=BM*BK/128u, ROW_THREADS=BK/COEFFS;\n"
"    constexpr uint INPUT_RUN=BK*BN/128u;\n"
"    const uint first_row=group.x*PAIRS,first_token=group.y*BN;\n"
"    simdgroup_float8x8 wf[MR],xf[NR],acc[MR*NR];\n"
"#pragma unroll\n"
"    for(uint i=0;i<MR*NR;i++)acc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);\n"
"    for(uint first_k=0;first_k<a.cols;first_k+=BK) {\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        const uint row=tid/ROW_THREADS,start_k=(tid%ROW_THREADS)*COEFFS;\n"
"        const uint pair_row=row%PAIRS;\n"
"        float scale=0.0f;\n"
"        uint4 bytes0=uint4(1u);\n"
"        if(first_row+pair_row<a.rows) {\n"
"            device const uchar *w=row<PAIRS?gate_w:up_w;\n"
"            device const uchar *block=w+ulong(first_row+pair_row)*a.row_bytes+(first_k/128u)*34u;\n"
"            const uint byte=2u+((first_k%128u)+start_k)/4u;\n"
"            scale=float(*(device const half*)block);\n"
"            bytes0=uint4(block[byte],block[byte+1u],block[byte+2u],block[byte+3u]);\n"
"        }\n"
"#pragma unroll\n"
"        for(uint j=0;j<COEFFS;j++) {\n"
"            const uint k=start_k+j;\n"
"            const uint byte=bytes0[j/4u];\n"
"            const uint dst=64u*((k/8u)*(BM/8u)+row/8u)+(k%8u)*8u+row%8u;\n"
"            weights[dst]=first_row+pair_row<a.rows&&first_k+k<a.cols\n"
"                ?scale*float(int((byte>>(2u*(j%4u)))&3u)-1):0.0f;\n"
"        }\n"
"        const uint token=tid/(BK/INPUT_RUN),input_k=(tid%(BK/INPUT_RUN))*INPUT_RUN;\n"
"#pragma unroll\n"
"        for(uint j=0;j<INPUT_RUN;j+=4u) {\n"
"            const uint k=input_k+j;\n"
"            float4 values=0.0f;\n"
"            if(first_token+token<a.n&&first_k+k+3u<a.cols)\n"
"                values=*(device const float4*)(x+ulong(first_token+token)*a.cols+first_k+k);\n"
"            const uint dst=64u*((k/8u)*(BN/8u)+token/8u)+(token%8u)*8u+k%8u;\n"
"            *(threadgroup float4*)(inputs+dst)=values;\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"#pragma unroll\n"
"        for(uint k=0;k<BK;k+=8u) {\n"
"#pragma unroll\n"
"            for(uint i=0;i<MR;i++)\n"
"                simdgroup_load(wf[i],weights+64u*((k/8u)*(BM/8u)+(sg%2u)*MR+i),8,0,false);\n"
"#pragma unroll\n"
"            for(uint j=0;j<NR;j++)\n"
"                simdgroup_load(xf[j],inputs+64u*((k/8u)*(BN/8u)+(sg/2u)*NR+j),8,0,false);\n"
"#pragma unroll\n"
"            for(uint j=0;j<NR;j++) {\n"
"#pragma unroll\n"
"                for(uint i=0;i<MR;i++)simdgroup_multiply_accumulate(acc[j*MR+i],xf[j],wf[i],acc[j*MR+i]);\n"
"            }\n"
"        }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for(uint part=0;part<2u;part++) {\n"
"        if(sg/2u==part) {\n"
"#pragma unroll\n"
"            for(uint j=0;j<NR;j++) {\n"
"#pragma unroll\n"
"                for(uint i=0;i<MR;i++)\n"
"                    simdgroup_store(acc[j*MR+i],weights+j*8u*BM+(sg%2u)*PAIRS+i*8u,BM,0,false);\n"
"            }\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for(uint i=tid;i<PAIRS*(BN/2u);i+=128u) {\n"
"            const uint t=part*(BN/2u)+i/PAIRS,r=i%PAIRS;\n"
"            if(first_token+t<a.n&&first_row+r<a.rows) {\n"
"                const uint src=(i/PAIRS)*BM+r;\n"
"                const float g=weights[src],u=weights[src+PAIRS];\n"
"                mid[ulong(first_token+t)*a.rows+first_row+r]=(g/(1.0f+exp(-g)))*u;\n"
"            }\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"}\n";


typedef struct {
    uint32_t n,rows,cols,type,row_bytes,pos,heads,kvheads,dim,rot,width,mode,groups;
    float eps,base;
} Args;
_Static_assert(sizeof(Args)==60,"Bonsai args");
typedef struct {const char *suffix;unsigned bm,bn;} Variant;
static const Variant variants[]={{"_frozen",64,32},{"_hoisted",64,32}};
enum {GUARD=64,ROUNDS=6};
static const NSUInteger offsets[]={80,112,96,128};
static void need(bool ok,const char *msg){if(!ok){fprintf(stderr,"FAIL %s\n",msg);exit(1);}}
static uint32_t hash(uint32_t x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
static id<MTLBuffer> buffer(id<MTLDevice> dev,size_t bytes,unsigned slot) {
    id<MTLBuffer> b=[dev newBufferWithLength:bytes+offsets[slot]+GUARD options:MTLResourceStorageModeShared];
    need(b!=nil,"buffer");memset(b.contents,0xa5,b.length);return b;
}
static void *data(id<MTLBuffer> b,unsigned slot){return (uint8_t *)b.contents+offsets[slot];}
static void poison(id<MTLBuffer> b){memset(data(b,3),0xff,b.length-offsets[3]-GUARD);}
static void guards(id<MTLBuffer> b,unsigned slot) {
    const uint8_t *p=b.contents;
    for(NSUInteger i=0;i<offsets[slot];i++)need(p[i]==0xa5,"prefix/offset canary");
    for(unsigned i=0;i<GUARD;i++)need(p[b.length-GUARD+i]==0xa5,"suffix canary");
}
static NSString *kernel_name(unsigned variant,bool pair) {
    return [NSString stringWithFormat:@"%s%s",variant==0?"frozen_":"",pair?"bonsai_mm_pq2_gate_up_tiled":"bonsai_mm_pq2_tiled"];
}
static double run(id<MTLCommandQueue> queue,NSDictionary *ps,Args a,
                  NSArray<id<MTLBuffer>> *b,unsigned variant,bool pair,unsigned repeats) {
    id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    id<MTLComputePipelineState> p=ps[kernel_name(variant,pair)];
    need(cb&&e&&p&&p.maxTotalThreadsPerThreadgroup>=128,"pipeline/command");
    [e setComputePipelineState:p];[e setBytes:&a length:sizeof(a) atIndex:0];
    NSArray<id<MTLBuffer>> *bound=pair?@[b[0],b[1],b[2],b[3]]:@[b[0],b[2],b[3]];
    const unsigned slots[]={0,pair?1:2,pair?2:3,3};
    for(NSUInteger i=0;i<bound.count;i++)[e setBuffer:bound[i] offset:offsets[slots[i]] atIndex:i+1];
    const unsigned mr=pair?variants[variant].bm/2:variants[variant].bm,bn=variants[variant].bn;
    for(unsigned i=0;i<repeats;i++)
        [e dispatchThreadgroups:MTLSizeMake((a.rows+mr-1)/mr,(a.n+bn-1)/bn,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [e endEncoding];[cb commit];[cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted)fprintf(stderr,"%s\n",cb.error.localizedDescription.UTF8String);
    need(cb.status==MTLCommandBufferStatusCompleted,"GPU completion");
    return (cb.GPUEndTime-cb.GPUStartTime)*1e6/repeats;
}
static void fill_weights(id<MTLBuffer> b,Args a,unsigned seed,unsigned slot) {
    const uint16_t scales[]={0x0000,0x8000,0x2800,0xac00,0x3000,0x0400,0x0001,0x3800};
    uint8_t *w=data(b,slot);
    for(unsigned r=0;r<a.rows;r++)for(unsigned block=0;block<a.cols/128;block++) {
        uint8_t *p=w+(size_t)r*a.row_bytes+block*34;
        // Keep a zero-code row when available, but force the final valid row
        // to contain nonzero weights, including single-row output tails.
        const uint16_t scale=r==a.rows-1 ? 0x3400 : scales[hash(r*17713+block*23+seed)%8];
        memcpy(p,&scale,2);
        for(unsigned j=0;j<32;j++)p[j+2]=(r==0&&a.rows>1)?0x55:(uint8_t)hash(r*2777+block*9391+j*13+seed);
    }
}
static void equal(id<MTLBuffer> b,const float *ref,size_t count,Args a,unsigned v,bool pair) {
    const float *y=data(b,3);
    for(size_t i=0;i<count;i++)if(!isfinite(y[i])||!isfinite(ref[i])||memcmp(y+i,ref+i,4)) {
        fprintf(stderr,"%s%s M%u K%u T%u i%zu got%.9g expected%.9g\n",pair?"pair":"plain",variants[v].suffix,a.rows,a.cols,a.n,i,y[i],ref[i]);
        need(false,"bitwise current-tile parity");
    }
}
static int compare_double(const void *a,const void *b){double x=*(const double *)a,y=*(const double *)b;return(x>y)-(x<y);}
static void fixture(id<MTLDevice> dev,id<MTLCommandQueue> queue,NSDictionary *ps,unsigned variantCount,
                    unsigned rows,unsigned cols,unsigned count,bool pair,bool timing) {
    @autoreleasepool {
        const Args a={.n=count,.rows=rows,.cols=cols,.type=142,.row_bytes=cols/128*34};
        const size_t outputs=(size_t)rows*count;
        NSArray<id<MTLBuffer>> *b=@[buffer(dev,(size_t)rows*a.row_bytes,0),buffer(dev,(size_t)rows*a.row_bytes,1),
            buffer(dev,(size_t)count*cols*4,2),buffer(dev,outputs*4,3)];
        fill_weights(b[0],a,11,0);fill_weights(b[1],a,83,1);
        float *x=data(b[2],2);
        for(size_t i=0;i<(size_t)count*cols;i++)x[i]=(float)(int32_t)hash((uint32_t)i+42)/2147483648.0f*.2f;
        if(count>1)memset(x+cols,0,cols*4);
        poison(b[3]);run(queue,ps,a,b,0,pair,1);
        float *ref=malloc(outputs*4);need(ref!=NULL,"reference");memcpy(ref,data(b[3],3),outputs*4);
        for(unsigned v=1;v<variantCount;v++) {
            poison(b[3]);run(queue,ps,a,b,v,pair,1);equal(b[3],ref,outputs,a,v,pair);
        }
        for(unsigned i=0;i<4;i++)guards(b[i],i);
        printf("PASS %s M%u K%u T%u variants%u exact=1\n",pair?"pair":"plain",rows,cols,count,variantCount);
        if(timing) {
            double times[4][ROUNDS];
            for(unsigned v=0;v<variantCount;v++)run(queue,ps,a,b,v,pair,1);
            for(unsigned trial=0;trial<ROUNDS;trial++)for(unsigned j=0;j<variantCount;j++) {
                const unsigned v=(trial+j)%variantCount;
                times[v][trial]=run(queue,ps,a,b,v,pair,2);
                equal(b[3],ref,outputs,a,v,pair);guards(b[3],3);
            }
            for(unsigned v=0;v<variantCount;v++)qsort(times[v],ROUNDS,sizeof(double),compare_double);
            const double base=(times[0][2]+times[0][3])*.5;
            for(unsigned v=1;v<variantCount;v++) {
                const double current=(times[v][2]+times[v][3])*.5;
                printf("BENCH %s%s M%u K%u T%u base_us%.3f variant_us%.3f speedup%.4f exact=1\n",
                       pair?"pair":"plain",variants[v].suffix,rows,cols,count,base,current,base/current);
            }
        }
        fflush(stdout);free(ref);
    }
}
int main(int argc,char **argv) {
    @autoreleasepool {
        bool timing=false;
        for(int i=1;i<argc;i++){if(!strcmp(argv[i],"--bench"))timing=true;else need(false,"usage test_bonsai_pq2_prefill_load [--bench]");}
        const unsigned variantCount=2;
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice();need(dev!=nil,"Metal device");
        id<MTLCommandQueue> queue=[dev newCommandQueue];need(queue!=nil,"command queue");
        NSError *error=nil;
        NSString *source=[[NSString alloc] initWithBytes:metal_bonsai_metal length:metal_bonsai_metal_len encoding:NSUTF8StringEncoding];
        need(source!=nil,"embedded production Metal");source=[source stringByAppendingString:frozen_source];
        MTLCompileOptions *options=[MTLCompileOptions new];
        if(@available(macOS 15.0,*))options.mathMode=MTLMathModeSafe;
        else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled=NO;
#pragma clang diagnostic pop
        }
        id<MTLLibrary> library=[dev newLibraryWithSource:source options:options error:&error];
        if(!library)fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);need(library!=nil,"shader compile");
        NSMutableDictionary *ps=[NSMutableDictionary dictionary];
        for(unsigned v=0;v<variantCount;v++)for(unsigned pair=0;pair<2;pair++) {
            NSString *name=kernel_name(v,pair);id<MTLComputePipelineState> p=[dev newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
            if(!p)fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);need(p!=nil,"pipeline");ps[name]=p;
            printf("PIPELINE %s max_threads%lu static_tg_bytes%lu\n",name.UTF8String,(unsigned long)p.maxTotalThreadsPerThreadgroup,(unsigned long)p.staticThreadgroupMemoryLength);
        }
        printf("DEVICE %s math=safe\n",dev.name.UTF8String);
        const unsigned rows[]={1,31,32,33,63,64,65,129},tokens[]={1,19,31,32,33,64,65,128};
        const unsigned widths[]={128,256,384};
        for(unsigned r=0;r<sizeof(rows)/sizeof(rows[0]);r++)for(unsigned t=0;t<sizeof(tokens)/sizeof(tokens[0]);t++)
            for(unsigned pair=0;pair<2;pair++)fixture(dev,queue,ps,variantCount,rows[r],widths[(r+t)%3],tokens[t],pair,false);
        fixture(dev,queue,ps,variantCount,65,384,129,false,false);
        fixture(dev,queue,ps,variantCount,65,384,129,true,false);
        fixture(dev,queue,ps,variantCount,17408,5120,19,false,false);
        fixture(dev,queue,ps,variantCount,17408,5120,19,true,false);
        fixture(dev,queue,ps,variantCount,5120,17408,19,false,false);
        for(unsigned n=32;n<=128;n*=2) {
            if(!timing&&n!=32)continue;
            fixture(dev,queue,ps,variantCount,17408,5120,n,false,timing);
            fixture(dev,queue,ps,variantCount,17408,5120,n,true,timing);
            fixture(dev,queue,ps,variantCount,5120,17408,n,false,timing);
        }
        puts("Exact PQ2 prefill load hoisting: PASS");return 0;
    }
}
