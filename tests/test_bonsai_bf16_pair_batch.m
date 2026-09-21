/* Exact BF16 alpha/beta prefill fusion; no model file.
 * MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ./tests/test_bonsai_bf16_pair_batch
 * Optional --bench measures alternating GPU medians without validation.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../metal/bonsai.metal.inc"

// Frozen type-30 branch of the original scalar GEMV. Never regenerate this
// oracle from production; short token counts retain the exact decode walk.
static NSString *frozen_source=
@"kernel void frozen_bf16_mv(constant BonsaiArgs &a [[buffer(0)]],\n"
"    device const uchar *w [[buffer(1)]], device const float *x [[buffer(2)]],\n"
"    device float *out [[buffer(3)]], uint group [[threadgroup_position_in_grid]],\n"
"    ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {\n"
"    uint row=group*4u+sg; float sum=0;\n"
"    if(row<a.rows) {\n"
"        device const uchar *wr=w+ulong(row)*a.row_bytes;\n"
"        device const ushort *wb=(device const ushort *)wr;\n"
"        for(uint k=lane;k<a.cols;k+=32u)\n"
"            sum+=as_type<float>(uint(wb[k])<<16)*x[k];\n"
"    }\n"
"    sum=simd_sum(sum);\n"
"    if(!lane&&row<a.rows)out[row]=sum;\n"
"}\n";

typedef struct {
    uint32_t n,rows,cols,type,row_bytes,pos,heads,kvheads,dim,rot,width,mode,groups;
    float eps,base;
} Args;
_Static_assert(sizeof(Args)==60,"Metal argument layout");
enum { WA, WB, X, YA, YB, RA, RB, BUFFER_COUNT };
// Different aligned binding offsets exercise each input/output independently.
static const NSUInteger offsets[BUFFER_COUNT]={80,112,96,128,160,144,176};
static const NSUInteger suffix=64;
static void need(bool ok,const char *message) {
    if(!ok) {fprintf(stderr,"FAIL %s\n",message);exit(1);}
}
static uint32_t hash(uint32_t n) {
    n^=n>>16;n*=0x7feb352du;n^=n>>15;n*=0x846ca68bu;return n^(n>>16);
}
static void *data(NSArray<id<MTLBuffer>> *buffers,unsigned i) {
    return (char *)buffers[i].contents+offsets[i];
}
static void guard_check(NSArray<id<MTLBuffer>> *buffers,unsigned i) {
    id<MTLBuffer>b=buffers[i];const unsigned char *p=b.contents;
    for(NSUInteger j=0;j<offsets[i];++j)need(p[j]==0xcd,"leading canary");
    for(NSUInteger j=0;j<suffix;++j)need(p[b.length-1-j]==0xcd,"trailing canary");
}
static void equal(NSArray<id<MTLBuffer>> *buffers,Args a) {
    for(unsigned projection=0;projection<2;++projection) {
        const float *reference=data(buffers,RA+projection),*actual=data(buffers,YA+projection);
        for(size_t i=0;i<(size_t)a.n*a.rows;++i) {
            if(!isfinite(reference[i])||!isfinite(actual[i])||memcmp(reference+i,actual+i,sizeof(float))) {
                fprintf(stderr,"M%u K%u T%u projection%u output%zu reference%.9g actual%.9g\n",
                        a.rows,a.cols,a.n,projection,i,reference[i],actual[i]);
                need(false,"bitwise parity");
            }
        }
    }
}
static void launch(id<MTLComputeCommandEncoder>enc,uint32_t rows,uint32_t tokens) {
    [enc dispatchThreadgroups:MTLSizeMake((rows+3u)/4u,(tokens+3u)/4u,1)
        threadsPerThreadgroup:MTLSizeMake(128,1,1)];
}
static double run(id<MTLCommandQueue>queue,NSDictionary *pipelines,
                  NSArray<id<MTLBuffer>> *buffers,Args a,bool fused,unsigned repeats) {
    id<MTLCommandBuffer>cb=[queue commandBuffer];id<MTLComputeCommandEncoder>enc=[cb computeCommandEncoder];
    need(cb&&enc,"command creation");
    [enc setBytes:&a length:sizeof(a) atIndex:0];
    for(unsigned repeat=0;repeat<repeats;++repeat) {
        if(fused) {
            [enc setComputePipelineState:pipelines[@"bonsai_bf16_pair_batch"]];
            for(unsigned i=0;i<5;++i)[enc setBuffer:buffers[i] offset:offsets[i] atIndex:i+1];
            launch(enc,a.rows,a.n);
        } else for(unsigned projection=0;projection<2;++projection) {
            [enc setComputePipelineState:pipelines[a.n<4?@"frozen_bf16_mv":@"bonsai_mm"]];
            [enc setBuffer:buffers[WA+projection] offset:offsets[WA+projection] atIndex:1];
            if(a.n<4) {
                for(unsigned token=0;token<a.n;++token) {
                    [enc setBuffer:buffers[X] offset:offsets[X]+(NSUInteger)token*a.cols*4u atIndex:2];
                    [enc setBuffer:buffers[RA+projection] offset:offsets[RA+projection]+(NSUInteger)token*a.rows*4u atIndex:3];
                    launch(enc,a.rows,1);
                }
            } else {
                [enc setBuffer:buffers[X] offset:offsets[X] atIndex:2];
                [enc setBuffer:buffers[RA+projection] offset:offsets[RA+projection] atIndex:3];
                launch(enc,a.rows,a.n);
            }
        }
    }
    [enc endEncoding];[cb commit];[cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted)fprintf(stderr,"%s\n",cb.error.localizedDescription.UTF8String);
    need(cb.status==MTLCommandBufferStatusCompleted,"GPU completion");
    return (cb.GPUEndTime-cb.GPUStartTime)*1e6/repeats;
}
static int compare_double(const void *a,const void *b) {
    const double x=*(const double *)a,y=*(const double *)b;return (x>y)-(x<y);
}
static void fixture(id<MTLDevice>dev,id<MTLCommandQueue>queue,NSDictionary *pipelines,
                    uint32_t rows,uint32_t cols,uint32_t tokens,unsigned mode,bool timing) {
    @autoreleasepool {
        Args a={.n=tokens,.rows=rows,.cols=cols,.type=30,.row_bytes=cols*2u};
        const size_t sizes[BUFFER_COUNT]={(size_t)rows*cols*2u,(size_t)rows*cols*2u,
            (size_t)tokens*cols*4u,(size_t)tokens*rows*4u,(size_t)tokens*rows*4u,
            (size_t)tokens*rows*4u,(size_t)tokens*rows*4u};
        NSMutableArray<id<MTLBuffer>> *buffers=[NSMutableArray array];
        for(unsigned i=0;i<BUFFER_COUNT;++i) {
            id<MTLBuffer>b=[dev newBufferWithLength:offsets[i]+sizes[i]+suffix options:MTLResourceStorageModeShared];
            need(b!=nil,"buffer allocation");memset(b.contents,0xcd,b.length);[buffers addObject:b];
        }
        for(unsigned projection=0;projection<2;++projection) {
            uint16_t *w=data(buffers,WA+projection);
            for(size_t i=0;i<(size_t)rows*cols;++i) {
                const float value=(float)(int32_t)hash((uint32_t)i+7919u*projection+23u)*0x1p-34f;
                uint32_t bits;memcpy(&bits,&value,4);
                w[i]=mode==2?(i%2?0x8000u:0u):(uint16_t)(bits>>16);
            }
        }
        float *x=data(buffers,X);
        for(unsigned t=0;t<tokens;++t)for(unsigned k=0;k<cols;++k)
            x[(size_t)t*cols+k]=mode==1?(k%2?-0.0f:0.0f):mode==3?(k==(t*31u+7u)%cols?1.0f:0.0f):
                (float)(int32_t)hash(t*cols+k+47u)*0x1p-33f;
        for(unsigned i=YA;i<BUFFER_COUNT;++i) {
            uint32_t *p=data(buffers,i);
            for(size_t j=0;j<sizes[i]/4u;++j)p[j]=0x7fc00000u;
        }
        void *saved[3];
        for(unsigned i=0;i<3;++i) {saved[i]=malloc(sizes[i]);need(saved[i]!=NULL,"input snapshot");memcpy(saved[i],data(buffers,i),sizes[i]);}
        run(queue,pipelines,buffers,a,false,1);run(queue,pipelines,buffers,a,true,1);equal(buffers,a);
        if(timing) {
            enum { ROUNDS=9 };double times[2][ROUNDS];
            const unsigned repeats=tokens<=32?5:3;
            for(unsigned v=0;v<2;++v)run(queue,pipelines,buffers,a,v!=0,2);
            for(unsigned trial=0;trial<ROUNDS;++trial)for(unsigned order=0;order<2;++order) {
                const unsigned v=(trial+order)%2;times[v][trial]=run(queue,pipelines,buffers,a,v!=0,repeats);
            }
            for(unsigned v=0;v<2;++v)qsort(times[v],ROUNDS,sizeof(double),compare_double);
            equal(buffers,a);
            printf("BENCH M%u K%u T%u separate_us=%.3f fused_us=%.3f speedup=%.4f exact=1\n",
                   rows,cols,tokens,times[0][4],times[1][4],times[0][4]/times[1][4]);
        } else printf("PASS M%u K%u T%u mode%u offsets/canaries bitexact\n",rows,cols,tokens,mode);
        for(unsigned i=0;i<3;++i) {need(!memcmp(saved[i],data(buffers,i),sizes[i]),"input unchanged");free(saved[i]);}
        for(unsigned i=0;i<BUFFER_COUNT;++i)guard_check(buffers,i);
        fflush(stdout);
    }
}
int main(int argc,char **argv) { @autoreleasepool {
    const bool timing=argc==2&&!strcmp(argv[1],"--bench");
    need(argc==1||timing,"usage test_bonsai_bf16_pair_batch [--bench]");
    id<MTLDevice>dev=MTLCreateSystemDefaultDevice();need(dev!=nil,"Metal device");
    id<MTLCommandQueue>queue=[dev newCommandQueue];need(queue!=nil,"queue");
    NSString *source=[[NSString alloc] initWithBytes:metal_bonsai_metal length:metal_bonsai_metal_len encoding:NSUTF8StringEncoding];
    source=[source stringByAppendingString:frozen_source];
    MTLCompileOptions *options=[MTLCompileOptions new];
    if(@available(macOS 15.0,*))options.mathMode=MTLMathModeSafe;
    else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled=NO;
#pragma clang diagnostic pop
    }
    NSError *error=nil;id<MTLLibrary>lib=[dev newLibraryWithSource:source options:options error:&error];
    if(!lib)fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);need(lib!=nil,"compile Metal");
    NSMutableDictionary *pipelines=[NSMutableDictionary dictionary];
    for(NSString *name in @[@"frozen_bf16_mv",@"bonsai_mm",@"bonsai_bf16_pair_batch"]) {
        id<MTLComputePipelineState>p=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&error];
        need(p&&p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=128,"pipeline");pipelines[name]=p;
    }
    const uint32_t rows[]={1,3,4,5,47,48,49},cols[]={1,31,32,33,47,127,128,129,5120};
    const uint32_t counts[]={1,2,3,4,5,15,16,17,31,32,33,64,65,127,128,129};
    unsigned fixtures=0;
    for(unsigned r=0;r<sizeof(rows)/sizeof(rows[0]);++r)
        for(unsigned t=0;t<sizeof(counts)/sizeof(counts[0]);++t) {
            fixture(dev,queue,pipelines,rows[r],cols[(r+t)%9],counts[t],0,false);++fixtures;
        }
    const uint32_t actualCounts[]={4,16,32,64,128};
    for(unsigned i=0;i<sizeof(actualCounts)/sizeof(actualCounts[0]);++i) {
        fixture(dev,queue,pipelines,48,5120,actualCounts[i],0,timing);++fixtures;
        for(unsigned mode=1;mode<=3;++mode) {
            fixture(dev,queue,pipelines,49,47,actualCounts[i]+1u,mode,false);++fixtures;
        }
    }
    printf("PASS %u BF16 paired prefill fixtures; exact four-token/scalar oracles, tails, offsets, zeros and guards\n",fixtures);
    return 0;
}}
