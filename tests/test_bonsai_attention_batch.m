/* Exact batched causal attention versus frozen single-query kernels.
 * No model is needed. Run correctness with Metal validation; --bench
 * requires an otherwise idle GPU and disabled validation.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../metal/bonsai.metal.inc"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Frozen original kernels; do not regenerate these from production changes.
static NSString *frozen_source=
@"kernel void frozen_bonsai_scores(constant BonsaiArgs &a [[buffer(0)]],\n"
"    device const float *q [[buffer(1)]], device const float *kc [[buffer(2)]],\n"
"    device float *scores [[buffer(3)]], uint2 i [[thread_position_in_grid]]) {\n"
"    if(i.x>a.pos||i.y>=a.heads)return;\n"
"    uint kh=i.y/(a.heads/a.kvheads);\n"
"    device const float *k=kc+(ulong(i.x)*a.kvheads+kh)*a.dim;\n"
"    float sum=0;\n"
"    for(uint d=0;d<a.dim;++d)sum+=q[i.y*a.dim+d]*k[d];\n"
"    scores[ulong(i.y)*a.width+i.x]=sum*rsqrt(float(a.dim));\n"
"}\n"
"kernel void frozen_bonsai_softmax(constant BonsaiArgs &a [[buffer(0)]],\n"
"    device float *scores [[buffer(1)]], uint h [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]]) {\n"
"    threadgroup float tmp[256];\n"
"    device float *row=scores+ulong(h)*a.width;\n"
"    float mx=-INFINITY;\n"
"    for(uint t=tid;t<=a.pos;t+=256)mx=max(mx,row[t]);\n"
"    tmp[tid]=mx;threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for(uint d=128;d;d/=2){if(tid<d)tmp[tid]=max(tmp[tid],tmp[tid+d]);threadgroup_barrier(mem_flags::mem_threadgroup);}\n"
"    mx=tmp[0];float sum=0;\n"
"    for(uint t=tid;t<=a.pos;t+=256){float v=exp(row[t]-mx);row[t]=v;sum+=v;}\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);tmp[tid]=sum;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for(uint d=128;d;d/=2){if(tid<d)tmp[tid]+=tmp[tid+d];threadgroup_barrier(mem_flags::mem_threadgroup);}\n"
"    sum=tmp[0];for(uint t=tid;t<=a.pos;t+=256)row[t]/=sum;\n"
"}\n"
"kernel void frozen_bonsai_attention(constant BonsaiArgs &a [[buffer(0)]],\n"
"    device const float *scores [[buffer(1)]], device const float *vc [[buffer(2)]],\n"
"    device float *out [[buffer(3)]], uint i [[thread_position_in_grid]]) {\n"
"    if(i>=a.heads*a.dim)return;\n"
"    uint h=i/a.dim,d=i%a.dim,kh=h/(a.heads/a.kvheads);\n"
"    float sum=0;\n"
"    for(uint t=0;t<=a.pos;++t)sum+=scores[ulong(h)*a.width+t]*vc[(ulong(t)*a.kvheads+kh)*a.dim+d];\n"
"    out[i]=sum;\n"
"}\n";

typedef struct {
    uint32_t n,rows,cols,type,row_bytes,pos,heads,kvheads,dim,rot,width,mode,groups;
    float eps,base;
} Args;
_Static_assert(sizeof(Args)==60,"Bonsai argument layout");
enum { GUARD=64, ROUNDS=6 };
static void need(bool ok,const char *message) {
    if(!ok){fprintf(stderr,"FAIL %s\n",message);exit(1);}
}
static uint32_t hash(uint32_t x) {
    x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);
}
@interface Guarded:NSObject {
@public
    id<MTLBuffer> buffer;
    NSUInteger offset,bytes;
}
@end
@implementation Guarded
@end
static Guarded *allocate(id<MTLDevice> dev,size_t count,unsigned salt) {
    Guarded *b=[Guarded new];b->offset=GUARD+(salt%4)*64;b->bytes=count*4;
    b->buffer=[dev newBufferWithLength:b->offset+b->bytes+GUARD options:MTLResourceStorageModeShared];
    need(b->buffer!=nil,"allocate guarded buffer");memset(b->buffer.contents,0xa5,b->buffer.length);return b;
}
static float *data(Guarded *b){return (float *)((uint8_t *)b->buffer.contents+b->offset);}
static void poison(Guarded *b){memset(data(b),0xff,b->bytes);}
static void guards(Guarded *b) {
    const uint8_t *p=b->buffer.contents;
    for(NSUInteger i=0;i<b->offset;i++)need(p[i]==0xa5,"prefix/offset canary");
    for(NSUInteger i=b->offset+b->bytes;i<b->buffer.length;i++)need(p[i]==0xa5,"suffix canary");
}
static void bind(id<MTLComputeCommandEncoder> e,id<MTLComputePipelineState> p,
                 Args a,NSArray<Guarded *> *buffers,const NSUInteger *extra) {
    need(p!=nil&&p.maxTotalThreadsPerThreadgroup>=256,"pipeline threads");
    [e setComputePipelineState:p];[e setBytes:&a length:sizeof(a) atIndex:0];
    for(NSUInteger i=0;i<buffers.count;i++) {
        Guarded *b=buffers[i];[e setBuffer:b->buffer offset:b->offset+(extra?extra[i]:0) atIndex:i+1];
    }
}
static void launch(id<MTLComputeCommandEncoder> e,unsigned x,unsigned y,unsigned z) {
    [e dispatchThreadgroups:MTLSizeMake(x,y,z) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
}
static double run(id<MTLCommandQueue> queue,NSDictionary *ps,Args a,
                  Guarded *q,Guarded *kc,Guarded *vc,Guarded *scores,Guarded *out,
                  bool batch,unsigned repeats) {
    id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    need(cb&&e,"command creation");
    for(unsigned repeat=0;repeat<repeats;repeat++) {
        if(batch) {
            bind(e,ps[@"bonsai_scores_batch"],a,@[q,kc,scores],NULL);
            launch(e,(a.pos+a.n+255)/256,a.heads,a.n);
            bind(e,ps[@"bonsai_softmax_batch"],a,@[scores],NULL);launch(e,a.heads,a.n,1);
            bind(e,ps[@"bonsai_attention_batch"],a,@[scores,vc,out],NULL);
            launch(e,(a.heads*a.dim+255)/256,a.n,1);
        } else for(unsigned row=0;row<a.n;row++) {
            Args one=a;one.pos+=row;
            const NSUInteger qoff=(NSUInteger)row*a.heads*a.dim*4,soff=(NSUInteger)row*a.heads*a.width*4;
            const NSUInteger scoreOffsets[]={qoff,0,soff},softOffsets[]={soff},outOffsets[]={soff,0,qoff};
            bind(e,ps[@"frozen_bonsai_scores"],one,@[q,kc,scores],scoreOffsets);
            launch(e,(one.pos+256)/256,a.heads,1);
            bind(e,ps[@"frozen_bonsai_softmax"],one,@[scores],softOffsets);launch(e,a.heads,1,1);
            bind(e,ps[@"frozen_bonsai_attention"],one,@[scores,vc,out],outOffsets);
            launch(e,(a.heads*a.dim+255)/256,1,1);
        }
    }
    [e endEncoding];[cb commit];[cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted)fprintf(stderr,"%s\n",cb.error.localizedDescription.UTF8String);
    need(cb.status==MTLCommandBufferStatusCompleted,"GPU completion");
    return (cb.GPUEndTime-cb.GPUStartTime)*1e6/repeats;
}
static void same(Guarded *actual,Guarded *expected,size_t count,const char *label,Args a) {
    const float *x=data(actual),*y=data(expected);
    for(size_t i=0;i<count;i++)if(!isfinite(x[i])||!isfinite(y[i])||memcmp(x+i,y+i,4)) {
        fprintf(stderr,"%s width%u pos%u N%u H%u K%u D%u i%zu got%.9g expected%.9g\n",
                label,a.width,a.pos,a.n,a.heads,a.kvheads,a.dim,i,x[i],y[i]);
        need(false,"bitwise causal attention parity");
    }
}
static void scores_same(Guarded *actual,Guarded *expected,Args a) {
    const float *x=data(actual),*y=data(expected);
    for(unsigned row=0;row<a.n;row++)for(unsigned h=0;h<a.heads;h++)for(unsigned t=0;t<a.width;t++) {
        const size_t i=((size_t)row*a.heads+h)*a.width+t;
        if(t<=a.pos+row)need(isfinite(x[i])&&isfinite(y[i])&&!memcmp(x+i,y+i,4),"bitwise softmax scores");
        else need(isnan(x[i])&&isnan(y[i]),"future score positions remain poisoned");
    }
}
static int compare_double(const void *a,const void *b) {
    const double x=*(const double *)a,y=*(const double *)b;return (x>y)-(x<y);
}
static void fixture(id<MTLDevice> dev,id<MTLCommandQueue> queue,NSDictionary *ps,
                    unsigned width,unsigned position,unsigned count,unsigned heads,unsigned kvheads,
                    unsigned dim,bool timing) {
    @autoreleasepool {
        const Args a={.n=count,.pos=position,.heads=heads,.kvheads=kvheads,.dim=dim,.width=width};
        need(count&&position+count<=width&&heads%kvheads==0,"valid fixture");
        const size_t qcount=(size_t)count*heads*dim,cachecount=(size_t)width*kvheads*dim;
        Guarded *q=allocate(dev,qcount,1),*kc=allocate(dev,cachecount,2),*vc=allocate(dev,cachecount,3);
        Guarded *scores[2]={allocate(dev,(size_t)count*heads*width,0),allocate(dev,(size_t)count*heads*width,2)};
        Guarded *out[2]={allocate(dev,qcount,3),allocate(dev,qcount,1)};
        float *query=data(q),*key=data(kc),*value=data(vc);
        for(size_t i=0;i<qcount;i++)query[i]=(float)(int32_t)hash((uint32_t)i+42)/2147483648.0f*.2f;
        for(size_t i=0;i<cachecount;i++) {
            key[i]=(float)(int32_t)hash((uint32_t)i+167)/2147483648.0f*.3f;
            value[i]=(float)(int32_t)hash((uint32_t)i+691)/2147483648.0f*.4f;
        }
        // Unwritten cache rows after the complete query block must never be read.
        const size_t future=(size_t)(position+count)*kvheads*dim;
        for(size_t i=future;i<cachecount;i++)key[i]=value[i]=NAN;
        for(unsigned v=0;v<2;v++) {
            poison(scores[v]);poison(out[v]);run(queue,ps,a,q,kc,vc,scores[v],out[v],v!=0,1);
        }
        same(out[1],out[0],qcount,"output",a);scores_same(scores[1],scores[0],a);
        // Poison every row after the first query, including valid later queries
        // from this block. The first output must remain unchanged and finite.
        if(count>1) {
            const size_t firstFuture=(size_t)(position+1)*kvheads*dim;
            for(size_t i=firstFuture;i<cachecount;i++)key[i]=value[i]=NAN;
            Args first=a;first.n=1;poison(scores[1]);poison(out[1]);
            run(queue,ps,first,q,kc,vc,scores[1],out[1],true,1);
            same(out[1],out[0],(size_t)heads*dim,"future-poison output",first);
            for(size_t i=(size_t)heads*dim;i<qcount;i++)need(isnan(data(out[1])[i]),"query output tail remains poisoned");
            // Restore the cache before optional timing.
            for(size_t i=firstFuture;i<future;i++) {
                key[i]=(float)(int32_t)hash((uint32_t)i+167)/2147483648.0f*.3f;
                value[i]=(float)(int32_t)hash((uint32_t)i+691)/2147483648.0f*.4f;
            }
        }
        for(Guarded *b in @[q,kc,vc,scores[0],scores[1],out[0],out[1]])guards(b);
        if(timing) {
            double times[2][ROUNDS];
            for(unsigned v=0;v<2;v++)run(queue,ps,a,q,kc,vc,scores[v],out[v],v!=0,1);
            for(unsigned trial=0;trial<ROUNDS;trial++)for(unsigned j=0;j<2;j++) {
                const unsigned v=(trial+j)%2;times[v][trial]=run(queue,ps,a,q,kc,vc,scores[v],out[v],v!=0,2);
            }
            for(unsigned v=0;v<2;v++)qsort(times[v],ROUNDS,sizeof(double),compare_double);
            const double before=(times[0][2]+times[0][3])*.5,after=(times[1][2]+times[1][3])*.5;
            printf("BENCH W%u P%u N%u H%u K%u D%u serial_us%.3f batch_us%.3f speedup%.4f dispatches%u->3 exact=1\n",
                   width,position,count,heads,kvheads,dim,before,after,before/after,3*count);
        } else printf("PASS W%u P%u N%u H%u K%u D%u scores/output/future/offsets exact=1\n",width,position,count,heads,kvheads,dim);
        fflush(stdout);
    }
}
int main(int argc,char **argv) {
    @autoreleasepool {
        const bool timing=argc==2&&!strcmp(argv[1],"--bench");
        need(argc==1||timing,"usage test_bonsai_attention_batch [--bench]");
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice();need(dev!=nil,"Metal device");
        id<MTLCommandQueue> queue=[dev newCommandQueue];need(queue!=nil,"command queue");
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
        NSError *error=nil;id<MTLLibrary> library=[dev newLibraryWithSource:source options:options error:&error];
        if(!library)fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);
        need(library!=nil,"shader library");
        NSMutableDictionary *ps=[NSMutableDictionary dictionary];
        for(NSString *name in @[@"frozen_bonsai_scores",@"frozen_bonsai_softmax",@"frozen_bonsai_attention",
            @"bonsai_scores_batch",@"bonsai_softmax_batch",@"bonsai_attention_batch"]) {
            id<MTLComputePipelineState> p=[dev newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
            need(p!=nil,"pipeline");ps[name]=p;
        }
        printf("DEVICE %s math=safe\n",dev.name.UTF8String);
        fixture(dev,queue,ps,1,0,1,3,1,7,false);
        fixture(dev,queue,ps,17,0,3,6,2,31,false);
        fixture(dev,queue,ps,19,1,8,6,2,129,false);
        fixture(dev,queue,ps,257,243,3,3,1,7,false);
        const unsigned contexts[]={255,256,257,2048,2049,4096},counts[]={1,3,8};
        for(unsigned w=0;w<sizeof(contexts)/sizeof(contexts[0]);w++)
            for(unsigned n=0;n<sizeof(counts)/sizeof(counts[0]);n++)
                fixture(dev,queue,ps,contexts[w],contexts[w]-counts[n],counts[n],24,4,256,false);
        if(timing) {
            fixture(dev,queue,ps,256,248,8,24,4,256,true);
            fixture(dev,queue,ps,2048,2040,8,24,4,256,true);
        }
        puts("Bonsai exact batched attention: PASS");return 0;
    }
}
