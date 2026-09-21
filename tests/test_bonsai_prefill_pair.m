/* Exact PQ2 prefill gate/up/SiLU fusion versus separate production tiled
 * projections and activation. No model is needed. Run correctness with
 * Metal validation; --bench timings require an otherwise idle GPU.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../metal/bonsai.metal.inc"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t n,rows,cols,type,row_bytes,pos,heads,kvheads,dim,rot,width,mode,groups;
    float eps,base;
} Args;
_Static_assert(sizeof(Args)==60,"Bonsai argument layout");
enum { GUARD=64, ROUNDS=6 };

static void need(bool ok,const char *message) {
    if(!ok) {fprintf(stderr,"FAIL %s\n",message);exit(1);}
}
static uint32_t hash(uint32_t x) {
    x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);
}
static id<MTLBuffer> buffer(id<MTLDevice> dev,size_t bytes) {
    id<MTLBuffer> b=[dev newBufferWithLength:bytes+2*GUARD options:MTLResourceStorageModeShared];
    need(b!=nil,"buffer allocation");memset(b.contents,0xa5,b.length);return b;
}
static void *data(id<MTLBuffer> b) {return (uint8_t *)b.contents+GUARD;}
static void poison(id<MTLBuffer> b) {memset(data(b),0xff,b.length-2*GUARD);}
static void guards(id<MTLBuffer> b) {
    const uint8_t *p=b.contents;
    for(unsigned i=0;i<GUARD;i++)
        need(p[i]==0xa5&&p[b.length-GUARD+i]==0xa5,"buffer canary");
}
static void bind(id<MTLComputeCommandEncoder> e,id<MTLComputePipelineState> p,
                 Args a,NSArray<id<MTLBuffer>> *buffers) {
    [e setComputePipelineState:p];[e setBytes:&a length:sizeof(a) atIndex:0];
    for(NSUInteger i=0;i<buffers.count;i++)[e setBuffer:buffers[i] offset:GUARD atIndex:i+1];
}
static void launch(id<MTLComputeCommandEncoder> e,unsigned gx,unsigned gy,unsigned threads) {
    [e dispatchThreadgroups:MTLSizeMake(gx,gy,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
}
static double run(id<MTLCommandQueue> queue,NSDictionary *pipelines,Args a,
                  NSArray<id<MTLBuffer>> *b,bool fused,unsigned repeats) {
    id<MTLCommandBuffer> cb=[queue commandBuffer];
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    for(unsigned i=0;i<repeats;i++) {
        if(fused) {
            bind(e,pipelines[@"bonsai_mm_pq2_gate_up_tiled"],a,@[b[0],b[1],b[2],b[5]]);
            launch(e,(a.rows+31)/32,(a.n+31)/32,128);
        } else {
            bind(e,pipelines[@"bonsai_mm_pq2_tiled"],a,@[b[0],b[2],b[3]]);
            launch(e,(a.rows+63)/64,(a.n+31)/32,128);
            bind(e,pipelines[@"bonsai_mm_pq2_tiled"],a,@[b[1],b[2],b[4]]);
            launch(e,(a.rows+63)/64,(a.n+31)/32,128);
            const Args element={.n=a.n*a.rows,.mode=1};
            bind(e,pipelines[@"bonsai_element"],element,@[b[3],b[4],b[5]]);
            launch(e,(element.n+255)/256,1,256);
        }
    }
    [e endEncoding];[cb commit];[cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted)fprintf(stderr,"%s\n",cb.error.localizedDescription.UTF8String);
    need(cb.status==MTLCommandBufferStatusCompleted,"GPU completion");
    return (cb.GPUEndTime-cb.GPUStartTime)*1e6/repeats;
}
static void fill_weights(id<MTLBuffer> b,Args a,unsigned seed,bool gate) {
    uint8_t *w=data(b);
    for(unsigned r=0;r<a.rows;r++)for(unsigned block=0;block<a.cols/128;block++) {
        uint8_t *p=w+(size_t)r*a.row_bytes+block*34;
        uint16_t scale=0x2000+(hash(r*17713+block*23+seed)&0xfff);
        if((r+block)%7==0)scale|=0x8000;
        if((r+block)%19==0)scale=0;
        memcpy(p,&scale,2);
        for(unsigned j=0;j<32;j++)
            p[j+2]=(gate?r==0:r==a.rows-1)?0x55:(uint8_t)hash(r*2777+block*9391+j*13+seed);
    }
}
static int compare_double(const void *a,const void *b) {
    const double x=*(const double *)a,y=*(const double *)b;return (x>y)-(x<y);
}
static void fixture(id<MTLDevice> dev,id<MTLCommandQueue> queue,NSDictionary *pipelines,
                    unsigned rows,unsigned cols,unsigned count,unsigned seed,bool timing) {
    @autoreleasepool {
        const Args a={.n=count,.rows=rows,.cols=cols,.type=142,.row_bytes=cols/128*34};
        const size_t outputs=(size_t)count*rows;
        NSArray<id<MTLBuffer>> *b=@[buffer(dev,(size_t)rows*a.row_bytes),buffer(dev,(size_t)rows*a.row_bytes),
            buffer(dev,(size_t)count*cols*4),buffer(dev,outputs*4),buffer(dev,outputs*4),buffer(dev,outputs*4)];
        fill_weights(b[0],a,seed,true);fill_weights(b[1],a,seed+29,false);
        float *input=data(b[2]);
        for(size_t i=0;i<(size_t)count*cols;i++)input[i]=(float)(int32_t)hash((uint32_t)i+seed+42)/2147483648.0f*.35f;
        // Include a completely zero activation token and zero projection rows.
        if(count>2)memset(input+cols,0,cols*4);
        for(unsigned i=3;i<6;i++)poison(b[i]);
        run(queue,pipelines,a,b,false,1);
        float *reference=malloc(outputs*4);need(reference!=NULL,"reference allocation");
        memcpy(reference,data(b[5]),outputs*4);
        for(unsigned i=3;i<6;i++)poison(b[i]);
        run(queue,pipelines,a,b,true,1);
        const float *actual=data(b[5]);
        for(size_t i=0;i<outputs;i++) {
            if(!isfinite(actual[i])||!isfinite(reference[i])||memcmp(actual+i,reference+i,4)) {
                fprintf(stderr,"M%u K%u T%u seed%u index%zu got%.9g expected%.9g\n",rows,cols,count,seed,i,actual[i],reference[i]);
                need(false,"bitwise tiled projection/SiLU parity");
            }
        }
        for(id<MTLBuffer> v in b)guards(v);
        if(timing) {
            double times[2][ROUNDS];
            for(unsigned v=0;v<2;v++)run(queue,pipelines,a,b,v,2);
            for(unsigned trial=0;trial<ROUNDS;trial++)for(unsigned j=0;j<2;j++) {
                const unsigned v=(trial+j)%2;times[v][trial]=run(queue,pipelines,a,b,v,3);
            }
            for(unsigned v=0;v<2;v++)qsort(times[v],ROUNDS,sizeof(double),compare_double);
            const double before=(times[0][2]+times[0][3])*.5,after=(times[1][2]+times[1][3])*.5;
            printf("BENCH M%u K%u T%u separate_us%.3f fused_us%.3f speedup%.4f exact=1\n",rows,cols,count,before,after,before/after);
        } else printf("PASS M%u K%u T%u seed%u count%zu exact=1\n",rows,cols,count,seed,outputs);
        fflush(stdout);free(reference);
    }
}
int main(int argc,char **argv) {
    @autoreleasepool {
        const bool timing=argc==2&&!strcmp(argv[1],"--bench");
        need(argc==1||timing,"usage test_bonsai_prefill_pair [--bench]");
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice();need(dev!=nil,"Metal device");
        id<MTLCommandQueue> queue=[dev newCommandQueue];need(queue!=nil,"command queue");
        NSString *source=[[NSString alloc] initWithBytes:metal_bonsai_metal length:metal_bonsai_metal_len encoding:NSUTF8StringEncoding];
        MTLCompileOptions *options=[MTLCompileOptions new];
        if(@available(macOS 15.0,*))options.mathMode=MTLMathModeSafe;
        else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled=NO;
#pragma clang diagnostic pop
        }
        NSError *error=nil;
        id<MTLLibrary> library=[dev newLibraryWithSource:source options:options error:&error];
        if(!library)fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);
        need(library!=nil,"shader library");
        NSMutableDictionary *pipelines=[NSMutableDictionary dictionary];
        for(NSString *name in @[@"bonsai_mm_pq2_tiled",@"bonsai_element",@"bonsai_mm_pq2_gate_up_tiled"]) {
            id<MTLComputePipelineState> p=[dev newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
            need(p!=nil,"pipeline");pipelines[name]=p;
            printf("PIPELINE %s max_threads%lu static_tg_bytes%lu\n",name.UTF8String,(unsigned long)p.maxTotalThreadsPerThreadgroup,(unsigned long)p.staticThreadgroupMemoryLength);
        }
        printf("DEVICE %s math=safe\n",dev.name.UTF8String);
        const unsigned rowCounts[]={1,9,31,32,33,63,64,65,129};
        const unsigned tokens[]={1,3,16,17,19,31,32,33,65};
        unsigned seed=1;
        for(unsigned r=0;r<sizeof(rowCounts)/sizeof(rowCounts[0]);r++)
            for(unsigned t=0;t<sizeof(tokens)/sizeof(tokens[0]);t++)
                fixture(dev,queue,pipelines,rowCounts[r],r%2?128:384,tokens[t],seed++,false);
        fixture(dev,queue,pipelines,129,5120,33,204,false);
        fixture(dev,queue,pipelines,17408,5120,19,310,false);
        fixture(dev,queue,pipelines,17408,5120,32,311,false);
        if(timing)for(unsigned n=16;n<=128;n*=2)
            fixture(dev,queue,pipelines,17408,5120,n,400+n,true);
        puts("PQ2 prefill gate/up/SiLU exact fusion: PASS");return 0;
    }
}
