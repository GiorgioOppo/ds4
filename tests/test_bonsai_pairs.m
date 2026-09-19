/* Exact decode fusions against separate production projections/activation.
 * No model is needed. Run correctness with Metal validation; --bench uses
 * eight alternating rounds and must run without another GPU workload.
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
_Static_assert(sizeof(Args)==60,"Bonsai arguments");
enum { GUARD=64, ROUNDS=8 };

static void need(bool ok,const char *message) {
    if (!ok) { fprintf(stderr,"FAIL %s\n",message); exit(1); }
}
static uint32_t hash(uint32_t x) {
    x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; return x^(x>>16);
}
static int compare_time(const void *a,const void *b) {
    const double x=*(const double *)a,y=*(const double *)b; return (x>y)-(x<y);
}
static id<MTLBuffer> buffer(id<MTLDevice> device,size_t bytes) {
    id<MTLBuffer> b=[device newBufferWithLength:bytes+2*GUARD options:MTLResourceStorageModeShared];
    need(b!=nil,"buffer allocation"); memset(b.contents,0xa5,b.length); return b;
}
static void *data(id<MTLBuffer> b) { return (uint8_t *)b.contents+GUARD; }
static void guards(id<MTLBuffer> b) {
    const uint8_t *p=b.contents;
    for (unsigned i=0;i<GUARD;++i)
        need(p[i]==0xa5 && p[b.length-GUARD+i]==0xa5,"buffer canary");
}
static void poison(id<MTLBuffer> b) { memset(data(b),0xff,b.length-2*GUARD); }
static void bind(id<MTLComputeCommandEncoder> e,id<MTLComputePipelineState> p,
                 Args a,NSArray<id<MTLBuffer>> *buffers) {
    [e setComputePipelineState:p]; [e setBytes:&a length:sizeof(a) atIndex:0];
    for (NSUInteger i=0;i<buffers.count;++i) [e setBuffer:buffers[i] offset:GUARD atIndex:i+1];
}
static void launch(id<MTLComputeCommandEncoder> e,unsigned groups,unsigned threads) {
    [e dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
}
static double run(id<MTLCommandQueue> queue,NSDictionary *pipelines,Args a,
                  NSArray<id<MTLBuffer>> *b,bool fused,unsigned repeat) {
    const bool bf16=a.type==30;
    id<MTLCommandBuffer> cb=[queue commandBuffer];
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    for (unsigned i=0;i<repeat;++i) {
        if (fused) {
            if (bf16) {
                bind(e,pipelines[@"bonsai_bf16_pair"],a,@[b[0],b[1],b[2],b[3],b[4]]);
                launch(e,(a.rows+3)/4,128);
            } else {
                bind(e,pipelines[@"bonsai_pq2_gate_up"],a,@[b[0],b[1],b[2],b[5]]);
                launch(e,(a.rows+7)/8,64);
            }
        } else {
            id<MTLComputePipelineState> p=pipelines[bf16?@"bonsai_mv":@"bonsai_pq2_mv"];
            bind(e,p,a,@[b[0],b[2],b[3]]); launch(e,(a.rows+(bf16?3:15))/(bf16?4:16),bf16?128:64);
            bind(e,p,a,@[b[1],b[2],b[4]]); launch(e,(a.rows+(bf16?3:15))/(bf16?4:16),bf16?128:64);
            if (!bf16) {
                bind(e,pipelines[@"bonsai_element"],(Args){.n=a.rows,.mode=1},@[b[3],b[4],b[5]]);
                launch(e,(a.rows+255)/256,256);
            }
        }
    }
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status!=MTLCommandBufferStatusCompleted) fprintf(stderr,"%s\n",cb.error.localizedDescription.UTF8String);
    need(cb.status==MTLCommandBufferStatusCompleted,"GPU completion");
    return (cb.GPUEndTime-cb.GPUStartTime)*1e6/repeat;
}
static void fill_weights(id<MTLBuffer> weight,Args a,unsigned seed,int basis) {
    uint8_t *w=data(weight);
    for (unsigned row=0;row<a.rows;++row) {
        if (a.type==30) {
            uint16_t *wr=(uint16_t *)(w+(size_t)row*a.row_bytes);
            for (unsigned k=0;k<a.cols;++k) {
                const float value=(float)(int32_t)hash(row*471+k+seed)/2147483648.0f*.1f;
                uint32_t bits; memcpy(&bits,&value,4); wr[k]=bits>>16;
            }
        } else {
            for (unsigned block=0;block<a.cols/128;++block) {
                uint8_t *p=w+(size_t)row*a.row_bytes+block*34;
                const uint16_t h=basis>=0?0x3800:0x2000+(hash(row*17713+block*23+seed)&0xfff);
                memcpy(p,&h,2);
                for (unsigned j=0;j<32;++j) {
                    if (basis>=0) p[2+j]=!block&&!j?(uint8_t)(seed?row^255:row):0x55;
                    else if (row==0 || row==3 || row==a.rows-1) p[2+j]=0x55;
                    else p[2+j]=(uint8_t)hash(row*2777+block*9391+j*13+seed);
                }
            }
        }
    }
}
static void equal(id<MTLBuffer> output,const float *reference,unsigned rows) {
    const float *actual=data(output);
    for (unsigned row=0;row<rows;++row) need(isfinite(actual[row]),"all rows written and finite");
    if (memcmp(actual,reference,rows*4)) {
        for (unsigned row=0;row<rows;++row) if (memcmp(actual+row,reference+row,4)) {
            fprintf(stderr,"row %u: %.9g != %.9g\n",row,actual[row],reference[row]); break;
        }
        need(false,"bitwise projection/activation parity");
    }
}
static void fixture(id<MTLDevice> device,id<MTLCommandQueue> queue,NSDictionary *pipelines,
                    unsigned rows,unsigned cols,bool bf16,int basis,bool timing) {
    @autoreleasepool {
        Args a={.rows=rows,.cols=cols,.type=bf16?30:142,.row_bytes=bf16?cols*2:cols/128*34};
        NSArray<id<MTLBuffer>> *b=@[buffer(device,(size_t)rows*a.row_bytes),buffer(device,(size_t)rows*a.row_bytes),
            buffer(device,cols*4),buffer(device,rows*4),buffer(device,rows*4),buffer(device,rows*4)];
        fill_weights(b[0],a,0,basis); fill_weights(b[1],a,29,basis);
        float *input=data(b[2]);
        for (unsigned k=0;k<cols;++k)
            input[k]=basis>=0?(k==(unsigned)basis?1:0):((float)(int32_t)hash(k+42)/2147483648.0f)*.35f;
        float *reference=malloc((size_t)rows*4*(bf16?2:1)); need(reference!=NULL,"reference allocation");
        for (unsigned i=3;i<6;++i) poison(b[i]);
        run(queue,pipelines,a,b,false,1);
        if (bf16) { memcpy(reference,data(b[3]),rows*4); memcpy(reference+rows,data(b[4]),rows*4); }
        else memcpy(reference,data(b[5]),rows*4);
        for (unsigned i=3;i<6;++i) poison(b[i]);
        run(queue,pipelines,a,b,true,1);
        if (bf16) { equal(b[3],reference,rows); equal(b[4],reference+rows,rows); }
        else equal(b[5],reference,rows);
        for (id<MTLBuffer> v in b) guards(v);
        if (timing) {
            double times[2][ROUNDS];
            for (unsigned v=0;v<2;++v) run(queue,pipelines,a,b,v,3);
            for (unsigned trial=0;trial<ROUNDS;++trial) for (unsigned j=0;j<2;++j) {
                const unsigned v=(trial+j)%2;
                times[v][trial]=run(queue,pipelines,a,b,v,20);
            }
            for (unsigned v=0;v<2;++v) qsort(times[v],ROUNDS,sizeof(double),compare_time);
            const double before=.5*(times[0][3]+times[0][4]),after=.5*(times[1][3]+times[1][4]);
            printf("%s,%u,%u,%.3f,%.3f,%.4f\n",bf16?"BF16 alpha/beta":"PQ2 gate/up",rows,cols,before,after,before/after);
        }
        free(reference);
    }
}
int main(int argc,char **argv) {
    @autoreleasepool {
        const bool timing=argc==2 && !strcmp(argv[1],"--bench");
        need(argc==1 || timing,"usage: test_bonsai_pairs [--bench]");
        id<MTLDevice> device=MTLCreateSystemDefaultDevice(); need(device!=nil,"Metal device");
        id<MTLCommandQueue> queue=[device newCommandQueue];
        NSString *source=[[NSString alloc] initWithBytes:metal_bonsai_metal length:metal_bonsai_metal_len encoding:NSUTF8StringEncoding];
        MTLCompileOptions *options=[MTLCompileOptions new];
        if (@available(macOS 15.0,*)) options.mathMode=MTLMathModeSafe;
        else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled=NO;
#pragma clang diagnostic pop
        }
        NSError *error=nil;
        id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
        if (!library) fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);
        need(library!=nil,"shader library");
        NSMutableDictionary *pipelines=[NSMutableDictionary dictionary];
        for (NSString *name in @[@"bonsai_mv",@"bonsai_pq2_mv",@"bonsai_element",@"bonsai_pq2_gate_up",@"bonsai_bf16_pair"]) {
            id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
            need(p!=nil,"pipeline"); pipelines[name]=p;
        }
        for (int basis=0;basis<4;++basis) fixture(device,queue,pipelines,256,384,false,basis,false);
        fixture(device,queue,pipelines,9,384,false,-1,false);
        fixture(device,queue,pipelines,17,17408,false,-1,false);
        fixture(device,queue,pipelines,4097,5120,false,-1,false);
        fixture(device,queue,pipelines,17408,5120,false,-1,timing);
        fixture(device,queue,pipelines,17,385,true,-1,false);
        fixture(device,queue,pipelines,48,5120,true,-1,timing);
        fixture(device,queue,pipelines,49,5120,true,-1,false);
        fprintf(stderr,"PASS Bonsai paired decode: exact gate/up activation and BF16 outputs, all byte bases, tails, poison and canaries.\n");
        return 0;
    }
}
