/* PQ2 tiled prefill against the independent FP64 dot-product oracle.
 * make tests/test_bonsai_mma
 * ./tests/test_bonsai_mma [--bench]
 * Run correctness with Metal validation; run timing without validation.
 * The fixed tolerances cover this deterministic fixture, not model quality.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../metal/bonsai.metal.inc"
#include "../bonsai_quant.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t n,rows,cols,type,row_bytes,pos,heads,kvheads,dim,rot,width,mode,groups;
    float eps,base;
} Args;
_Static_assert(sizeof(Args)==60,"Bonsai arguments");
static const NSUInteger guard_bytes=64;

static void need(bool ok,const char *message) {
    if (!ok) { fprintf(stderr,"FAIL: %s\n",message); exit(1); }
}
static id<MTLBuffer> buffer(id<MTLDevice> dev,size_t bytes) {
    id<MTLBuffer> b=[dev newBufferWithLength:bytes+2*guard_bytes options:MTLResourceStorageModeShared];
    need(b!=nil,"allocate guarded buffer"); memset(b.contents,0xcd,b.length); return b;
}
static void guards(id<MTLBuffer>b) {
    const uint8_t*p=b.contents;
    for (NSUInteger i=0;i<guard_bytes;++i)
        need(p[i]==0xcd && p[b.length-1-i]==0xcd,"buffer guard corruption");
}
static void *data(id<MTLBuffer>b) { return (uint8_t*)b.contents+guard_bytes; }

static double run(id<MTLCommandQueue>queue,id<MTLComputePipelineState>pipeline,Args a,
                  id<MTLBuffer>w,id<MTLBuffer>x,id<MTLBuffer>y,uint32_t tile,uint32_t repeats) {
    id<MTLCommandBuffer>cb=[queue commandBuffer]; id<MTLComputeCommandEncoder>enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline]; [enc setBytes:&a length:sizeof(a) atIndex:0];
    [enc setBuffer:w offset:guard_bytes atIndex:1];
    for (uint32_t r=0;r<repeats;++r) {
        [enc setBuffer:x offset:guard_bytes atIndex:2]; [enc setBuffer:y offset:guard_bytes atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((a.rows+(tile==4?3:63))/(tile==4?4:64),(a.n+tile-1)/tile,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    }
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status!=MTLCommandBufferStatusCompleted) fprintf(stderr,"%s\n",cb.error.localizedDescription.UTF8String);
    need(cb.status==MTLCommandBufferStatusCompleted,"GPU completion");
    return (cb.GPUEndTime-cb.GPUStartTime)*1e6/repeats;
}

static void weights_fill(id<MTLBuffer>w,Args a) {
    uint8_t*bytes=data(w);
    if (a.type==0 || a.type==1 || a.type==30) {
        for (size_t i=0;i<(size_t)a.rows*a.cols;++i) {
            float v=sinf((float)(i%1048573)*.071f)*.1f;
            if (a.type==0) ((float*)bytes)[i]=v;
            else if (a.type==1) ((_Float16*)bytes)[i]=(_Float16)v;
            else { uint32_t bits; memcpy(&bits,&v,4); ((uint16_t*)bytes)[i]=bits>>16; }
        }
        return;
    }
    const uint32_t width=a.type==8?32:128, block_bytes=a.type==143?28:34;
    const uint16_t scales[]={0x0000,0x8000,0x2800,0xac00,0x3000,0x0400,0x0001,0x3800};
    for (uint32_t r=0;r<a.rows;++r) for (uint32_t b=0;b<a.cols/width;++b) {
        uint8_t*p=bytes+(size_t)r*a.row_bytes+b*block_bytes;
        for (uint32_t j=0;j<block_bytes;++j) p[j]=(uint8_t)(r*73+b*91+j*17);
        uint16_t d=scales[(r+b)%8]; uint32_t off=a.type==143?26:0;
        p[off]=d&255; p[off+1]=d>>8;
    }
}
static void poison(id<MTLBuffer>b,size_t count) {uint32_t*p=data(b);for(size_t i=0;i<count;i++)p[i]=0x7fc00000u;}
static int compare_double(const void*a,const void*b) {
    double x=*(const double*)a,y=*(const double*)b; return (x>y)-(x<y);
}

static void test_shape(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                       NSArray *pipelines, uint32_t rows, uint32_t cols,
                       uint32_t tokens, bool timing) {
    Args a={.n=tokens,.rows=rows,.cols=cols,.type=142,.row_bytes=cols/128*34};
    const size_t count=(size_t)rows*tokens;
    id<MTLBuffer> w=buffer(dev,(size_t)rows*a.row_bytes);
    id<MTLBuffer> x=buffer(dev,(size_t)tokens*cols*4);
    id<MTLBuffer> reference=buffer(dev,count*4), out=buffer(dev,count*4);
    weights_fill(w,a);
    float *input=data(x);
    for (uint32_t t=0;t<tokens;t++) for (uint32_t k=0;k<cols;k++)
        input[(size_t)t*cols+k]=sinf((k+t*19)*.031f)*.25f;
    poison(reference,count);
    poison(out,count);
    run(queue,pipelines[0],a,w,x,reference,4,1);
    run(queue,pipelines[1],a,w,x,out,32,1);
    const float *r=data(reference), *y=data(out);
    double delta_ss=0,delta_max=0;
    for (size_t i=0;i<count;i++) {
        need(isfinite(r[i]) && isfinite(y[i]),"finite complete output after NaN poisoning");
        const double d=(double)y[i]-r[i];
        delta_ss+=d*d;delta_max=fmax(delta_max,fabs(d));
    }
    double error_ss=0,reference_ss=0,expected_ss=0,error_max=0;
    size_t samples=0;
    float *decoded=malloc(cols*sizeof(float));
    need(decoded!=NULL,"oracle row allocation");
    // Check every row for small tail fixtures, 17 distributed rows for full FFN.
    const uint32_t sampled_rows=rows<=129?rows:17;
    for (uint32_t ri=0;ri<sampled_rows;ri++) {
        const uint32_t row=rows<=129?ri:(uint32_t)(((uint64_t)ri*2654435761u)%rows);
        need(ds4_bonsai_dequantize_row(142,(const uint8_t*)data(w)+(size_t)row*a.row_bytes,
                                      decoded,cols),"oracle row dequantization");
        for (uint32_t t=0;t<tokens;t++) {
            double expected=0;
            for (uint32_t k=0;k<cols;k++)
                expected+=(double)decoded[k]*input[(size_t)t*cols+k];
            const double d=y[(size_t)t*rows+row]-expected;
            const double rd=r[(size_t)t*rows+row]-expected;
            const double limit=2e-4+2e-5*fabs(expected);
            if (fabs(d)>limit) {
                fprintf(stderr,"FP64 tolerance exceeded M%u K%u T%u row%u token%u: %.9g > %.9g\n",
                        rows,cols,tokens,row,t,fabs(d),limit);
                exit(1);
            }
            error_ss+=d*d;reference_ss+=rd*rd;expected_ss+=expected*expected;
            error_max=fmax(error_max,fabs(d));samples++;
        }
    }
    free(decoded);
    const double relative_rmse=sqrt(error_ss/fmax(expected_ss,1e-30));
    need(relative_rmse<=1e-5,"relative RMS FP64 tolerance");
    guards(w);guards(x);guards(reference);guards(out);
    printf("PASS PQ2 M=%u K=%u T=%u oracle_rmse=%.9g oracle_max=%.9g relative_rmse=%.9g "
           "serial_rmse=%.9g difference_rmse=%.9g difference_max=%.9g\n",
           rows,cols,tokens,sqrt(error_ss/samples),error_max,relative_rmse,
           sqrt(reference_ss/samples),sqrt(delta_ss/count),delta_max);
    if (timing) {
        double times[2][6];
        const uint32_t repeats=tokens>=64?1:3;
        for (uint32_t trial=0;trial<6;trial++) for (uint32_t j=0;j<2;j++) {
            const uint32_t v=trial%2?1-j:j;
            poison(out,count);
            times[v][trial]=run(queue,pipelines[v],a,w,x,out,v?32:4,repeats);
            for (size_t i=0;i<count;i++) need(isfinite(((float*)data(out))[i]),"complete timed output");
            guards(out);
        }
        for (uint32_t v=0;v<2;v++) qsort(times[v],6,sizeof(double),compare_double);
        const double serial_us=(times[0][2]+times[0][3])*.5;
        const double tiled_us=(times[1][2]+times[1][3])*.5;
        printf("BENCH M=%u K=%u T=%u mm4_us=%.3f tiled_us=%.3f speedup=%.3f\n",
               rows,cols,tokens,serial_us,tiled_us,serial_us/tiled_us);
    }
    fflush(stdout);
}

int main(int argc,char **argv) { @autoreleasepool {
    const bool timing=argc==2 && !strcmp(argv[1],"--bench");
    need(argc==1 || timing,"usage: test_bonsai_mma [--bench]");
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice();need(dev!=nil,"Metal device");
    id<MTLCommandQueue> queue=[dev newCommandQueue];need(queue!=nil,"Metal queue");
    NSString *source=[[NSString alloc] initWithBytes:metal_bonsai_metal
                        length:metal_bonsai_metal_len encoding:NSUTF8StringEncoding];
    MTLCompileOptions *options=[MTLCompileOptions new];
    if (@available(macOS 15.0,*)) options.mathMode=MTLMathModeSafe;
    else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled=NO;
#pragma clang diagnostic pop
    }
    NSError *error=nil;
    id<MTLLibrary> library=[dev newLibraryWithSource:source options:options error:&error];
    if (!library) fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);
    need(library!=nil,"shader library");
    NSMutableArray *pipelines=[NSMutableArray array];
    for (NSString *name in @[@"bonsai_mm",@"bonsai_mm_pq2_tiled"]) {
        id<MTLComputePipelineState> pipeline=[dev newComputePipelineStateWithFunction:
                                               [library newFunctionWithName:name] error:&error];
        if (!pipeline) fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);
        need(pipeline!=nil && pipeline.maxTotalThreadsPerThreadgroup>=128,"pipeline");
        [pipelines addObject:pipeline];
    }
    puts("FP64 limits: absolute 2e-4 + relative 2e-5 per sampled output; relative RMSE <= 1e-5.");
    const uint32_t counts[]={1,3,17,31,32,33,64,128};
    for (uint32_t i=0;i<sizeof(counts)/sizeof(counts[0]);i++)
        test_shape(dev,queue,pipelines,65,384,counts[i],false);
    test_shape(dev,queue,pipelines,9,128,3,false);
    test_shape(dev,queue,pipelines,129,384,33,false);
    test_shape(dev,queue,pipelines,129,384,65,false);
    const uint32_t shapes[][2]={{17408,5120},{5120,17408}};
    for (uint32_t s=0;s<2;s++) { @autoreleasepool {
        test_shape(dev,queue,pipelines,shapes[s][0],shapes[s][1],32,timing);
        if (timing) test_shape(dev,queue,pipelines,shapes[s][0],shapes[s][1],128,true);
    } }
    puts("Bonsai PQ2 tiled FP32 prefill tests: PASS");
    return 0;
} }
