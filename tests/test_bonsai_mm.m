/* Exact four-token prefill GEMM versus frozen serial GEMV. No model file.
 * make metal/bonsai.metal.inc
 * clang -O2 -Wall -Wextra -fobjc-arc tests/test_bonsai_mm.m \
 *   -framework Foundation -framework Metal -o /tmp/test_bonsai_mm
 * /tmp/test_bonsai_mm [--bench]  (run benchmarks without Metal validation)
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
static const NSUInteger guard_bytes=64;
// Frozen decode kernel: changes to production must not move this oracle.
static NSString *frozen_source=
@"kernel void frozen_bonsai_mv(constant BonsaiArgs &a [[buffer(0)]],\n"
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
"            for (uint k = lane; k < a.cols; k += 32u) sum += bs_weight(wr, k, a.type) * x[k];\n"
"        }\n"
"    }\n"
"    sum = simd_sum(sum);\n"
"    if (!lane && row < a.rows) out[row] = sum;\n"
"}\n"
"\n";

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
                  id<MTLBuffer>w,id<MTLBuffer>x,id<MTLBuffer>y,bool serial,uint32_t repeats) {
    id<MTLCommandBuffer>cb=[queue commandBuffer]; id<MTLComputeCommandEncoder>enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline]; [enc setBytes:&a length:sizeof(a) atIndex:0];
    [enc setBuffer:w offset:guard_bytes atIndex:1];
    for (uint32_t r=0;r<repeats;++r) {
        if (serial) {
            for (uint32_t t=0;t<a.n;++t) {
                [enc setBuffer:x offset:guard_bytes+(NSUInteger)t*a.cols*4 atIndex:2];
                [enc setBuffer:y offset:guard_bytes+(NSUInteger)t*a.rows*4 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((a.rows+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            }
        } else {
            [enc setBuffer:x offset:guard_bytes atIndex:2]; [enc setBuffer:y offset:guard_bytes atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((a.rows+3)/4,(a.n+3)/4,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        }
    }
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status!=MTLCommandBufferStatusCompleted) fprintf(stderr,"%s\n",cb.error.localizedDescription.UTF8String);
    need(cb.status==MTLCommandBufferStatusCompleted,"GPU completion");
    return (cb.GPUEndTime-cb.GPUStartTime)*1e6/repeats;
}
static uint32_t row_bytes(uint32_t type,uint32_t cols) {
    return type==142?cols/128*34:type==143?cols/128*28:type==8?cols/32*34:cols*(type==0?4:2);
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
static int compare_double(const void*a,const void*b) {
    double x=*(const double*)a,y=*(const double*)b; return (x>y)-(x<y);
}
static void test_shape(id<MTLDevice>dev,id<MTLCommandQueue>queue,id<MTLComputePipelineState>mv,
                       id<MTLComputePipelineState>mm,uint32_t type,uint32_t rows,uint32_t cols,
                       uint32_t tokens,bool timing) {
    Args a={.n=tokens,.rows=rows,.cols=cols,.type=type,.row_bytes=row_bytes(type,cols)};
    id<MTLBuffer>w=buffer(dev,(size_t)rows*a.row_bytes),x=buffer(dev,(size_t)tokens*cols*4);
    id<MTLBuffer>ref=buffer(dev,(size_t)tokens*rows*4),out=buffer(dev,(size_t)tokens*rows*4);
    weights_fill(w,a); float*input=data(x);
    for (uint32_t t=0;t<tokens;++t) for (uint32_t k=0;k<cols;++k)
        input[(size_t)t*cols+k]=sinf((k+t*19)*.031f)*.25f;
    run(queue,mv,a,w,x,ref,true,1); run(queue,mm,a,w,x,out,false,1);
    const float*r=data(ref),*y=data(out);
    for (size_t i=0;i<(size_t)tokens*rows;++i) {
        if (!isfinite(r[i]) || !isfinite(y[i]) || memcmp(r+i,y+i,4)) {
            fprintf(stderr,"type%u M%u K%u T%u index%zu got%.9g expected%.9g\n",type,rows,cols,tokens,i,y[i],r[i]);
            need(false,"frozen serial bitwise parity");
        }
    }
    guards(w); guards(x); guards(ref); guards(out);
    if (timing) {
        double times[2][8]; const uint32_t repeats=tokens>=16?2:5;
        for (uint32_t trial=0;trial<8;++trial) for (uint32_t j=0;j<2;++j) {
            uint32_t v=trial%2?1-j:j;
            times[v][trial]=run(queue,v?mm:mv,a,w,x,out,!v,repeats);
        }
        for (uint32_t v=0;v<2;++v) qsort(times[v],8,sizeof(double),compare_double);
        double serial=(times[0][3]+times[0][4])/2,batch=(times[1][3]+times[1][4])/2;
        printf("BENCH type=%u M=%u K=%u T=%u serial_us=%.3f mm_us=%.3f speedup=%.3f exact=1\n",type,rows,cols,tokens,serial,batch,serial/batch);
    } else printf("PASS type=%u M=%u K=%u T=%u exact=1\n",type,rows,cols,tokens);
    fflush(stdout);
}

int main(int argc,char**argv) { @autoreleasepool {
    const bool timing=argc==2 && !strcmp(argv[1],"--bench");
    need(argc==1 || timing,"usage: test_bonsai_mm [--bench]");
    id<MTLDevice>dev=MTLCreateSystemDefaultDevice(); need(dev!=nil,"Metal device");
    id<MTLCommandQueue>queue=[dev newCommandQueue]; need(queue!=nil,"queue");
    NSString*source=[[NSString alloc]initWithBytes:metal_bonsai_metal length:metal_bonsai_metal_len encoding:NSUTF8StringEncoding];
    source=[source stringByAppendingString:frozen_source];
    MTLCompileOptions*options=[MTLCompileOptions new];
    if (@available(macOS 15.0,*)) options.mathMode=MTLMathModeSafe;
    else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled=NO;
#pragma clang diagnostic pop
    }
    NSError*error=nil; id<MTLLibrary>library=[dev newLibraryWithSource:source options:options error:&error];
    if (!library) fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);
    need(library!=nil,"shader library");
    id<MTLComputePipelineState>mv=[dev newComputePipelineStateWithFunction:[library newFunctionWithName:@"frozen_bonsai_mv"] error:&error];
    id<MTLComputePipelineState>mm=[dev newComputePipelineStateWithFunction:[library newFunctionWithName:@"bonsai_mm"] error:&error];
    need(mv && mm,"pipelines");
    const uint32_t types[]={0,1,30,8,142,143},counts[]={1,2,3,4,5,7,8,15,31,32,33,64};
    for (size_t f=0;f<sizeof(types)/sizeof(types[0]);++f)
        for (size_t t=0;t<sizeof(counts)/sizeof(counts[0]);++t) {@autoreleasepool {
            uint32_t k=types[f]==8?96:types[f]>=142?384:47;
            test_shape(dev,queue,mv,mm,types[f],9,k,counts[t],false);
        }}
    const uint32_t shapes[][2]={{17408,5120},{5120,17408}},full_counts[]={1,4,16,64};
    for (size_t s=0;s<2;++s) for (size_t t=0;t<4;++t) {@autoreleasepool {
        test_shape(dev,queue,mv,mm,142,shapes[s][0],shapes[s][1],full_counts[t],timing);
    }}
    puts("PASS Bonsai MM: frozen serial parity, token/row/K tails, guards, all storage types");
    return 0;
} }
