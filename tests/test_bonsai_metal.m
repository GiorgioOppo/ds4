/* Standalone codec/rotation parity tests; no model file is needed.
 * clang -O2 -Wall -Wextra -fobjc-arc tests/test_bonsai_metal.m \
 *   -framework Foundation -framework Metal -o /tmp/test_bonsai_metal
 * Run with MTL_DEBUG_LAYER=1 for Metal API validation. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../bonsai_quant.h"
#include "../metal/bonsai.metal.inc"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t n,rows,cols,type,row_bytes,pos,heads,kvheads;
    uint32_t dim,rot,width,mode,groups;
    float eps,base;
} Args;
_Static_assert(sizeof(Args)==60,"shader arguments");

static void need(bool ok,const char *message) {
    if (!ok) { fprintf(stderr,"FAIL: %s\n",message); exit(1); }
}
static id<MTLBuffer> buffer(id<MTLDevice> dev,size_t bytes) {
    id<MTLBuffer> b=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    need(b!=nil,"allocate buffer"); return b;
}
static void run_grid(id<MTLCommandQueue> queue,id<MTLLibrary> library,NSString *name,Args a,
                     NSArray<id<MTLBuffer>> *buffers,MTLSize grid,NSUInteger threads) {
    NSError *error=nil;
    id<MTLComputePipelineState> pipeline=[queue.device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
    if (!pipeline) fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);
    need(pipeline!=nil,"compile pipeline");
    id<MTLCommandBuffer> cb=[queue commandBuffer];
    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline]; [enc setBytes:&a length:sizeof(a) atIndex:0];
    for (NSUInteger i=0;i<buffers.count;++i) [enc setBuffer:buffers[i] offset:0 atIndex:i+1];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status!=MTLCommandBufferStatusCompleted) fprintf(stderr,"%s\n",cb.error.localizedDescription.UTF8String);
    need(cb.status==MTLCommandBufferStatusCompleted,"kernel execution");
}
static void run(id<MTLCommandQueue> queue,id<MTLLibrary> library,NSString *name,Args a,
                NSArray<id<MTLBuffer>> *buffers,NSUInteger groups,NSUInteger threads) {
    run_grid(queue,library,name,a,buffers,MTLSizeMake(groups,1,1),threads);
}
static void close_array(const float *actual,const float *expected,size_t n,float atol,const char *name) {
    float maxerror=0;
    for (size_t i=0;i<n;++i) {
        float error=fabsf(actual[i]-expected[i]);
        if (!isfinite(actual[i]) || error>atol) {
            fprintf(stderr,"%s[%zu]: got %.9g want %.9g, abs %.9g limit %.9g\n",name,i,actual[i],expected[i],error,atol);
            exit(1);
        }
        maxerror=fmaxf(maxerror,error);
    }
    printf("%-30s n=%zu max_abs=%.9g\n",name,n,maxerror);
}

static void test_quant(id<MTLDevice> dev,id<MTLCommandQueue> queue,id<MTLLibrary> library,uint32_t type,uint32_t cols) {
    const uint32_t rows=9, block_bytes=type==142?34:28, stride=cols/128*block_bytes;
    id<MTLBuffer> weights=buffer(dev,(size_t)rows*stride), input=buffer(dev,cols*4), output=buffer(dev,MAX(rows,cols)*4);
    uint8_t *w=weights.contents;
    for (uint32_t r=0;r<rows;++r) for (uint32_t b=0;b<cols/128;++b) {
        uint8_t *p=w+(size_t)r*stride+b*block_bytes;
        for (uint32_t j=0;j<block_bytes;++j) p[j]=(uint8_t)(r*73+b*91+j*17);
        const uint16_t scales[]={0x3800,0x3c00,0xc000,0x0000};
        uint16_t h=scales[(r+b)%4]; uint32_t off=type==142?0:26;
        p[off]=h&255u; p[off+1]=h>>8;
    }
    float *x=input.contents, *decoded=malloc(cols*sizeof(float)), expected[9];
    need(decoded!=NULL,"row oracle");
    for (uint32_t j=0;j<cols;++j) x[j]=sinf((float)j*0.31f)*0.1f;
    for (uint32_t r=0;r<rows;++r) {
        need(ds4_bonsai_dequantize_row(type,w+(size_t)r*stride,decoded,cols),"decode oracle row");
        double sum=0; for (uint32_t j=0;j<cols;++j) sum+=(double)decoded[j]*x[j];
        expected[r]=(float)sum;
    }
    Args a={.rows=rows,.cols=cols,.type=type,.row_bytes=stride};
    run(queue,library,@"bonsai_mv",a,@[weights,input,output],(rows+3)/4,128);
    char label[80]; snprintf(label,sizeof(label),"type%u GEMV cols%u tail9",type,cols);
    close_array(output.contents,expected,rows,2e-5f,label);
    for (uint32_t r=0;r<rows;++r) {
        a.pos=r;
        need(ds4_bonsai_dequantize_row(type,w+(size_t)r*stride,decoded,cols),"decode lookup oracle");
        run(queue,library,@"bonsai_embed",a,@[weights,output],(cols+255)/256,256);
        snprintf(label,sizeof(label),"type%u lookup row%u cols%u",type,r,cols);
        close_array(output.contents,decoded,cols,0.0f,label);
    }
    free(decoded);
}

static void test_transform(id<MTLDevice> dev,id<MTLCommandQueue> queue,id<MTLLibrary> library) {
    const uint32_t width=2048,heads=16,kh=2,dim=128;
    id<MTLBuffer> input=buffer(dev,width*4),signs=buffer(dev,width*4),output=buffer(dev,width*4);
    float *x=input.contents; int32_t *s=signs.contents;
    float *expected=malloc(width*4),*permuted=malloc(width*4);
    need(expected && permuted,"transform oracle");
    for (uint32_t i=0;i<width;++i) { x[i]=sinf(i*.023f)+cosf(i*.017f); s[i]=(i*13u%7u)<3?-1:1; }
    for (uint32_t inverse=0;inverse<2;++inverse) {
        need(ds4_bonsai_hadamard_transform(expected,x,s,width,inverse),"transform oracle");
        run(queue,library,@"bonsai_hadamard",(Args){.n=width,.mode=inverse},@[input,signs,output],width/1024,256);
        close_array(output.contents,expected,width,0.0f,inverse?"inverse Hadamard":"forward Hadamard");
    }
    for (uint32_t h=0;h<heads;++h) for (uint32_t d=0;d<dim;++d)
        permuted[((h%kh)*(heads/kh)+h/kh)*dim+d]=x[h*dim+d];
    need(ds4_bonsai_hadamard_forward(expected,permuted,s,width),"grouped oracle");
    run(queue,library,@"bonsai_hadamard",(Args){.n=width,.heads=heads,.dim=dim,.groups=kh},@[input,signs,output],width/1024,256);
    close_array(output.contents,expected,width,0.0f,"grouped Hadamard 16/2");
    free(expected); free(permuted);

    const uint32_t shapes[][3]={{5120,19,0},{6144,32,16},{17408,33,0}};
    for (unsigned shape=0;shape<sizeof(shapes)/sizeof(shapes[0]);++shape) {
        const uint32_t n=shapes[shape][0],count=shapes[shape][1],groups=shapes[shape][2];
        const size_t elements=(size_t)n*count;
        id<MTLBuffer> bx=buffer(dev,elements*4),bs=buffer(dev,n*4),by=buffer(dev,(elements+16)*4);
        float *inputRows=bx.contents,*outputRows=by.contents;
        int32_t *signTable=bs.contents;
        float *oracle=malloc(elements*4),*ordered=malloc(n*4);
        need(oracle && ordered,"batched transform oracle");
        for (uint32_t i=0;i<n;++i) signTable[i]=(i*17u%11u)<5 ? -1 : 1;
        for (size_t i=0;i<elements;++i) inputRows[i]=sinf((float)i*.029f)+cosf((float)i*.011f);
        for (uint32_t inverse=0;inverse<2;++inverse) {
            if (groups && inverse) continue;
            for (size_t i=0;i<elements+16;++i) outputRows[i]=12345.0f;
            for (uint32_t row=0;row<count;++row) {
                const float *src=inputRows+(size_t)row*n;
                if (groups) {
                    for (uint32_t head=0;head<48;++head) for (uint32_t d=0;d<128;++d)
                        ordered[((head%groups)*(48/groups)+head/groups)*128+d]=src[head*128+d];
                    src=ordered;
                }
                need(ds4_bonsai_hadamard_transform(oracle+(size_t)row*n,src,signTable,n,inverse),"batch Hadamard oracle");
            }
            run_grid(queue,library,@"bonsai_hadamard",(Args){.n=n,.mode=inverse,.groups=groups,.heads=48,.dim=128},
                     @[bx,bs,by],MTLSizeMake(n/1024,count,1),256);
            close_array(outputRows,oracle,elements,0.0f,"batched Hadamard");
            need(!memcmp(outputRows,oracle,elements*4),"batched Hadamard bit-exact");
            for (size_t i=elements;i<elements+16;++i) need(outputRows[i]==12345.0f,"batch output canary");
        }
        free(oracle);free(ordered);
    }
}

static void test_bf16(id<MTLDevice> dev,id<MTLCommandQueue> queue,id<MTLLibrary> library) {
    const uint32_t shapes[][2]={{9,384},{48,5120},{49,5120}};
    for (unsigned sh=0;sh<sizeof(shapes)/sizeof(shapes[0]);++sh) {
        const uint32_t rows=shapes[sh][0],cols=shapes[sh][1];
        id<MTLBuffer> weights=buffer(dev,(size_t)rows*cols*2),input=buffer(dev,cols*4);
        id<MTLBuffer> output=buffer(dev,rows*4),reference=buffer(dev,rows*4);
        uint16_t *w=weights.contents;
        float *x=input.contents,*expected=malloc(rows*4);
        need(expected!=NULL,"BF16 oracle allocation");
        for (uint32_t j=0;j<cols;++j) x[j]=sinf(j*.013f)*.1f;
        for (uint32_t r=0;r<rows;++r) {
            double sum=0;
            for (uint32_t j=0;j<cols;++j) {
                const float value=sinf((r*cols+j)*.153f);
                uint32_t bits; memcpy(&bits,&value,4);
                w[r*cols+j]=(uint16_t)(bits>>16);
                bits&=0xffff0000u;
                float decoded; memcpy(&decoded,&bits,4);
                sum+=(double)decoded*x[j];
            }
            expected[r]=(float)sum;
        }
        Args a={.rows=rows,.cols=cols,.type=30,.row_bytes=cols*2};
        run(queue,library,@"bonsai_mv_reference",a,@[weights,input,reference],(rows+3)/4,128);
        run(queue,library,@"bonsai_mv",a,@[weights,input,output],(rows+3)/4,128);
        need(!memcmp(output.contents,reference.contents,rows*4),"BF16 unchanged lane walk is bit-exact");
        close_array(output.contents,expected,rows,5e-5f,"BF16 GEMV vs double oracle");
        free(expected);
    }
}

int main(void) { @autoreleasepool {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice(); need(device!=nil,"Metal device");
    id<MTLCommandQueue> queue=[device newCommandQueue]; need(queue!=nil,"Metal queue");
    NSString *source=[[NSString alloc] initWithBytes:metal_bonsai_metal length:metal_bonsai_metal_len encoding:NSUTF8StringEncoding];
    // Frozen original generic loop: typed dispatch must retain its arithmetic.
    source=[source stringByAppendingString:@"\n"
        "kernel void bonsai_mv_reference(constant BonsaiArgs&a [[buffer(0)]],"
        "device const uchar*w [[buffer(1)]],device const float*x [[buffer(2)]],"
        "device float*out [[buffer(3)]],uint group [[threadgroup_position_in_grid]],"
        "ushort lane [[thread_index_in_simdgroup]],ushort sg [[simdgroup_index_in_threadgroup]]) {"
        "uint row=group*4u+sg;float sum=0;if(row<a.rows){"
        "device const uchar*wr=w+ulong(row)*a.row_bytes;"
        "for(uint k=lane;k<a.cols;k+=32u)sum+=bs_weight(wr,k,a.type)*x[k];}"
        "sum=simd_sum(sum);if(!lane&&row<a.rows)out[row]=sum;}\n"];
    MTLCompileOptions *options=[MTLCompileOptions new];
    if (@available(macOS 15.0,*)) options.mathMode=MTLMathModeSafe;
    else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled=NO;
#pragma clang diagnostic pop
    }
    NSError *error=nil; id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
    if (!library) fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);
    need(library!=nil,"compile shader library");
    test_quant(device,queue,library,142,384); test_quant(device,queue,library,143,384);
    test_quant(device,queue,library,142,1024); test_quant(device,queue,library,143,1024);
    test_transform(device,queue,library);
    test_bf16(device,queue,library);
    puts("Bonsai Metal codec and rotation tests passed");
    return 0;
} }
