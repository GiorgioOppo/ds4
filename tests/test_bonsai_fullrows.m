/* PQ2 full-row specialization: frozen pre-change shader reference, complete
 * output bit parity, independent FP64 samples, NaN poisoning and canaries.
 * make tests/test_bonsai_fullrows
 * ./tests/test_bonsai_fullrows [--bench]
 * Use Metal validation for correctness, without validation for benchmark.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../bonsai_quant.h"
#include "../metal/bonsai.metal.inc"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct {uint32_t n,rows,cols,type,row_bytes,pos,heads,kvheads,dim,rot,width,mode,groups;float eps,base;} Args;
_Static_assert(sizeof(Args)==60,"Args");
static void need(bool b,const char*m){if(!b){fprintf(stderr,"FAIL %s\n",m);exit(1);}}
static id<MTLBuffer> buf(id<MTLDevice>d,size_t n){id<MTLBuffer>b=[d newBufferWithLength:n+128 options:MTLResourceStorageModeShared];need(b!=nil,"buffer");memset(b.contents,0xcd,b.length);return b;}
static void*data(id<MTLBuffer>b){return(uint8_t*)b.contents+64;}
static void guards(id<MTLBuffer>b){const uint8_t*p=b.contents;for(int i=0;i<64;i++)need(p[i]==0xcd&&p[b.length-1-i]==0xcd,"canary");}
static void poison(id<MTLBuffer>b,size_t n){uint32_t*p=data(b);for(size_t i=0;i<n;i++)p[i]=0x7fc00000u;}
static int cmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return(x>y)-(x<y);}
static NSString *reference_source =
@"kernel void reference_bonsai_pq2_mv(constant BonsaiArgs &a [[buffer(0)]],\n"
"                          device const uchar *weights [[buffer(1)]],\n"
"                          device const float *x [[buffer(2)]], device float *out [[buffer(3)]],\n"
"                          uint group [[threadgroup_position_in_grid]],\n"
"                          ushort lane [[thread_index_in_simdgroup]],\n"
"                          ushort sg [[simdgroup_index_in_threadgroup]]) {\n"
"    constexpr uint NR = 8;\n"
"    const uint first_row = (group * 2u + sg) * NR;\n"
"    const uint stripe = lane / 8u, first_col = (lane % 8u) * 16u;\n"
"    float sums[NR];\n"
"#pragma unroll\n"
"    for (uint row = 0; row < NR; ++row) sums[row] = 0.0f;\n"
"\n"
"    for (uint block = stripe; block < a.cols / 128u; block += 4u) {\n"
"        float coeff[16];\n"
"        device const packed_float4 *input =\n"
"            (device const packed_float4 *)(x + block * 128u + first_col);\n"
"#pragma unroll\n"
"        for (uint j = 0; j < 4; ++j) {\n"
"            const float4 v = float4(input[j]);\n"
"            coeff[4u*j] = v.w - 4.0f * v.z;\n"
"            coeff[4u*j+1u] = v.z - 4.0f * v.y;\n"
"            coeff[4u*j+2u] = v.y - 4.0f * v.x;\n"
"            coeff[4u*j+3u] = v.x;\n"
"        }\n"
"#pragma unroll\n"
"        for (uint row = 0; row < NR; ++row) {\n"
"            if (first_row + row >= a.rows) continue;\n"
"            device const uchar *p = weights + ulong(first_row + row) * a.row_bytes + block * 34u;\n"
"            float partial = 0.0f;\n"
"#pragma unroll\n"
"            for (uint j = 0; j < 4; ++j) {\n"
"                const float c = float(p[2u + first_col / 4u + j]) - 85.0f;\n"
"                partial += floor(fma(c, 1.0f/64.0f, 21.0f/64.0f)) * coeff[4u*j];\n"
"                partial += floor(fma(c, 1.0f/16.0f, 5.0f/16.0f)) * coeff[4u*j+1u];\n"
"                partial += floor(fma(c, 1.0f/4.0f, 1.0f/4.0f)) * coeff[4u*j+2u];\n"
"                partial += c * coeff[4u*j+3u];\n"
"            }\n"
"            sums[row] += float(*(device const half *)p) * partial;\n"
"        }\n"
"    }\n"
"#pragma unroll\n"
"    for (uint row = 0; row < NR; ++row) {\n"
"        const float total = simd_sum(sums[row]);\n"
"        if (!lane && first_row + row < a.rows) out[first_row + row] = total;\n"
"    }\n"
"}\n"
"\n"
"kernel void reference_bonsai_pq2_gate_up(constant BonsaiArgs &a [[buffer(0)]],\n"
"    device const uchar *gate_w [[buffer(1)]], device const uchar *up_w [[buffer(2)]],\n"
"    device const float *x [[buffer(3)]], device float *mid [[buffer(4)]],\n"
"    uint group [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],\n"
"    ushort sg [[simdgroup_index_in_threadgroup]]) {\n"
"    constexpr uint NR=4;\n"
"    const uint first_row=(group*2u+sg)*NR;\n"
"    const uint stripe=lane/8u, first_col=(lane%8u)*16u;\n"
"    float sums_g[NR],sums_u[NR];\n"
"#pragma unroll\n"
"    for(uint r=0;r<NR;++r) {sums_g[r]=0.0f;sums_u[r]=0.0f;}\n"
"    for(uint block=stripe;block<a.cols/128u;block+=4u) {\n"
"        float coeff[16];\n"
"        device const packed_float4 *input=(device const packed_float4 *)(x+block*128u+first_col);\n"
"#pragma unroll\n"
"        for(uint j=0;j<4;++j) {\n"
"            const float4 v=float4(input[j]);\n"
"            coeff[4u*j]=v.w-4.0f*v.z;\n"
"            coeff[4u*j+1u]=v.z-4.0f*v.y;\n"
"            coeff[4u*j+2u]=v.y-4.0f*v.x;\n"
"            coeff[4u*j+3u]=v.x;\n"
"        }\n"
"#pragma unroll\n"
"        for(uint r=0;r<NR;++r) {\n"
"            if(first_row+r>=a.rows)continue;\n"
"            ulong offset=ulong(first_row+r)*a.row_bytes+block*34u;\n"
"            device const uchar *g=gate_w+offset,*u=up_w+offset;\n"
"            float partial_g=0.0f,partial_u=0.0f;\n"
"#pragma unroll\n"
"            for(uint j=0;j<4;++j) {\n"
"                const float cg=float(g[2u+first_col/4u+j])-85.0f;\n"
"                const float cu=float(u[2u+first_col/4u+j])-85.0f;\n"
"                partial_g+=floor(fma(cg,1.0f/64.0f,21.0f/64.0f))*coeff[4u*j];\n"
"                partial_g+=floor(fma(cg,1.0f/16.0f,5.0f/16.0f))*coeff[4u*j+1u];\n"
"                partial_g+=floor(fma(cg,1.0f/4.0f,1.0f/4.0f))*coeff[4u*j+2u];\n"
"                partial_g+=cg*coeff[4u*j+3u];\n"
"                partial_u+=floor(fma(cu,1.0f/64.0f,21.0f/64.0f))*coeff[4u*j];\n"
"                partial_u+=floor(fma(cu,1.0f/16.0f,5.0f/16.0f))*coeff[4u*j+1u];\n"
"                partial_u+=floor(fma(cu,1.0f/4.0f,1.0f/4.0f))*coeff[4u*j+2u];\n"
"                partial_u+=cu*coeff[4u*j+3u];\n"
"            }\n"
"            sums_g[r]+=float(*(device const half *)g)*partial_g;\n"
"            sums_u[r]+=float(*(device const half *)u)*partial_u;\n"
"        }\n"
"    }\n"
"#pragma unroll\n"
"    for(uint r=0;r<NR;++r) {\n"
"        const float g=simd_sum(sums_g[r]),u=simd_sum(sums_u[r]);\n"
"        if(!lane&&first_row+r<a.rows) {\n"
"            mid[first_row+r]=(g/(1.0f+exp(-g)))*u;\n"
"        }\n"
"    }\n"
"}\n"
;
static uint selected(Args a,bool fused,uint v) {
    const uint group=fused?8u:16u;
    if (v==2 && a.rows%group) v=1;
    return (fused?3u:0u)+v;
}

static double run(id<MTLCommandQueue>q,NSArray*ps,Args a,id<MTLBuffer>g,id<MTLBuffer>u,id<MTLBuffer>x,id<MTLBuffer>y,bool fused,uint variant,uint repeats){
 id<MTLCommandBuffer>cb=[q commandBuffer];id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];[e setComputePipelineState:ps[selected(a,fused,variant)]];[e setBytes:&a length:sizeof(a) atIndex:0];[e setBuffer:g offset:64 atIndex:1];if(fused)[e setBuffer:u offset:64 atIndex:2];[e setBuffer:x offset:64 atIndex:fused?3:2];[e setBuffer:y offset:64 atIndex:fused?4:3];uint group=fused?8u:16u;for(uint i=0;i<repeats;i++)[e dispatchThreadgroups:MTLSizeMake((a.rows+group-1u)/group,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)fprintf(stderr,"%s\n",cb.error.localizedDescription.UTF8String);need(cb.status==MTLCommandBufferStatusCompleted,"GPU completion");return(cb.GPUEndTime-cb.GPUStartTime)*1e6/repeats;
}
static void fill(id<MTLBuffer>b,Args a,uint seed){uint8_t*w=data(b);const uint16_t scales[]={0x0000,0x8000,0x2800,0xac00,0x3000,0x0400,0x0001,0x3800};for(uint row=0;row<a.rows;row++)for(uint k=0;k<a.cols/128u;k++){uint8_t*z=w+(size_t)row*a.row_bytes+k*34u;uint16_t scale=scales[(row+k+seed)%8];memcpy(z,&scale,2);for(uint j=0;j<32;j++)z[2+j]=(uint8_t)(row*13+k*29+j*37+seed*97);}}
static void shape(id<MTLDevice>d,id<MTLCommandQueue>q,NSArray*ps,uint rows,uint cols,bool fused,bool timing){
 Args a={.n=1,.rows=rows,.cols=cols,.type=142,.row_bytes=cols/128u*34u};size_t bytes=(size_t)rows*a.row_bytes;
 id<MTLBuffer>g=buf(d,bytes),u=buf(d,bytes),x=buf(d,cols*4u),y=buf(d,rows*4u),ref=buf(d,rows*4u);fill(g,a,0);fill(u,a,1);float*input=data(x);for(uint k=0;k<cols;k++)input[k]=sinf(k*.013f)*.25f;
 poison(ref,rows);run(q,ps,a,g,u,x,ref,fused,0,1);float*rv=data(ref);
 for(uint v=0;v<3;v++){
  poison(y,rows);run(q,ps,a,g,u,x,y,fused,v,1);float*values=data(y);for(uint i=0;i<rows;i++)need(isfinite(values[i])&&isfinite(rv[i]),"finite output after poisoning");need(!memcmp(values,rv,rows*4u),"bit-exact full rows/template");
  float*dg=malloc(cols*4u),*du=malloc(cols*4u);need(dg&&du,"CPU oracle allocations");double ss=0,expected_ss=0,maxabs=0;uint ns=rows<33?rows:33;
  for(uint j=0;j<ns;j++){uint row=rows<33?j:(uint)(((uint64_t)j*2654435761u)%rows);need(ds4_bonsai_dequantize_row(142,(uint8_t*)data(g)+(size_t)row*a.row_bytes,dg,cols),"gate dequant");need(ds4_bonsai_dequantize_row(142,(uint8_t*)data(u)+(size_t)row*a.row_bytes,du,cols),"up dequant");double eg=0,eu=0;for(uint k=0;k<cols;k++){eg+=(double)dg[k]*input[k];eu+=(double)du[k]*input[k];}double expected=fused?(eg/(1.0+exp(-eg)))*eu:eg,err=values[row]-expected;need(fabs(err)<=(fused?4e-3:2e-4)+5e-5*fabs(expected),"FP64 tolerance");ss+=err*err;expected_ss+=expected*expected;maxabs=fmax(maxabs,fabs(err));}
  free(dg);free(du);need(sqrt(ss/fmax(expected_ss,1e-30))<=(fused?2e-5:1e-5),"FP64 relative RMS tolerance");guards(g);guards(u);guards(x);guards(y);guards(ref);printf("CHECK %s M%u K%u V%u PSO%u bitexact oracleRMSE%.9g oracleMAX%.9g\n",fused?"fused":"mv",rows,cols,v,selected(a,fused,v),sqrt(ss/ns),maxabs);fflush(stdout);
 }
 if (!timing) return;
 double times[3][8];uint repeats=rows>100000?5:rows>100?100:30;
 for(uint trial=0;trial<8;trial++)for(uint j=0;j<3;j++){uint v=trial%2?2-j:j;poison(y,rows);times[v][trial]=run(q,ps,a,g,u,x,y,fused,v,repeats);for(uint i=0;i<rows;i++)need(isfinite(((float*)data(y))[i]),"complete timed output");guards(y);}
 for(uint v=0;v<3;v++)qsort(times[v],8,sizeof(double),cmp);
 for(uint v=0;v<3;v++)printf("BENCH %s M%u K%u V%u median_us%.3f min_us%.3f max_us%.3f speedup%.4f\n",fused?"fused":"mv",rows,cols,v,(times[v][3]+times[v][4])*.5,times[v][0],times[v][7],(times[0][3]+times[0][4])/(times[v][3]+times[v][4]));fflush(stdout);
}
int main(int argc,char**argv){@autoreleasepool{
 bool timing=argc==2&&!strcmp(argv[1],"--bench");need(argc==1||timing,"usage: test_bonsai_fullrows [--bench]");id<MTLDevice>d=MTLCreateSystemDefaultDevice();need(d!=nil,"Metal device");id<MTLCommandQueue>q=[d newCommandQueue];NSError*error=nil;NSString*s=[[NSString alloc]initWithBytes:metal_bonsai_metal length:metal_bonsai_metal_len encoding:NSUTF8StringEncoding];s=[s stringByAppendingString:reference_source];MTLCompileOptions*o=[MTLCompileOptions new];if(@available(macOS 15.0,*))o.mathMode=MTLMathModeSafe;else{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
 o.fastMathEnabled=NO;
#pragma clang diagnostic pop
 }
 id<MTLLibrary>lib=[d newLibraryWithSource:s options:o error:&error];if(!lib)fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);need(lib!=nil,"shader library");NSMutableArray*ps=[NSMutableArray array];for(NSString*n in @[@"reference_bonsai_pq2_mv",@"bonsai_pq2_mv",@"bonsai_pq2_mv_full",@"reference_bonsai_pq2_gate_up",@"bonsai_pq2_gate_up",@"bonsai_pq2_gate_up_full"]){id<MTLComputePipelineState>p=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:n] error:&error];if(!p)fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);need(p!=nil,"pipeline");[ps addObject:p];}
 puts("SAFE FP32: template/fallback/full results must be bit-identical to frozen baseline; output poisoned, guarded buffers, FP64 samples.");
 const uint tails[][2]={{9,384},{8,512},{16,640},{17,5120}};for(uint i=0;i<4;i++)for(uint f=0;f<2;f++){@autoreleasepool{shape(d,q,ps,tails[i][0],tails[i][1],f,timing);}}
 shape(d,q,ps,17408,5120,true,timing);
 const uint projections[][2]={{5120,17408},{10240,5120},{6144,5120},{5120,6144},{12288,5120},{248320,5120}};
 for(uint i=0;i<sizeof(projections)/sizeof(projections[0]);i++){@autoreleasepool{shape(d,q,ps,projections[i][0],projections[i][1],false,timing);}}
 return 0;
}}
