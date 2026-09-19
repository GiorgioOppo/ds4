/* PQ2 GEMV microbenchmark. No model file. Usage: bench [shader-source] [--bf16].
 * The generic baseline is embedded here so production changes remain measurable.
 * clang -O2 -fobjc-arc tests/bench_bonsai_pq2.m -framework Foundation -framework Metal -o /tmp/bench_bonsai_pq2
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint32_t n,rows,cols,type,row_bytes,pos,heads,kvheads,dim,rot,width,mode,groups; float eps,base; } Args;
_Static_assert(sizeof(Args)==60,"shader arguments");
static NSString *variants =
@"\nkernel void bench_baseline(constant BonsaiArgs &a [[buffer(0)]],device const uchar*w [[buffer(1)]],device const float*x [[buffer(2)]],device float*y [[buffer(3)]],uint tg [[threadgroup_position_in_grid]],ushort lane [[thread_index_in_simdgroup]],ushort sg [[simdgroup_index_in_threadgroup]]) {\n"
"uint row=tg*4+sg; float sum=0; if(row<a.rows) { device const uchar*wr=w+ulong(row)*a.row_bytes; for(uint k=lane;k<a.cols;k+=32) sum+=bs_weight(wr,k,a.type)*x[k]; } sum=simd_sum(sum); if(!lane&&row<a.rows)y[row]=sum; }\n";
static void need(bool b,const char*m) { if(!b) {fprintf(stderr,"FAIL %s\n",m);exit(1);} }
static int fcmp(const void*a,const void*b) {double x=*(const double*)a,y=*(const double*)b;return(x>y)-(x<y);}
static double run(id<MTLCommandQueue>q,id<MTLComputePipelineState>p,Args a,id<MTLBuffer>w,id<MTLBuffer>x,id<MTLBuffer>y,uint32_t nr,uint32_t repeats) {
 id<MTLCommandBuffer>cb=[q commandBuffer]; id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];
 [e setComputePipelineState:p]; [e setBytes:&a length:sizeof(a) atIndex:0]; [e setBuffer:w offset:0 atIndex:1]; [e setBuffer:x offset:0 atIndex:2]; [e setBuffer:y offset:0 atIndex:3];
 for(uint32_t r=0;r<repeats;r++) [e dispatchThreadgroups:MTLSizeMake((a.rows+4*nr-1)/(4*nr),1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
 [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
 if(cb.status!=MTLCommandBufferStatusCompleted)fprintf(stderr,"%s\n",cb.error.localizedDescription.UTF8String);
 need(cb.status==MTLCommandBufferStatusCompleted,"GPU completion");return(cb.GPUEndTime-cb.GPUStartTime)*1e6/repeats;
}
int main(int argc,char**argv) { @autoreleasepool {
 id<MTLDevice>d=MTLCreateSystemDefaultDevice();need(d!=nil,"Metal device");id<MTLCommandQueue>q=[d newCommandQueue];
 NSError*error=nil; NSString*path=argc>1?[NSString stringWithUTF8String:argv[1]]:@"metal/bonsai.metal";
 NSString*source=[NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];need(source!=nil,"shader file");source=[source stringByAppendingString:variants];
 MTLCompileOptions*o=[MTLCompileOptions new];
 if (@available(macOS 15.0,*)) o.mathMode=MTLMathModeSafe;
 else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  o.fastMathEnabled=NO;
#pragma clang diagnostic pop
 }
 id<MTLLibrary>lib=[d newLibraryWithSource:source options:o error:&error];if(!lib)fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);need(lib!=nil,"library");
 NSArray<NSString*>*names=@[@"bench_baseline",@"bonsai_mv"];
 const uint32_t nrs[]={1,1};NSMutableArray*ps=[NSMutableArray array];
 for(NSString*n in names){id<MTLComputePipelineState>p=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:n] error:&error];need(p!=nil,"pipeline");[ps addObject:p];}
 const uint32_t pq_shapes[][2]={{9,384},{17408,5120},{5120,17408},{10240,5120},{6144,5120},{5120,6144},{12288,5120},{248320,5120}};
 const uint32_t bf_shapes[][2]={{48,5120},{49,5120},{128,5120},{17,384}};
 const bool bf16=argc>2 && !strcmp(argv[2],"--bf16");
 const uint32_t (*shapes)[2]=bf16?bf_shapes:pq_shapes;
 const uint32_t nshapes=bf16?4u:8u;
 fprintf(stderr,"Bonsai matvec benchmark: %s; 8 alternating A/B rounds, median GPU time.\n",d.name.UTF8String);
 printf("shape,kernel,median_us,min_us,max_us,speedup,max_abs,neq\n");fflush(stdout);
 for(uint32_t sh=0;sh<nshapes;sh++) {@autoreleasepool {
  uint32_t rows=shapes[sh][0],cols=shapes[sh][1],stride=bf16?cols*2:cols/128*34;Args a={.rows=rows,.cols=cols,.row_bytes=stride,.type=bf16?30:142};
  id<MTLBuffer>w=[d newBufferWithLength:(NSUInteger)rows*stride options:MTLResourceStorageModeShared]; id<MTLBuffer>x=[d newBufferWithLength:cols*4 options:MTLResourceStorageModeShared];id<MTLBuffer>y=[d newBufferWithLength:rows*4 options:MTLResourceStorageModeShared];need(w&&x&&y,"buffers");
  if(bf16) {
   uint16_t *wb=w.contents;
   for(uint32_t r=0;r<rows;r++)for(uint32_t j=0;j<cols;j++) {
    float value=sinf((r*cols+j)*.153f);uint32_t bits;memcpy(&bits,&value,4);wb[r*cols+j]=bits>>16;
   }
  } else {
  uint8_t*wb=w.contents; for(uint32_t r=0;r<rows;r++)for(uint32_t b=0;b<cols/128;b++){uint8_t*p=wb+(size_t)r*stride+b*34; p[0]=(r+b*11)&255;p[1]=0x2c+(r+b)%8;for(uint32_t j=0;j<32;j++)p[2+j]=(uint8_t)(r*13+b*29+j*37);}
  }
  float*xv=x.contents;for(uint32_t j=0;j<cols;j++)xv[j]=sinf(j*.013f)*.25f;
  float*ref=malloc(rows*4);need(ref!=NULL,"reference allocation");run(q,ps[0],a,w,x,y,1,2);memcpy(ref,y.contents,rows*4);
  double times[2][8]={{0}},errs[2]={0};uint32_t neq[2]={0};
  for(uint32_t v=0;v<2;v++){run(q,ps[v],a,w,x,y,nrs[v],2);float*z=y.contents;for(uint32_t r=0;r<rows;r++){need(isfinite(z[r]),"finite output");errs[v]=fmax(errs[v],fabs(z[r]-ref[r]));neq[v]+=memcmp(z+r,ref+r,4)!=0;}need(!neq[v],"preserved reduction exactness");}
  for(uint32_t trial=0;trial<8;trial++)for(uint32_t j=0;j<2;j++){uint32_t v=trial%2?1-j:j;times[v][trial]=run(q,ps[v],a,w,x,y,nrs[v],sh==7?5:20);}
  for(uint32_t v=0;v<2;v++)qsort(times[v],8,sizeof(double),fcmp);
  for(uint32_t v=0;v<2;v++){printf("%ux%u,%s,%.3f,%.3f,%.3f,%.4f,%.9g,%u\n",rows,cols,names[v].UTF8String,(times[v][3]+times[v][4])*.5,times[v][0],times[v][7],(times[0][3]+times[0][4])/(times[v][3]+times[v][4]),errs[v],neq[v]);}fflush(stdout);free(ref);
 }}
 return 0;
} }
