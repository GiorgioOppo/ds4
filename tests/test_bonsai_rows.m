/* Model-free Bonsai original-space embedding injection and interleaved MRoPE.
 * Link ds4_bonsai.o + ds4_bonsai_metal.o and Foundation/Metal.
 * Run with MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../ds4_bonsai.h"
#include "../bonsai_quant.h"
extern unsigned char metal_bonsai_metal[];
extern unsigned int metal_bonsai_metal_len;
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void need(bool ok,const char *what) {
    if(!ok){fprintf(stderr,"FAIL Bonsai rows: %s\n",what);exit(1);}
}
static uint32_t mix(uint32_t x) {x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
enum { EMB=1024,VOCAB=8,FF=32,CONTEXT=12,TOKENS=7 };
typedef struct {
    ds4_bonsai_model m;
    void *owned[64];unsigned count;
    float *original;
} Fixture;
static void *own(Fixture *f,size_t bytes) {
    need(f->count<64,"allocation inventory");void *p=calloc(1,bytes);need(p!=NULL,"allocate fixture");
    f->owned[f->count++]=p;return p;
}
static ds4_bonsai_tensor tensor(Fixture *f,uint32_t cols,uint32_t rows,float scale) {
    float *p=own(f,(size_t)cols*rows*sizeof(float));
    for(size_t i=0;i<(size_t)cols*rows;++i)p[i]=((int)(mix((uint32_t)i+97u*f->count)%17)-8)*scale;
    return (ds4_bonsai_tensor){.data=p,.bytes=(uint64_t)cols*rows*4u,.type=0,.cols=cols,.rows=rows};
}
static ds4_bonsai_tensor norm(Fixture *f,uint32_t cols) {
    ds4_bonsai_tensor t=tensor(f,cols,1,0);for(unsigned i=0;i<cols;++i)((float *)t.data)[i]=1;return t;
}
static void fixture_init(Fixture *f,bool folded) {
    memset(f,0,sizeof(*f));
    f->m=(ds4_bonsai_model){.n_layer=2,.n_embd=EMB,.n_vocab=VOCAB,.n_ff=FF,
        .n_head=2,.n_kv_head=1,.head_dim=80,.n_rot=64,.n_k_head=1,.n_v_head=2,
        .ssm_dim=8,.conv_width=4,.full_interval=2,.context=CONTEXT,.eps=1e-6f,.rope_base=10000000.f};
    ds4_bonsai_model *m=&f->m;
    m->embedding=tensor(f,EMB,VOCAB,1.f/64.f);m->output=tensor(f,EMB,VOCAB,1.f/512.f);m->output_norm=norm(f,EMB);
    // Nonzero dyadic entries keep forward/inverse H1024 exactly representable
    // without making signed zero part of the embedding-identity contract.
    for(size_t i=0;i<(size_t)EMB*VOCAB;++i)((float *)m->embedding.data)[i]+=1.f/128.f;
    f->original=own(f,(size_t)EMB*VOCAB*4u);memcpy(f->original,m->embedding.data,(size_t)EMB*VOCAB*4u);
    if(folded) {
        int32_t *signs=own(f,EMB*sizeof(int32_t));for(unsigned i=0;i<EMB;++i)signs[i]=i%3?1:-1;
        for(unsigned row=0;row<VOCAB;++row)
            need(ds4_bonsai_hadamard_forward((float *)m->embedding.data+(size_t)row*EMB,
                f->original+(size_t)row*EMB,signs,EMB),"fold word embeddings");
        m->embedding.signs=signs;
    }
    for(unsigned i=0;i<2;++i) {
        ds4_bonsai_layer *l=&m->layer[i];
        l->norm=norm(f,EMB);l->post_norm=norm(f,EMB);
        l->gate=tensor(f,EMB,FF,1.f/512.f);l->up=tensor(f,EMB,FF,1.f/512.f);l->down=tensor(f,FF,EMB,1.f/128.f);
        if(i==0) {
            l->qkv=tensor(f,EMB,32,1.f/512.f);l->z=tensor(f,EMB,16,1.f/512.f);
            l->alpha=tensor(f,EMB,2,1.f/512.f);l->beta=tensor(f,EMB,2,1.f/512.f);
            l->out=tensor(f,16,EMB,1.f/128.f);l->conv=tensor(f,4,32,1.f/64.f);
            for(unsigned c=0;c<32;++c)((float *)l->conv.data)[c*4+3]+=.5f;
            l->a=tensor(f,2,1,0);((float *)l->a.data)[0]=-.25f;((float *)l->a.data)[1]=-.5f;
            l->dt=tensor(f,2,1,1.f/16.f);l->ssm_norm=norm(f,8);
        } else {
            l->q=tensor(f,EMB,320,1.f/512.f);l->k=tensor(f,EMB,80,1.f/512.f);
            l->v=tensor(f,EMB,80,1.f/512.f);l->out=tensor(f,160,EMB,1.f/256.f);
            l->q_norm=norm(f,80);l->k_norm=norm(f,80);
        }
    }
    need(ds4_bonsai_model_valid(m),"synthetic model geometry");
}
static void fixture_free(Fixture *f){for(unsigned i=0;i<f->count;++i)free(f->owned[i]);}
static void same_logits(const float *a,const float *b,const char *what) {
    for(unsigned i=0;i<VOCAB;++i)need(isfinite(a[i])&&isfinite(b[i]),"finite logits");
    need(!memcmp(a,b,VOCAB*sizeof(float)),what);
}
static void check_rows(bool folded) {
    Fixture f;fixture_init(&f,folded);
    ds4_bonsai_metal *reference=ds4_bonsai_metal_create(&f.m,CONTEXT),*actual=ds4_bonsai_metal_create(&f.m,CONTEXT);
    need(reference&&actual,"create row sessions");
    const int tokens[TOKENS]={0,1,2,3,4,5,6};
    float expected[CONTEXT][VOCAB],logits[VOCAB];
    const float *identity[TOKENS];for(unsigned t=0;t<TOKENS;++t)identity[t]=f.original+(size_t)tokens[t]*EMB;
    for(unsigned t=0;t<TOKENS;++t) {
        need(ds4_bonsai_metal_eval(reference,tokens[t],expected[t]),"normal word reference");
        need(ds4_bonsai_metal_eval_row(actual,tokens[t],identity[t],NULL,logits),"inject original word row");
        same_logits(logits,expected[t],"original-space embedding equals word lookup");
    }
    ds4_bonsai_metal_reset(actual);
    need(ds4_bonsai_metal_prefill_rows(actual,tokens,TOKENS,NULL,NULL,logits),"NULL rows and positions fallback");
    same_logits(logits,expected[TOKENS-1],"fallback prefill equals text decode");
    ds4_bonsai_metal_reset(actual);
    need(ds4_bonsai_metal_prefill_rows(actual,tokens,TOKENS,identity,NULL,logits),"batch inject original rows");
    same_logits(logits,expected[TOKENS-1],"injected prefill equals text decode");
    ds4_bonsai_metal_reset(reference);
    need(ds4_bonsai_metal_prefill(reference,tokens,TOKENS,logits),"old text prefill wrapper");
    same_logits(logits,expected[TOKENS-1],"old prefill behavior retained");

    // An injected row deliberately belongs to a different token, proving the
    // override is used. Alternate normal and injected rows within each chunk.
    int substituted[TOKENS];const float *mixed[TOKENS];int32_t positions[TOKENS*3];
    for(unsigned t=0;t<TOKENS;++t) {
        substituted[t]=(t%2==0)?(tokens[t]+3)%VOCAB:tokens[t];
        mixed[t]=t%2==0?f.original+(size_t)substituted[t]*EMB:NULL;
        positions[3*t]=(int32_t)(t/3)+4;positions[3*t+1]=(int32_t)(t/2)+13;positions[3*t+2]=(int32_t)(t%3)-2;
    }
    ds4_bonsai_metal_reset(reference);
    for(unsigned t=0;t<TOKENS;++t)
        need(ds4_bonsai_metal_eval_row(reference,substituted[t],NULL,positions+3*t,expected[t]),"MRoPE serial word oracle");
    const int32_t nextpos[3]={23,23,23};
    need(ds4_bonsai_metal_eval_row(reference,1,NULL,nextpos,expected[TOKENS]),"continued custom text coordinate");
    need(ds4_bonsai_metal_eval(reference,2,expected[TOKENS+1]),"plain coordinate after custom row");
    const unsigned chunk_sizes[]={1,2,3,7};
    for(unsigned c=0;c<sizeof(chunk_sizes)/sizeof(chunk_sizes[0]);++c) {
        ds4_bonsai_metal_reset(actual);unsigned begin=0;
        while(begin<TOKENS) {
            unsigned count=chunk_sizes[c];if(count>TOKENS-begin)count=TOKENS-begin;
            need(ds4_bonsai_metal_prefill_rows(actual,tokens+begin,count,mixed+begin,positions+3*begin,logits),"mixed custom prefill");
            begin+=count;same_logits(logits,expected[begin-1],"custom prefix bit parity");
        }
        need(ds4_bonsai_metal_eval_row(actual,1,NULL,nextpos,logits),"custom decode after prefill");
        same_logits(logits,expected[TOKENS],"custom decode state parity");
        need(ds4_bonsai_metal_eval(actual,2,logits),"plain decode after custom positions");
        same_logits(logits,expected[TOKENS+1],"positions do not persist or change KV addressing");
    }
    ds4_bonsai_metal_reset(actual);ds4_bonsai_metal_reset(reference);
    need(ds4_bonsai_metal_eval(actual,0,logits)&&ds4_bonsai_metal_eval(reference,0,expected[0]),"validation prefix");
    float saved[VOCAB];memcpy(saved,logits,sizeof(saved));
    float *bad=malloc(EMB*sizeof(float));need(bad!=NULL,"invalid row allocation");memcpy(bad,identity[1],EMB*sizeof(float));
    bad[EMB-1]=NAN;const float *bad_rows[2]={NULL,bad};const int valid[2]={1,2},bad_tokens[2]={1,VOCAB};
    need(!ds4_bonsai_metal_eval_row(actual,-1,identity[1],nextpos,logits),"invalid injected token rejected");
    need(!ds4_bonsai_metal_prefill_rows(actual,bad_tokens,2,NULL,positions,logits),"late invalid token rejected");
    need(!ds4_bonsai_metal_prefill_rows(actual,valid,2,bad_rows,positions,logits),"late NaN embedding rejected");
    bad[EMB-1]=INFINITY;
    need(!ds4_bonsai_metal_eval_row(actual,1,bad,nextpos,logits),"infinite decode embedding rejected");
    need(!ds4_bonsai_metal_prefill_rows(actual,NULL,1,NULL,NULL,logits)&&
         !ds4_bonsai_metal_prefill_rows(actual,valid,0,NULL,NULL,logits)&&
         !ds4_bonsai_metal_prefill_rows(actual,valid,DS4_BONSAI_METAL_PREFILL_CAP+1,NULL,NULL,logits),"invalid batch bounds");
    same_logits(logits,saved,"invalid calls leave logits intact");
    need(ds4_bonsai_metal_eval_row(actual,1,identity[1],nextpos,logits)&&
         ds4_bonsai_metal_eval_row(reference,1,NULL,nextpos,expected[1]),"resume after validation failures");
    same_logits(logits,expected[1],"validation happens before recurrent state mutation");
    free(bad);ds4_bonsai_metal_free(reference);ds4_bonsai_metal_free(actual);fixture_free(&f);
    printf("PASS embedding rows folded%d identity/mixed/chunks/custom-decode/validation\n",folded);
}

typedef struct {
    uint32_t n,rows,cols,type,row_bytes,pos,heads,kvheads,dim,rot,width,mode,groups;
    float eps,base;
} Args;
_Static_assert(sizeof(Args)==60,"Bonsai argument layout");
static void check_mrope(void) {
    id<MTLDevice>dev=MTLCreateSystemDefaultDevice();need(dev!=nil,"Metal device");
    id<MTLCommandQueue>queue=[dev newCommandQueue];need(queue!=nil,"Metal queue");
    NSString *source=[[NSString alloc] initWithBytes:metal_bonsai_metal length:metal_bonsai_metal_len encoding:NSUTF8StringEncoding];
    MTLCompileOptions *options=[MTLCompileOptions new];
    if(@available(macOS 15.0,*))options.mathMode=MTLMathModeSafe;
    else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled=NO;
#pragma clang diagnostic pop
    }
    NSError *error=nil;id<MTLLibrary>lib=[dev newLibraryWithSource:source options:options error:&error];
    if(!lib)fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);need(lib!=nil,"MRoPE compile");
    id<MTLComputePipelineState>plain=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bonsai_rope"] error:&error];
    id<MTLComputePipelineState>multi=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bonsai_mrope"] error:&error];
    need(plain&&multi,"RoPE pipelines");
    enum { N=7,H=3,D=256,GUARD=64 };
    const size_t bytes=(size_t)N*H*D*sizeof(float);
    id<MTLBuffer>a=[dev newBufferWithLength:bytes+2*GUARD options:MTLResourceStorageModeShared];
    id<MTLBuffer>b=[dev newBufferWithLength:bytes+2*GUARD options:MTLResourceStorageModeShared];
    need(a&&b,"RoPE buffers");float *original=malloc(bytes);need(original!=NULL,"RoPE oracle input");
    for(size_t i=0;i<bytes/4;++i)original[i]=(float)((int)(mix((uint32_t)i)%31)-15)/32.f;
    Args args={.n=N,.pos=17,.heads=H,.dim=D,.rot=64,.base=10000000.f};
    // Independent coordinate list from IMRoPE sections [11,11,10,0].
    static const unsigned axes[32]={0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,
                                   1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1};
    for(unsigned mode=0;mode<2;++mode) {
        int32_t positions[N*3];
        for(unsigned t=0;t<N;++t)for(unsigned axis=0;axis<3;++axis)
            positions[t*3+axis]=mode?(int32_t)(t*7+axis*29)-3:(int32_t)(args.pos+t);
        memset(a.contents,0xa5,a.length);memset(b.contents,0xa5,b.length);
        memcpy((char *)a.contents+GUARD,original,bytes);memcpy((char *)b.contents+GUARD,original,bytes);
        id<MTLCommandBuffer>cb=[queue commandBuffer];id<MTLComputeCommandEncoder>enc=[cb computeCommandEncoder];
        need(cb&&enc,"RoPE command");[enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setComputePipelineState:multi];[enc setBuffer:a offset:GUARD atIndex:1];
        [enc setBytes:positions length:sizeof(positions) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1,N,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if(!mode) {
            [enc setComputePipelineState:plain];[enc setBuffer:b offset:GUARD atIndex:1];
            [enc dispatchThreadgroups:MTLSizeMake(1,N,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        }
        [enc endEncoding];[cb commit];[cb waitUntilCompleted];
        if(cb.status!=MTLCommandBufferStatusCompleted)fprintf(stderr,"%s\n",cb.error.localizedDescription.UTF8String);
        need(cb.status==MTLCommandBufferStatusCompleted,"RoPE completion");
        const float *out=(const float *)((char *)a.contents+GUARD);
        if(!mode)need(!memcmp(out,(char *)b.contents+GUARD,bytes),"equal axes retain exact text RoPE");
        for(unsigned t=0;t<N;++t)for(unsigned h=0;h<H;++h)for(unsigned j=0;j<D;++j) {
            const size_t index=((size_t)t*H+h)*D+j;
            if(j>=64)need(!memcmp(out+index,original+index,4),"unrotated head tail unchanged");
            else {
                const unsigned pair=j%32;const size_t first=index-j+pair;
                const float angle=(float)positions[3*t+axes[pair]]*powf(args.base,-(float)pair/32.f);
                const float c=cosf(angle),s=sinf(angle),p=original[first],q=original[first+32];
                const float expected=j<32?p*c-q*s:p*s+q*c;
                need(isfinite(out[index])&&fabsf(out[index]-expected)<=3e-5f,"MRoPE independent coordinate/rotation oracle");
            }
        }
        for(unsigned i=0;i<GUARD;++i) {
            need(((uint8_t *)a.contents)[i]==0xa5&&((uint8_t *)a.contents)[a.length-1-i]==0xa5,"MRoPE guards");
            need(((uint8_t *)b.contents)[i]==0xa5&&((uint8_t *)b.contents)[b.length-1-i]==0xa5,"text RoPE guards");
        }
    }
    free(original);puts("PASS MRoPE sections[11,11,10], signed coordinates, exact text fallback, head tails and guards");
}
int main(void) { @autoreleasepool {
    check_mrope();check_rows(false);check_rows(true);puts("PASS Bonsai embedding rows and MRoPE");return 0;
}}
