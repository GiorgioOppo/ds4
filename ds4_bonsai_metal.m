#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "ds4_bonsai.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "metal/bonsai.metal.inc"

typedef struct {
    uint32_t n, rows, cols, type, row_bytes, pos, heads, kvheads;
    uint32_t dim, rot, width, mode, groups;
    float eps, base;
} BonsaiArgs;
_Static_assert(sizeof(BonsaiArgs) == 60, "Bonsai Metal argument layout");

@interface BSMWeight : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic, strong) id<MTLBuffer> signs;
@property(nonatomic) NSUInteger offset;
@property(nonatomic) uint32_t rowBytes;
@end
@implementation BSMWeight
@end

enum { BS_X, BS_NORM, BS_RESULT, BS_QKV, BS_Z, BS_ALPHA, BS_BETA,
       BS_CONV, BS_GDN, BS_Q, BS_K, BS_V, BS_ATTN, BS_GATE, BS_UP,
       BS_MID, BS_ROT, BS_LOGITS, BS_SCORES, BS_BUFFER_COUNT };

static uint64_t bsm_reserved_bytes;

@interface BSMContext : NSObject {
@public
    const ds4_bonsai_model *model;
    uint32_t context, position, batchCapacity, attentionBatchCapacity;
    bool failed;
    uint64_t reservedBytes;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
    NSMutableDictionary<NSValue *, BSMWeight *> *weights;
    NSMutableDictionary<NSString *, id<MTLBuffer>> *signBuffers;
    id<MTLBuffer> scratch[BS_BUFFER_COUNT];
    id<MTLBuffer> batch[BS_BUFFER_COUNT];
    id<MTLBuffer> state[DS4_BONSAI_LAYERS], history[DS4_BONSAI_LAYERS];
    id<MTLBuffer> keyCache[DS4_BONSAI_LAYERS], valueCache[DS4_BONSAI_LAYERS];
}
@end
@implementation BSMContext
- (void)dealloc {
    @synchronized([BSMContext class]) {
        bsm_reserved_bytes -= reservedBytes;
    }
}
@end

struct ds4_bonsai_metal { void *implementation; };

static uint64_t bsm_row_bytes(uint32_t type, uint32_t cols) {
    switch (type) {
        case 0: return (uint64_t)cols * 4;
        case 1: case 30: return (uint64_t)cols * 2;
        case 8: return cols % 32 ? 0 : (uint64_t)(cols / 32) * 34;
        case 142: return cols % 128 ? 0 : (uint64_t)(cols / 128) * 34;
        case 143: return cols % 128 ? 0 : (uint64_t)(cols / 128) * 28;
        default: return 0;
    }
}

static bool bsm_scalar_type(uint32_t type) { return type == 0 || type == 1 || type == 30; }

static BSMWeight *bsm_weight(BSMContext *s, const ds4_bonsai_tensor *t) {
    if (!t || !t->data || !t->cols || !t->rows) return nil;
    NSValue *key = [NSValue valueWithPointer:t];
    BSMWeight *w = s->weights[key];
    if (w) return w;
    const uint64_t row = bsm_row_bytes(t->type, t->cols);
    if (!row || row > UINT32_MAX || row > UINT64_MAX / t->rows ||
        t->bytes < row * t->rows || t->bytes > NSUIntegerMax ||
        (t->signs && t->cols % 1024u)) return nil;
    w = [BSMWeight new];
    w.rowBytes = (uint32_t)row;
    const NSUInteger bytes = (NSUInteger)(row * t->rows);
    if (bytes < 65536u) {
        // Tiny scalar/vector tables may be ordinary malloc buffers.
        w.buffer = [s->device newBufferWithBytes:t->data length:bytes options:MTLResourceStorageModeShared];
    } else {
        // Register each mmap-backed tensor range without copying the model.
        // The owner keeps the borrowed GGUF mapping alive until session free.
        const uintptr_t page = (uintptr_t)getpagesize();
        const uintptr_t ptr = (uintptr_t)t->data, base = ptr & ~(page - 1u);
        const NSUInteger offset = (NSUInteger)(ptr - base);
        if (bytes > NSUIntegerMax - offset - (page - 1u)) return nil;
        const NSUInteger length = (bytes + offset + page - 1u) & ~(page - 1u);
        if (length > s->device.maxBufferLength) return nil;
        w.buffer = [s->device newBufferWithBytesNoCopy:(void *)base length:length
                  options:MTLResourceStorageModeShared deallocator:nil];
        w.offset = offset;
    }
    if (!w.buffer) return nil;
    if (t->signs) {
        NSString *sk = [NSString stringWithFormat:@"%p/%u", (const void *)t->signs, t->cols];
        w.signs = s->signBuffers[sk];
        if (!w.signs) {
            w.signs = [s->device newBufferWithBytes:t->signs length:(NSUInteger)t->cols * sizeof(int32_t)
                       options:MTLResourceStorageModeShared];
            if (!w.signs) return nil;
            s->signBuffers[sk] = w.signs;
        }
    }
    s->weights[key] = w;
    return w;
}

static bool bsm_shape(BSMContext *s, const ds4_bonsai_tensor *t, uint32_t cols, uint32_t rows, bool scalar) {
    return t && t->cols == cols && t->rows == rows && (!scalar || bsm_scalar_type(t->type)) && bsm_weight(s, t) != nil;
}

static NSString *bsm_source(void) {
    return [[NSString alloc] initWithBytes:metal_bonsai_metal length:metal_bonsai_metal_len
                                 encoding:NSUTF8StringEncoding];
}

// Account for the complete session before creating even mmap-backed weight
// buffers. The process-wide reservation also bounds multiple Bonsai sessions.
// It intentionally counts shared model weights again for a second session:
// rejecting a marginal allocation is preferable to exhausting unified memory.
static bool bsm_memory_admit(BSMContext *s, const uint64_t *sizes) {
    const ds4_bonsai_model *m=s->model;
    NSMutableDictionary<NSValue *, NSNumber *> *unique=[NSMutableDictionary dictionary];
    NSMutableDictionary<NSValue *, NSNumber *> *signs=[NSMutableDictionary dictionary];
    const ds4_bonsai_tensor *ts[3+19*DS4_BONSAI_LAYERS];
    size_t count=0;
    ts[count++]=&m->embedding; ts[count++]=&m->output; ts[count++]=&m->output_norm;
    for (uint32_t i=0;i<m->n_layer;++i) {
        const ds4_bonsai_layer *l=&m->layer[i];
        ts[count++]=&l->norm; ts[count++]=&l->post_norm;
        ts[count++]=&l->gate; ts[count++]=&l->up; ts[count++]=&l->down; ts[count++]=&l->out;
        if ((i+1u)%m->full_interval==0) {
            ts[count++]=&l->q; ts[count++]=&l->k; ts[count++]=&l->v;
            ts[count++]=&l->q_norm; ts[count++]=&l->k_norm;
        } else {
            ts[count++]=&l->qkv; ts[count++]=&l->z; ts[count++]=&l->alpha; ts[count++]=&l->beta;
            ts[count++]=&l->conv; ts[count++]=&l->a; ts[count++]=&l->dt; ts[count++]=&l->ssm_norm;
        }
    }
    const uint64_t page=(uint64_t)getpagesize();
    for (size_t i=0;i<count;++i) {
        const ds4_bonsai_tensor *t=ts[i];
        uint64_t row=bsm_row_bytes(t->type,t->cols);
        if (!row || row>UINT64_MAX/t->rows) return false;
        uint64_t bytes=row*t->rows;
        if (bytes>=65536u) {
            uint64_t offset=(uintptr_t)t->data&(page-1u);
            if (bytes>UINT64_MAX-offset-(page-1u)) return false;
            bytes=(bytes+offset+page-1u)&~(page-1u);
        }
        if (bytes>s->device.maxBufferLength) return false;
        NSValue *key=[NSValue valueWithPointer:t->data];
        if (bytes>[unique[key] unsignedLongLongValue]) unique[key]=@(bytes);
        if (t->signs) {
            key=[NSValue valueWithPointer:t->signs]; bytes=(uint64_t)t->cols*sizeof(int32_t);
            if (bytes>[signs[key] unsignedLongLongValue]) signs[key]=@(bytes);
        }
    }
    uint64_t total=0;
#define BSM_ADD_BYTES(value) do { uint64_t n_=(value); if (n_>UINT64_MAX-total) return false; total+=n_; } while(0)
    for (NSNumber *bytes in unique.allValues) BSM_ADD_BYTES(bytes.unsignedLongLongValue);
    for (NSNumber *bytes in signs.allValues) BSM_ADD_BYTES(bytes.unsignedLongLongValue);
    for (uint32_t i=0;i<BS_BUFFER_COUNT;++i) {
        if (sizes[i]>UINT64_MAX/4u || MAX(sizes[i],1u)*4u>s->device.maxBufferLength) return false;
        BSM_ADD_BYTES(MAX(sizes[i],1u)*4u);
        if (i < BS_LOGITS) {
            if (sizes[i]>UINT64_MAX/s->batchCapacity) return false;
            uint64_t batch=sizes[i]*s->batchCapacity;
            if (batch>UINT64_MAX/4u || MAX(batch,1u)*4u>s->device.maxBufferLength) return false;
            BSM_ADD_BYTES(MAX(batch,1u)*4u);
        }
    }
    const uint64_t v=(uint64_t)m->n_v_head*m->ssm_dim;
    const uint64_t c=(2ull*m->n_k_head+m->n_v_head)*m->ssm_dim;
    const uint64_t kv=(uint64_t)m->n_kv_head*m->head_dim;
    for (uint32_t i=0;i<m->n_layer;++i) {
        uint64_t a,b;
        if ((i+1u)%m->full_interval==0) { a=(uint64_t)s->context*kv; b=a; }
        else { a=v*m->ssm_dim; b=c*(m->conv_width-1u); }
        if (a>UINT64_MAX/4u || b>UINT64_MAX/4u || MAX(a,1u)*4u>s->device.maxBufferLength ||
            MAX(b,1u)*4u>s->device.maxBufferLength) return false;
        BSM_ADD_BYTES(MAX(a,1u)*4u); BSM_ADD_BYTES(MAX(b,1u)*4u);
    }
#undef BSM_ADD_BYTES
    @synchronized([BSMContext class]) {
        uint64_t budget=s->device.recommendedMaxWorkingSetSize;
        uint64_t used=MAX(bsm_reserved_bytes,(uint64_t)s->device.currentAllocatedSize);
        if (used>budget || total>budget-used) {
            fprintf(stderr,"ds4: Bonsai Metal needs %.2f GiB; %.2f GiB available within recommended working set\n",
                    total/1073741824.0,(budget>used ? budget-used : 0)/1073741824.0);
            return false;
        }
        bsm_reserved_bytes+=total; s->reservedBytes=total;
    }
    return true;
}

static bool bsm_dispatch(BSMContext *s, id<MTLComputeCommandEncoder> enc, NSString *name,
                         BonsaiArgs a, NSArray<id<MTLBuffer>> *buffers, const NSUInteger *offsets,
                         MTLSize grid, MTLSize threads) {
    id<MTLComputePipelineState> pipeline = s->pipelines[name];
    if (!pipeline || threads.width * threads.height > pipeline.maxTotalThreadsPerThreadgroup) return false;
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&a length:sizeof(a) atIndex:0];
    for (NSUInteger i = 0; i < buffers.count; ++i)
        [enc setBuffer:buffers[i] offset:offsets ? offsets[i] : 0 atIndex:i + 1];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    return true;
}

static bool bsm_vector(BSMContext *s, id<MTLComputeCommandEncoder> enc, NSString *name,
                       BonsaiArgs a, NSArray<id<MTLBuffer>> *buffers, const NSUInteger *offsets, uint32_t n) {
    return bsm_dispatch(s, enc, name, a, buffers, offsets, MTLSizeMake(((uint64_t)n + 255u) / 256u, 1, 1), MTLSizeMake(256, 1, 1));
}

static bool bsm_transform_at(BSMContext *s, id<MTLComputeCommandEncoder> enc,
                          const ds4_bonsai_tensor *t, id<MTLBuffer> x, id<MTLBuffer> out,
                          bool inverse, bool grouped, NSUInteger xOffset, NSUInteger outOffset) {
    BSMWeight *w = bsm_weight(s, t);
    if (!w || !w.signs || t->cols % 1024u) return false;
    BonsaiArgs a = {.n=t->cols, .heads=s->model->n_v_head, .dim=s->model->ssm_dim,
                    .mode=inverse, .groups=grouped ? s->model->n_k_head : 0};
    const NSUInteger offsets[]={xOffset,0,outOffset};
    return bsm_dispatch(s, enc, @"bonsai_hadamard", a, @[x,w.signs,out], offsets,
                        MTLSizeMake(t->cols / 1024u, 1, 1), MTLSizeMake(256, 1, 1));
}

static bool bsm_transform(BSMContext *s, id<MTLComputeCommandEncoder> enc,
                          const ds4_bonsai_tensor *t, id<MTLBuffer> x, id<MTLBuffer> out,
                          bool inverse, bool grouped) {
    return bsm_transform_at(s,enc,t,x,out,inverse,grouped,0,0);
}

static bool bsm_mv_prepared(BSMContext *s, id<MTLComputeCommandEncoder> enc,
                            const ds4_bonsai_tensor *t, id<MTLBuffer> x, id<MTLBuffer> out) {
    BSMWeight *w = bsm_weight(s, t);
    if (!w) return false;
    BonsaiArgs a = {.rows=t->rows, .cols=t->cols, .type=t->type, .row_bytes=w.rowBytes};
    const NSUInteger offsets[] = {w.offset,0,0};
    // Amortize each activation tile over eight output rows. Small matrices
    // retain the original mapping to keep enough independent threadgroups.
    if (t->type==142u && t->rows>=4096u) {
        NSString *name=t->rows%16u==0 ? @"bonsai_pq2_mv_full" : @"bonsai_pq2_mv";
        return bsm_dispatch(s,enc,name,a,@[w.buffer,x,out],offsets,
                            MTLSizeMake(((uint64_t)t->rows+15u)/16u,1,1),MTLSizeMake(64,1,1));
    }
    return bsm_dispatch(s, enc, @"bonsai_mv", a, @[w.buffer,x,out], offsets,
                        MTLSizeMake(((uint64_t)t->rows + 3u) / 4u, 1, 1), MTLSizeMake(128, 1, 1));
}

static bool bsm_mv(BSMContext *s, id<MTLComputeCommandEncoder> enc,
                   const ds4_bonsai_tensor *t, id<MTLBuffer> x, id<MTLBuffer> out, bool grouped) {
    if (t->signs) {
        if (!bsm_transform(s, enc, t, x, s->scratch[BS_ROT], false, grouped)) return false;
        x = s->scratch[BS_ROT];
    }
    return bsm_mv_prepared(s,enc,t,x,out);
}

// Sibling projections consume the same immutable input. Reuse its rotation
// only for the identical sign table and width; ordinary BF16 projections
// still consume x. Keep the reuse local to this call, because BS_ROT and x
// are scratch buffers overwritten by subsequent graph operations.
static bool bsm_mv_siblings(BSMContext *s, id<MTLComputeCommandEncoder> enc,
                            const ds4_bonsai_tensor *const *tensors, id<MTLBuffer> x,
                            const unsigned *outputs, unsigned count) {
    const int32_t *signs = NULL;
    uint32_t cols = 0;
    for (unsigned i=0;i<count;++i) {
        const ds4_bonsai_tensor *t=tensors[i];
        id<MTLBuffer> input=x;
        if (t->signs) {
            if (t->signs!=signs || t->cols!=cols) {
                if (!bsm_transform(s,enc,t,x,s->scratch[BS_ROT],false,false)) return false;
                signs=t->signs; cols=t->cols;
            }
            input=s->scratch[BS_ROT];
        }
        if (!bsm_mv_prepared(s,enc,t,input,s->scratch[outputs[i]])) return false;
    }
    return true;
}

static bool bsm_norm_at(BSMContext *s, id<MTLComputeCommandEncoder> enc,
                     const ds4_bonsai_tensor *t, id<MTLBuffer> x, id<MTLBuffer> out,
                     uint32_t heads, uint32_t width, uint32_t stride, bool l2,
                     NSUInteger xOffset, NSUInteger outOffset) {
    BSMWeight *w = l2 ? nil : bsm_weight(s, t);
    if (!l2 && !w) return false;
    BonsaiArgs a = {.cols=width, .width=stride, .mode=l2, .type=l2 ? 0 : t->type, .eps=s->model->eps};
    const NSUInteger offsets[] = {xOffset,l2 ? 0 : w.offset,outOffset};
    return bsm_dispatch(s, enc, @"bonsai_norm", a, @[x,l2 ? x : w.buffer,out], offsets,
                        MTLSizeMake(heads,1,1), MTLSizeMake(256,1,1));
}

static bool bsm_norm(BSMContext *s, id<MTLComputeCommandEncoder> enc,
                     const ds4_bonsai_tensor *t, id<MTLBuffer> x, id<MTLBuffer> out,
                     uint32_t heads, uint32_t width, uint32_t stride, bool l2) {
    return bsm_norm_at(s,enc,t,x,out,heads,width,stride,l2,0,0);
}

static bool bsm_element_at(BSMContext *s, id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x,
                        id<MTLBuffer> y, id<MTLBuffer> out, uint32_t n, uint32_t mode, uint32_t dim,
                        NSUInteger xOffset, NSUInteger yOffset, NSUInteger outOffset) {
    const NSUInteger offsets[]={xOffset,yOffset,outOffset};
    return bsm_vector(s,enc,@"bonsai_element",(BonsaiArgs){.n=n,.mode=mode,.dim=dim},@[x,y,out],offsets,n);
}

static bool bsm_element(BSMContext *s, id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x,
                        id<MTLBuffer> y, id<MTLBuffer> out, uint32_t n, uint32_t mode, uint32_t dim) {
    return bsm_element_at(s,enc,x,y,out,n,mode,dim,0,0,0);
}

// These fusions are decode-only. Keep the existing sibling path for
// layouts that would otherwise change the reduction or prepared input.
static bool bsm_gate_up(BSMContext *s, id<MTLComputeCommandEncoder> enc,
                       const ds4_bonsai_tensor *gate, const ds4_bonsai_tensor *up,
                       id<MTLBuffer> x, id<MTLBuffer> mid) {
    if (gate->type!=142u || up->type!=142u || gate->rows<4096u ||
        gate->rows!=up->rows || gate->cols!=up->cols || gate->signs!=up->signs) {
        const ds4_bonsai_tensor *tensors[]={gate,up};
        const unsigned outputs[]={BS_GATE,BS_UP};
        return bsm_mv_siblings(s,enc,tensors,x,outputs,2) &&
            bsm_element(s,enc,s->scratch[BS_GATE],s->scratch[BS_UP],mid,gate->rows,1,0);
    }
    BSMWeight *g=bsm_weight(s,gate), *u=bsm_weight(s,up);
    if (!g || !u || g.rowBytes!=u.rowBytes) return false;
    if (gate->signs) {
        if (!bsm_transform(s,enc,gate,x,s->scratch[BS_ROT],false,false)) return false;
        x=s->scratch[BS_ROT];
    }
    BonsaiArgs a={.rows=gate->rows,.cols=gate->cols,.type=142u,.row_bytes=g.rowBytes};
    const NSUInteger offsets[]={g.offset,u.offset,0,0};
    NSString *name=gate->rows%8u==0 ? @"bonsai_pq2_gate_up_full" : @"bonsai_pq2_gate_up";
    return bsm_dispatch(s,enc,name,a,@[g.buffer,u.buffer,x,mid],offsets,
                        MTLSizeMake(((uint64_t)gate->rows+7u)/8u,1,1),MTLSizeMake(64,1,1));
}

static bool bsm_alpha_beta(BSMContext *s, id<MTLComputeCommandEncoder> enc,
                          const ds4_bonsai_layer *l, id<MTLBuffer> x) {
    const ds4_bonsai_tensor *alpha=&l->alpha, *beta=&l->beta;
    if (alpha->type!=30u || beta->type!=30u || alpha->signs || beta->signs ||
        alpha->rows!=beta->rows || alpha->cols!=beta->cols) {
        const ds4_bonsai_tensor *tensors[]={alpha,beta};
        const unsigned outputs[]={BS_ALPHA,BS_BETA};
        return bsm_mv_siblings(s,enc,tensors,x,outputs,2);
    }
    BSMWeight *a=bsm_weight(s,alpha), *b=bsm_weight(s,beta);
    if (!a || !b) return false;
    BonsaiArgs args={.rows=alpha->rows,.cols=alpha->cols,.type=30u,.row_bytes=a.rowBytes};
    const NSUInteger offsets[]={a.offset,b.offset,0,0,0};
    return bsm_dispatch(s,enc,@"bonsai_bf16_pair",args,
                        @[a.buffer,b.buffer,x,s->scratch[BS_ALPHA],s->scratch[BS_BETA]],offsets,
                        MTLSizeMake(((uint64_t)alpha->rows+3u)/4u,1,1),MTLSizeMake(128,1,1));
}

static bool bsm_gdn(BSMContext *s, id<MTLComputeCommandEncoder> enc, uint32_t il) {
    const ds4_bonsai_model *m = s->model;
    const ds4_bonsai_layer *l = &m->layer[il];
    const uint32_t d = m->ssm_dim, vh = m->n_v_head, kh = m->n_k_head;
    const uint32_t channels = (2u * kh + vh) * d;
    const ds4_bonsai_tensor *projections[]={&l->qkv,&l->z};
    const unsigned outputs[]={BS_QKV,BS_Z};
    if (!bsm_mv_siblings(s,enc,projections,s->scratch[BS_NORM],outputs,2) ||
        !bsm_alpha_beta(s,enc,l,s->scratch[BS_NORM])) return false;
    BSMWeight *conv=bsm_weight(s,&l->conv), *A=bsm_weight(s,&l->a), *dt=bsm_weight(s,&l->dt);
    const NSUInteger co[] = {0,conv.offset,0,0};
    if (!bsm_vector(s,enc,@"bonsai_conv",(BonsaiArgs){.n=channels,.width=m->conv_width,.type=l->conv.type},
         @[s->scratch[BS_QKV],conv.buffer,s->history[il],s->scratch[BS_CONV]],co,channels) ||
        !bsm_norm(s,enc,NULL,s->scratch[BS_CONV],s->scratch[BS_CONV],2u*kh,d,d,true)) return false;
    const NSUInteger so[] = {0,0,0,A.offset,dt.offset,0,0};
    if (!bsm_dispatch(s,enc,d==128u ? @"bonsai_gdn_128" : @"bonsai_gdn",(BonsaiArgs){.dim=d,.heads=vh,.kvheads=kh,.type=l->a.type,.mode=l->dt.type},
         @[s->scratch[BS_CONV],s->scratch[BS_ALPHA],s->scratch[BS_BETA],A.buffer,dt.buffer,s->state[il],s->scratch[BS_GDN]],so,
         MTLSizeMake(vh,1,1),MTLSizeMake(d,1,1)) ||
        !bsm_norm(s,enc,&l->ssm_norm,s->scratch[BS_GDN],s->scratch[BS_ATTN],vh,d,d,false) ||
        !bsm_element(s,enc,s->scratch[BS_Z],s->scratch[BS_ATTN],s->scratch[BS_ATTN],vh*d,1,0)) return false;
    return bsm_mv(s,enc,&l->out,s->scratch[BS_ATTN],s->scratch[BS_RESULT],m->gdn_v_grouped);
}

static bool bsm_attention(BSMContext *s, id<MTLComputeCommandEncoder> enc, uint32_t il) {
    const ds4_bonsai_model *m=s->model;
    const ds4_bonsai_layer *l=&m->layer[il];
    const uint32_t d=m->head_dim, h=m->n_head, kh=m->n_kv_head, kv=kh*d;
    const ds4_bonsai_tensor *projections[]={&l->q,&l->k,&l->v};
    const unsigned outputs[]={BS_QKV,BS_K,BS_V};
    if (!bsm_mv_siblings(s,enc,projections,s->scratch[BS_NORM],outputs,3) ||
        !bsm_norm(s,enc,&l->q_norm,s->scratch[BS_QKV],s->scratch[BS_Q],h,d,2u*d,false) ||
        !bsm_norm(s,enc,&l->k_norm,s->scratch[BS_K],s->scratch[BS_K],kh,d,d,false)) return false;
    BonsaiArgs a={.n=kv,.pos=s->position,.heads=h,.kvheads=kh,.dim=d,.rot=m->n_rot,.width=s->context,.base=m->rope_base};
    if (m->n_rot) {
        if (!bsm_vector(s,enc,@"bonsai_rope",a,@[s->scratch[BS_Q]],NULL,h*m->n_rot/2u)) return false;
        BonsaiArgs kargs=a; kargs.heads=kh;
        if (!bsm_vector(s,enc,@"bonsai_rope",kargs,@[s->scratch[BS_K]],NULL,kh*m->n_rot/2u)) return false;
    }
    return bsm_vector(s,enc,@"bonsai_cache",a,@[s->scratch[BS_K],s->scratch[BS_V],s->keyCache[il],s->valueCache[il]],NULL,kv) &&
        bsm_dispatch(s,enc,@"bonsai_scores",a,@[s->scratch[BS_Q],s->keyCache[il],s->scratch[BS_SCORES]],NULL,
                     MTLSizeMake(((uint64_t)s->position+256u)/256u,h,1),MTLSizeMake(256,1,1)) &&
        bsm_dispatch(s,enc,@"bonsai_softmax",a,@[s->scratch[BS_SCORES]],NULL,MTLSizeMake(h,1,1),MTLSizeMake(256,1,1)) &&
        bsm_vector(s,enc,@"bonsai_attention",a,@[s->scratch[BS_SCORES],s->valueCache[il],s->scratch[BS_ATTN]],NULL,h*d) &&
        bsm_element(s,enc,s->scratch[BS_ATTN],s->scratch[BS_QKV],s->scratch[BS_ATTN],h*d,2,d) &&
        bsm_mv(s,enc,&l->out,s->scratch[BS_ATTN],s->scratch[BS_RESULT],false);
}

static NSUInteger bsm_row_offset(uint32_t row, uint32_t width) {
    return (NSUInteger)row * width * sizeof(float);
}

/* Residual addition and SwiGLU are independent at every element. Collapse
 * contiguous prompt rows without changing their arithmetic. Keep the count
 * widened until the dispatch argument is known to fit its 32-bit field. */
static bool bsm_element_rows(BSMContext *s, id<MTLComputeCommandEncoder> enc,
                             id<MTLBuffer> x, id<MTLBuffer> y, id<MTLBuffer> out,
                             uint32_t count, uint32_t width, uint32_t mode) {
    const uint64_t elements=(uint64_t)count*width;
    if (elements<=UINT32_MAX)
        return bsm_element(s,enc,x,y,out,(uint32_t)elements,mode,0);
    for (uint32_t row=0;row<count;++row) {
        const NSUInteger off=bsm_row_offset(row,width);
        if (!bsm_element_at(s,enc,x,y,out,width,mode,0,off,off,off)) return false;
    }
    return true;
}

static bool bsm_mm_prepared(BSMContext *s, id<MTLComputeCommandEncoder> enc,
                            const ds4_bonsai_tensor *t, id<MTLBuffer> x,
                            id<MTLBuffer> out, uint32_t count) {
    BSMWeight *w=bsm_weight(s,t);
    if (!w) return false;
    BonsaiArgs a={.n=count,.rows=t->rows,.cols=t->cols,.type=t->type,.row_bytes=w.rowBytes};
    const NSUInteger offsets[]={w.offset,0,0};
    // Large PQ2 prompt projections reuse a 64-output x 32-token tile.
    // Both operands and the accumulator stay FP32. Short tails and small
    // projections retain the exact four-token or serial lane reductions.
    if (t->type==142u && t->rows>=4096u && count>=16u)
        return bsm_dispatch(s,enc,@"bonsai_mm_pq2_tiled",a,@[w.buffer,x,out],offsets,
                            MTLSizeMake(((uint64_t)t->rows+63u)/64u,(count+31u)/32u,1),MTLSizeMake(128,1,1));
    if (count<4u) {
        for (uint32_t row=0;row<count;++row) {
            const NSUInteger rowOffsets[]={w.offset,bsm_row_offset(row,t->cols),bsm_row_offset(row,t->rows)};
            if (!bsm_dispatch(s,enc,@"bonsai_mv",a,@[w.buffer,x,out],rowOffsets,
                              MTLSizeMake(((uint64_t)t->rows+3u)/4u,1,1),MTLSizeMake(128,1,1))) return false;
        }
        return true;
    }
    return bsm_dispatch(s,enc,@"bonsai_mm",a,@[w.buffer,x,out],offsets,
                        MTLSizeMake(((uint64_t)t->rows+3u)/4u,(count+3u)/4u,1),MTLSizeMake(128,1,1));
}

static bool bsm_transform_batch(BSMContext *s, id<MTLComputeCommandEncoder> enc,
                                const ds4_bonsai_tensor *t, id<MTLBuffer> x,
                                uint32_t count, bool grouped) {
    BSMWeight *w=bsm_weight(s,t);
    if (!w || !w.signs || t->cols%1024u) return false;
    const BonsaiArgs a={.n=t->cols,.heads=s->model->n_v_head,.dim=s->model->ssm_dim,
                        .groups=grouped ? s->model->n_k_head : 0};
    // Rows have independent butterflies and reuse the same sign table.
    return bsm_dispatch(s,enc,@"bonsai_hadamard",a,@[x,w.signs,s->batch[BS_ROT]],NULL,
                        MTLSizeMake(t->cols/1024u,count,1),MTLSizeMake(256,1,1));
}

static bool bsm_mm(BSMContext *s, id<MTLComputeCommandEncoder> enc,
                   const ds4_bonsai_tensor *t, id<MTLBuffer> x, id<MTLBuffer> out,
                   uint32_t count, bool grouped) {
    if (t->signs) {
        if (!bsm_transform_batch(s,enc,t,x,count,grouped)) return false;
        x=s->batch[BS_ROT];
    }
    return bsm_mm_prepared(s,enc,t,x,out,count);
}

static bool bsm_mm_siblings(BSMContext *s, id<MTLComputeCommandEncoder> enc,
                            const ds4_bonsai_tensor *const *tensors, id<MTLBuffer> x,
                            const unsigned *outputs, unsigned tensorCount, uint32_t count) {
    const int32_t *signs=NULL;
    uint32_t cols=0;
    for (unsigned i=0;i<tensorCount;++i) {
        const ds4_bonsai_tensor *t=tensors[i];
        id<MTLBuffer> input=x;
        if (t->signs) {
            if (t->signs!=signs || t->cols!=cols) {
                if (!bsm_transform_batch(s,enc,t,x,count,false)) return false;
                signs=t->signs; cols=t->cols;
            }
            input=s->batch[BS_ROT];
        }
        if (!bsm_mm_prepared(s,enc,t,input,s->batch[outputs[i]],count)) return false;
    }
    return true;
}

/* Pair the two tiled PQ2 projections and publish only the SwiGLU result.
 * Keep the original projection path for short chunks and other layouts. */
static bool bsm_gate_up_batch(BSMContext *s, id<MTLComputeCommandEncoder> enc,
                              const ds4_bonsai_layer *l, uint32_t count) {
    const ds4_bonsai_tensor *gate=&l->gate, *up=&l->up;
    if (count<16u || gate->type!=142u || up->type!=142u || gate->rows<4096u ||
        gate->rows!=up->rows || gate->cols!=up->cols || gate->signs!=up->signs) {
        const ds4_bonsai_tensor *projections[]={gate,up};
        const unsigned outputs[]={BS_GATE,BS_UP};
        return bsm_mm_siblings(s,enc,projections,s->batch[BS_NORM],outputs,2,count) &&
            bsm_element_rows(s,enc,s->batch[BS_GATE],s->batch[BS_UP],s->batch[BS_MID],count,gate->rows,1);
    }
    BSMWeight *g=bsm_weight(s,gate), *u=bsm_weight(s,up);
    if (!g || !u || g.rowBytes!=u.rowBytes) return false;
    id<MTLBuffer> x=s->batch[BS_NORM];
    if (gate->signs) {
        if (!bsm_transform_batch(s,enc,gate,x,count,false)) return false;
        x=s->batch[BS_ROT];
    }
    const BonsaiArgs a={.n=count,.rows=gate->rows,.cols=gate->cols,.type=142u,.row_bytes=g.rowBytes};
    const NSUInteger offsets[]={g.offset,u.offset,0,0};
    return bsm_dispatch(s,enc,@"bonsai_mm_pq2_gate_up_tiled",a,@[g.buffer,u.buffer,x,s->batch[BS_MID]],offsets,
                        MTLSizeMake(((uint64_t)gate->rows+31u)/32u,(count+31u)/32u,1),MTLSizeMake(128,1,1));
}

static bool bsm_alpha_beta_batch(BSMContext *s, id<MTLComputeCommandEncoder> enc,
                                 const ds4_bonsai_layer *l, uint32_t count) {
    const ds4_bonsai_tensor *alpha=&l->alpha, *beta=&l->beta;
    if (count<4u || alpha->type!=30u || beta->type!=30u || alpha->signs || beta->signs ||
        alpha->rows!=beta->rows || alpha->cols!=beta->cols) {
        const ds4_bonsai_tensor *projections[]={alpha,beta};
        const unsigned outputs[]={BS_ALPHA,BS_BETA};
        return bsm_mm_siblings(s,enc,projections,s->batch[BS_NORM],outputs,2,count);
    }
    BSMWeight *a=bsm_weight(s,alpha), *b=bsm_weight(s,beta);
    if (!a || !b) return false;
    const BonsaiArgs args={.n=count,.rows=alpha->rows,.cols=alpha->cols,.type=30u,.row_bytes=a.rowBytes};
    const NSUInteger offsets[]={a.offset,b.offset,0,0,0};
    return bsm_dispatch(s,enc,@"bonsai_bf16_pair_batch",args,
                        @[a.buffer,b.buffer,s->batch[BS_NORM],s->batch[BS_ALPHA],s->batch[BS_BETA]],offsets,
                        MTLSizeMake(((uint64_t)alpha->rows+3u)/4u,(count+3u)/4u,1),MTLSizeMake(128,1,1));
}

/* Long chunks share conv/L2/scan dispatches. Every recurrent update retains
 * the single-token arithmetic and order; short chunks use the original loop. */
static bool bsm_gdn_batch(BSMContext *s, id<MTLComputeCommandEncoder> enc,
                         uint32_t il, uint32_t count) {
    const ds4_bonsai_model *m=s->model;
    const ds4_bonsai_layer *l=&m->layer[il];
    const uint32_t d=m->ssm_dim,vh=m->n_v_head,kh=m->n_k_head;
    const uint32_t channels=(2u*kh+vh)*d,v=vh*d;
    const ds4_bonsai_tensor *projections[]={&l->qkv,&l->z};
    const unsigned outputs[]={BS_QKV,BS_Z};
    if (!bsm_mm_siblings(s,enc,projections,s->batch[BS_NORM],outputs,2,count) ||
        !bsm_alpha_beta_batch(s,enc,l,count)) return false;
    BSMWeight *conv=bsm_weight(s,&l->conv),*A=bsm_weight(s,&l->a),*dt=bsm_weight(s,&l->dt);
    if (!conv || !A || !dt) return false;
    if (count>=16u) {
        const NSUInteger co[]={0,conv.offset,0,0};
        if (!bsm_vector(s,enc,@"bonsai_conv_batch",
             (BonsaiArgs){.n=count,.cols=channels,.width=m->conv_width,.type=l->conv.type},
             @[s->batch[BS_QKV],conv.buffer,s->history[il],s->batch[BS_CONV]],co,channels) ||
            !bsm_dispatch(s,enc,@"bonsai_l2_batch",
             (BonsaiArgs){.n=count,.cols=d,.heads=2u*kh,.width=channels,.eps=m->eps},
             @[s->batch[BS_CONV]],NULL,MTLSizeMake(2u*kh,count,1),MTLSizeMake(256,1,1))) return false;
        const NSUInteger so[]={0,0,0,A.offset,dt.offset,0,0};
        if (!bsm_dispatch(s,enc,d==128u ? @"bonsai_gdn_batch_128" : @"bonsai_gdn_batch",
             (BonsaiArgs){.n=count,.dim=d,.heads=vh,.kvheads=kh,.type=l->a.type,.mode=l->dt.type},
             @[s->batch[BS_CONV],s->batch[BS_ALPHA],s->batch[BS_BETA],A.buffer,dt.buffer,s->state[il],s->batch[BS_GDN]],so,
             MTLSizeMake(vh,1,1),MTLSizeMake(d,1,1))) return false;
    } else {
        for (uint32_t row=0;row<count;++row) {
            const NSUInteger cOff=bsm_row_offset(row,channels),vOff=bsm_row_offset(row,v);
            const NSUInteger hOff=bsm_row_offset(row,vh);
            const NSUInteger co[]={cOff,conv.offset,0,cOff};
            if (!bsm_vector(s,enc,@"bonsai_conv",(BonsaiArgs){.n=channels,.width=m->conv_width,.type=l->conv.type},
                 @[s->batch[BS_QKV],conv.buffer,s->history[il],s->batch[BS_CONV]],co,channels) ||
                !bsm_norm_at(s,enc,NULL,s->batch[BS_CONV],s->batch[BS_CONV],2u*kh,d,d,true,cOff,cOff)) return false;
            const NSUInteger so[]={cOff,hOff,hOff,A.offset,dt.offset,0,vOff};
            if (!bsm_dispatch(s,enc,d==128u ? @"bonsai_gdn_128" : @"bonsai_gdn",(BonsaiArgs){.dim=d,.heads=vh,.kvheads=kh,.type=l->a.type,.mode=l->dt.type},
                 @[s->batch[BS_CONV],s->batch[BS_ALPHA],s->batch[BS_BETA],A.buffer,dt.buffer,s->state[il],s->batch[BS_GDN]],so,
                 MTLSizeMake(vh,1,1),MTLSizeMake(d,1,1))) return false;
        }
    }
    /* Output normalization/gating does not feed the recurrent state, so it
     * can follow the complete scan. Each head retains its own reduction. */
    const uint64_t heads=(uint64_t)count*vh;
    if (heads<=UINT32_MAX) {
        if (!bsm_norm(s,enc,&l->ssm_norm,s->batch[BS_GDN],s->batch[BS_ATTN],(uint32_t)heads,d,d,false)) return false;
    } else {
        for (uint32_t row=0;row<count;++row) {
            const NSUInteger off=bsm_row_offset(row,v);
            if (!bsm_norm_at(s,enc,&l->ssm_norm,s->batch[BS_GDN],s->batch[BS_ATTN],vh,d,d,false,off,off)) return false;
        }
    }
    return bsm_element_rows(s,enc,s->batch[BS_Z],s->batch[BS_ATTN],s->batch[BS_ATTN],count,v,1) &&
        bsm_mm(s,enc,&l->out,s->batch[BS_ATTN],s->batch[BS_RESULT],count,m->gdn_v_grouped);
}

static bool bsm_attention_batch(BSMContext *s, id<MTLComputeCommandEncoder> enc,
                               uint32_t il, uint32_t count) {
    const ds4_bonsai_model *m=s->model;
    const ds4_bonsai_layer *l=&m->layer[il];
    const uint32_t d=m->head_dim,h=m->n_head,kh=m->n_kv_head,kv=kh*d,q=h*d;
    const ds4_bonsai_tensor *projections[]={&l->q,&l->k,&l->v};
    const unsigned outputs[]={BS_QKV,BS_K,BS_V};
    if (!bsm_mm_siblings(s,enc,projections,s->batch[BS_NORM],outputs,3,count)) return false;
    const uint64_t queryHeads=(uint64_t)count*h,keyHeads=(uint64_t)count*kh;
    if (queryHeads>UINT32_MAX || keyHeads>UINT32_MAX ||
        !bsm_norm(s,enc,&l->q_norm,s->batch[BS_QKV],s->batch[BS_Q],(uint32_t)queryHeads,d,2u*d,false) ||
        !bsm_norm(s,enc,&l->k_norm,s->batch[BS_K],s->batch[BS_K],(uint32_t)keyHeads,d,d,false)) return false;
    BonsaiArgs batchArgs={.n=kv,.pos=s->position,.heads=h,.kvheads=kh,.dim=d,
                          .rot=m->n_rot,.width=s->context,.base=m->rope_base};
    if (m->n_rot) {
        if (!bsm_dispatch(s,enc,@"bonsai_rope",batchArgs,@[s->batch[BS_Q]],NULL,
                          MTLSizeMake(((uint64_t)h*m->n_rot/2u+255u)/256u,count,1),MTLSizeMake(256,1,1))) return false;
        BonsaiArgs ka=batchArgs;ka.heads=kh;
        if (!bsm_dispatch(s,enc,@"bonsai_rope",ka,@[s->batch[BS_K]],NULL,
                          MTLSizeMake(((uint64_t)kh*m->n_rot/2u+255u)/256u,count,1),MTLSizeMake(256,1,1))) return false;
    }
    // Cache all rows now; each query below still reads only through its own
    // causal position, so future rows cannot influence earlier outputs.
    if (!bsm_dispatch(s,enc,@"bonsai_cache",batchArgs,
                      @[s->batch[BS_K],s->batch[BS_V],s->keyCache[il],s->valueCache[il]],NULL,
                      MTLSizeMake(((uint64_t)kv+255u)/256u,count,1),MTLSizeMake(256,1,1))) return false;
    for (uint32_t row=0;row<count;row+=s->attentionBatchCapacity) {
        const NSUInteger qOff=bsm_row_offset(row,q);
        BonsaiArgs a=batchArgs;a.pos+=row;a.n=MIN(count-row,s->attentionBatchCapacity);
        const NSUInteger scoreOff[]={qOff,0,0},attnOff[]={0,0,qOff};
        if (!bsm_dispatch(s,enc,@"bonsai_scores_batch",a,@[s->batch[BS_Q],s->keyCache[il],s->scratch[BS_SCORES]],scoreOff,
                          MTLSizeMake(((uint64_t)a.pos+a.n+255u)/256u,h,a.n),MTLSizeMake(256,1,1)) ||
            !bsm_dispatch(s,enc,@"bonsai_softmax_batch",a,@[s->scratch[BS_SCORES]],NULL,MTLSizeMake(h,a.n,1),MTLSizeMake(256,1,1)) ||
            !bsm_dispatch(s,enc,@"bonsai_attention_batch",a,@[s->scratch[BS_SCORES],s->valueCache[il],s->batch[BS_ATTN]],attnOff,
                          MTLSizeMake(((uint64_t)q+255u)/256u,a.n,1),MTLSizeMake(256,1,1))) return false;
    }
    const uint64_t elements=(uint64_t)count*q;
    if (elements>UINT32_MAX ||
        !bsm_element(s,enc,s->batch[BS_ATTN],s->batch[BS_QKV],s->batch[BS_ATTN],(uint32_t)elements,2,d)) return false;
    return bsm_mm(s,enc,&l->out,s->batch[BS_ATTN],s->batch[BS_RESULT],count,false);
}

static bool bsm_validate(BSMContext *s) {
    const ds4_bonsai_model *m=s->model;
    if (!m || !m->n_layer || m->n_layer>DS4_BONSAI_LAYERS || !m->n_embd || !m->n_vocab || !m->n_ff ||
        !m->n_head || !m->n_kv_head || m->n_head % m->n_kv_head || !m->head_dim ||
        !m->n_k_head || !m->n_v_head || m->n_v_head % m->n_k_head || !m->ssm_dim || m->ssm_dim>256 ||
        !m->conv_width || !m->full_interval || m->n_rot>m->head_dim || m->n_rot%2 ||
        !isfinite(m->eps) || m->eps<=0 || !isfinite(m->rope_base) || m->rope_base<=0 ||
        (uint64_t)m->n_head*m->head_dim>UINT32_MAX/2u ||
        (2ull*m->n_k_head+m->n_v_head)*m->ssm_dim>UINT32_MAX) return false;
    uint32_t e=m->n_embd, f=m->n_ff, q=m->n_head*m->head_dim, kv=m->n_kv_head*m->head_dim;
    uint32_t v=m->n_v_head*m->ssm_dim, c=(2u*m->n_k_head+m->n_v_head)*m->ssm_dim;
    if (!bsm_shape(s,&m->embedding,e,m->n_vocab,false) || !bsm_shape(s,&m->output,e,m->n_vocab,false) ||
        !bsm_shape(s,&m->output_norm,e,1,true)) return false;
    for (uint32_t il=0;il<m->n_layer;++il) {
        const ds4_bonsai_layer *l=&m->layer[il];
        if (!bsm_shape(s,&l->norm,e,1,true) || !bsm_shape(s,&l->post_norm,e,1,true) ||
            !bsm_shape(s,&l->gate,e,f,false) || !bsm_shape(s,&l->up,e,f,false) || !bsm_shape(s,&l->down,f,e,false)) return false;
        if ((il+1u)%m->full_interval==0) {
            if (!bsm_shape(s,&l->q,e,2u*q,false) || !bsm_shape(s,&l->k,e,kv,false) || !bsm_shape(s,&l->v,e,kv,false) ||
                !bsm_shape(s,&l->out,q,e,false) || !bsm_shape(s,&l->q_norm,m->head_dim,1,true) ||
                !bsm_shape(s,&l->k_norm,m->head_dim,1,true)) return false;
        } else if (!bsm_shape(s,&l->qkv,e,c,false) || !bsm_shape(s,&l->z,e,v,false) ||
                   !bsm_shape(s,&l->alpha,e,m->n_v_head,false) || !bsm_shape(s,&l->beta,e,m->n_v_head,false) ||
                   !bsm_shape(s,&l->conv,m->conv_width,c,true) || !bsm_shape(s,&l->a,m->n_v_head,1,true) ||
                   !bsm_shape(s,&l->dt,m->n_v_head,1,true) || !bsm_shape(s,&l->ssm_norm,m->ssm_dim,1,true) ||
                   !bsm_shape(s,&l->out,v,e,false)) return false;
    }
    return true;
}

static id<MTLBuffer> bsm_alloc(BSMContext *s, uint64_t elements) {
    if (elements>NSUIntegerMax/sizeof(float) || elements*4u>s->device.maxBufferLength) return nil;
    return [s->device newBufferWithLength:(NSUInteger)MAX(elements,1u)*sizeof(float) options:MTLResourceStorageModeShared];
}

ds4_bonsai_metal *ds4_bonsai_metal_create(const ds4_bonsai_model *m, uint32_t ctx) {
    @autoreleasepool {
        if (!ds4_bonsai_model_valid(m) || !ctx || (m->context && ctx>m->context)) return NULL;
        BSMContext *s=[BSMContext new]; s->model=m; s->context=ctx;
        s->batchCapacity=MIN(ctx,DS4_BONSAI_METAL_PREFILL_CAP);
        // Reuse a bounded workspace for up to eight causal queries. At very
        // long contexts, reduce the query batch to keep scores within 32 MiB
        // (or the original single-query allocation if that is already larger).
        const uint64_t scoreRow=(uint64_t)ctx*m->n_head;
        s->attentionBatchCapacity=ds4_bonsai_metal_score_capacity(ctx,m->n_head);
        s->device=MTLCreateSystemDefaultDevice(); s->queue=[s->device newCommandQueue];
        s->weights=[NSMutableDictionary dictionary]; s->signBuffers=[NSMutableDictionary dictionary];
        s->pipelines=[NSMutableDictionary dictionary];
        const uint64_t e=m->n_embd,f=m->n_ff,q=(uint64_t)m->n_head*m->head_dim,kv=(uint64_t)m->n_kv_head*m->head_dim;
        const uint64_t v=(uint64_t)m->n_v_head*m->ssm_dim,c=(2ull*m->n_k_head+m->n_v_head)*m->ssm_dim;
        const uint64_t sizes[BS_BUFFER_COUNT]={e,e,e,MAX(c,2u*q),v,m->n_v_head,m->n_v_head,c,v,q,kv,kv,MAX(q,v),f,f,f,MAX(MAX(e,f),MAX(q,v)),m->n_vocab,scoreRow*s->attentionBatchCapacity};
        if (!s->device || !s->queue || !bsm_memory_admit(s,sizes) || !bsm_validate(s)) {
            fprintf(stderr,"ds4: Bonsai Metal unsupported model shape, tensor layout, or device\n"); return NULL;
        }
        NSString *source=bsm_source(); if (!source) return NULL;
        MTLCompileOptions *options=[MTLCompileOptions new];
        if (@available(macOS 15.0,*)) options.mathMode=MTLMathModeSafe;
        else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled=NO;
#pragma clang diagnostic pop
        }
        NSError *error=nil;
        s->library=[s->device newLibraryWithSource:source options:options error:&error];
        if (!s->library) { fprintf(stderr,"ds4: Bonsai Metal compilation failed: %s\n",error.localizedDescription.UTF8String); return NULL; }
        NSArray<NSString *> *names=@[@"bonsai_embed",@"bonsai_mv",@"bonsai_pq2_mv",@"bonsai_pq2_mv_full",@"bonsai_pq2_gate_up",@"bonsai_pq2_gate_up_full",@"bonsai_bf16_pair",@"bonsai_mm",@"bonsai_mm_pq2_tiled",@"bonsai_hadamard",@"bonsai_norm",@"bonsai_element",
            @"bonsai_conv",@"bonsai_gdn",@"bonsai_gdn_128",@"bonsai_conv_batch",@"bonsai_l2_batch",@"bonsai_gdn_batch",@"bonsai_gdn_batch_128",@"bonsai_mm_pq2_gate_up_tiled",
            @"bonsai_rope",@"bonsai_cache",@"bonsai_scores",@"bonsai_softmax",@"bonsai_attention",
            @"bonsai_scores_batch",@"bonsai_softmax_batch",@"bonsai_attention_batch",@"bonsai_bf16_pair_batch"];
        for (NSString *name in names) {
            id<MTLFunction> function=[s->library newFunctionWithName:name];
            id<MTLComputePipelineState> pipeline=function ? [s->device newComputePipelineStateWithFunction:function error:&error] : nil;
            const NSUInteger requiredThreads=([name isEqualToString:@"bonsai_gdn_128"] ||
                [name isEqualToString:@"bonsai_gdn_batch_128"] || [name isEqualToString:@"bonsai_mm_pq2_gate_up_tiled"] ||
                [name isEqualToString:@"bonsai_bf16_pair_batch"]) ? 128u : 256u;
            if (!pipeline || pipeline.threadExecutionWidth!=32 || pipeline.maxTotalThreadsPerThreadgroup<requiredThreads) {
                fprintf(stderr,"ds4: Bonsai Metal pipeline unavailable: %s\n",name.UTF8String); return NULL;
            }
            s->pipelines[name]=pipeline;
        }
        for (uint32_t i=0;i<BS_BUFFER_COUNT;++i) if (!(s->scratch[i]=bsm_alloc(s,sizes[i]))) return NULL;
        for (uint32_t i=0;i<BS_LOGITS;++i)
            if (!(s->batch[i]=bsm_alloc(s,sizes[i]*s->batchCapacity))) return NULL;
        for (uint32_t il=0;il<m->n_layer;++il) {
            if ((il+1u)%m->full_interval==0) {
                s->keyCache[il]=bsm_alloc(s,(uint64_t)ctx*kv); s->valueCache[il]=bsm_alloc(s,(uint64_t)ctx*kv);
                if (!s->keyCache[il] || !s->valueCache[il]) return NULL;
            } else {
                s->state[il]=bsm_alloc(s,v*m->ssm_dim); s->history[il]=bsm_alloc(s,c*(m->conv_width-1u));
                if (!s->state[il] || !s->history[il]) return NULL;
            }
        }
        ds4_bonsai_metal *result=calloc(1,sizeof(*result)); if (!result) return NULL;
        result->implementation=(__bridge_retained void *)s;
        ds4_bonsai_metal_reset(result);
        fprintf(stderr,"ds4: native Bonsai Metal: %s, %u layers, context %u, packed ternary weights\n",s->device.name.UTF8String,m->n_layer,ctx);
        return result;
    }
}

void ds4_bonsai_metal_free(ds4_bonsai_metal *s) {
    if (!s) return;
    @autoreleasepool { BSMContext *owner=(__bridge_transfer BSMContext *)s->implementation; (void)owner; }
    free(s);
}

void ds4_bonsai_metal_reset(ds4_bonsai_metal *handle) {
    if (!handle) return;
    BSMContext *s=(__bridge BSMContext *)handle->implementation;
    for (uint32_t il=0;il<s->model->n_layer;++il) {
        if (s->state[il]) memset(s->state[il].contents,0,s->state[il].length);
        if (s->history[il]) memset(s->history[il].contents,0,s->history[il].length);
    }
    // KV rows beyond the new frontier are never read and are overwritten.
    s->position=0; s->failed=false;
}

bool ds4_bonsai_metal_eval(ds4_bonsai_metal *handle, int token, float *logits) {
    if (!handle) return false;
    @autoreleasepool {
        BSMContext *s=(__bridge BSMContext *)handle->implementation;
        const ds4_bonsai_model *m=s->model;
        if (s->failed || token<0 || (uint32_t)token>=m->n_vocab || s->position>=s->context) return false;
        id<MTLCommandBuffer> cb=[s->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
        if (!cb || !enc) return false;
        BSMWeight *embedding=bsm_weight(s,&m->embedding);
        const NSUInteger eo[]={embedding.offset,0};
        bool ok=bsm_vector(s,enc,@"bonsai_embed",(BonsaiArgs){.cols=m->n_embd,.type=m->embedding.type,.row_bytes=embedding.rowBytes,.pos=(uint32_t)token},
             @[embedding.buffer,m->embedding.signs ? s->scratch[BS_NORM] : s->scratch[BS_X]],eo,m->n_embd);
        if (ok && m->embedding.signs) ok=bsm_transform(s,enc,&m->embedding,s->scratch[BS_NORM],s->scratch[BS_X],true,false);
        for (uint32_t il=0;ok && il<m->n_layer;++il) {
            const ds4_bonsai_layer *l=&m->layer[il];

            ok=bsm_norm(s,enc,&l->norm,s->scratch[BS_X],s->scratch[BS_NORM],1,m->n_embd,m->n_embd,false) &&
                ((il+1u)%m->full_interval==0 ? bsm_attention(s,enc,il) : bsm_gdn(s,enc,il)) &&
                bsm_element(s,enc,s->scratch[BS_X],s->scratch[BS_RESULT],s->scratch[BS_X],m->n_embd,0,0) &&
                bsm_norm(s,enc,&l->post_norm,s->scratch[BS_X],s->scratch[BS_NORM],1,m->n_embd,m->n_embd,false) &&
                bsm_gate_up(s,enc,&l->gate,&l->up,s->scratch[BS_NORM],s->scratch[BS_MID]) &&
                bsm_mv(s,enc,&l->down,s->scratch[BS_MID],s->scratch[BS_RESULT],false) &&
                bsm_element(s,enc,s->scratch[BS_X],s->scratch[BS_RESULT],s->scratch[BS_X],m->n_embd,0,0);
        }
        if (ok && logits) ok=bsm_norm(s,enc,&m->output_norm,s->scratch[BS_X],s->scratch[BS_NORM],1,m->n_embd,m->n_embd,false) &&
            bsm_mv(s,enc,&m->output,s->scratch[BS_NORM],s->scratch[BS_LOGITS],false);
        [enc endEncoding];
        if (!ok) return false; // No command has been committed yet.
        [cb commit]; [cb waitUntilCompleted];
        if (cb.status!=MTLCommandBufferStatusCompleted) {
            s->failed=true;
            fprintf(stderr,"ds4: Bonsai Metal token failed: %s\n",cb.error.localizedDescription.UTF8String);
            return false;
        }
        if (logits) {
            memcpy(logits,s->scratch[BS_LOGITS].contents,(size_t)m->n_vocab*sizeof(float));
            for (uint32_t i=0;i<m->n_vocab;++i) if (!isfinite(logits[i])) { s->failed=true; return false; }
        }
        s->position++;
        return true;
    }
}

bool ds4_bonsai_metal_prefill(ds4_bonsai_metal *handle, const int *tokens,
                             uint32_t count, float *logits) {
    if (!handle || !tokens || !count) return false;
    if (count==1u) return ds4_bonsai_metal_eval(handle,tokens[0],logits);
    @autoreleasepool {
        BSMContext *s=(__bridge BSMContext *)handle->implementation;
        const ds4_bonsai_model *m=s->model;
        if (s->failed || count>s->batchCapacity || s->position>s->context || count>s->context-s->position)
            return false;
        for (uint32_t row=0;row<count;++row)
            if (tokens[row]<0 || (uint32_t)tokens[row]>=m->n_vocab) return false;

        id<MTLCommandBuffer> cb=[s->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
        if (!cb || !enc) return false;
        BSMWeight *embedding=bsm_weight(s,&m->embedding);
        bool ok=true;
        for (uint32_t row=0;ok && row<count;++row) {
            const NSUInteger off=bsm_row_offset(row,m->n_embd);
            const NSUInteger offsets[]={embedding.offset,off};
            ok=bsm_vector(s,enc,@"bonsai_embed",(BonsaiArgs){.cols=m->n_embd,.type=m->embedding.type,
                 .row_bytes=embedding.rowBytes,.pos=(uint32_t)tokens[row]},
                 @[embedding.buffer,m->embedding.signs ? s->batch[BS_NORM] : s->batch[BS_X]],offsets,m->n_embd);
            if (ok && m->embedding.signs)
                ok=bsm_transform_at(s,enc,&m->embedding,s->batch[BS_NORM],s->batch[BS_X],true,false,off,off);
        }
        for (uint32_t il=0;ok && il<m->n_layer;++il) {
            const ds4_bonsai_layer *l=&m->layer[il];
            ok=bsm_norm(s,enc,&l->norm,s->batch[BS_X],s->batch[BS_NORM],
                        count,m->n_embd,m->n_embd,false);
            if (ok) ok=(il+1u)%m->full_interval==0 ?
                bsm_attention_batch(s,enc,il,count) : bsm_gdn_batch(s,enc,il,count);
            if (ok) ok=bsm_element_rows(s,enc,s->batch[BS_X],s->batch[BS_RESULT],s->batch[BS_X],count,m->n_embd,0) &&
                bsm_norm(s,enc,&l->post_norm,s->batch[BS_X],s->batch[BS_NORM],count,m->n_embd,m->n_embd,false);
            if (ok) ok=bsm_gate_up_batch(s,enc,l,count);
            if (ok) ok=bsm_mm(s,enc,&l->down,s->batch[BS_MID],s->batch[BS_RESULT],count,false);
            if (ok) ok=bsm_element_rows(s,enc,s->batch[BS_X],s->batch[BS_RESULT],s->batch[BS_X],count,m->n_embd,0);
        }
        if (ok && logits)
            ok=bsm_norm_at(s,enc,&m->output_norm,s->batch[BS_X],s->scratch[BS_NORM],
                           1,m->n_embd,m->n_embd,false,bsm_row_offset(count-1u,m->n_embd),0) &&
               bsm_mv(s,enc,&m->output,s->scratch[BS_NORM],s->scratch[BS_LOGITS],false);
        [enc endEncoding];
        if (!ok) return false; // Nothing was submitted, so the old state is intact.
        [cb commit]; [cb waitUntilCompleted];
        if (cb.status!=MTLCommandBufferStatusCompleted) {
            s->failed=true;
            fprintf(stderr,"ds4: Bonsai Metal prefill failed: %s\n",cb.error.localizedDescription.UTF8String);
            return false;
        }
        if (logits) {
            memcpy(logits,s->scratch[BS_LOGITS].contents,(size_t)m->n_vocab*sizeof(float));
            for (uint32_t i=0;i<m->n_vocab;++i) if (!isfinite(logits[i])) { s->failed=true; return false; }
        }
        s->position+=count;
        return true;
    }
}
