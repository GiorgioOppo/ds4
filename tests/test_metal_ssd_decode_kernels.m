#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// GGUF-free comparison of the production legacy and SSD-specialized kernels.
// Synthetic buffers mimic streamed cache slots, including nonzero offsets and
// indirect addresses. This is a correctness test, not an SSD throughput test.
enum { GUARD = 256, SLOTS = 6 };
static const uint32_t poison = 0x7fc12345u;
typedef struct {
    int32_t ne00, ne01, ne02;
    uint64_t nb00, nb01, nb02, nb03;
    int32_t ne10, ne11, ne12;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne0, ne1, nr0;
    int16_t r2, r3;
} mv_args;
typedef struct {
    int32_t nei0, nei1;
    uint64_t nbi1;
    int32_t ne00, ne01, ne02;
    uint64_t nb00, nb01, nb02;
    int32_t ne10, ne11, ne12, ne13;
    uint64_t nb10, nb11, nb12;
    int32_t ne0, ne1;
    uint64_t nb1;
    int32_t nr0, tp_rank, tp_world, tp_addend, tp_expert_base;
} id_args;
typedef struct {
    uint32_t width, rows;
    uint64_t gate_row_stride, up_row_stride, mid_row_stride, weight_stride;
    uint32_t write_clamped;
    float clamp_value;
} act_args;
typedef struct { uint16_t d, dmin; uint8_t scales[12], qs[128]; } q4_block;
typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
_Static_assert(sizeof(mv_args) == 112, "mul_mv ABI");
_Static_assert(sizeof(id_args) == 136, "mul_mv_id ABI");
_Static_assert(sizeof(act_args) == 48, "SwiGLU ABI");
_Static_assert(sizeof(q4_block) == 144 && sizeof(q8_block) == 34, "quant ABI");

static void require(int ok, const char *message) {
    if (!ok) { fprintf(stderr, "SSD kernel test: %s\n", message); exit(1); }
}

static NSString *metal_prelude(void) {
    return @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "#define MAX(x, y) ((x) > (y) ? (x) : (y))\n"
            "#define MIN(x, y) ((x) < (y) ? (x) : (y))\n"
            "#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }\n"
            "#define QK8_0 32\n"
            "#ifndef QK_K\n#define QK_K 256\n#endif\n"
            "#define N_SIMDWIDTH 32\n"
            "#define N_R0_Q8_0 2\n"
            "#define N_SG_Q8_0 4\n"
            "#define FC_MUL_MV 600\n"
            "#define FC_MUL_MM 700\n"
            "#define FC_BIN 1300\n"
            "#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"
            "#define M_PI_F 3.14159265358979323846f\n"
            "enum ds4_sort_order { DS4_SORT_ORDER_ASC, DS4_SORT_ORDER_DESC };\n"
            "struct block_q8_0 { half d; int8_t qs[QK8_0]; };\n"
            "struct block_q8_K { float d; int8_t qs[QK_K]; "
            "int16_t bsums[QK_K / 16]; };\n";
}

/* Keep the same concatenation order as ds4_metal.m. The test specializes
 * and dispatches the checked-in production kernels; it carries no kernel copy. */
static NSString *load_metal_source(void) {
    static const char *paths[] = {
        "metal/activations.metal",
        "metal/flash_attn.metal",
        "metal/dense.metal",
        "metal/moe.metal",
        "metal/dsv4_hc.metal",
        "metal/unary.metal",
        "metal/dsv4_kv.metal",
        "metal/dsv4_rope.metal",
        "metal/dsv4_misc.metal",
        "metal/argsort.metal",
        "metal/cpy.metal",
        "metal/concat.metal",
        "metal/get_rows.metal",
        "metal/sum_rows.metal",
        "metal/softmax.metal",
        "metal/repeat.metal",
        "metal/glu.metal",
        "metal/norm.metal",
        "metal/bin.metal",
        "metal/set_rows.metal",
    };
    const char *root_env = getenv("DS4_SOURCE_ROOT");
    NSString *root = root_env && root_env[0]
        ? [NSString stringWithUTF8String:root_env]
        : @".";
    NSMutableString *source = [NSMutableString stringWithString:metal_prelude()];
    for (size_t i = 0; i < sizeof(paths) / sizeof(paths[0]); i++) {
        NSString *relative = [NSString stringWithUTF8String:paths[i]];
        NSString *path = [root stringByAppendingPathComponent:relative];
        NSError *error = nil;
        NSString *part = [NSString stringWithContentsOfFile:path
                                                   encoding:NSUTF8StringEncoding
                                                      error:&error];
        if (!part) {
            fprintf(stderr, "test-metal-ssd-decode-kernels: cannot read %s: %s\n",
                    [path fileSystemRepresentation],
                    [[error localizedDescription] UTF8String]);
            return nil;
        }
        [source appendFormat:@"\n// appended %@\n%@\n", relative, part];
    }
    return source;
}

static id<MTLComputePipelineState> pipeline(id<MTLDevice> dev, id<MTLLibrary> lib,
                                           NSString *name, int16_t nsg) {
    MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
    NSError *error = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name constantValues:constants error:&error];
    if (!fn) fprintf(stderr, "%s: %s\n", name.UTF8String, error.description.UTF8String);
    require(fn != nil, "function specialization failed");
    id<MTLComputePipelineState> result = [dev newComputePipelineStateWithFunction:fn error:&error];
    if (!result) fprintf(stderr, "%s: %s\n", name.UTF8String, error.description.UTF8String);
    require(result != nil, "pipeline compilation failed");
    return result;
}
static uint32_t random_u32(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}
static void *data(id<MTLBuffer> b) { return (char *)b.contents + GUARD; }
static id<MTLBuffer> buffer(id<MTLDevice> dev, NSUInteger bytes) {
    id<MTLBuffer> b = [dev newBufferWithLength:GUARD * 2 + bytes options:MTLResourceStorageModeShared];
    require(b != nil, "buffer allocation failed");
    memset(b.contents, 0xa5, b.length);
    return b;
}
static id<MTLBuffer> output(id<MTLDevice> dev, NSUInteger count) {
    id<MTLBuffer> b = buffer(dev, count * sizeof(float));
    uint32_t *p = data(b);
    for (NSUInteger i = 0; i < count; ++i) p[i] = poison;
    return b;
}
static uint64_t hash(id<MTLBuffer> b) {
    const uint8_t *p = b.contents;
    uint64_t h = 14695981039346656037ull;
    for (NSUInteger i = 0; i < b.length; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}
static void equal_output(id<MTLBuffer> a, id<MTLBuffer> b, NSUInteger rows,
                         NSUInteger width, NSUInteger stride) {
    require(a.length == b.length, "output lengths differ");
    const uint8_t *ap = a.contents, *bp = b.contents;
    for (NSUInteger i = 0; i < GUARD; ++i)
        require(ap[i] == 0xa5 && bp[i] == 0xa5 &&
                ap[a.length - GUARD + i] == 0xa5 && bp[b.length - GUARD + i] == 0xa5,
                "output guard overwritten");
    const uint32_t *av = data(a), *bv = data(b);
    for (NSUInteger r = 0; r < rows; ++r) {
        for (NSUInteger c = 0; c < stride; ++c) {
            NSUInteger i = r * stride + c;
            if (c < width) {
                // Integer checks remain valid when host is built with fast-math.
                require((av[i] & 0x7f800000u) != 0x7f800000u &&
                        (bv[i] & 0x7f800000u) != 0x7f800000u, "nonfinite/unwritten output");
            } else require(av[i] == poison && bv[i] == poison, "stride padding overwritten");
            if (av[i] != bv[i]) {
                fprintf(stderr, "row=%lu col=%lu old=%08x new=%08x\n",
                        (unsigned long)r, (unsigned long)c, av[i], bv[i]);
                require(0, "output is not bitwise equal");
            }
        }
    }
}
static void finish(id<MTLCommandBuffer> cb) {
    [cb commit]; [cb waitUntilCompleted];
    if (cb.error) fprintf(stderr, "%s\n", cb.error.description.UTF8String);
    require(cb.status == MTLCommandBufferStatusCompleted, "GPU execution failed");
}
static void fill_x(id<MTLBuffer> x, NSUInteger count, uint32_t *seed) {
    float *p = data(x);
    for (NSUInteger i = 0; i < count; ++i)
        p[i] = i % 11 == 0 ? 0.f : ((int)(random_u32(seed) % 8193) - 4096) / 4096.f;
}
static void test_q4(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                    id<MTLComputePipelineState> oldp, id<MTLComputePipelineState> newp,
                    BOOL addr, int k, int rows, int tokens, float clamp) {
    uint32_t seed = 17u + k + rows + tokens;
    NSUInteger wb = (NSUInteger)(k / 256) * 144 * rows;
    id<MTLBuffer> gate[SLOTS], up[SLOTS];
    uint64_t original[2 * SLOTS];
    for (int s = 0; s < SLOTS; ++s) {
        gate[s] = buffer(dev, wb); up[s] = buffer(dev, wb);
        for (int arm = 0; arm < 2; ++arm) {
            q4_block *w = data(arm ? up[s] : gate[s]);
            for (NSUInteger i = 0; i < wb / 144; ++i) {
                w[i].d = 0x1800 + (random_u32(&seed) & 0x3ff);
                w[i].dmin = 0x1400 + (random_u32(&seed) & 0x3ff);
                for (int c = 0; c < 12; ++c) w[i].scales[c] = random_u32(&seed) >> 24;
                for (int c = 0; c < 128; ++c) w[i].qs[c] = random_u32(&seed) >> 24;
            }
        }
    }
    // Repeated routing id: two slots must address the same expert.
    gate[3] = gate[1]; up[3] = up[1];
    for (int s = 0; s < SLOTS; ++s) { original[s] = hash(gate[s]); original[6+s] = hash(up[s]); }
    int32_t expert[SLOTS] = {0, 17, 383, 17, 255, 1};
    id<MTLBuffer> ids = buffer(dev, tokens * SLOTS * 4);
    id<MTLBuffer> ga = buffer(dev, 384 * 8), ua = buffer(dev, 384 * 8);
    memset(data(ga), 0, 384 * 8); memset(data(ua), 0, 384 * 8);
    for (int s = 0; s < SLOTS; ++s) {
        ((uint64_t *)data(ga))[expert[s]] = gate[s].gpuAddress + GUARD;
        ((uint64_t *)data(ua))[expert[s]] = up[s].gpuAddress + GUARD;
        for (int t = 0; t < tokens; ++t) ((int32_t *)data(ids))[t*SLOTS+s] = expert[s];
    }
    id<MTLBuffer> x = buffer(dev, k * tokens * 4), weights = buffer(dev, SLOTS * tokens * 16);
    fill_x(x, k * tokens, &seed);
    for (int r = 0; r < tokens*SLOTS; ++r) ((float *)data(weights))[r*4] = (r%6) / 7.f;
    uint64_t xhash = hash(x), whash = hash(weights), ihash = hash(ids);
    const NSUInteger mid_stride = rows + 8;
    id<MTLBuffer> out[2][3];
    for (int arm = 0; arm < 2; ++arm)
        for (int d = 0; d < 3; ++d)
            out[arm][d] = output(dev, tokens * SLOTS * (d == 2 ? mid_stride : rows));
    id_args args = {
        .nei0 = SLOTS, .nei1 = tokens, .nbi1 = SLOTS * 4,
        .ne00 = k, .ne01 = rows, .ne02 = 384, .nb00 = 144,
        .nb01 = (uint64_t)k/256*144, .nb02 = wb,
        .ne10 = k, .ne11 = 1, .ne12 = tokens, .ne13 = 1,
        .nb10 = 4, .nb11 = (uint64_t)k*4, .nb12 = (uint64_t)k*4,
        .ne0 = rows, .ne1 = SLOTS, .nb1 = (uint64_t)rows*4, .nr0 = 2, .tp_world = 1,
    };
    act_args act = {.width = rows, .rows = tokens*SLOTS,
        .gate_row_stride = (uint64_t)rows*4, .up_row_stride = (uint64_t)rows*4,
        .mid_row_stride = mid_stride*4, .weight_stride = 16, .clamp_value = clamp};
    for (int arm = 0; arm < 2; ++arm) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:arm ? newp : oldp];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBytes:&act length:sizeof(act) atIndex:1];
        if (addr) {
            [enc setBuffer:ga offset:GUARD atIndex:2];
            [enc setBuffer:ua offset:GUARD atIndex:3];
            [enc setBuffer:x offset:GUARD atIndex:4];
            for (int d = 0; d < 3; ++d) [enc setBuffer:out[arm][d] offset:GUARD atIndex:5+d];
            [enc setBuffer:ids offset:GUARD atIndex:8];
            [enc setBuffer:weights offset:GUARD atIndex:9];
            for (int s = 0; s < SLOTS; ++s) {
                [enc useResource:gate[s] usage:MTLResourceUsageRead];
                [enc useResource:up[s] usage:MTLResourceUsageRead];
            }
        } else {
            for (int s = 0; s < SLOTS; ++s) {
                [enc setBuffer:gate[s] offset:GUARD atIndex:2+s];
                [enc setBuffer:up[s] offset:GUARD atIndex:8+s];
            }
            [enc setBuffer:x offset:GUARD atIndex:14];
            for (int d = 0; d < 3; ++d) [enc setBuffer:out[arm][d] offset:GUARD atIndex:15+d];
            [enc setBuffer:weights offset:GUARD atIndex:18];
        }
        [enc setThreadgroupMemoryLength:256 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(rows/4, 1, tokens*SLOTS)
             threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
        [enc endEncoding]; finish(cb);
    }
    for (int d = 0; d < 3; ++d)
        equal_output(out[0][d], out[1][d], tokens*SLOTS, rows, d == 2 ? mid_stride : rows);
    for (int s = 0; s < SLOTS; ++s)
        require(hash(gate[s]) == original[s] && hash(up[s]) == original[6+s], "Q4 weight mutated");
    require(hash(x) == xhash && hash(weights) == whash && hash(ids) == ihash, "Q4 input mutated");
}

static void test_q8(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                    id<MTLComputePipelineState> oldp, id<MTLComputePipelineState> newp,
                    int nsg, BOOL store, int k, int rows, float clamp) {
    uint32_t seed = 41u + k + rows;
    NSUInteger wb = (NSUInteger)(k / 32) * 34 * rows;
    id<MTLBuffer> gate = buffer(dev, wb), up = buffer(dev, wb), x = buffer(dev, k*4);
    for (int arm = 0; arm < 2; ++arm) {
        q8_block *w = data(arm ? up : gate);
        for (NSUInteger i = 0; i < wb/34; ++i) {
            w[i].d = 0x1800 + (random_u32(&seed) & 0x3ff);
            for (int c = 0; c < 32; ++c) w[i].qs[c] = (int)(random_u32(&seed) >> 24) - 128;
        }
    }
    fill_x(x, k, &seed);
    uint64_t gh = hash(gate), uh = hash(up);
    mv_args args = {.ne00=k, .ne01=rows, .ne02=1, .nb00=34,
        .nb01=(uint64_t)k/32*34, .nb02=wb, .nb03=wb,
        .ne10=k, .ne11=1, .ne12=1, .nb10=4, .nb11=(uint64_t)k*4,
        .nb12=(uint64_t)k*4, .nb13=(uint64_t)k*4, .ne0=rows, .ne1=1,
        .nr0=2, .r2=1, .r3=1};
    id<MTLBuffer> out[2][3];
    for (int arm = 0; arm < 2; ++arm)
        for (int d = 0; d < 3; ++d) out[arm][d] = output(dev, rows);
    // Repeated dispatches exercise recycled threadgroup memory. In mid-only
    // mode both unused diagnostic outputs alias mid, just like production.
    for (int repeat = 0; repeat < 5; ++repeat) {
        fill_x(x, k, &seed);
        uint64_t xh = hash(x);
        for (int arm = 0; arm < 2; ++arm)
            for (int d = 0; d < 3; ++d)
                for (int row = 0; row < rows; ++row)
                    ((uint32_t *)data(out[arm][d]))[row] = poison;
        for (int arm = 0; arm < 2; ++arm) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:arm ? newp : oldp];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:gate offset:GUARD atIndex:1];
            [enc setBuffer:up offset:GUARD atIndex:2];
            [enc setBuffer:x offset:GUARD atIndex:3];
            for (int d = 0; d < 3; ++d)
                [enc setBuffer:out[arm][store ? d : 2] offset:GUARD atIndex:4+d];
            [enc setBytes:&clamp length:sizeof(clamp) atIndex:7];
            [enc setThreadgroupMemoryLength:512 atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(rows/2, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
            [enc endEncoding]; finish(cb);
        }
        for (int d = store ? 0 : 2; d < 3; ++d)
            equal_output(out[0][d], out[1][d], 1, rows, rows);
        require(hash(x) == xh, "Q8 input mutated");
    }
    require(hash(gate) == gh && hash(up) == uh, "Q8 weight mutated");
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        require(dev != nil, "no Metal device (run on a Mac with GPU access)");
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        require(queue != nil, "command queue creation failed");
        NSError *error = nil;
        NSString *source = load_metal_source();
        require(source != nil, "shader source loading failed");
        id<MTLLibrary> lib = [dev newLibraryWithSource:source options:[MTLCompileOptions new] error:&error];
        if (!lib) fprintf(stderr, "%s\n", error.description.UTF8String);
        require(lib != nil, "Metal library compilation failed");
        int cases = 0;
        for (int addr = 0; addr < 2; ++addr) {
            NSString *base = addr ? @"kernel_mul_mv_addr_q4_K_pair_swiglu_f32" :
                                    @"kernel_mul_mv_slots6_q4_K_pair_swiglu_f32";
            id<MTLComputePipelineState> oldp = pipeline(dev, lib, base, 2);
            id<MTLComputePipelineState> newp = pipeline(dev, lib, [base stringByAppendingString:@"_shared_x"], 2);
            const int shapes[][3] = {{256,4,1}, {1024,32,3}, {4096,2048,1}, {7168,32,1}};
            for (unsigned s = 0; s < sizeof(shapes)/sizeof(shapes[0]); ++s)
                for (int clamp = 0; clamp < 2; ++clamp) {
                    test_q4(dev, queue, oldp, newp, addr, shapes[s][0], shapes[s][1], shapes[s][2], clamp*4.f);
                    ++cases;
                }
        }
        printf("Q4: 16 bitwise cases passed (slots/address, offsets, routing, clamp).\n");
        for (int nsg = 4; nsg <= 8; nsg += 4) for (int store = 0; store < 2; ++store) {
            NSString *base = store ? @"kernel_dsv4_shared_gate_up_swiglu_q8_0" :
                                     @"kernel_dsv4_shared_mid_swiglu_q8_0";
            id<MTLComputePipelineState> oldp = pipeline(dev, lib, base, nsg);
            id<MTLComputePipelineState> newp = pipeline(dev, lib, [base stringByAppendingString:@"_single_barrier"], nsg);
            const int shapes[][2] = {{32,2}, {256,32}, {4096,2048}, {7168,32}};
            for (unsigned s = 0; s < sizeof(shapes)/sizeof(shapes[0]); ++s)
                for (int clamp = 0; clamp < 2; ++clamp) {
                    test_q8(dev, queue, oldp, newp, nsg, store, shapes[s][0], shapes[s][1], clamp*4.f);
                    ++cases;
                }
        }
        printf("Q8: 32 bitwise cases passed (4/8 simdgroups, mid-only alias, clamp, repeated dispatch).\n");
        printf("PASS: %d SSD decode kernel cases on %s; no model or SSD I/O measured.\n",
               cases, dev.name.UTF8String);
    }
    return 0;
}
