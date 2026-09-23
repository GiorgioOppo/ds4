/* Independent production Q4_K attention kernel regression and benchmark.
 * Build (from the repository root):
 *   clang -O2 -Wall -Wextra -fobjc-arc -framework Foundation -framework Metal \
 *     tests/test_metal_qwen4_q4_attention.m -o /tmp/test_metal_qwen4_q4_attention
 * Run with --reference-repo PATH to compile the classic NR2 oracle from a
 * frozen repository/metal snapshot. --repo PATH selects candidate sources.
 * Both libraries use identical default-fast or safe MSL compilation options.
 * --compile-only creates pipelines without a queue or GPU dispatch.
 * --benchmark reports GPU command timestamps for balanced ABBA/BAAB samples.
 * Pair geometry is NR4, matching the production wrapper.
 * Small tail fixtures pad physical weight rows through a full eight-row tile;
 * only logical output rows may be written. Pair fixtures have complete tiles.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int32_t ne00, ne01, ne02;
    uint64_t nb00, nb01, nb02, nb03;
    int32_t ne10, ne11, ne12;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne0, ne1, nr0;
    int16_t r2, r3;
} mv_args;
typedef struct { _Float16 d, dmin; uint8_t scales[12], qs[128]; } q4_block;
typedef struct { uint32_t k, rows, tokens; } shape;
_Static_assert(sizeof(mv_args) == 112, "production matvec argument ABI");
_Static_assert(offsetof(mv_args, nb10) == 64, "activation stride ABI");
_Static_assert(sizeof(q4_block) == 144, "GGUF Q4_K block ABI");
_Static_assert(offsetof(q4_block, qs) == 16, "GGUF Q4_K nibble offset");

static const uint32_t kCanary = UINT32_C(0x7f24d13b);
static const NSUInteger kOffset = 68; /* Four-byte aligned, not float4 aligned. */
static const uint32_t kTokens[] = {1, 2, 3, 8};
static const shape kSmall[] = {
    {256, 1, 1}, {512, 3, 1}, {768, 7, 1}, {1024, 8, 1},
    {1280, 9, 1}, {2560, 15, 1}, {6144, 17, 1}, {2560, 31, 1},
};
static const shape kReal[] = {
    {2560, 10240, 1}, {2560, 6144, 1},
    {2560, 12288, 1}, {6144, 2560, 1},
};
static uint64_t values_checked, cases_checked;

static void fail(NSString *message) {
    fprintf(stderr, "FAIL Qwen Q4 attention: %s\n", message.UTF8String);
    exit(1);
}
static NSString *read_source(NSString *path) {
    NSError *error = nil;
    NSString *text = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
    if (!text) fail([NSString stringWithFormat:@"read %@: %@", path, error]);
    return text;
}
static NSString *source_path(NSString *repo, NSString *name) {
    NSString *path = [[repo stringByAppendingPathComponent:@"metal"] stringByAppendingPathComponent:name];
    if (![[NSFileManager defaultManager] fileExistsAtPath:path])
        path = [repo stringByAppendingPathComponent:name];
    return path;
}
static NSString *between(NSString *text, NSString *begin, NSString *end) {
    if ([text componentsSeparatedByString:begin].count != 2 ||
        [text componentsSeparatedByString:end].count != 2)
        fail([NSString stringWithFormat:@"source anchor missing/nonunique: %@ ... %@", begin, end]);
    NSRange a = [text rangeOfString:begin], b = [text rangeOfString:end];
    if (b.location <= a.location) fail(@"source anchors out of order");
    return [text substringWithRange:NSMakeRange(a.location, b.location - a.location)];
}
static NSString *kernel_source(NSString *text, NSString *name) {
    NSString *anchor = [NSString stringWithFormat:@"kernel void %@(", name];
    if ([text componentsSeparatedByString:anchor].count != 2)
        fail([NSString stringWithFormat:@"kernel missing/nonunique: %@", name]);
    NSUInteger start = [text rangeOfString:anchor].location;
    NSRange opening = [text rangeOfString:@"{" options:0 range:NSMakeRange(start, text.length - start)];
    if (opening.location == NSNotFound) fail(@"missing kernel body");
    int depth = 0;
    for (NSUInteger i = opening.location; i < text.length; ++i) {
        unichar ch = [text characterAtIndex:i];
        if (ch == '{') ++depth;
        if (ch == '}' && --depth == 0)
            return [[text substringWithRange:NSMakeRange(start, i + 1 - start)] stringByAppendingString:@"\n"];
    }
    fail(@"unterminated kernel body");
    return nil;
}
static NSString *shader_source(NSString *repo, bool candidate) {
    NSString *dense = read_source(source_path(repo, @"dense.metal"));
    NSString *moe = read_source(source_path(repo, @"moe.metal"));
    NSMutableString *body = [NSMutableString stringWithString:
        @"#include <metal_stdlib>\nusing namespace metal;\n"
         "#define FC_MUL_MV 600\n#define QK_K 256\n#define N_R0_Q4_K 2\n"
         "#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"];
    [body appendString:between(dense, @"constant short FC_mul_mv_nsg", @"struct ds4_metal_args_compressor_pair_store {")];
    [body appendString:between(moe, @"struct block_q4_K {", @"struct block_q5_K {")];
    [body appendString:between(moe, @"template<int nr0, typename args_t>\nvoid kernel_mul_mv_q4_K_f32_impl(",
                                   @"template<int nr0, typename args_t>\nvoid kernel_mul_mv_mxfp4_f32_impl(")];
    [body appendString:kernel_source(moe, @"kernel_mul_mv_q4_K_dense_f32")];
    if (candidate) {
        [body appendString:kernel_source(moe, @"kernel_qwen4_attn_q4_K_nr4_f32")];
        [body appendString:kernel_source(moe, @"kernel_qwen4_attn_q4_K_pair_f32")];
    }
    return body;
}
static uint64_t source_hash(NSString *source) {
    NSData *data = [source dataUsingEncoding:NSUTF8StringEncoding];
    const uint8_t *p = data.bytes;
    uint64_t hash = UINT64_C(14695981039346656037);
    for (NSUInteger i = 0; i < data.length; ++i) hash = (hash ^ p[i]) * UINT64_C(1099511628211);
    return hash;
}
static id<MTLLibrary> library(id<MTLDevice> device, NSString *source, bool safe) {
    MTLCompileOptions *options = [MTLCompileOptions new];
    if (safe) {
        if (@available(macOS 15.0, *)) options.mathMode = MTLMathModeSafe;
        else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
    }
    NSError *error = nil;
    id<MTLLibrary> result = [device newLibraryWithSource:source options:options error:&error];
    if (!result) fail([NSString stringWithFormat:@"MSL compile (%@): %@", safe ? @"safe" : @"default-fast", error]);
    return result;
}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> lib, NSString *name) {
    short nsg = 2;
    MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
    NSError *error = nil;
    id<MTLFunction> function = [lib newFunctionWithName:name constantValues:constants error:&error];
    id<MTLComputePipelineState> result = function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
    if (!result) fail([NSString stringWithFormat:@"pipeline %@: %@", name, error]);
    if (result.threadExecutionWidth != 32 || result.maxTotalThreadsPerThreadgroup < 64)
        fail(@"device does not support NSG=2 / 64-thread geometry");
    return result;
}
static uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= UINT32_C(0x7feb352d);
    x ^= x >> 15; x *= UINT32_C(0x846ca68b);
    return x ^ (x >> 16);
}
static void *payload(id<MTLBuffer> buffer) { return (char *)buffer.contents + kOffset; }
static id<MTLBuffer> buffer(id<MTLDevice> device, size_t bytes) {
    bytes = (bytes + 3u) & ~(size_t)3u;
    id<MTLBuffer> result = [device newBufferWithLength:bytes + 2*kOffset options:MTLResourceStorageModeShared];
    if (!result) fail(@"buffer allocation");
    uint32_t *words = result.contents;
    for (NSUInteger i = 0; i < result.length / 4; ++i) words[i] = kCanary;
    return result;
}
static void guards(id<MTLBuffer> buffer, uint64_t words) {
    const uint32_t *p = buffer.contents;
    for (NSUInteger i = 0; i < buffer.length / 4; ++i)
        if ((i < kOffset/4 || i >= kOffset/4 + words) && p[i] != kCanary)
            fail(@"output prefix/suffix/tail guard overwritten");
}
static void fill_weights(id<MTLBuffer> buffer, shape s, int pattern, uint32_t seed) {
    q4_block *weights = payload(buffer);
    uint32_t blocks = s.k/256, physical_rows = (s.rows + 7u) & ~7u;
    for (uint32_t row = 0; row < physical_rows; ++row) for (uint32_t b = 0; b < blocks; ++b) {
        q4_block *block = &weights[(uint64_t)row*blocks + b];
        uint32_t random = mix(row*65537u + b*313u + seed);
        float sign = (random & 1u) ? -1.f : 1.f;
        block->d = (_Float16)ldexpf(0.75f + (random % 997u)/997.f, (int)(b % 9u) - 12);
        block->dmin = (_Float16)ldexpf(0.5f + (random % 113u)/127.f, (int)(b % 7u) - 12);
        if (pattern == 0 || pattern == 2) { block->d = (_Float16)0.5f; block->dmin = (_Float16)0.25f; }
        if (pattern == 3) {
            block->d = (_Float16)(sign * ldexpf(1.f + (random % 17u)/32.f, (int)(b % 20u) - 14));
            block->dmin = (_Float16)(-sign * ldexpf(1.f, (int)(b % 17u) - 15));
        }
        if (pattern == 4) { block->d = (_Float16)(sign*0.f); block->dmin = (_Float16)(-sign*0.f); }
        uint8_t scales[8], minima[8];
        for (uint32_t j = 0; j < 8; ++j) {
            scales[j] = (uint8_t)(mix(random + j*811u) & 63u);
            minima[j] = (uint8_t)(mix(random + j*3571u + 29u) & 63u);
            if (pattern == 0 || pattern == 2) { scales[j] = (uint8_t)(1 + row % 7); minima[j] = 1; }
            if (pattern == 5) { scales[j] = (j & 1) ? 63 : 16; minima[j] = (j & 1) ? 32 : 15; }
        }
        for (uint32_t j = 0; j < 4; ++j) {
            block->scales[j] = scales[j] | ((scales[j + 4] >> 4) << 6);
            block->scales[j + 4] = minima[j] | ((minima[j + 4] >> 4) << 6);
            block->scales[j + 8] = (scales[j + 4] & 15u) | ((minima[j + 4] & 15u) << 4);
        }
        for (uint32_t q = 0; q < 128; ++q) {
            block->qs[q] = (uint8_t)mix(random + q*7919u);
            if (pattern == 0 || pattern == 2) block->qs[q] = 0x11;
            if (pattern == 5) block->qs[q] = q & 1u ? 0xf0 : 0x0f;
        }
    }
}
static void fill_input(id<MTLBuffer> buffer, shape s, int pattern) {
    float *x = payload(buffer);
    for (uint32_t token = 0; token < s.tokens; ++token) for (uint32_t k = 0; k < s.k; ++k) {
        uint32_t random = mix(token*65537u + k*7919u + 123u);
        float value = ((int)(random % 65521u) - 32760) / 16384.f + 0.00000017f;
        if (pattern == 0) value = (token + 1u) * 0.125f;
        if (pattern == 1 && k % 11u == 0) value *= 16.f;
        if (pattern == 2) {
            const float cancellation[] = {0x1p20f, 0.06250001f, -0x1p20f, 1.00000012f,
                                          -0x1p18f, -0.25000003f, 0x1p18f, -0.50000006f};
            value = cancellation[k % 8u] * (token + 1u);
        }
        if (pattern == 3) value = (k / 32u % 2u ? -1.f : 1.f) * (1.f + (k % 31u)/31.f);
        if (pattern == 4) value = (k % 2u ? -1.f : 1.f) * (k % 5u ? 0.f : 0x1p-140f);
        if (pattern == 5) value = ldexpf(value, (int)(k % 25u) - 12);
        x[(uint64_t)token*s.k + k] = value;
    }
}
static mv_args arguments(shape s, uint32_t nr0) {
    uint64_t row_bytes = (uint64_t)(s.k/256u)*sizeof(q4_block);
    return (mv_args){
        .ne00=s.k, .ne01=s.rows, .ne02=1,
        .nb00=sizeof(q4_block), .nb01=row_bytes, .nb02=row_bytes*s.rows, .nb03=row_bytes*s.rows,
        .ne10=s.k, .ne11=s.tokens, .ne12=1,
        .nb10=4, .nb11=(uint64_t)s.k*4, .nb12=(uint64_t)s.k*s.tokens*4, .nb13=(uint64_t)s.k*s.tokens*4,
        .ne0=s.rows, .ne1=s.tokens, .nr0=nr0, .r2=1, .r3=1,
    };
}
static void encode_single(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipe,
                          id<MTLBuffer> weights, id<MTLBuffer> input, id<MTLBuffer> output,
                          shape s, uint32_t nr0) {
    mv_args args = arguments(s, nr0);
    [encoder setComputePipelineState:pipe];
    [encoder setBytes:&args length:sizeof(args) atIndex:0];
    [encoder setBuffer:weights offset:kOffset atIndex:1];
    [encoder setBuffer:input offset:kOffset atIndex:2];
    [encoder setBuffer:output offset:kOffset atIndex:3];
    [encoder setThreadgroupMemoryLength:32 atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake((s.rows + 2*nr0 - 1u)/(2*nr0), s.tokens, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
}
static void encode_pair(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipe,
                        id<MTLBuffer> w0, id<MTLBuffer> w1, id<MTLBuffer> input,
                        id<MTLBuffer> out0, id<MTLBuffer> out1, shape s0, shape s1, uint32_t nr0) {
    if (s0.k != s1.k || s0.tokens != s1.tokens || s0.rows % (2*nr0) || s1.rows % (2*nr0))
        fail(@"invalid pair fixture geometry");
    mv_args a0 = arguments(s0, nr0), a1 = arguments(s1, nr0);
    [encoder setComputePipelineState:pipe];
    [encoder setBytes:&a0 length:sizeof(a0) atIndex:0];
    [encoder setBytes:&a1 length:sizeof(a1) atIndex:1];
    [encoder setBuffer:w0 offset:kOffset atIndex:2];
    [encoder setBuffer:w1 offset:kOffset atIndex:3];
    [encoder setBuffer:input offset:kOffset atIndex:4];
    [encoder setBuffer:out0 offset:kOffset atIndex:5];
    [encoder setBuffer:out1 offset:kOffset atIndex:6];
    [encoder setThreadgroupMemoryLength:32 atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake((s0.rows + s1.rows)/(2*nr0), s0.tokens, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
}
static void complete(id<MTLCommandBuffer> command) {
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        fail([NSString stringWithFormat:@"GPU execution: %@", command.error]);
}
static void compare(id<MTLBuffer> expected, id<MTLBuffer> actual, shape s, bool safe,
                    int pattern, NSString *label) {
    uint64_t n = (uint64_t)s.rows*s.tokens;
    guards(expected, n); guards(actual, n);
    const uint32_t *want = payload(expected), *got = payload(actual);
    const float *wf = (const float *)want, *gf = (const float *)got;
    for (uint64_t i = 0; i < n; ++i) {
        if (want[i] == kCanary || got[i] == kCanary || !isfinite(wf[i]) || !isfinite(gf[i]))
            fail(@"unwritten or nonfinite output from finite fixture");
        if (want[i] != got[i]) fail([NSString stringWithFormat:
            @"%@ %@ K=%u rows=%u T=%u pattern=%d at=%llu: NR2=%08x %.9g candidate=%08x %.9g",
            safe ? @"safe" : @"default-fast", label, s.k, s.rows, s.tokens, pattern,
            (unsigned long long)i, want[i], wf[i], got[i], gf[i]]);
    }
    values_checked += n;
}
static void unchanged(id<MTLBuffer> buffer, NSData *snapshot) {
    if (memcmp(buffer.contents, snapshot.bytes, buffer.length)) fail(@"input/weight buffer or its guards modified");
}
static void run_case(id<MTLDevice> device, id<MTLCommandQueue> queue,
                     id<MTLComputePipelineState> reference, id<MTLComputePipelineState> current,
                     id<MTLComputePipelineState> nr4, shape s, bool safe, int pattern) {
    id<MTLBuffer> weights = buffer(device, (uint64_t)((s.rows+7u)&~7u)*(s.k/256u)*144u);
    id<MTLBuffer> input = buffer(device, (uint64_t)s.k*s.tokens*4u);
    id<MTLBuffer> expected = buffer(device, (uint64_t)s.rows*s.tokens*4u);
    id<MTLBuffer> out2 = buffer(device, (uint64_t)s.rows*s.tokens*4u);
    id<MTLBuffer> out4 = buffer(device, (uint64_t)s.rows*s.tokens*4u);
    fill_weights(weights, s, pattern, 471); fill_input(input, s, pattern);
    NSData *wcopy = [NSData dataWithBytes:weights.contents length:weights.length];
    NSData *xcopy = [NSData dataWithBytes:input.contents length:input.length];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!command || !encoder) fail(@"command allocation");
    encode_single(encoder, reference, weights, input, expected, s, 2);
    encode_single(encoder, current, weights, input, out2, s, 2);
    encode_single(encoder, nr4, weights, input, out4, s, 4);
    [encoder endEncoding]; complete(command);
    compare(expected, out2, s, safe, pattern, @"current NR2");
    compare(expected, out4, s, safe, pattern, @"NR4");
    unchanged(weights, wcopy); unchanged(input, xcopy);
    ++cases_checked;
}
static void run_pair(id<MTLDevice> device, id<MTLCommandQueue> queue,
                     id<MTLComputePipelineState> reference, id<MTLComputePipelineState> pair,
                     shape s0, shape s1, uint32_t nr0, bool safe, int pattern) {
    id<MTLBuffer> w0 = buffer(device, (uint64_t)s0.rows*(s0.k/256u)*144u);
    id<MTLBuffer> w1 = buffer(device, (uint64_t)s1.rows*(s1.k/256u)*144u);
    id<MTLBuffer> input = buffer(device, (uint64_t)s0.k*s0.tokens*4u);
    id<MTLBuffer> expected0 = buffer(device, (uint64_t)s0.rows*s0.tokens*4u);
    id<MTLBuffer> expected1 = buffer(device, (uint64_t)s1.rows*s1.tokens*4u);
    id<MTLBuffer> out0 = buffer(device, (uint64_t)s0.rows*s0.tokens*4u);
    id<MTLBuffer> out1 = buffer(device, (uint64_t)s1.rows*s1.tokens*4u);
    fill_weights(w0, s0, pattern, 471); fill_weights(w1, s1, pattern, 9721);
    fill_input(input, s0, pattern);
    NSData *copy0 = [NSData dataWithBytes:w0.contents length:w0.length];
    NSData *copy1 = [NSData dataWithBytes:w1.contents length:w1.length];
    NSData *xcopy = [NSData dataWithBytes:input.contents length:input.length];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!command || !encoder) fail(@"pair command allocation");
    encode_single(encoder, reference, w0, input, expected0, s0, 2);
    encode_single(encoder, reference, w1, input, expected1, s1, 2);
    encode_pair(encoder, pair, w0, w1, input, out0, out1, s0, s1, nr0);
    [encoder endEncoding]; complete(command);
    compare(expected0, out0, s0, safe, pattern, @"pair output0");
    compare(expected1, out1, s1, safe, pattern, @"pair output1");
    unchanged(w0, copy0); unchanged(w1, copy1); unchanged(input, xcopy);
    ++cases_checked;
}

/* Each timing command contains many independent writes to the same outputs,
 * matching the baseline/candidate memory dependencies. Two separate matvecs
 * share one encoder, so the pair benchmark does not inflate encoder overhead. */
typedef double (^TimedArm)(uint32_t iterations);
static double gpu_us(id<MTLCommandBuffer> command, uint32_t iterations) {
    complete(command);
    double seconds = command.GPUEndTime - command.GPUStartTime;
    if (!(seconds > 0) || !isfinite(seconds)) fail(@"GPU timestamps unavailable");
    return seconds * 1.e6 / iterations;
}
static double median(NSArray<NSNumber *> *values) {
    NSArray<NSNumber *> *sorted = [values sortedArrayUsingSelector:@selector(compare:)];
    NSUInteger n = sorted.count;
    return (sorted[(n-1)/2].doubleValue + sorted[n/2].doubleValue)*0.5;
}
static void balanced_benchmark(NSString *label, TimedArm first, TimedArm second,
                               uint32_t rounds, uint32_t iterations) {
    for (uint32_t i = 0; i < 3; ++i) { (void)first(iterations); (void)second(iterations); }
    NSMutableArray<NSNumber *> *a = [NSMutableArray new], *b = [NSMutableArray new], *ratios = [NSMutableArray new];
    for (uint32_t round = 0; round < rounds; ++round) {
        double aa[2], bb[2];
        if (round & 1u) { bb[0]=second(iterations); aa[0]=first(iterations); aa[1]=first(iterations); bb[1]=second(iterations); }
        else { aa[0]=first(iterations); bb[0]=second(iterations); bb[1]=second(iterations); aa[1]=first(iterations); }
        double av = (aa[0]+aa[1])*0.5, bv = (bb[0]+bb[1])*0.5;
        [a addObject:@(av)]; [b addObject:@(bv)]; [ratios addObject:@(av/bv)];
    }
    printf("BENCH %s: first %.3f us, second %.3f us, median paired speedup %.4fx (%u balanced rounds, %u iterations/sample)\n",
           label.UTF8String, median(a), median(b), median(ratios), rounds, iterations);
}
static void bench_single(id<MTLDevice> device, id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> nr2, id<MTLComputePipelineState> nr4,
                         shape s, uint32_t rounds, uint32_t iterations) {
    id<MTLBuffer> weights = buffer(device, (uint64_t)s.rows*(s.k/256u)*144u);
    id<MTLBuffer> input = buffer(device, (uint64_t)s.k*s.tokens*4u);
    id<MTLBuffer> out2 = buffer(device, (uint64_t)s.rows*s.tokens*4u), out4 = buffer(device, (uint64_t)s.rows*s.tokens*4u);
    fill_weights(weights, s, 1, 471); fill_input(input, s, 1);
    TimedArm arm2 = ^double(uint32_t count) {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) fail(@"benchmark command allocation");
        for (uint32_t i = 0; i < count; ++i) encode_single(encoder, nr2, weights, input, out2, s, 2);
        [encoder endEncoding]; return gpu_us(command, count);
    };
    TimedArm arm4 = ^double(uint32_t count) {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) fail(@"benchmark command allocation");
        for (uint32_t i = 0; i < count; ++i) encode_single(encoder, nr4, weights, input, out4, s, 4);
        [encoder endEncoding]; return gpu_us(command, count);
    };
    balanced_benchmark([NSString stringWithFormat:@"NR2/NR4 NSG=2 K=%u rows=%u T=%u", s.k, s.rows, s.tokens], arm2, arm4, rounds, iterations);
    compare(out2, out4, s, false, 1, @"benchmark NR4");
}
static void bench_pair(id<MTLDevice> device, id<MTLCommandQueue> queue,
                       id<MTLComputePipelineState> separate, uint32_t separate_nr0,
                       id<MTLComputePipelineState> pair, uint32_t pair_nr0,
                       uint32_t tokens, uint32_t rounds, uint32_t iterations) {
    shape s0 = {2560, 10240, tokens}, s1 = {2560, 6144, tokens};
    id<MTLBuffer> w0 = buffer(device, (uint64_t)s0.rows*10u*144u), w1 = buffer(device, (uint64_t)s1.rows*10u*144u);
    id<MTLBuffer> input = buffer(device, (uint64_t)s0.k*tokens*4u);
    id<MTLBuffer> sep0 = buffer(device, (uint64_t)s0.rows*tokens*4u), sep1 = buffer(device, (uint64_t)s1.rows*tokens*4u);
    id<MTLBuffer> fused0 = buffer(device, (uint64_t)s0.rows*tokens*4u), fused1 = buffer(device, (uint64_t)s1.rows*tokens*4u);
    fill_weights(w0, s0, 1, 471); fill_weights(w1, s1, 1, 9721); fill_input(input, s0, 1);
    TimedArm separate_arm = ^double(uint32_t count) {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) fail(@"pair benchmark command allocation");
        for (uint32_t i = 0; i < count; ++i) {
            encode_single(encoder, separate, w0, input, sep0, s0, separate_nr0);
            encode_single(encoder, separate, w1, input, sep1, s1, separate_nr0);
        }
        [encoder endEncoding]; return gpu_us(command, count);
    };
    TimedArm pair_arm = ^double(uint32_t count) {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) fail(@"pair benchmark command allocation");
        for (uint32_t i = 0; i < count; ++i) encode_pair(encoder, pair, w0, w1, input, fused0, fused1, s0, s1, pair_nr0);
        [encoder endEncoding]; return gpu_us(command, count);
    };
    balanced_benchmark([NSString stringWithFormat:@"separate2-NR%u/pair-NR%u qkv10240+gate6144 K=2560 T=%u",
                        separate_nr0, pair_nr0, tokens], separate_arm, pair_arm, rounds, iterations);
    compare(sep0, fused0, s0, false, 1, @"benchmark pair0");
    compare(sep1, fused1, s1, false, 1, @"benchmark pair1");
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        NSString *repo = @".", *reference_repo = nil;
        bool benchmark = false, compile_only = false;
        uint32_t rounds = 8, iterations = 32;
        const uint32_t pair_nr0 = 4;
        for (int i = 1; i < argc; ++i) {
            if (!strcmp(argv[i], "--repo") && i + 1 < argc) repo = @(argv[++i]);
            else if (!strcmp(argv[i], "--reference-repo") && i + 1 < argc) reference_repo = @(argv[++i]);
            else if (!strcmp(argv[i], "--benchmark")) benchmark = true;
            else if (!strcmp(argv[i], "--compile-only")) compile_only = true;
            else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (!strcmp(argv[i], "--iterations") && i + 1 < argc) iterations = (uint32_t)strtoul(argv[++i], NULL, 10);
            else {
                fprintf(stderr, "Usage: %s [--repo PATH] [--reference-repo PATH] [--compile-only] [--benchmark] [--rounds N] [--iterations N]\n", argv[0]);
                return 2;
            }
        }
        if (!rounds || rounds > 1000 || !iterations || iterations > 10000) {
            fprintf(stderr, "rounds must be 1..1000; iterations 1..10000\n");
            return 2;
        }
        reference_repo = reference_repo ?: repo;
        NSString *ref_source = shader_source(reference_repo, false), *candidate_source = shader_source(repo, true);
        printf("Q4 attention source hashes (FNV64): reference %016llx candidate %016llx; reference=%s; pairNR%u; view offset=%lu\n",
               (unsigned long long)source_hash(ref_source), (unsigned long long)source_hash(candidate_source),
               reference_repo.UTF8String, pair_nr0, (unsigned long)kOffset);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) fail(@"Metal device unavailable");
        NSMutableArray<id<MTLComputePipelineState>> *pipelines = [NSMutableArray new];
        for (int mode = 0; mode < 2; ++mode) {
            id<MTLLibrary> ref = library(device, ref_source, mode != 0), current = library(device, candidate_source, mode != 0);
            [pipelines addObject:pipeline(device, ref, @"kernel_mul_mv_q4_K_dense_f32")];
            [pipelines addObject:pipeline(device, current, @"kernel_mul_mv_q4_K_dense_f32")];
            [pipelines addObject:pipeline(device, current, @"kernel_qwen4_attn_q4_K_nr4_f32")];
            [pipelines addObject:pipeline(device, current, @"kernel_qwen4_attn_q4_K_pair_f32")];
            printf("PASS compile %s: reference NR2, current NR2, NR4, pair; NSG=2\n", mode ? "safe" : "default-fast");
        }
        if (compile_only) { puts("PASS compile-only: 8 pipelines; no queue or dispatch"); return 0; }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) fail(@"command queue allocation");
        for (int mode = 0; mode < 2; ++mode) {
            NSUInteger pi = (NSUInteger)mode*4;
            for (size_t t = 0; t < sizeof(kTokens)/sizeof(kTokens[0]); ++t) {
                for (size_t j = 0; j < sizeof(kSmall)/sizeof(kSmall[0]); ++j)
                    for (int pattern = 0; pattern < 6; ++pattern) @autoreleasepool {
                        shape s = kSmall[j]; s.tokens = kTokens[t];
                        run_case(device, queue, pipelines[pi], pipelines[pi+1], pipelines[pi+2], s, mode != 0, pattern);
                    }
                for (size_t j = 0; j < sizeof(kReal)/sizeof(kReal[0]); ++j)
                    for (int pattern = 1; pattern <= 2; ++pattern) @autoreleasepool {
                        shape s = kReal[j]; s.tokens = kTokens[t];
                        run_case(device, queue, pipelines[pi], pipelines[pi+1], pipelines[pi+2], s, mode != 0, pattern);
                    }
                const uint32_t pair_rows[][2] = {{8, 16}, {24, 8}};
                for (size_t j = 0; j < sizeof(pair_rows)/sizeof(pair_rows[0]); ++j)
                    for (int pattern = 0; pattern < 6; ++pattern) @autoreleasepool {
                        shape a = {j ? 2560 : 256, pair_rows[j][0], kTokens[t]};
                        shape b = {a.k, pair_rows[j][1], a.tokens};
                        run_pair(device, queue, pipelines[pi], pipelines[pi+3], a, b, pair_nr0, mode != 0, pattern);
                    }
                for (int pattern = 1; pattern <= 2; ++pattern) @autoreleasepool {
                    shape a = {2560, 10240, kTokens[t]}, b = {2560, 6144, kTokens[t]};
                    run_pair(device, queue, pipelines[pi], pipelines[pi+3], a, b, pair_nr0, mode != 0, pattern);
                }
            }
            printf("PASS bitwise %s: single/pair, T=1/2/3/8, real shapes, cancellation/scales, tails and input/output guards\n",
                   mode ? "safe" : "default-fast");
        }
        if (benchmark) {
            printf("BENCH device=%s; resident weights; default-fast; GPU timestamps; ABBA/BAAB\n", device.name.UTF8String);
            for (size_t t = 0; t < sizeof(kTokens)/sizeof(kTokens[0]); ++t) {
                for (size_t j = 0; j < sizeof(kReal)/sizeof(kReal[0]); ++j) @autoreleasepool {
                    shape s = kReal[j]; s.tokens = kTokens[t];
                    bench_single(device, queue, pipelines[1], pipelines[2], s, rounds, iterations);
                }
                for (uint32_t nr0 = 2; nr0 <= 4; nr0 += 2) @autoreleasepool {
                    bench_pair(device, queue, pipelines[nr0/2], nr0, pipelines[3], pair_nr0,
                               kTokens[t], rounds, iterations);
                }
            }
        }
        printf("PASS Qwen Q4 attention: %llu fixtures, %llu bitwise values; weights/inputs unchanged; canaries intact\n",
               (unsigned long long)cases_checked, (unsigned long long)values_checked);
        return 0;
    }
}
