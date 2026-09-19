/* Bitwise regression checks and microbenchmarks for Bonsai normalization and
 * Hadamard transforms. No model file is required. The original implementations
 * are embedded so later production changes retain a fixed arithmetic oracle.
 *
 * clang -O2 -fobjc-arc tests/bench_bonsai_elementwise.m -framework Foundation \
 *       -framework Metal -o /tmp/bench_bonsai_elementwise
 * /tmp/bench_bonsai_elementwise [metal/bonsai.metal]
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t n, rows, cols, type, row_bytes, pos, heads, kvheads;
    uint32_t dim, rot, width, mode, groups;
    float eps, base;
} Args;
_Static_assert(sizeof(Args) == 60, "shader arguments");

static NSString *baseline =
@"kernel void bench_hadamard_baseline(constant BonsaiArgs &a [[buffer(0)]],\n"
"                            device const float *x [[buffer(1)]], device const int *signs [[buffer(2)]],\n"
"                            device float *out [[buffer(3)]], uint block [[threadgroup_position_in_grid]],\n"
"                            uint tid [[thread_index_in_threadgroup]]) {\n"
"    threadgroup float v[1024];\n"
"    for (uint j = tid; j < 1024; j += 256) {\n"
"        uint dst = block * 1024u + j, src = dst;\n"
"        if (a.groups) {\n"
"            // Output head order: [rep][key head] -> [key head][rep].\n"
"            uint head = dst / a.dim, d = dst % a.dim, rep = a.heads / a.groups;\n"
"            src = ((head % rep) * a.groups + head / rep) * a.dim + d;\n"
"        }\n"
"        v[j] = x[src] * (a.mode ? 1.0f : float(signs[dst]));\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint stride = 1; stride < 1024; stride *= 2) {\n"
"        for (uint pair = tid; pair < 512; pair += 256) {\n"
"            uint i = (pair / stride) * (2u * stride) + pair % stride;\n"
"            float p = v[i], q = v[i + stride];\n"
"            v[i] = p + q; v[i + stride] = p - q;\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    for (uint j = tid; j < 1024; j += 256) {\n"
"        uint i = block * 1024u + j;\n"
"        out[i] = (v[j] * (1.0f / 32.0f)) * (a.mode ? float(signs[i]) : 1.0f);\n"
"    }\n"
"}\n"
"\n"
"// width is the input stride; output rows are contiguous. L2 deliberately\n"
"// uses max(sqrt(sum), epsilon), not rsqrt(sum + epsilon).\n"
"kernel void bench_norm_baseline(constant BonsaiArgs &a [[buffer(0)]],\n"
"                        device const float *x [[buffer(1)]], device const uchar *w [[buffer(2)]],\n"
"                        device float *out [[buffer(3)]], uint row [[threadgroup_position_in_grid]],\n"
"                        uint tid [[thread_index_in_threadgroup]]) {\n"
"    threadgroup float tmp[256];\n"
"    float sum = 0;\n"
"    for (uint i = tid; i < a.cols; i += 256) { float v = x[ulong(row) * a.width + i]; sum += v * v; }\n"
"    tmp[tid] = sum;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint d = 128; d; d /= 2) {\n"
"        if (tid < d) tmp[tid] += tmp[tid + d];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const float scale = a.mode ? 1.0f / max(sqrt(tmp[0]), a.eps) : rsqrt(tmp[0] / float(a.cols) + a.eps);\n"
"    for (uint i = tid; i < a.cols; i += 256)\n"
"        out[ulong(row) * a.cols + i] = x[ulong(row) * a.width + i] * scale * (a.mode ? 1.0f : bs_scalar(w, i, a.type));\n"
"}\n";

static void need(bool ok, const char *message) {
    if (!ok) { fprintf(stderr, "FAIL %s\n", message); exit(1); }
}

static int compare_time(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static id<MTLBuffer> make_buffer(id<MTLDevice> device, size_t bytes) {
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    need(buffer != nil, "buffer allocation");
    return buffer;
}

static void fill(id<MTLBuffer> buffer, unsigned pattern, float scale) {
    float *p = buffer.contents;
    for (size_t i = 0; i < buffer.length / sizeof(float); ++i) {
        if (pattern == 0) p[i] = (sinf((float)i * .137f) + cosf((float)i * .321f)) * scale;
        else if (pattern == 1) p[i] = ldexpf((float)((int)(i % 31u) - 15), (int)(i % 17u) - 12) * scale;
        else p[i] = i % 19u ? (i % 2u ? 0.0f : -0.0f) : (i % 3u ? scale : -scale);
    }
}

static double run(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                  Args args, NSArray<id<MTLBuffer>> *buffers, uint32_t groups, uint32_t repeats) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBytes:&args length:sizeof(args) atIndex:0];
    for (NSUInteger i = 0; i < buffers.count; ++i)
        [encoder setBuffer:buffers[i] offset:0 atIndex:i + 1];
    for (uint32_t i = 0; i < repeats; ++i)
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted)
        fprintf(stderr, "%s\n", cb.error.localizedDescription.UTF8String);
    need(cb.status == MTLCommandBufferStatusCompleted, "GPU completion");
    return (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / repeats;
}

static void check_exact(id<MTLBuffer> output, const float *expected, size_t bytes, const char *label) {
    const float *actual = output.contents;
    size_t differences = 0;
    double max_abs = 0;
    for (size_t i = 0; i < bytes / sizeof(float); ++i) {
        need(isfinite(actual[i]) && isfinite(expected[i]), "finite reference and output");
        differences += memcmp(actual + i, expected + i, sizeof(float)) != 0;
        max_abs = fmax(max_abs, fabs(actual[i] - expected[i]));
    }
    if (differences)
        fprintf(stderr, "%s: %zu differing floats, max_abs=%.9g\n", label, differences, max_abs);
    need(differences == 0, "bitwise arithmetic parity");
}

static NSArray<id<MTLComputePipelineState>> *make_pipelines(id<MTLDevice> device, id<MTLLibrary> library,
                                                          NSArray<NSString *> *names) {
    NSMutableArray *pipelines = [NSMutableArray array];
    for (NSString *name in names) {
        NSError *error = nil;
        id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:[library newFunctionWithName:name]
                                                                           error:&error];
        if (!p) fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
        need(p != nil, "pipeline compilation");
        [pipelines addObject:p];
    }
    return pipelines;
}

static void benchmark(id<MTLCommandQueue> queue, NSArray<id<MTLComputePipelineState>> *pipelines,
                      NSArray<NSString *> *names, Args args, NSArray<id<MTLBuffer>> *buffers,
                      uint32_t groups, NSString *label) {
    double times[2][8];
    // Warm both pipelines before alternating their order across eight rounds.
    for (unsigned v = 0; v < 2; ++v) run(queue, pipelines[v], args, buffers, groups, 3);
    for (unsigned trial = 0; trial < 8; ++trial)
        for (unsigned j = 0; j < 2; ++j) {
            unsigned v = (trial + j) % 2;
            times[v][trial] = run(queue, pipelines[v], args, buffers, groups, 100);
        }
    for (unsigned v = 0; v < 2; ++v) qsort(times[v], 8, sizeof(double), compare_time);
    for (unsigned v = 0; v < 2; ++v)
        printf("%s,%s,%.3f,%.3f,%.3f,%.4f,0\n", label.UTF8String, names[v].UTF8String,
               (times[v][3] + times[v][4]) * .5, times[v][0], times[v][7],
               (times[0][3] + times[0][4]) / (times[v][3] + times[v][4]));
    fflush(stdout);
}

static void test_norm(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library) {
    NSArray *names = @[@"bench_norm_baseline", @"bonsai_norm"];
    NSArray *pipelines = make_pipelines(device, library, names);
    const uint32_t shapes[][4] = {
        {1, 5120, 5120, 0}, {48, 128, 128, 0}, {24, 256, 512, 0},
        {32, 128, 128, 1}, {4, 256, 256, 0}, {1, 17408, 17408, 0},
        {6, 8, 8, 1}, {3, 123, 128, 0}, {1, 1024, 1024, 0},
    };
    for (unsigned shape = 0; shape < sizeof(shapes) / sizeof(shapes[0]); ++shape) {
        @autoreleasepool {
            Args a = {.rows=shapes[shape][0], .cols=shapes[shape][1],
                      .width=shapes[shape][2], .mode=shapes[shape][3], .eps=1e-6f};
            size_t bytes = (size_t)a.rows * a.cols * sizeof(float);
            id<MTLBuffer> x = make_buffer(device, (size_t)a.rows * a.width * sizeof(float));
            id<MTLBuffer> weight = make_buffer(device, a.cols * sizeof(float));
            id<MTLBuffer> y = make_buffer(device, bytes);
            float *expected = malloc(bytes);
            need(expected != NULL, "reference allocation");
            fill(weight, 0, .8f);
            NSArray *buffers = @[x, weight, y];
            for (unsigned pattern = 0; pattern < 3; ++pattern) {
                fill(x, pattern, .5f);
                run(queue, pipelines[0], a, buffers, a.rows, 1);
                memcpy(expected, y.contents, bytes);
                memset(y.contents, 0xff, bytes); // A missing write must not pass parity.
                run(queue, pipelines[1], a, buffers, a.rows, 1);
                check_exact(y, expected, bytes, "normalization");
                // GDN Q/K normalization uses contiguous in-place rows.
                if (a.cols == a.width) {
                    run(queue, pipelines[1], a, @[x, weight, x], a.rows, 1);
                    check_exact(x, expected, bytes, "in-place normalization");
                }
            }
            fill(x, 0, .5f);
            benchmark(queue, pipelines, names, a, buffers, a.rows,
                      [NSString stringWithFormat:@"norm%ux%u_stride%u_L2%u", a.rows, a.cols, a.width, a.mode]);
            free(expected);
        }
    }
}

static void test_hadamard(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library) {
    NSArray *names = @[@"bench_hadamard_baseline", @"bonsai_hadamard"];
    NSArray *pipelines = make_pipelines(device, library, names);
    const uint32_t shapes[][3] = {
        {5120, 0, 0}, {17408, 0, 0}, {6144, 0, 16},
        {5120, 1, 0}, {6144, 1, 16}, {1024, 0, 0}, {1024, 1, 0},
    };
    for (unsigned shape = 0; shape < sizeof(shapes) / sizeof(shapes[0]); ++shape) {
        @autoreleasepool {
            Args a = {.n=shapes[shape][0], .mode=shapes[shape][1],
                      .groups=shapes[shape][2], .heads=48, .dim=128};
            size_t bytes = a.n * sizeof(float);
            id<MTLBuffer> x = make_buffer(device, bytes), signs = make_buffer(device, bytes);
            id<MTLBuffer> y = make_buffer(device, bytes);
            int32_t *s = signs.contents;
            for (uint32_t i = 0; i < a.n; ++i) s[i] = i % 7u ? 1 : -1;
            float *expected = malloc(bytes);
            need(expected != NULL, "reference allocation");
            NSArray *buffers = @[x, signs, y];
            for (unsigned pattern = 0; pattern < 3; ++pattern) {
                fill(x, pattern, .7f);
                run(queue, pipelines[0], a, buffers, a.n / 1024, 1);
                memcpy(expected, y.contents, bytes);
                memset(y.contents, 0xff, bytes);
                run(queue, pipelines[1], a, buffers, a.n / 1024, 1);
                check_exact(y, expected, bytes, "Hadamard");
            }
            fill(x, 0, .7f);
            benchmark(queue, pipelines, names, a, buffers, a.n / 1024,
                      [NSString stringWithFormat:@"had%u_mode%u_group%u", a.n, a.mode, a.groups]);
            free(expected);
        }
    }
}

int main(int argc, char **argv) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        need(device != nil, "Metal device");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *error = nil;
        NSString *path = argc > 1 ? [NSString stringWithUTF8String:argv[1]] : @"metal/bonsai.metal";
        NSString *source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
        need(source != nil, "shader source");
        MTLCompileOptions *options = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) options.mathMode = MTLMathModeSafe;
        else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        id<MTLLibrary> library = [device newLibraryWithSource:[source stringByAppendingString:baseline]
                                                    options:options error:&error];
        if (!library) fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
        need(library != nil, "shader library compilation");
        fprintf(stderr, "Bonsai elementwise benchmark: %s; bitwise checks, 8 alternating A/B rounds.\n", device.name.UTF8String);
        printf("shape,kernel,median_us,min_us,max_us,speedup,neq\n");
        test_norm(device, queue, library);
        test_hadamard(device, queue, library);
        fprintf(stderr, "PASS all normalization and Hadamard checks are bitwise exact.\n");
    }
    return 0;
}
