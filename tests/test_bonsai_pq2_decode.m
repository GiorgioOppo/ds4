/* Centered PQ2 decode regression and optional benchmark; no model is required.
 * Compile with -I. -fobjc-arc -framework Foundation -framework Metal.
 * Usage: test_bonsai_pq2_decode [metal/bonsai.metal] [--bench]
 *
 * The FP64 oracle decodes original packed weights directly. Numerical limits
 * apply to these fixtures: |x| <= .35, finite scales in [.0078,.125), K <=17408.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "bonsai_quant.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t n, rows, cols, type, row_bytes, pos, heads, kvheads;
    uint32_t dim, rot, width, mode, groups;
    float eps, base;
} Args;
_Static_assert(sizeof(Args) == 60, "shader argument layout");
enum { GUARD = 64 };

static void need(bool ok, const char *message) {
    if (!ok) { fprintf(stderr, "FAIL %s\n", message); exit(1); }
}
static uint32_t hash(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16);
}
static int compare_time(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y);
}
static id<MTLBuffer> make_buffer(id<MTLDevice> device, size_t bytes) {
    id<MTLBuffer> b = [device newBufferWithLength:bytes + 2 * GUARD options:MTLResourceStorageModeShared];
    need(b != nil, "buffer allocation"); memset(b.contents, 0xa5, b.length); return b;
}
static void check_guard(id<MTLBuffer> buffer) {
    const uint8_t *p = buffer.contents;
    for (unsigned i = 0; i < GUARD; ++i)
        need(p[i] == 0xa5 && p[buffer.length - GUARD + i] == 0xa5, "buffer canary");
}
static double run(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                  Args a, NSArray<id<MTLBuffer>> *buffers, unsigned nr, unsigned repeats) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:pipeline]; [e setBytes:&a length:sizeof(a) atIndex:0];
    for (unsigned i = 0; i < 3; ++i) [e setBuffer:buffers[i] offset:GUARD atIndex:i + 1];
    const unsigned rows_per_group = nr ? 2 * nr : 4, threads = nr ? 64 : 128;
    for (unsigned i = 0; i < repeats; ++i)
        [e dispatchThreadgroups:MTLSizeMake((a.rows + rows_per_group - 1) / rows_per_group, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) fprintf(stderr, "%s\n", cb.error.localizedDescription.UTF8String);
    need(cb.status == MTLCommandBufferStatusCompleted, "GPU completion");
    return (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / repeats;
}
static double oracle(const uint8_t *weights, const float *x, Args a, unsigned row) {
    double sum = 0;
    for (unsigned block = 0; block < a.cols / 128; ++block) {
        const uint8_t *p = weights + (size_t)row * a.row_bytes + block * 34;
        uint16_t h; memcpy(&h, p, sizeof(h));
        const double scale = ds4_bonsai_fp16_to_f32(h);
        for (unsigned j = 0; j < 128; ++j)
            sum += scale * (double)((int)((p[2 + j / 4] >> (2 * (j % 4))) & 3) - 1) * (double)x[block * 128 + j];
    }
    return sum;
}

static void fixture(id<MTLDevice> device, id<MTLCommandQueue> queue,
                    NSArray<id<MTLComputePipelineState>> *pipelines, NSArray<NSString *> *names,
                    unsigned rows, unsigned cols, int basis, unsigned pattern, bool benchmark) {
    Args a = {.rows=rows, .cols=cols, .row_bytes=cols / 128 * 34, .type=142};
    id<MTLBuffer> w = make_buffer(device, (size_t)rows * a.row_bytes);
    id<MTLBuffer> x = make_buffer(device, cols * sizeof(float));
    id<MTLBuffer> y = make_buffer(device, rows * sizeof(float));
    uint8_t *weights = (uint8_t *)w.contents + GUARD;
    float *input = (float *)((uint8_t *)x.contents + GUARD);
    float *output = (float *)((uint8_t *)y.contents + GUARD);
    for (unsigned row = 0; row < rows; ++row) for (unsigned block = 0; block < cols / 128; ++block) {
        uint8_t *p = weights + (size_t)row * a.row_bytes + block * 34;
        const uint16_t h = basis >= 0 ? 0x3800 : 0x2000 + (hash(row * 17713 + block * 23) & 0xfff);
        memcpy(p, &h, sizeof(h));
        for (unsigned j = 0; j < 32; ++j) {
            const uint32_t r = hash(row * 2777 + block * 9391 + j * 13 + pattern * 317);
            if (basis >= 0) p[2 + j] = !block && !j ? row : 0x55;
            else if (row == 0 || row == 3 || row == rows - 1) p[2 + j] = 0x55;
            else if (pattern) p[2 + j] = r; // Includes every code, including +2.
            else p[2 + j] = (r % 3) | (((r >> 8) % 3) << 2) | (((r >> 16) % 3) << 4) | (((r >> 24) % 3) << 6);
        }
    }
    for (unsigned j = 0; j < cols; ++j)
        input[j] = basis >= 0 ? (j == (unsigned)basis ? 1.0f : 0.0f) : ((float)(int32_t)hash(j + 42 + pattern * 17) / 2147483648.0f) * .35f;

    const unsigned nref = rows < 1024 ? rows : 1024;
    double *expected = malloc(nref * sizeof(double)); unsigned *indices = malloc(nref * sizeof(unsigned));
    need(expected && indices, "oracle allocation");
    for (unsigned i = 0; i < nref; ++i) {
        indices[i] = (uint64_t)i * (rows - 1) / (nref - 1);
        expected[i] = oracle(weights, input, a, indices[i]);
    }
    NSArray *buffers = @[w,x,y]; const unsigned nr[] = {0,8};
    for (unsigned v = 1; v < 2; ++v) {
        memset(output, 0xff, rows * sizeof(float)); // Missing writes stay NaN.
        run(queue, pipelines[v], a, buffers, nr[v], 1);
        check_guard(w); check_guard(x); check_guard(y);
        double max_abs = 0, sum_error = 0, sum_reference = 0;
        for (unsigned row = 0; row < rows; ++row) {
            need(isfinite(output[row]), "all output rows written and finite");
            if (basis < 0 && (row == 0 || row == 3 || row == rows - 1)) need(output[row] == 0.0f, "zero row is exactly zero");
        }
        for (unsigned i = 0; i < nref; ++i) {
            const double difference = (double)output[indices[i]] - expected[i];
            if (basis >= 0) need(difference == 0.0, "all 256 bytes against unit inputs are exact");
            max_abs = fmax(max_abs, fabs(difference)); sum_error += difference * difference; sum_reference += expected[i] * expected[i];
        }
        const double relative_l2 = sqrt(sum_error / fmax(sum_reference, 1e-300));
        if (max_abs > 1e-5 || relative_l2 > 2e-6)
            fprintf(stderr, "%s M%u K%u pattern%u: maxabs %.9g relativeL2 %.9g\n", names[v].UTF8String, rows, cols, pattern, max_abs, relative_l2);
        need(max_abs <= 1e-5 && relative_l2 <= 2e-6, "FP64 error bounds");
    }
    if (benchmark) {
        double time[2][8];
        for (unsigned v = 0; v < 2; ++v) run(queue, pipelines[v], a, buffers, nr[v], 3);
        for (unsigned trial = 0; trial < 8; ++trial) for (unsigned j = 0; j < 2; ++j) {
            unsigned v = (trial + j) % 2;
            time[v][trial] = run(queue, pipelines[v], a, buffers, nr[v], rows > 100000 ? 3 : 15);
        }
        for (unsigned v = 0; v < 2; ++v) qsort(time[v], 8, sizeof(double), compare_time);
        for (unsigned v = 0; v < 2; ++v)
            printf("%ux%u,%s,%.3f,%.3f,%.3f,%.4f\n", rows, cols, names[v].UTF8String,
                   .5 * (time[v][3] + time[v][4]), time[v][0], time[v][7],
                   (time[0][3] + time[0][4]) / (time[v][3] + time[v][4]));
        fflush(stdout);
    }
    free(expected); free(indices);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        bool bench = false; NSString *path = @"metal/bonsai.metal";
        for (int i = 1; i < argc; ++i) {
            if (!strcmp(argv[i], "--bench")) bench = true;
            else path = [NSString stringWithUTF8String:argv[i]];
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice(); need(device != nil, "Metal device");
        id<MTLCommandQueue> queue = [device newCommandQueue]; NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error]; need(source != nil, "shader source");
        MTLCompileOptions *options = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) options.mathMode = MTLMathModeSafe;
        else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (!library) fprintf(stderr, "%s\n", error.localizedDescription.UTF8String); need(library != nil, "library");
        NSArray *names = @[@"bonsai_mv",@"bonsai_pq2_mv"];
        NSMutableArray *pipelines = [NSMutableArray array];
        for (NSString *name in names) {
            id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
            need(p != nil, "pipeline"); [pipelines addObject:p];
        }
        for (int basis = 0; basis < 4; ++basis) fixture(device,queue,pipelines,names,256,384,basis,0,false);
        const unsigned shapes[][2] = {{9,384},{17,5120},{17,17408},{257,5120},{65,17408}};
        for (unsigned i = 0; i < sizeof(shapes)/sizeof(shapes[0]); ++i)
            for (unsigned pattern = 0; pattern < 2; ++pattern)
                fixture(device,queue,pipelines,names,shapes[i][0],shapes[i][1],-1,pattern,false);
        fprintf(stderr, "PASS centered PQ2: all bytes/bases, zero rows, +2 codes, tails, canaries, FP64 bounds.\n");
        if (bench) {
            printf("shape,kernel,median_us,min_us,max_us,speedup\n");
            const unsigned large[][2] = {{17408,5120},{5120,17408},{10240,5120},{6144,5120},{5120,6144},{12288,5120},{1024,5120},{248320,5120}};
            for (unsigned i = 0; i < sizeof(large)/sizeof(large[0]); ++i)
                fixture(device,queue,pipelines,names,large[i][0],large[i][1],-1,0,true);
        }
    }
    return 0;
}
