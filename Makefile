CC ?= cc
UNAME_S := $(shell uname -s)
.DEFAULT_GOAL := all

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif
SAMPLING_TEST := tests/test_sampling
GLM53_KDA_TEST := tests/test_glm53_kda
GLM53_KDA_ROCM_TEST := tests/test_glm53_kda_rocm

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc
QUALITY_CFLAGS ?= -O3 $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c11

LDLIBS ?= -lm -pthread
METAL_SRCS := $(wildcard metal/*.metal)
ROCM_SRCS := $(wildcard rocm/*.cuh)
Q4_CPU_TEST_DEPS := ds4_image.c ds4_distributed.c ds4_tp.c ds4_ssd.c ds4_layer_pack.c
DS4_TEST_MODEL ?= ds4flash.gguf
DS4_TEST_MTP ?= gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
DS4_DSPARK_MODEL ?= $(DS4_TEST_MODEL)
DS4_DSPARK_SUPPORT ?= gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_metal.o ds4_layer_pack.o
CPU_CORE_OBJS = ds4_cpu.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o
else
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
CUDA_HOME ?= $(shell if [ -x /usr/local/cuda/bin/nvcc ]; then \
	printf '%s' /usr/local/cuda; \
	elif command -v nvcc >/dev/null 2>&1; then \
	dirname "$$(dirname "$$(command -v nvcc)")"; \
	else \
	printf '%s' /usr/local/cuda; \
	fi)
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
ifneq ($(filter sm_120 sm_120a,$(strip $(CUDA_ARCH))),)
NVCC_ARCH_FLAGS := -gencode arch=compute_120a,code=sm_120a -DDS4_CUDA_HAVE_MXF4=1
else ifneq ($(filter sm_121 sm_121a,$(strip $(CUDA_ARCH))),)
NVCC_ARCH_FLAGS := -gencode arch=compute_121a,code=sm_121a -DDS4_CUDA_HAVE_MXF4=1
else
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif

endif
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
# Vendored llama.cpp mmq prefill tier (cuda/mmq/, see cuda/mmq/VENDOR.md).
MMQ_INCLUDES := -Icuda/mmq
MMQ_OBJS := cuda/mmq/ds4_ggml_stubs.o cuda/mmq/ds4_mmq.o cuda/mmq/ds4_mmq_d2r.o cuda/mmq/quantize.o cuda/mmq/mmid.o cuda/mmq/mmvq.o cuda/mmq/ds4_repack.o
CORE_OBJS = ds4.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_cuda.o ds4_layer_pack.o $(MMQ_OBJS)
CPU_CORE_OBJS = ds4_cpu.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
HIPCC ?= $(shell command -v hipcc 2>/dev/null || echo /opt/rocm/bin/hipcc)
ROCM_ARCH ?= gfx1151
ROCM_HOST_CFLAGS ?= -fPIC
ROCM_CFLAGS ?= -O3 -ffast-math -g -fno-finite-math-only -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument --offload-arch=$(ROCM_ARCH)
ROCM_LDLIBS ?= -lm -pthread -lhipblas -lhipblaslt -lrocblas
ROCM_MMQ_Y ?= 64
ROCM_MMQ_FLAGS := $(ROCM_CFLAGS) -std=c++17 -DGGML_USE_HIP -DDS4_HIP_MMQ_Y=$(ROCM_MMQ_Y) $(MMQ_INCLUDES)
ROCM_MMQ_OBJS := cuda/mmq/ds4_ggml_stubs.rocm.o cuda/mmq/ds4_mmq.rocm.o cuda/mmq/quantize.rocm.o cuda/mmq/mmid.rocm.o cuda/mmq/mmvq.rocm.o cuda/mmq/d2r_stubs.rocm.o
DS4_LINK ?= $(NVCC) $(NVCCFLAGS)
DS4_LINK_LIBS ?= $(CUDA_LDLIBS)
METAL_LDLIBS := $(LDLIBS)
endif

.PHONY: all help clean test \
	test-rocm test-glm53-kda-rocm test-metal-session-batch test-mxfp4-cuda \
	test-mxfp4-rocm test-cuda-session-batch test-cuda-mixed-batch dspark-acceptance \
	dspark-verify-depth mtp-verify-depth cpu cuda \
	cuda-spark cuda-generic cuda-regression strix-halo \
	rocm

.PHONY: test-cpu-q4 test-quantizer-indexer-q4
tests/test_cpu_q4_dense: tests/test_cpu_q4_dense.c ds4.c ds4.h $(Q4_CPU_TEST_DEPS)
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -I. -o $@ tests/test_cpu_q4_dense.c $(Q4_CPU_TEST_DEPS) $(LDLIBS)

test-cpu-q4: tests/test_cpu_q4_dense q4k-dot-test
	./tests/test_cpu_q4_dense

gguf-tools/deepseek4-quantize: gguf-tools/deepseek4-quantize.c gguf-tools/quants.c gguf-tools/quants.h
	$(MAKE) -C gguf-tools deepseek4-quantize

tests/test_quantizer_indexer_q4: tests/test_quantizer_indexer_q4.c gguf-tools/quants.c gguf-tools/quants.h
	$(CC) -O2 -Wall -Wextra -std=c99 -Igguf-tools -o $@ tests/test_quantizer_indexer_q4.c gguf-tools/quants.c $(LDLIBS)

test-quantizer-indexer-q4: gguf-tools/deepseek4-quantize tests/test_quantizer_indexer_q4
	./tests/test_quantizer_indexer_q4 ./gguf-tools/deepseek4-quantize

ifeq ($(UNAME_S),Darwin)
.PHONY: metal-decode-schedule-bench metal-prefill-variant-bench check-mxfp4-half-lut
.PHONY: test-metal-moe-prefill test-metal-dense-mpp

all: ds4 ds4-server ds4-bench ds4-eval ds4-agent

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test         Build and run tests"
	@echo "  make metal-decode-schedule-bench  Build the balanced Metal decode schedule benchmark"
	@echo "  make metal-prefill-variant-bench  Build the balanced Metal prefill variant benchmark"
	@echo "  make check-mxfp4-half-lut  Verify the checked-in MXFP4 half LUT matches the generator"
	@echo "  make test-mxfp4-metal  Check the MXFP4 half LUT, then run Metal MXFP4 exactness tests"
	@echo "  make dspark-verify-depth  Run DSpark speculative verification smoke if support GGUF is present"
	@echo "  make mtp-verify-depth  Run legacy MTP speculative verification smoke if MTP GGUF is present"
	@echo "  make clean        Remove build outputs"

ds4: ds4_cli.o ds4_help.o ds4_prompt_prefix.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o ds4_help.o ds4_prompt_prefix.o linenoise.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-bench: ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-eval: ds4_eval.o ds4_eval_cases.o ds4_help.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_eval.o ds4_eval_cases.o ds4_help.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_agent.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

gguf-tools/quality-testing/score_official: gguf-tools/quality-testing/score_official.c ds4.h ds4_distributed.h ds4_tp.h $(CORE_OBJS) rax.o ds4_gpu_args.o
	$(CC) $(QUALITY_CFLAGS) -I. -o $@ gguf-tools/quality-testing/score_official.c $(CORE_OBJS) rax.o ds4_gpu_args.o $(METAL_LDLIBS)

tests/test_metal_session_batch.o: tests/test_metal_session_batch.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_metal_session_batch.c

tests/test_metal_session_batch: tests/test_metal_session_batch.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

tests/test_metal_tp_spec.o: tests/test_metal_tp_spec.c ds4.h ds4_tp.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_metal_tp_spec: tests/test_metal_tp_spec.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-session-batch: tests/test_metal_session_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_metal_session_batch

speed-bench/metal_decode_schedule_bench.o: speed-bench/metal_decode_schedule_bench.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

speed-bench/metal_decode_schedule_bench: speed-bench/metal_decode_schedule_bench.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

metal-decode-schedule-bench: speed-bench/metal_decode_schedule_bench

speed-bench/metal_prefill_variant_bench.o: speed-bench/metal_prefill_variant_bench.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

speed-bench/metal_prefill_variant_bench: speed-bench/metal_prefill_variant_bench.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

metal-prefill-variant-bench: speed-bench/metal_prefill_variant_bench

tests/test_mxfp4_metal.o: tests/test_mxfp4_metal.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_mxfp4_metal: tests/test_mxfp4_metal.o ds4_metal.o ds4_image.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

check-mxfp4-half-lut:
	python3 metal/generate_mxfp4_half_lut.py --check

test-mxfp4-metal: check-mxfp4-half-lut tests/test_mxfp4_metal
	./tests/test_mxfp4_metal

tests/test_metal_moe_prefill.o: tests/test_metal_moe_prefill.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -I. -c -o $@ $<

tests/test_metal_moe_prefill: tests/test_metal_moe_prefill.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-moe-prefill: tests/test_metal_moe_prefill
	./tests/test_metal_moe_prefill

tests/test_metal_ssd_experts.o: tests/test_metal_ssd_experts.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -I. -c -o $@ $<

tests/test_metal_ssd_experts: tests/test_metal_ssd_experts.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

.PHONY: test-metal-ssd-experts
test-metal-ssd-experts: tests/test_metal_ssd_experts
	./tests/test_metal_ssd_experts

tests/test_metal_dense_mpp.o: tests/test_metal_dense_mpp.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -I. -c -o $@ $<

tests/test_metal_dense_mpp: tests/test_metal_dense_mpp.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-dense-mpp: tests/test_metal_dense_mpp
	./tests/test_metal_dense_mpp

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_eval_cases.o ds4_agent_cpu.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o ds4_prompt_prefix.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_eval_cases.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression:
	@echo "cuda-regression requires a CUDA build"
else
all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make strix-halo          Build ROCm for Strix Halo / gfx1151"
	@echo "  make rocm                Alias for make strix-halo"
	@echo "  make test-mxfp4-rocm     Build and run the synthetic ROCm MXFP4 MoE test"
	@echo "  make test-rocm           Core regression suite on ROCm-only hosts"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test                Build and run tests"
	@echo "  make dspark-verify-depth Run DSpark speculative verification smoke if support GGUF is present"
	@echo "  make mtp-verify-depth    Run legacy MTP speculative verification smoke if MTP GGUF is present"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=sm_121

cuda-generic:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH="$(CUDA_ARCH)"

strix-halo:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent \
		CORE_OBJS="ds4.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o ds4_layer_pack.o $(ROCM_MMQ_OBJS)" \
		CFLAGS="$(CFLAGS) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD" \
		DS4_LINK="$(HIPCC) $(ROCM_CFLAGS)" \
		DS4_LINK_LIBS="$(ROCM_LDLIBS)"

rocm: strix-halo

# Core regression suite for ROCm-only hosts: the CUDA-specific binaries
# (tests/test_sampling, the CUDA session/mixed-batch oracles) are not part
# of this target; run them through `make test` / `make cuda-regression` on
# CUDA hosts.  Everything else mirrors `make test`.
test-rocm:
	$(MAKE) -B ds4_test ds4_agent_test ds4-eval q4k-dot-test mxfp4-dot-test \
		test-session-state \
		tests/test_layer_pack tests/test_engine_mgpu_placement tests/test_gpu_args tests/test_prompt_prefix \
		ds4 ds4-server ds4-bench ds4-agent \
		CORE_OBJS="ds4.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o ds4_layer_pack.o $(ROCM_MMQ_OBJS)" \
		CFLAGS="$(CFLAGS) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD" \
		DS4_LINK="$(HIPCC) $(ROCM_CFLAGS)" \
		DS4_LINK_LIBS="$(ROCM_LDLIBS)"
	./ds4-eval --self-test-extractors
	./ds4_agent_test
	./ds4_test
	./tests/test_layer_pack
	./tests/test_engine_mgpu_placement
	./tests/test_gpu_args
	./tests/test_gpu_args_cli.sh
	./tests/test_prompt_prefix

ds4: ds4_cli.o ds4_help.o ds4_prompt_prefix.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-bench: ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-eval: ds4_eval.o ds4_eval_cases.o ds4_help.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

gguf-tools/quality-testing/score_official.o: gguf-tools/quality-testing/score_official.c ds4.h
	$(CC) $(filter-out -ffast-math,$(QUALITY_CFLAGS)) $(ROCM_HOST_CFLAGS) -I. -c -o $@ $<

gguf-tools/quality-testing/score_official: gguf-tools/quality-testing/score_official.o $(CORE_OBJS) rax.o ds4_gpu_args.o
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_cuda_q8_scratch.o: tests/test_cuda_q8_scratch.cu cuda/mmq/ds4_mmq.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Icuda/mmq -c -o $@ $<

tests/test_cuda_q8_scratch: tests/test_cuda_q8_scratch.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

.PHONY: test-cuda-q8-scratch
test-cuda-q8-scratch: tests/test_cuda_q8_scratch
	./tests/test_cuda_q8_scratch

tests/test_cuda_dspark_moe.o: cuda/mmq/test/test_iq2_aligned_entry.cu cuda/mmq/ds4_mmq.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Icuda/mmq -c -o $@ $<

tests/test_cuda_dspark_moe: tests/test_cuda_dspark_moe.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

.PHONY: test-cuda-dspark-moe
test-cuda-dspark-moe: tests/test_cuda_dspark_moe
	./tests/test_cuda_dspark_moe

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_eval_cases.o ds4_agent_cpu.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o ds4_prompt_prefix.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_eval_cases.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke

tests/test_mxfp4_cuda: tests/test_mxfp4_cuda.cu $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -o $@ $^ $(CUDA_LDLIBS)

test-mxfp4-cuda: tests/test_mxfp4_cuda
	./tests/test_mxfp4_cuda
endif

ds4.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h ds4_gpu.h ds4_linux_memory.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_image.o: ds4_image.c ds4_image.h third_party/iris/jpeg.h third_party/iris/png.h
	$(CC) $(CFLAGS) -c -o $@ ds4_image.c

ds4_ssd.o: ds4_ssd.c ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_ssd.c

ds4_cli.o: ds4_cli.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_prompt_prefix.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_distributed.o: ds4_distributed.c ds4_distributed.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_distributed.c

ds4_tp.o: ds4_tp.c ds4_tp.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_tp.c

ds4_help.o: ds4_help.c ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_help.c

ds4_prompt_prefix.o: ds4_prompt_prefix.c ds4_prompt_prefix.h ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_prompt_prefix.c

ds4_gpu_args.o: ds4_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4_gpu_args.c

ds4_server.o: ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4_eval_cases.h ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_eval_cases.o: ds4_eval_cases.c ds4_eval_cases.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval_cases.c

ds4_agent.o: ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_tp.h ds4_help.h ds4_prompt_prefix.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_agent.c

ds4_web.o: ds4_web.c ds4_web.h
	$(CC) $(CFLAGS) -c -o $@ ds4_web.c

ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvstore.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

ds4_agent_test.o: tests/ds4_agent_test.c ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_tp.h ds4_help.h ds4_prompt_prefix.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_agent_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h ds4_gpu.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cli_cpu.o: ds4_cli.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_prompt_prefix.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_gpu_args_cpu.o: ds4_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_gpu_args.c

ds4_server_cpu.o: ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c ds4_eval_cases.h ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_agent_cpu.o: ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_prompt_prefix.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_agent.c

ds4_metal.o: ds4_metal.m ds4_gpu.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

tests/test_glm53_kda.o: tests/test_glm53_kda.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_glm53_kda.c

tests/test_glm53_vision_engine.o: tests/test_glm53_vision_engine.c ds4.h ds4_image.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -I. -c -o $@ tests/test_glm53_vision_engine.c

tests/test_glm53_vision_engine: tests/test_glm53_vision_engine.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
endif

tests/test_glm53_vision_prompt.o: tests/test_glm53_vision_prompt.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_glm53_vision_prompt.c

tests/test_glm53_vision_prompt: tests/test_glm53_vision_prompt.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
endif

tests/test_deepseek4_vision_image.o: tests/test_deepseek4_vision_image.c ds4_image.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_deepseek4_vision_image: tests/test_deepseek4_vision_image.o ds4_image.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

ifeq ($(UNAME_S),Darwin)
$(GLM53_KDA_TEST): tests/test_glm53_kda.o ds4_metal.o ds4_image.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
else
$(GLM53_KDA_TEST): tests/test_glm53_kda.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
endif

.PHONY: test-glm53-kda
test-glm53-kda: $(GLM53_KDA_TEST)
	./$(GLM53_KDA_TEST)

tests/test_glm53_kda_rocm.o: tests/test_glm53_kda.c ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

$(GLM53_KDA_ROCM_TEST): tests/test_glm53_kda_rocm.o ds4_rocm.o ds4_image.o $(ROCM_MMQ_OBJS)
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-glm53-kda-rocm: $(GLM53_KDA_ROCM_TEST)
	./$(GLM53_KDA_ROCM_TEST)

tests/test_glm_attention_rocm.o: tests/test_glm_attention.c ds4.h ds4_gpu.h ds4_linux_memory.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_glm_attention_rocm: tests/test_glm_attention_rocm.o ds4_rocm.o ds4_image.o $(ROCM_MMQ_OBJS)
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

.PHONY: test-glm-attention-rocm
test-glm-attention-rocm: tests/test_glm_attention_rocm
	./tests/test_glm_attention_rocm

tests/test_linux_memory: tests/test_linux_memory.c ds4_linux_memory.h
	$(CC) $(CFLAGS) -I. -o $@ $<

tests/test_rocm_memory: tests/test_rocm_memory.cu ds4_rocm_memory.h ds4_linux_memory.h
	$(HIPCC) $(ROCM_CFLAGS) -I. -o $@ $<

.PHONY: test-linux-memory test-rocm-memory
test-linux-memory: tests/test_linux_memory
	./tests/test_linux_memory

test-rocm-memory: tests/test_rocm_memory
	./tests/test_rocm_memory

tests/test_glm_attention.o: tests/test_glm_attention.c ds4.h ds4_gpu.h ds4_linux_memory.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -I. -c -o $@ $<

ifeq ($(UNAME_S),Darwin)
tests/test_glm_attention: tests/test_glm_attention.o ds4_metal.o ds4_image.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
else
tests/test_glm_attention: tests/test_glm_attention.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
endif

.PHONY: test-glm-attention
test-glm-attention: tests/test_glm_attention
	./tests/test_glm_attention

tests/test_ssd_cache: tests/test_ssd_cache.c ds4_ssd.c ds4_ssd.h
	$(CC) $(CFLAGS) -I. -o $@ tests/test_ssd_cache.c ds4_ssd.c

.PHONY: test-ssd-cache
test-ssd-cache: tests/test_ssd_cache
	./tests/test_ssd_cache

ds4_cuda.o: ds4_cuda.cu cuda/ds4_q4_dequant_layout.h cuda/ds4_q4_dequant_vec.cuh cuda/ds4_q4_prefill_reduce.h cuda/ds4_q8_quantize.cuh cuda/ds4_f16_compressor.cuh ds4_gpu.h ds4_gpu_mgpu.h ds4_glm53_vision_gpu.cuh ds4_deepseek4_vision_gpu.cuh ds4_image.h ds4_iq2_tables_cuda.inc cuda/mmq/ds4_mmq.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

# Vendored mmq pieces (see cuda/mmq/VENDOR.md).  ds4_mmq.cu transitively
# pulls in mmq.cuh which has heavy template instantiation -- each piece
# compiles in its own TU and links in.
cuda/mmq/ds4_ggml_stubs.o: cuda/mmq/ds4_ggml_stubs.cu cuda/mmq/ds4_ggml_stubs.h cuda/mmq/common.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_mmq.o: cuda/mmq/ds4_mmq.cu cuda/mmq/mmvq.cuh cuda/mmq/ds4_q4_mmvq_epilogue.h cuda/mmq/ds4_mmq.h cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/quantize.cuh cuda/mmq/mmid.cuh cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_mmq_d2r.o: cuda/mmq/ds4_mmq_d2r.cu cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/quantize.o: cuda/mmq/quantize.cu cuda/mmq/quantize.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/mmq.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/mmid.o: cuda/mmq/mmid.cu cuda/mmq/mmid.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/mmvq.o: cuda/mmq/mmvq.cu cuda/mmq/ds4_q4_mmvq_epilogue.h cuda/mmq/mmvq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/quantize.cuh cuda/mmq/vecdotq.cuh cuda/mmq/unary.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_repack.o: cuda/mmq/ds4_repack.cu cuda/mmq/ds4_repack.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -c -o $@ $<

ds4_rocm.o: ds4_rocm.cu cuda/ds4_q4_dequant_layout.h cuda/ds4_q4_dequant_vec.cuh cuda/ds4_q8_k_bsum.h cuda/ds4_q8_k_reduce.h ds4_rocm.h ds4_rocm_memory.h ds4_linux_memory.h ds4_gpu.h ds4_glm53_vision_gpu.cuh ds4_deepseek4_vision_gpu.cuh ds4_image.h ds4_iq2_tables_cuda.inc $(ROCM_SRCS)
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm.cu

cuda/mmq/ds4_ggml_stubs.rocm.o: cuda/mmq/ds4_ggml_stubs.cu cuda/mmq/ds4_ggml_stubs.h cuda/mmq/common.cuh cuda/mmq/vendors/hip.h ds4_rocm_memory.h ds4_linux_memory.h
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

cuda/mmq/ds4_mmq.rocm.o: cuda/mmq/ds4_mmq.cu cuda/mmq/mmvq.cuh cuda/mmq/ds4_q4_mmvq_epilogue.h cuda/mmq/ds4_mmq.h cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/quantize.cuh cuda/mmq/mmid.cuh cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh cuda/mmq/vendors/hip.h
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

cuda/mmq/quantize.rocm.o: cuda/mmq/quantize.cu cuda/mmq/quantize.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/mmq.cuh cuda/mmq/vendors/hip.h
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

cuda/mmq/mmid.rocm.o: cuda/mmq/mmid.cu cuda/mmq/mmid.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/vendors/hip.h
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

cuda/mmq/mmvq.rocm.o: cuda/mmq/mmvq.cu cuda/mmq/ds4_q4_mmvq_epilogue.h cuda/mmq/mmvq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/quantize.cuh cuda/mmq/vecdotq.cuh cuda/mmq/unary.cuh cuda/mmq/vendors/hip.h
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

cuda/mmq/d2r_stubs.rocm.o: cuda/mmq/test/d2r_stubs.cu cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/vendors/hip.h
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

tests/test_mxfp4_rocm.o: tests/test_mxfp4_rocm.c ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_mxfp4_rocm: tests/test_mxfp4_rocm.o ds4_rocm.o ds4_image.o $(ROCM_MMQ_OBJS)
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

tests/bench_mxfp4_rocm.o: tests/bench_mxfp4_rocm.c ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/bench_mxfp4_rocm: tests/bench_mxfp4_rocm.o ds4_rocm.o ds4_image.o $(ROCM_MMQ_OBJS)
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-mxfp4-rocm: tests/test_mxfp4_rocm
	./tests/test_mxfp4_rocm

ds4_rocm_compat.o: ds4_rocm_compat.cu ds4_gpu.h ds4_gpu_mgpu.h ds4_gpu_args.h ds4_rocm_memory.h ds4_linux_memory.h
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm_compat.cu

ds4_rocm_unavailable.o: ds4_rocm_unavailable.cu
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm_unavailable.cu

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_layer_pack.o: tests/test_layer_pack.c ds4_layer_pack.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_layer_pack: tests/test_layer_pack.o ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_gpu_args.o: tests/test_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -DDS4_NO_GPU -c -o $@ $<

tests/test_gpu_args: tests/test_gpu_args.o ds4_gpu_args_cpu.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ds4_cpu_test_hooks.o: ds4.c ds4.h ds4_image.h ds4_gpu.h ds4_gpu_mgpu.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -DDS4_TEST_HOOKS -c -o $@ ds4.c

tests/test_engine_mgpu_placement.o: tests/test_engine_mgpu_placement.c ds4.h ds4_gpu_mgpu.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_engine_mgpu_placement: tests/test_engine_mgpu_placement.o ds4_cpu_test_hooks.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_sampling.o: tests/test_sampling.c ds4.h
	$(CC) $(CFLAGS) -fno-finite-math-only -DDS4_TEST_HOOKS -I. -c -o $@ $<

tests/test_sampling: tests/test_sampling.o ds4_cpu_test_hooks.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_session_state.o: tests/test_session_state.c ds4.c ds4.h ds4_gpu.h ds4_image.h ds4_tp.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -I. -c -o $@ $<

tests/test_session_state: tests/test_session_state.o $(filter-out ds4_cpu.o,$(CPU_CORE_OBJS))
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_session_state_gpu.o: tests/test_session_state.c ds4.c ds4.h ds4_gpu.h ds4_image.h ds4_tp.h
	$(CC) $(CFLAGS) -Wno-unused-function -I. -c -o $@ $<

tests/test_session_state_gpu: tests/test_session_state_gpu.o $(filter-out ds4.o,$(CORE_OBJS))
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
endif

tests/test_tp_commands.o: tests/test_tp_commands.c ds4_tp.c ds4_tp.h ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_tp_commands: tests/test_tp_commands.o $(filter-out ds4_tp.o,$(CPU_CORE_OBJS))
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: test-session-state
test-session-state: tests/test_session_state tests/test_tp_commands
	./tests/test_session_state
	./tests/test_tp_commands

ifneq ($(UNAME_S),Darwin)
tests/test_gpu_xdev.o: tests/test_gpu_xdev.c ds4_gpu.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_xdev: tests/test_gpu_xdev.o ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_gpu_model_cache.o: tests/test_gpu_model_cache.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_model_cache: tests/test_gpu_model_cache.o ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_gpu_lookup_cache_strict.o: tests/test_gpu_lookup_cache_strict.c ds4_gpu.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_lookup_cache_strict: tests/test_gpu_lookup_cache_strict.o ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_cuda_test_hooks.o: ds4.c ds4.h ds4_gpu.h ds4_gpu_mgpu.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_TEST_HOOKS -I$(CUDA_HOME)/include -c -o $@ ds4.c

tests/test_engine_mgpu_refusal.o: tests/test_engine_mgpu_refusal.c ds4.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_mgpu_refusal: tests/test_engine_mgpu_refusal.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_engine_mgpu_runtime.o: tests/test_engine_mgpu_runtime.c ds4.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_mgpu_runtime: tests/test_engine_mgpu_runtime.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_cuda.o ds4_layer_pack.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_engine_correctness.o: tests/test_engine_correctness.c ds4.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_correctness: tests/test_engine_correctness.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_cuda_session_batch.o: tests/test_cuda_session_batch.c ds4.h ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_cuda_session_batch: tests/test_cuda_session_batch.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-session-batch: tests/test_cuda_session_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_cuda_session_batch

tests/test_cuda_mixed_batch.o: tests/test_cuda_mixed_batch.c ds4.h ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_cuda_mixed_batch: tests/test_cuda_mixed_batch.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_cuda.o ds4_layer_pack.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-mixed-batch: tests/test_cuda_mixed_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_cuda_mixed_batch
endif

ds4_test: ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(DS4_LINK_LIBS)
endif

ds4_agent_test: ds4_agent_test.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_agent_test.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ ds4_agent_test.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(DS4_LINK_LIBS)
endif

tests/test_prompt_prefix.o: tests/test_prompt_prefix.c ds4_prompt_prefix.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_prompt_prefix: tests/test_prompt_prefix.o ds4_prompt_prefix.o
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: test-frontends
test-frontends: ds4_test ds4_agent_test
	./ds4_test --server
	./ds4_agent_test

test: ds4_test ds4_agent_test ds4-eval q4k-dot-test mxfp4-dot-test test-session-state test-linux-memory \
	tests/test_layer_pack tests/test_engine_mgpu_placement tests/test_gpu_args \
	tests/test_deepseek4_vision_image tests/test_prompt_prefix $(SAMPLING_TEST) ds4 ds4-server ds4-bench ds4-agent
	./ds4-eval --validate-cases
	./ds4-eval --self-test-extractors
	./ds4_agent_test
	./ds4_test
	./tests/test_layer_pack
	./tests/test_engine_mgpu_placement
	./tests/test_gpu_args
	./tests/test_gpu_args_cli.sh
	./tests/test_prompt_prefix
	./tests/test_sampling
	./tests/test_deepseek4_vision_image

dspark-acceptance: ds4
	DS4_DSPARK_MODEL="$(DS4_DSPARK_MODEL)" \
	DS4_DSPARK_SUPPORT="$(DS4_DSPARK_SUPPORT)" \
	sh tests/dspark_acceptance_fixture.sh

dspark-verify-depth: ds4_test
	@if [ ! -f "$(DS4_TEST_MODEL)" ]; then \
		echo "dspark-verify-depth: skipped, missing model $(DS4_TEST_MODEL)"; \
	elif [ ! -f "$(DS4_DSPARK_SUPPORT)" ]; then \
		echo "dspark-verify-depth: skipped, missing DSpark support $(DS4_DSPARK_SUPPORT)"; \
		echo "dspark-verify-depth: run ./download_model.sh ds4f-dspark or set DS4_DSPARK_SUPPORT=FILE"; \
	else \
		DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_DSPARK="$(DS4_DSPARK_SUPPORT)" ./ds4_test --dspark-verify-depth; \
	fi

mtp-verify-depth: ds4_test
	@if [ ! -f "$(DS4_TEST_MODEL)" ]; then \
		echo "mtp-verify-depth: skipped, missing model $(DS4_TEST_MODEL)"; \
	elif [ ! -f "$(DS4_TEST_MTP)" ]; then \
		echo "mtp-verify-depth: skipped, missing MTP support $(DS4_TEST_MTP)"; \
		echo "mtp-verify-depth: set DS4_TEST_MTP=FILE to a legacy support GGUF"; \
	else \
		DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_MTP="$(DS4_TEST_MTP)" ./ds4_test --mtp-verify-depth; \
	fi

q4k-dot-test: tests/test_q4k_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_q4k_dot tests/test_q4k_dot.c -lm -pthread
	./tests/test_q4k_dot

mxfp4-dot-test: tests/test_mxfp4_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_mxfp4_dot tests/test_mxfp4_dot.c -lm
	./tests/test_mxfp4_dot

.PHONY: test-quality-api
# Only the scorer's JSON parser is needed; discard the unused engine entry point.
test-quality-api: tests/test_quality_api.c gguf-tools/quality-testing/score_official.c
	$(CC) $(QUALITY_CFLAGS) -I. -ffunction-sections -fdata-sections -o tests/test_quality_api tests/test_quality_api.c -Wl,$(if $(filter Darwin,$(UNAME_S)),-dead_strip,--gc-sections) -lm
	./tests/test_quality_api

ds4.o ds4_cpu.o ds4_agent.o ds4_agent_cpu.o ds4_server.o ds4_server_cpu.o \
ds4_test.o ds4_agent_test.o \
ds4_cpu_test_hooks.o ds4_cuda_test_hooks.o tests/test_session_state.o \
tests/test_session_state_gpu.o: ds4_tool_text.h

clean:
	rm -f tests/test_metal_ssd_experts
	rm -f tests/test_cuda_q8_scratch
	rm -f tests/test_cuda_dspark_moe
	rm -f tests/test_quality_api
	rm -f tests/test_linux_memory tests/test_rocm_memory
	rm -f tests/test_glm_attention tests/test_glm_attention_rocm
	rm -f tests/test_ssd_cache
	rm -f tests/test_session_state tests/test_session_state_gpu tests/test_tp_commands
	rm -f tests/test_metal_tp_spec
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_cpu ds4_native ds4_server_test ds4_test ds4_agent_test gguf-tools/quality-testing/score_official gguf-tools/quality-testing/score_official.o speed-bench/metal_decode_schedule_bench speed-bench/metal_prefill_variant_bench speed-bench/*.o tests/test_q4k_dot tests/test_mxfp4_dot tests/test_mxfp4_metal tests/test_mxfp4_rocm tests/test_mxfp4_cuda tests/test_metal_session_batch tests/test_metal_moe_prefill tests/test_metal_dense_mpp tests/test_glm53_kda tests/test_glm53_kda_rocm tests/test_glm53_vision_engine tests/test_glm53_vision_prompt tests/test_deepseek4_vision_image tests/test_prompt_prefix tests/test_gpu_xdev tests/test_gpu_model_cache tests/test_gpu_lookup_cache_strict tests/test_engine_mgpu_refusal tests/test_engine_mgpu_runtime tests/test_engine_correctness tests/test_sampling tests/test_cuda_session_batch tests/test_cuda_mixed_batch tests/*.o *.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o

# Q4 attention validation. See docs/Q4_ATTENTION.md for scope and hardware limits.
.PHONY: test-q4-preflight-host
test-q4-preflight-host: tests/test_q4_preflight.py tests/kernel_source.py ds4.c ds4_gpu.h
	python3 tests/test_q4_preflight.py

GPU_IQ2_SIGN_DEPS := tests/test_gpu_iq2_signs.py tests/test_gpu_iq2_signs.cpp \
	tests/kernel_source.py ds4_iq2_tables_cuda.inc ds4_cuda.cu ds4_rocm.cu \
	ds4_rocm.h rocm/ds4_rocm_moe.cuh
.PHONY: test-gpu-iq2-signs-host test-cuda-iq2-signs test-rocm-iq2-signs
test-gpu-iq2-signs-host: $(GPU_IQ2_SIGN_DEPS)
	python3 tests/test_gpu_iq2_signs.py

test-cuda-iq2-signs: $(GPU_IQ2_SIGN_DEPS)
	NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" python3 tests/test_gpu_iq2_signs.py --cuda

test-rocm-iq2-signs: $(GPU_IQ2_SIGN_DEPS)
	HIPCC="$(HIPCC)" ROCM_CFLAGS="$(ROCM_CFLAGS)" python3 tests/test_gpu_iq2_signs.py --rocm

.PHONY: test-cuda-hc-split-norm-host test-cuda-hc-split-norm test-cuda-q8-quantize-host test-cuda-q8-quantize test-rocm-raw-kv-store-host
test-cuda-hc-split-norm-host: tests/test_cuda_hc_split_norm.py tests/kernel_source.py ds4_cuda.cu
	python3 tests/test_cuda_hc_split_norm.py

test-cuda-hc-split-norm: tests/test_cuda_hc_split_norm.py tests/kernel_source.py ds4_cuda.cu
	NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" python3 tests/test_cuda_hc_split_norm.py --cuda

test-cuda-q8-quantize-host: tests/test_cuda_q8_quantize.py tests/test_cuda_q8_quantize.cpp tests/kernel_source.py ds4_cuda.cu cuda/ds4_q8_quantize.cuh
	python3 tests/test_cuda_q8_quantize.py

test-cuda-q8-quantize: tests/test_cuda_q8_quantize.py tests/test_cuda_q8_quantize.cpp ds4_cuda.cu cuda/ds4_q8_quantize.cuh
	NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" python3 tests/test_cuda_q8_quantize.py --cuda

F16_COMPRESSOR_DEPS := tests/test_cuda_f16_compressor.py tests/test_cuda_f16_compressor.cpp \
	tests/kernel_source.py ds4_cuda.cu cuda/ds4_f16_compressor.cuh
.PHONY: test-cuda-f16-compressor-host test-cuda-f16-compressor bench-cuda-f16-compressor
test-cuda-f16-compressor-host: $(F16_COMPRESSOR_DEPS) tests/test_cuda_f16_compressor_policy.py
	python3 tests/test_cuda_f16_compressor.py
	python3 tests/test_cuda_f16_compressor_policy.py

test-cuda-f16-compressor: $(F16_COMPRESSOR_DEPS)
	NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" python3 tests/test_cuda_f16_compressor.py --cuda

bench-cuda-f16-compressor: $(F16_COMPRESSOR_DEPS)
	NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" python3 tests/test_cuda_f16_compressor.py --cuda --bench

ROCM_F16_COMPRESSOR_DEPS := tests/test_rocm_f16_compressor.py tests/test_rocm_f16_compressor.cpp \
	tests/kernel_source.py rocm/ds4_rocm_common.cuh rocm/ds4_rocm_matmul.cuh \
	rocm/ds4_rocm_compressor.cuh rocm/ds4_rocm_q8.cuh rocm/ds4_rocm_norm_rope.cuh \
	rocm/ds4_rocm_runtime.cuh
.PHONY: test-rocm-f16-compressor-host test-rocm-f16-compressor bench-rocm-f16-compressor
test-rocm-f16-compressor-host: $(ROCM_F16_COMPRESSOR_DEPS) tests/test_rocm_f16_compressor_policy.py
	python3 tests/test_rocm_f16_compressor.py
	python3 tests/test_rocm_f16_compressor_policy.py

test-rocm-f16-compressor: $(ROCM_F16_COMPRESSOR_DEPS)
	HIPCC="$(HIPCC)" ROCM_CFLAGS="$(ROCM_CFLAGS)" python3 tests/test_rocm_f16_compressor.py --rocm

bench-rocm-f16-compressor: $(ROCM_F16_COMPRESSOR_DEPS)
	HIPCC="$(HIPCC)" ROCM_CFLAGS="$(ROCM_CFLAGS)" python3 tests/test_rocm_f16_compressor.py --rocm --bench

Q8_HC_ALIGNED_DEPS := tests/test_cuda_q8_hc_aligned.py tests/test_cuda_q8_hc_aligned.cpp \
	tests/kernel_source.py ds4_cuda.cu cuda/ds4_q8_quantize.cuh
.PHONY: test-cuda-q8-hc-aligned-host test-cuda-q8-hc-aligned
test-cuda-q8-hc-aligned-host: $(Q8_HC_ALIGNED_DEPS)
	python3 tests/test_cuda_q8_hc_aligned.py

test-cuda-q8-hc-aligned: $(Q8_HC_ALIGNED_DEPS)
	NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" python3 tests/test_cuda_q8_hc_aligned.py --cuda

test-rocm-raw-kv-store-host: tests/test_rocm_raw_kv_store.py tests/kernel_source.py ds4_gpu.h rocm/ds4_rocm_fp8_kv.cuh rocm/ds4_rocm_attention_launch.cuh
	python3 tests/test_rocm_raw_kv_store.py

Q4_TOK8_DEPS := tests/test_cuda_q4_grouped_tok8.py tests/kernel_source.py \
	speed-bench/cuda_q4_grouped_tok8_bench.cu cuda/ds4_q4_grouped_tok8_candidate.cuh ds4_cuda.cu
Q4_TOK8_TOKENS ?= 512
Q4_TOK8_DEVICE ?= 0
.PHONY: test-cuda-q4-grouped-tok8-host test-cuda-q4-grouped-tok8 bench-cuda-q4-grouped-tok8
test-cuda-q4-grouped-tok8-host: $(Q4_TOK8_DEPS)
	python3 tests/test_cuda_q4_grouped_tok8.py

test-cuda-q4-grouped-tok8: $(Q4_TOK8_DEPS) $(MMQ_OBJS)
	NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" python3 tests/test_cuda_q4_grouped_tok8.py \
		--cuda --device $(Q4_TOK8_DEVICE) --link-objects "$(MMQ_OBJS)"

bench-cuda-q4-grouped-tok8: $(Q4_TOK8_DEPS) $(MMQ_OBJS)
	NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" python3 tests/test_cuda_q4_grouped_tok8.py \
		--cuda --device $(Q4_TOK8_DEVICE) --link-objects "$(MMQ_OBJS)" \
		--bench --tokens $(Q4_TOK8_TOKENS)

.PHONY: test-q4-epilogue-host test-cuda-q4-epilogue
tests/test_q4_epilogue_host: tests/test_cuda_q4_epilogue.cpp cuda/mmq/ds4_q4_mmvq_epilogue.h
	$(CXX) -O2 -Wall -Wextra -std=c++17 -o $@ $<

test-q4-epilogue-host: tests/test_q4_epilogue_host
	./tests/test_q4_epilogue_host

.PHONY: test-q4-prefill-dequant-host test-cuda-q4-prefill-dequant test-rocm-q4-prefill-dequant bench-cuda-q4-prefill-dequant bench-rocm-q4-prefill-dequant
Q4_PREFILL_DEQUANT_HEADERS := cuda/ds4_q4_dequant_layout.h cuda/ds4_q4_dequant_vec.cuh

tests/test_q4_prefill_dequant_host: tests/test_q4_prefill_dequant.cpp $(Q4_PREFILL_DEQUANT_HEADERS)
	$(CXX) -O2 -Wall -Wextra -std=c++17 -o $@ $<

test-q4-prefill-dequant-host: tests/test_q4_prefill_dequant_host
	./tests/test_q4_prefill_dequant_host

tests/test_cuda_q4_prefill_dequant: tests/test_q4_prefill_dequant.cpp $(Q4_PREFILL_DEQUANT_HEADERS)
	@if [ -z "$(NVCC)" ] || ! command -v "$(NVCC)" >/dev/null 2>&1; then \
		echo "error: native Q4 dequant tests require nvcc"; exit 1; fi
	$(NVCC) $(NVCCFLAGS) -std=c++17 -x cu -o $@ $<

tests/test_rocm_q4_prefill_dequant: tests/test_q4_prefill_dequant.cpp $(Q4_PREFILL_DEQUANT_HEADERS)
	@if [ -z "$(HIPCC)" ] || ! command -v "$(HIPCC)" >/dev/null 2>&1; then \
		echo "error: native Q4 dequant tests require hipcc"; exit 1; fi
	$(HIPCC) $(ROCM_CFLAGS) -std=c++17 -x hip -o $@ $<

test-cuda-q4-prefill-dequant: tests/test_cuda_q4_prefill_dequant
	./tests/test_cuda_q4_prefill_dequant

test-rocm-q4-prefill-dequant: tests/test_rocm_q4_prefill_dequant
	./tests/test_rocm_q4_prefill_dequant

bench-cuda-q4-prefill-dequant: tests/test_cuda_q4_prefill_dequant
	./tests/test_cuda_q4_prefill_dequant --bench

bench-rocm-q4-prefill-dequant: tests/test_rocm_q4_prefill_dequant
	./tests/test_rocm_q4_prefill_dequant --bench
test-cuda-q4-prefill-norm-host:
	python3 tests/test_cuda_q4_prefill_norm.py

test-cuda-q4-prefill-norm:
	NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" python3 tests/test_cuda_q4_prefill_norm.py --cuda

test-cuda-q4-dequant-flat-host:
	python3 tests/test_cuda_q4_dequant_flat.py

test-rocm-q4-dequant-flat-host:
	python3 tests/test_cuda_q4_dequant_flat.py --backend rocm

test-cuda-q4-dequant-flat:
	NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" python3 tests/test_cuda_q4_dequant_flat.py --gpu

test-rocm-q4-dequant-flat:
	HIPCC="$(HIPCC)" ROCM_CFLAGS="$(ROCM_CFLAGS)" python3 tests/test_cuda_q4_dequant_flat.py --backend rocm --gpu

.PHONY: test-q4-prefill-reduce-host test-cuda-q4-prefill-reduce bench-cuda-q4-prefill-reduce
tests/test_q4_prefill_reduce_host: tests/test_cuda_q4_prefill_reduce.cpp cuda/ds4_q4_prefill_reduce.h
	$(CXX) -O2 -Wall -Wextra -std=c++17 -fno-fast-math -o $@ $<

tests/test_q4_prefill_reduce_host_fast: tests/test_cuda_q4_prefill_reduce.cpp cuda/ds4_q4_prefill_reduce.h
	$(CXX) -O3 -Wall -Wextra -std=c++17 -ffast-math -fno-finite-math-only -o $@ $<

test-q4-prefill-reduce-host: tests/test_q4_prefill_reduce_host tests/test_q4_prefill_reduce_host_fast
	./tests/test_q4_prefill_reduce_host
	./tests/test_q4_prefill_reduce_host_fast

tests/test_cuda_q4_prefill_reduce: tests/test_cuda_q4_prefill_reduce.cpp cuda/ds4_q4_prefill_reduce.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -x cu -o $@ $<

CUDA_Q4_PREFILL_REDUCE_TEST_ARGS ?=
test-cuda-q4-prefill-reduce:
	@reduce_nvcc="$(strip $(NVCC))"; \
	if [ -z "$$reduce_nvcc" ]; then reduce_nvcc="$$(command -v nvcc 2>/dev/null || true)"; fi; \
	reduce_probe="$${reduce_nvcc%% *}"; \
	if [ -z "$$reduce_probe" ] || ! command -v "$$reduce_probe" >/dev/null 2>&1; then \
		echo "CUDA Q4 prefill reduction: FAIL (nvcc and CUDA device required)"; exit 1; \
	fi; \
	$(MAKE) --no-print-directory tests/test_cuda_q4_prefill_reduce NVCC="$$reduce_nvcc" || exit $$?; \
	./tests/test_cuda_q4_prefill_reduce $(CUDA_Q4_PREFILL_REDUCE_TEST_ARGS)

bench-cuda-q4-prefill-reduce:
	$(MAKE) test-cuda-q4-prefill-reduce CUDA_Q4_PREFILL_REDUCE_TEST_ARGS=--bench

.PHONY: test-rocm-q4-dot-host
.PHONY: test-rocm-q4-prefill-dispatch-host
test-rocm-q4-prefill-dispatch-host: tests/test_rocm_q4_prefill_dispatch.py tests/kernel_source.py \
	rocm/ds4_rocm_q4.cuh rocm/ds4_rocm_runtime.cuh
	python3 tests/test_rocm_q4_prefill_dispatch.py

ROCM_Q4_DOT_HEADERS = rocm/ds4_rocm_q4_dot.cuh rocm/ds4_rocm_q4_lds.cuh
tests/test_rocm_q4_dot_host: tests/test_rocm_q4_dot_host.cpp $(ROCM_Q4_DOT_HEADERS)
	$(CXX) -O2 -Wall -Wextra -std=c++17 -fno-fast-math -I. -o $@ $<

tests/test_rocm_q4_dot_host_fast: tests/test_rocm_q4_dot_host.cpp $(ROCM_Q4_DOT_HEADERS)
	$(CXX) -O3 -Wall -Wextra -std=c++17 -ffast-math -fno-finite-math-only -I. -o $@ $<

test-rocm-q4-dot-host: tests/test_rocm_q4_dot_host tests/test_rocm_q4_dot_host_fast
	./tests/test_rocm_q4_dot_host
	./tests/test_rocm_q4_dot_host_fast

.PHONY: test-rocm-q4-lds-host
tests/test_rocm_q4_lds_host: tests/test_rocm_q4_lds_host.cpp rocm/ds4_rocm_q4_lds.cuh
	$(CXX) -O2 -Wall -Wextra -std=c++17 -I. -o $@ $<

test-rocm-q4-lds-host: tests/test_rocm_q4_lds_host
	./tests/test_rocm_q4_lds_host

.PHONY: test-rocm-q4-lds-aligned-host
tests/test_rocm_q4_lds_aligned_host: tests/test_rocm_q4_lds_aligned_host.cpp rocm/ds4_rocm_q4_lds.cuh
	$(CXX) -O2 -Wall -Wextra -std=c++17 -fno-fast-math -I. -o $@ $<

tests/test_rocm_q4_lds_aligned_host_fast: tests/test_rocm_q4_lds_aligned_host.cpp rocm/ds4_rocm_q4_lds.cuh
	$(CXX) -O3 -Wall -Wextra -std=c++17 -ffast-math -fno-finite-math-only -I. -o $@ $<

test-rocm-q4-lds-aligned-host: tests/test_rocm_q4_lds_aligned_host tests/test_rocm_q4_lds_aligned_host_fast
	./tests/test_rocm_q4_lds_aligned_host
	./tests/test_rocm_q4_lds_aligned_host_fast

.PHONY: test-rocm-q4-lds-aligned bench-rocm-q4-lds-aligned
tests/test_rocm_q4_lds_aligned.o: tests/test_rocm_q4_lds_aligned.cpp ds4_gpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -std=c++17 -fno-fast-math -I. -c -o $@ $<

tests/test_rocm_q4_lds_aligned: tests/test_rocm_q4_lds_aligned.o ds4_image.o ds4_rocm.o $(ROCM_MMQ_OBJS) ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

ROCM_Q4_LDS_ALIGNED_TEST_ARGS ?=
test-rocm-q4-lds-aligned:
	@lds_hipcc="$(strip $(HIPCC))"; \
	if [ -z "$$lds_hipcc" ]; then lds_hipcc="$$(command -v hipcc 2>/dev/null || true)"; fi; \
	lds_probe="$${lds_hipcc%% *}"; \
	if [ -z "$$lds_probe" ] || ! command -v "$$lds_probe" >/dev/null 2>&1; then \
		echo "ROCm Q4 aligned LDS: FAIL (hipcc and gfx1151 device required)"; exit 1; \
	fi; \
	$(MAKE) --no-print-directory tests/test_rocm_q4_lds_aligned HIPCC="$$lds_hipcc" || exit $$?; \
	./tests/test_rocm_q4_lds_aligned $(ROCM_Q4_LDS_ALIGNED_TEST_ARGS)

bench-rocm-q4-lds-aligned:
	$(MAKE) test-rocm-q4-lds-aligned ROCM_Q4_LDS_ALIGNED_TEST_ARGS="--bench $(ROCM_Q4_LDS_ALIGNED_TEST_ARGS)"

.PHONY: test-rocm-q4-wmma-load-host
tests/test_rocm_q4_wmma_load_host: tests/test_rocm_q4_wmma_load_host.cpp rocm/ds4_rocm_q4_wmma_load.cuh
	$(CXX) -O2 -Wall -Wextra -std=c++17 -I. -o $@ $<

test-rocm-q4-wmma-load-host: tests/test_rocm_q4_wmma_load_host
	./tests/test_rocm_q4_wmma_load_host

.PHONY: test-rocm-q4-qb-epilogue-host test-rocm-q4-qb-epilogue bench-rocm-q4-qb-epilogue
tests/test_rocm_q4_qb_epilogue_host: tests/test_rocm_q4_qb_epilogue_host.cpp rocm/ds4_rocm_q4_qb_epilogue_layout.cuh
	$(CXX) -O2 -Wall -Wextra -std=c++17 -fno-fast-math -ffp-contract=off -I. -o $@ $<

# ROCm uses Clang FP pragmas; exercise them under fast-math even without HIP.
ROCM_Q4_EPILOGUE_HOST_CLANG ?= clang++
tests/test_rocm_q4_qb_epilogue_host_fast: tests/test_rocm_q4_qb_epilogue_host.cpp rocm/ds4_rocm_q4_qb_epilogue_layout.cuh
	$(ROCM_Q4_EPILOGUE_HOST_CLANG) -O3 -Wall -Wextra -std=c++17 -ffast-math -fno-finite-math-only -I. -o $@ $<

test-rocm-q4-qb-epilogue-host: tests/test_rocm_q4_qb_epilogue_host tests/test_rocm_q4_qb_epilogue_host_fast
	./tests/test_rocm_q4_qb_epilogue_host
	./tests/test_rocm_q4_qb_epilogue_host_fast

tests/test_rocm_q4_qb_epilogue.o: tests/test_rocm_q4_qb_epilogue.cpp ds4_gpu.h rocm/ds4_rocm_q4_qb_epilogue_layout.cuh
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -std=c++17 -fno-fast-math -I. -c -o $@ $<

tests/test_rocm_q4_qb_epilogue: tests/test_rocm_q4_qb_epilogue.o ds4_image.o ds4_rocm.o $(ROCM_MMQ_OBJS) ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

ROCM_Q4_EPILOGUE_TEST_ARGS ?=
test-rocm-q4-qb-epilogue:
	@epilogue_hipcc="$(strip $(HIPCC))"; \
	if [ -z "$$epilogue_hipcc" ]; then epilogue_hipcc="$$(command -v hipcc 2>/dev/null || true)"; fi; \
	epilogue_probe="$${epilogue_hipcc%% *}"; \
	if [ -z "$$epilogue_probe" ] || ! command -v "$$epilogue_probe" >/dev/null 2>&1; then \
		if [ -n "$(strip $(DS4_TEST_REQUIRE_ROCM_DEVICE))" ] && [ "$(strip $(DS4_TEST_REQUIRE_ROCM_DEVICE))" != "0" ]; then \
			echo "ROCm Q4 F32 epilogue: FAIL (hipcc not found, device required)"; exit 1; \
		fi; \
		echo "ROCm Q4 F32 epilogue: SKIP (hipcc not found)"; exit 0; \
	fi; \
	$(MAKE) --no-print-directory tests/test_rocm_q4_qb_epilogue HIPCC="$$epilogue_hipcc" || exit $$?; \
	DS4_TEST_REQUIRE_ROCM_DEVICE="$(strip $(DS4_TEST_REQUIRE_ROCM_DEVICE))" \
		./tests/test_rocm_q4_qb_epilogue $(ROCM_Q4_EPILOGUE_TEST_ARGS); \
	rc=$$?; \
	if [ $$rc -eq 77 ]; then echo "ROCm Q4 F32 epilogue: SKIP (no HIP device)"; exit 0; fi; \
	exit $$rc

bench-rocm-q4-qb-epilogue:
	$(MAKE) test-rocm-q4-qb-epilogue ROCM_Q4_EPILOGUE_TEST_ARGS="--bench $(ROCM_Q4_EPILOGUE_TEST_ARGS)"

tests/test_rocm_q4_dense_pair.o: tests/test_rocm_q4_dense_pair.cpp ds4_gpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -std=c++17 -fno-fast-math -I. -c -o $@ $<

tests/test_rocm_q4_dense_pair: tests/test_rocm_q4_dense_pair.o ds4_image.o ds4_rocm.o $(ROCM_MMQ_OBJS) ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

# Keep the public test target usable on development hosts without ROCm.  The
# binary itself exits 77 when HIP is installed but no device is visible; an
# explicitly required Strix run converts that condition into a hard failure.
ROCM_Q4_TEST_ARGS ?= --all
test-rocm-q4-parity:
	@rocm_test_hipcc="$(strip $(HIPCC))"; \
	if [ -z "$$rocm_test_hipcc" ]; then \
		rocm_test_hipcc="$$(command -v hipcc 2>/dev/null || true)"; \
	fi; \
	rocm_test_probe="$${rocm_test_hipcc%% *}"; \
	if [ -z "$$rocm_test_probe" ] || ! command -v "$$rocm_test_probe" >/dev/null 2>&1; then \
		if [ -n "$(strip $(DS4_TEST_REQUIRE_ROCM_DEVICE))" ] && [ "$(strip $(DS4_TEST_REQUIRE_ROCM_DEVICE))" != "0" ]; then \
			echo "ROCm Q4 dense/pair/prefill oracle: FAIL (hipcc not found, device required)"; \
			exit 1; \
		fi; \
		echo "ROCm Q4 dense/pair/prefill oracle: SKIP (hipcc not found)"; exit 0; \
	fi; \
	$(MAKE) --no-print-directory tests/test_rocm_q4_dense_pair HIPCC="$$rocm_test_hipcc" || exit $$?; \
	if [ -n "$(strip $(DS4_TEST_REQUIRE_ROCM_DEVICE))" ] && [ "$(strip $(DS4_TEST_REQUIRE_ROCM_DEVICE))" != "0" ]; then \
		DS4_TEST_REQUIRE_ROCM_DEVICE="$(strip $(DS4_TEST_REQUIRE_ROCM_DEVICE))" \
			./tests/test_rocm_q4_dense_pair $(ROCM_Q4_TEST_ARGS); \
	else \
		env -u DS4_TEST_REQUIRE_ROCM_DEVICE \
			./tests/test_rocm_q4_dense_pair $(ROCM_Q4_TEST_ARGS); \
	fi; \
	rc=$$?; \
	if [ $$rc -eq 77 ]; then \
		echo "ROCm Q4 dense/pair/prefill oracle: SKIP (no visible HIP device)"; \
		exit 0; \
	fi; \
	exit $$rc

test-rocm-q4-dense:
	$(MAKE) --no-print-directory test-rocm-q4-parity ROCM_Q4_TEST_ARGS=--dense

test-rocm-q4-pair:
	$(MAKE) --no-print-directory test-rocm-q4-parity ROCM_Q4_TEST_ARGS=--pair

test-rocm-q4-prefill:
	$(MAKE) --no-print-directory test-rocm-q4-parity ROCM_Q4_TEST_ARGS=--prefill

.PHONY: test-rocm-q4-prefill-load4
test-rocm-q4-prefill-load4:
	$(MAKE) --no-print-directory test-rocm-q4-parity ROCM_Q4_TEST_ARGS=--prefill-wmma-load4

test-strix-rocm-q4-parity:
	$(MAKE) --no-print-directory -B test-rocm-q4-parity ROCM_ARCH=gfx1151 DS4_TEST_REQUIRE_ROCM_DEVICE=1

test-strix-rocm-q4-prefill:
	$(MAKE) --no-print-directory -B test-rocm-q4-parity ROCM_ARCH=gfx1151 \
		DS4_TEST_REQUIRE_ROCM_DEVICE=1 ROCM_Q4_TEST_ARGS=--prefill
	$(MAKE) --no-print-directory test-rocm-q4-qb-epilogue ROCM_ARCH=gfx1151 \
		DS4_TEST_REQUIRE_ROCM_DEVICE=1
	$(MAKE) --no-print-directory test-rocm-q4-lds-aligned ROCM_ARCH=gfx1151

test-strix-rocm-q4-prefill-long:
	$(MAKE) --no-print-directory -B test-rocm-q4-parity ROCM_ARCH=gfx1151 \
		DS4_TEST_REQUIRE_ROCM_DEVICE=1 ROCM_Q4_TEST_ARGS=--prefill-long


test-cuda-mmq-dense-ids-host:
	python3 tests/test_cuda_mmq_dense_ids.py

ifeq ($(UNAME_S),Darwin)
tests/test_metal_q4_prefill_pair.o: tests/test_metal_q4_prefill_pair.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_metal_q4_prefill_pair: tests/test_metal_q4_prefill_pair.o ds4_image.o ds4_metal.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-q4-prefill-pair: tests/test_metal_q4_prefill_pair
	env \
		-u DS4_METAL_DISABLE_Q4_PREFILL_PAIR_F16_RHS \
		-u DS4_METAL_REQUIRE_Q4_PREFILL_PAIR_F16_RHS \
		-u DS4_METAL_DISABLE_Q4_DENSE_PAIR \
		-u DS4_METAL_DISABLE_CONTIG_F32_F16_COPY \
		-u DS4_METAL_MODEL_UNTRACKED \
		-u DS4_METAL_UNRETAINED_COMMAND_BUFFERS \
		./tests/test_metal_q4_prefill_pair
	env \
		-u DS4_METAL_DISABLE_Q4_PREFILL_PAIR_F16_RHS \
		-u DS4_METAL_REQUIRE_Q4_PREFILL_PAIR_F16_RHS \
		-u DS4_METAL_DISABLE_Q4_DENSE_PAIR \
		-u DS4_METAL_DISABLE_CONTIG_F32_F16_COPY \
		-u DS4_METAL_MODEL_UNTRACKED \
		DS4_METAL_UNRETAINED_COMMAND_BUFFERS=1 \
		./tests/test_metal_q4_prefill_pair

tests/test_metal_indexer_q4.o: tests/test_metal_indexer_q4.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_metal_indexer_q4: tests/test_metal_indexer_q4.o ds4_image.o ds4_metal.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-indexer-q4: tests/test_metal_indexer_q4
	./tests/test_metal_indexer_q4

tests/test_metal_q4_attn_out_a_direct.o: tests/test_metal_q4_attn_out_a_direct.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_metal_q4_attn_out_a_direct: tests/test_metal_q4_attn_out_a_direct.o ds4_image.o ds4_metal.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-q4-attn-out-a-direct: tests/test_metal_q4_attn_out_a_direct
	env -u DS4_METAL_DISABLE_Q4_ATTN_OUT_A_DIRECT \
		-u DS4_METAL_REQUIRE_Q4_ATTN_OUT_A_DIRECT \
		-u DS4_METAL_DISABLE_Q4_ATTN_OUT_B_F16_RHS \
		-u DS4_METAL_REQUIRE_Q4_ATTN_OUT_B_F16_RHS \
		./tests/test_metal_q4_attn_out_a_direct

tests/test_metal_q4_qb_f16_cache.o: tests/test_metal_q4_qb_f16_cache.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_metal_q4_qb_f16_cache: tests/test_metal_q4_qb_f16_cache.o ds4_image.o ds4_metal.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-q4-qb-f16-cache: tests/test_metal_q4_qb_f16_cache
	env -u DS4_METAL_DISABLE_Q4_ATTN_Q_B_F16_CACHE \
		-u DS4_METAL_DISABLE_Q4_ATTN_Q_B_F16_RHS \
		-u DS4_METAL_DISABLE_Q4_ATTN_Q_B_TRANSIENT_F16 \
		-u DS4_METAL_Q4_ATTN_Q_B_TRANSIENT_F16_MIN_TOKENS \
		-u DS4_METAL_UNRETAINED_COMMAND_BUFFERS \
		-u DS4_TEST_METAL_Q4_QB_F16_CACHE_TIMING \
		-u DS4_TEST_METAL_Q4_QB_F16_CACHE_TIMING_TOKENS \
		DS4_METAL_Q4_ATTN_Q_B_F16_CACHE_MIN_TOKENS=32 \
		DS4_METAL_REQUIRE_Q4_ATTN_Q_B_F16_CACHE=1 \
		./tests/test_metal_q4_qb_f16_cache
	env -u DS4_METAL_DISABLE_Q4_ATTN_Q_B_F16_CACHE \
		-u DS4_METAL_DISABLE_Q4_ATTN_Q_B_F16_RHS \
		-u DS4_METAL_DISABLE_Q4_ATTN_Q_B_TRANSIENT_F16 \
		-u DS4_METAL_Q4_ATTN_Q_B_TRANSIENT_F16_MIN_TOKENS \
		-u DS4_TEST_METAL_Q4_QB_F16_CACHE_TIMING \
		-u DS4_TEST_METAL_Q4_QB_F16_CACHE_TIMING_TOKENS \
		DS4_METAL_UNRETAINED_COMMAND_BUFFERS=1 \
		DS4_METAL_Q4_ATTN_Q_B_F16_CACHE_MIN_TOKENS=32 \
		DS4_METAL_REQUIRE_Q4_ATTN_Q_B_F16_CACHE=1 \
		./tests/test_metal_q4_qb_f16_cache

.PHONY: test-metal-q4-qb-token-pair
tests/test_metal_q4_qb_token_pair: tests/test_metal_q4_qb_token_pair.m ds4_gpu.h ds4_image.o ds4_metal.o
	$(CC) $(OBJCFLAGS) -I. -o $@ $< ds4_image.o ds4_metal.o $(METAL_LDLIBS)

test-metal-q4-qb-token-pair: tests/test_metal_q4_qb_token_pair
	./tests/test_metal_q4_qb_token_pair

.PHONY: test-metal-q4-hc
tests/test_metal_q4_hc: tests/test_metal_q4_hc.c ds4_gpu.h ds4_image.o ds4_metal.o
	$(CC) $(CFLAGS) -I. -o $@ $< ds4_image.o ds4_metal.o $(METAL_LDLIBS)

test-metal-q4-hc: tests/test_metal_q4_hc
	./tests/test_metal_q4_hc

.PHONY: test-metal-q4-prefill-long
# Large synthetic fixtures run sequentially to bound peak GPU memory usage.
test-metal-q4-prefill-long: tests/test_metal_q4_attn_out_a_direct tests/test_metal_q4_hc
	./tests/test_metal_q4_attn_out_a_direct --tokens 8191
	./tests/test_metal_q4_attn_out_a_direct --tokens 8192
	./tests/test_metal_q4_hc --tokens 8192

.PHONY: test-metal-decode-fusions test-metal-f16-compressor
test-metal-decode-fusions: tests/test_metal_decode_fusions.py tests/test_metal_decode_fusions.m \
		tests/kernel_source.py metal/dense.metal metal/dsv4_hc.metal
	python3 tests/test_metal_decode_fusions.py

.PHONY: test-metal-iq2-signs
test-metal-iq2-signs: tests/test_metal_iq2_signs.py tests/test_metal_iq2_signs.m \
		tests/kernel_source.py metal/moe.metal metal/dense.metal
	python3 tests/test_metal_iq2_signs.py

tests/test_metal_f16_compressor: tests/test_metal_f16_compressor.c ds4_gpu.h ds4_image.o ds4_metal.o
	$(CC) $(CFLAGS) -I. -o $@ $< ds4_image.o ds4_metal.o $(METAL_LDLIBS)

test-metal-f16-compressor: tests/test_metal_f16_compressor
	./tests/test_metal_f16_compressor

.PHONY: test-metal-decode-defaults
tests/test_metal_decode_defaults: tests/test_metal_decode_defaults.m ds4_metal.m ds4_gpu.h $(METAL_SRCS)
	$(CC) -O2 -fobjc-arc -fblocks -DDS4_USE_METAL -o $@ $< $(METAL_LDLIBS) -framework Accelerate

test-metal-decode-defaults: tests/test_metal_decode_defaults
	./tests/test_metal_decode_defaults

speed-bench/metal_q4_dense_pair_bench: speed-bench/metal_q4_dense_pair_bench.m $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -o $@ $< $(METAL_LDLIBS)
metal-q4-dense-pair-bench: speed-bench/metal_q4_dense_pair_bench

speed-bench/metal_q4_prefill_pair_bench: speed-bench/metal_q4_prefill_pair_bench.m $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -o $@ $< $(METAL_LDLIBS)

metal-q4-prefill-pair-bench: speed-bench/metal_q4_prefill_pair_bench

speed-bench/metal_q4_mm_tail_cull_bench: speed-bench/metal_q4_mm_tail_cull_bench.m $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -o $@ $< $(METAL_LDLIBS)

metal-q4-mm-tail-cull-bench: speed-bench/metal_q4_mm_tail_cull_bench

speed-bench/metal_q4_attn_out_a_direct_bench: speed-bench/metal_q4_attn_out_a_direct_bench.m $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -o $@ $< $(METAL_LDLIBS)

metal-q4-attn-out-a-direct-bench: speed-bench/metal_q4_attn_out_a_direct_bench

endif

ifneq ($(UNAME_S),Darwin)
cuda/mmq/test/test_mmq_parity: cuda/mmq/test/test_mmq_parity.cu cuda/mmq/test/iq2_host_tables.h cuda/mmq/ggml-common.h cuda/mmq/ds4_mmq.h $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -o $@ $< $(MMQ_OBJS) $(CUDA_LDLIBS)

test-mmq-parity-cuda: cuda/mmq/test/test_mmq_parity
	./cuda/mmq/test/test_mmq_parity

test-mmq-q4-grouped-q81-cuda: cuda/mmq/test/test_mmq_parity
	./cuda/mmq/test/test_mmq_parity --q4-grouped-q81

tests/test_cuda_q4_epilogue.o: tests/test_cuda_q4_epilogue.cpp cuda/mmq/ds4_q4_mmvq_epilogue.h cuda/mmq/ds4_mmq.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -x cu $(MMQ_INCLUDES) -c -o $@ $<

tests/test_cuda_q4_epilogue: tests/test_cuda_q4_epilogue.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-q4-epilogue: tests/test_cuda_q4_epilogue
	./tests/test_cuda_q4_epilogue
speed-bench/rocm_q4_prefill_bench.o: speed-bench/rocm_q4_prefill_bench.cpp ds4_gpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -std=c++17 -fno-fast-math -I. -c -o $@ $<

speed-bench/rocm_q4_prefill_bench: speed-bench/rocm_q4_prefill_bench.o ds4_image.o ds4_rocm.o $(ROCM_MMQ_OBJS) ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

rocm-q4-prefill-bench:
	$(MAKE) --no-print-directory -B speed-bench/rocm_q4_prefill_bench ROCM_ARCH="$(ROCM_ARCH)"
speed-bench/cuda_q4_prefill_bench.o: speed-bench/cuda_q4_prefill_bench.cu ds4_gpu.h cuda/mmq/ds4_mmq.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -DDS4_BENCH_CUDA -I. -c -o $@ $<

speed-bench/cuda_q4_prefill_bench: speed-bench/cuda_q4_prefill_bench.o ds4_image.o ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -o $@ $^ $(CUDA_LDLIBS)

cuda-q4-prefill-bench:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda-q4-prefill-bench CUDA_ARCH=sm_121"; \
		exit 2; \
	fi
	$(MAKE) --no-print-directory -B speed-bench/cuda_q4_prefill_bench CUDA_ARCH="$(CUDA_ARCH)"

endif

.PHONY: test-q4-epilogue-host test-q4-prefill-dequant-host test-cuda-q4-prefill-dequant test-rocm-q4-prefill-dequant \
	bench-cuda-q4-prefill-dequant bench-rocm-q4-prefill-dequant test-cuda-q4-prefill-norm-host test-cuda-q4-prefill-norm \
	test-cuda-q4-dequant-flat-host test-rocm-q4-dequant-flat-host test-cuda-q4-dequant-flat test-rocm-q4-dequant-flat \
	test-q4-prefill-reduce-host test-cuda-q4-prefill-reduce bench-cuda-q4-prefill-reduce test-rocm-q4-dot-host \
	test-rocm-q4-lds-host test-rocm-q4-lds-aligned-host test-rocm-q4-lds-aligned bench-rocm-q4-lds-aligned \
	test-rocm-q4-wmma-load-host test-rocm-q4-qb-epilogue-host test-rocm-q4-qb-epilogue bench-rocm-q4-qb-epilogue \
	test-rocm-q4-parity test-rocm-q4-dense test-rocm-q4-pair test-rocm-q4-prefill \
	test-rocm-q4-prefill-load4 test-strix-rocm-q4-parity test-strix-rocm-q4-prefill test-strix-rocm-q4-prefill-long \
	test-cuda-mmq-dense-ids-host test-metal-q4-prefill-pair test-metal-indexer-q4 test-metal-q4-attn-out-a-direct \
	test-metal-q4-qb-f16-cache test-metal-q4-qb-token-pair metal-q4-dense-pair-bench metal-q4-prefill-pair-bench \
	metal-q4-mm-tail-cull-bench metal-q4-attn-out-a-direct-bench test-mmq-parity-cuda test-mmq-q4-grouped-q81-cuda \
	test-cuda-q4-epilogue rocm-q4-prefill-bench cuda-q4-prefill-bench

# Only generated Q4 executables are removed by this auxiliary clean target.
Q4_BUILD_PRODUCTS := tests/test_cpu_q4_dense tests/test_quantizer_indexer_q4 tests/test_q4_epilogue_host \
	tests/test_q4_prefill_dequant_host tests/test_cuda_q4_prefill_dequant tests/test_rocm_q4_prefill_dequant \
	tests/test_q4_prefill_reduce_host tests/test_q4_prefill_reduce_host_fast tests/test_cuda_q4_prefill_reduce \
	tests/test_rocm_q4_dot_host tests/test_rocm_q4_dot_host_fast tests/test_rocm_q4_lds_host \
	tests/test_rocm_q4_lds_aligned_host tests/test_rocm_q4_lds_aligned_host_fast tests/test_rocm_q4_lds_aligned \
	tests/test_rocm_q4_wmma_load_host tests/test_rocm_q4_qb_epilogue_host tests/test_rocm_q4_qb_epilogue_host_fast \
	tests/test_rocm_q4_qb_epilogue tests/test_rocm_q4_dense_pair tests/test_metal_q4_prefill_pair \
	tests/test_metal_indexer_q4 tests/test_metal_q4_attn_out_a_direct tests/test_metal_q4_qb_f16_cache \
	tests/test_metal_q4_qb_token_pair tests/test_metal_q4_hc tests/test_metal_decode_defaults tests/test_metal_f16_compressor \
	speed-bench/metal_q4_dense_pair_bench \
	speed-bench/metal_q4_prefill_pair_bench speed-bench/metal_q4_mm_tail_cull_bench speed-bench/metal_q4_attn_out_a_direct_bench \
	cuda/mmq/test/test_mmq_parity tests/test_cuda_q4_epilogue speed-bench/rocm_q4_prefill_bench \
	speed-bench/cuda_q4_prefill_bench

.PHONY: clean-q4
clean: clean-q4
clean-q4:
	rm -f $(Q4_BUILD_PRODUCTS)
