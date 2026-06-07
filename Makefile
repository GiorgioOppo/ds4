# DS4-gui build orchestrator.
#
# This builds the existing ds4 C/Objective-C engine (UNCHANGED, from the parent
# directory) into a static library `enginelib/libDS4Engine.a`, then drives the
# SwiftPM build of the Swift bridge (DS4Kit) and the GUI smoke test.
#
# We deliberately reuse the exact compiler flags from the parent Makefile so the
# in-process engine behaves byte-for-byte like the upstream ./ds4 binary. The
# project's #1 rule is correctness; the GUI must not perturb the inference path.

# Root of the upstream ds4 project (the parent of this folder).
DS4_ROOT ?= ..

CC ?= cc
NATIVE_CPU_FLAG ?= -mcpu=native
DEBUG_FLAGS ?= -g

# Deployment target must match the SwiftPM platform (see Package.swift) so the
# linker does not warn about mixing object minimum-OS versions.
MACOS_MIN ?= -mmacosx-version-min=14.0

# Same as parent Makefile (Darwin/Metal path), plus the deployment target.
CFLAGS    ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) $(MACOS_MIN) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) $(MACOS_MIN) -Wall -Wextra -fobjc-arc

OBJDIR := enginelib/obj
LIB    := enginelib/libDS4Engine.a

# CORE_OBJS for the Metal backend, from the parent Makefile:
#   ds4.o ds4_distributed.o ds4_ssd.o ds4_metal.o
ENGINE_OBJS := \
	$(OBJDIR)/ds4.o \
	$(OBJDIR)/ds4_distributed.o \
	$(OBJDIR)/ds4_ssd.o \
	$(OBJDIR)/ds4_metal.o \
	$(OBJDIR)/ds4_kvstore.o

.PHONY: all engine swift smoke app xcode xcodeproj embed-kernels clean

all: swift

# Regenerate the embedded kernel sources (Sources/DS4Metal/KernelSources.swift)
# from metal/*.metal. Run after editing any kernel.
embed-kernels:
	sh scripts/embed_kernels.sh

# (Re)generate the standalone DwarfStar.xcodeproj for the pure-Swift engine
# (DS4Core + DS4Metal + DS4Demo) — NO external links — via xcodegen (project.yml).
xcodeproj:
	xcodegen generate

# Open the pure-Swift engine project in Xcode (clickable .xcodeproj, builds with
# no external links). The full SwiftUI GUI (DwarfStar app, which drives the C
# ./ds4 engine) opens instead via `xed .` / Package.swift.
xcode: xcodeproj
	open DwarfStar.xcodeproj

# Assemble a distributable DwarfStar.app (release build + bundled metal/).
app:
	sh packaging/make_app.sh

# Build the Swift package (depends on the prebuilt engine static lib).
swift: $(LIB)
	swift build

# Convenience: build + run the Phase 0 smoke test.
smoke: $(LIB)
	swift run ds4gui-smoke

engine: $(LIB)

$(LIB): $(ENGINE_OBJS)
	@mkdir -p enginelib
	ar rcs $@ $(ENGINE_OBJS)
	@echo "built $@"

$(OBJDIR)/ds4.o: $(DS4_ROOT)/ds4.c $(DS4_ROOT)/ds4.h $(DS4_ROOT)/ds4_ssd.h $(DS4_ROOT)/ds4_distributed.h $(DS4_ROOT)/ds4_gpu.h
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -I$(DS4_ROOT) -c -o $@ $(DS4_ROOT)/ds4.c

$(OBJDIR)/ds4_distributed.o: $(DS4_ROOT)/ds4_distributed.c $(DS4_ROOT)/ds4_distributed.h $(DS4_ROOT)/ds4.h $(DS4_ROOT)/ds4_ssd.h
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -I$(DS4_ROOT) -c -o $@ $(DS4_ROOT)/ds4_distributed.c

$(OBJDIR)/ds4_ssd.o: $(DS4_ROOT)/ds4_ssd.c $(DS4_ROOT)/ds4_ssd.h
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -I$(DS4_ROOT) -c -o $@ $(DS4_ROOT)/ds4_ssd.c

$(OBJDIR)/ds4_metal.o: $(DS4_ROOT)/ds4_metal.m $(DS4_ROOT)/ds4_gpu.h
	@mkdir -p $(OBJDIR)
	$(CC) $(OBJCFLAGS) -I$(DS4_ROOT) -c -o $@ $(DS4_ROOT)/ds4_metal.m

$(OBJDIR)/ds4_kvstore.o: $(DS4_ROOT)/ds4_kvstore.c $(DS4_ROOT)/ds4_kvstore.h $(DS4_ROOT)/ds4.h
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -I$(DS4_ROOT) -c -o $@ $(DS4_ROOT)/ds4_kvstore.c

clean:
	rm -rf $(OBJDIR) $(LIB) .build
