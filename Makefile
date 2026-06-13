# DS4-gui build orchestrator.
#
# The engine is a pure-Swift reimplementation (DS4Core + DS4Metal): no C engine,
# no prebuilt static lib, no external links. This just drives SwiftPM / xcodegen
# and the packaging script. The project's #1 rule is correctness; the Metal
# kernels (metal/*.metal) are embedded in the binary at build time.

.PHONY: all swift test app xcode xcodeproj embed-kernels clean

all: swift

# Build the Swift package.
swift:
	swift build

# Run the test suite.
test:
	swift test

# Regenerate the embedded kernel sources (Sources/DS4Metal/Runtime/KernelSources.swift)
# from metal/*.metal. Run after editing any kernel.
embed-kernels:
	sh scripts/embed_kernels.sh

# (Re)generate the standalone DwarfStar.xcodeproj (DS4Core + DS4Metal + DS4Engine
# + DwarfStar + DS4Demo) — NO external links — via xcodegen (project.yml).
# Bumps CURRENT_PROJECT_VERSION (the build number) in project.yml in-place first,
# so every regeneration yields a higher build. Display name (DwarfStar) and the
# Utilities app category come from project.yml.
xcodeproj:
	@perl -i -pe 's/(CURRENT_PROJECT_VERSION: ")(\d+)(")/"$$1".($$2+1)."$$3"/e' project.yml
	@echo "build $$(perl -ne 'print $$1 if /CURRENT_PROJECT_VERSION: \"(\d+)\"/' project.yml)"
	xcodegen generate

# Generate + open the project in Xcode (clickable .xcodeproj, no external links).
xcode: xcodeproj
	open DwarfStar.xcodeproj

# Assemble a distributable DwarfStar.app (release build + bundled metal/).
app:
	sh packaging/make_app.sh

clean:
	rm -rf .build
