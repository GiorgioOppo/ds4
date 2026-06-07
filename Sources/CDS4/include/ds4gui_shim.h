#ifndef DS4GUI_SHIM_H
#define DS4GUI_SHIM_H

/* Helpers that live on the C side of the bridge so Swift does not have to
 * reproduce engine-private knowledge.
 *
 * The ds4 Metal backend compiles its kernels at runtime from the per-kernel
 * .metal source files under metal/, resolved relative to the working directory
 * or via a set of DS4_METAL_*_SOURCE environment overrides (see the function
 * ds4_gpu_full_source in ds4_metal.m).
 * A bundled .app has no useful working directory, so before opening the engine
 * the GUI points every override at the metal/ folder it ships in its Resources.
 *
 * Call this BEFORE ds4_engine_open(). dir is the absolute path to a directory
 * that contains the .metal files (e.g. <App>.app/Contents/Resources/metal).
 * Returns 0 on success, or -1 if dir is NULL/empty. */
int ds4gui_set_metal_source_dir(const char *dir);

/* Tier-A per-layer streaming knobs.
 *
 * Together with --ssd-streaming these tell the engine to stop pinning the model
 * in RAM, so macOS can page out layers that have not been touched recently.
 * Combined with the routed-expert SSD cache this approximates what an explicit
 * per-layer streaming loader would do, without modifying the engine's graph.
 *
 *   DS4_METAL_NO_RESIDENCY      skip the GPU residency set on the model mmap.
 *   DS4_METAL_NO_MODEL_WARMUP   skip the startup pass that touches every page.
 *
 * Call before ds4_engine_open(). Pass enabled=0 to leave the env unchanged. */
void ds4gui_set_minimum_ram_mode(int enabled);

#endif
