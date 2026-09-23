#ifndef DS4_METAL_DEVICE_H
#define DS4_METAL_DEVICE_H

#include <string.h>

/* Share the M5 tuning policy with M6. Keep the generation token bounded so
 * untested future names such as M50 and M60 do not inherit these defaults.
 * Callers must still check runtime and kernel capabilities where required. */
static inline int ds4_metal_device_name_is_m5_or_m6(const char *name) {
    return name && strncmp(name, "Apple M", 7) == 0 &&
           (name[7] == '5' || name[7] == '6') &&
           (name[8] == '\0' || name[8] == ' ');
}

#endif
