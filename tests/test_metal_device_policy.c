#include <stdio.h>

#include "../ds4_metal_device.h"

int main(void) {
    const struct {
        const char *name;
        int expected;
    } cases[] = {
        {NULL, 0},
        {"", 0},
        {"Apple", 0},
        {"Apple M", 0},
        {"Apple M1 Max", 0},
        {"Apple M2", 0},
        {"Apple M3 Ultra", 0},
        {"Apple M4 Pro", 0},
        {"Apple M5", 1},
        {"Apple M5 Pro", 1},
        {"Apple M5 Max", 1},
        {"Apple M5 Ultra", 1},
        {"Apple M6", 1},
        {"Apple M6 Pro", 1},
        {"Apple M6 Max", 1},
        {"Apple M6 Ultra", 1},
        {"Apple M7", 0},
        {"Apple M50", 0},
        {"Apple M60 Max", 0},
        {"Apple M5x", 0},
        {"Apple M6x", 0},
        {"Apple A19", 0},
        {"Apple A20", 0},
        {"AMD M5", 0},
        {"M6", 0},
    };
    for (unsigned i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        const int actual = ds4_metal_device_name_is_m5_or_m6(cases[i].name);
        if (actual != cases[i].expected) {
            fprintf(stderr, "Metal device policy: %s: expected %d, got %d\n",
                    cases[i].name ? cases[i].name : "(null)",
                    cases[i].expected, actual);
            return 1;
        }
    }
    puts("Metal device policy: M5/M6 names and generation boundaries passed");
    return 0;
}
