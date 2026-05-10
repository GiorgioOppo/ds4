#include <metal_stdlib>
using namespace metal;

// Convert a packed bf16 stored as ushort to float.
inline float bf16_to_float(ushort x) {
    uint u = (uint)x << 16;
    return as_type<float>(u);
}

inline ushort float_to_bf16(float x) {
    uint u = as_type<uint>(x);
    // round-to-nearest-even
    uint rounded = u + ((u >> 16) & 1u) + 0x7FFFu;
    return (ushort)(rounded >> 16);
}
