// Ported from GiorgioOppo/ds4 acf5c16bb6a01f8c4f027a0db265ae270d8907ab.
// Upstream license: MIT (see metal/qwen38/LICENSE).
static constant float ds4_metal_mxfp4_values[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};
static inline float ds4_metal_e8m0_to_f32(uchar e) {
    const uint bits = e == 0 ? 0x00400000u : (uint)e << 23;
    return as_type<float>(bits);
}
