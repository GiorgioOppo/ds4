import Foundation

// Helper di conversione condivisi fra i quantizzatori INT2/INT4/INT8
// e fra `CalibratedQuant`. Originariamente duplicati `private` in ogni
// file `Int{2,4,8}Quant.swift`; estratti qui per evitare collisioni
// quando uno dei file deve esportare la stessa funzione a un caller
// esterno (vedi `CalibratedQuant.swift`).

/// IEEE-754 F32 → F16 con round-to-nearest-even. Equivalente alla
/// vecchia `floatToF16Local` privata di `Int8Quant.swift`. Usato
/// dai quantizzatori per scrivere le per-row, per-group `scaleF16`
/// nel safetensors di output.
@inline(__always)
internal func floatToF16Local(_ f: Float) -> UInt16 {
    let bits = f.bitPattern
    let sign = (bits >> 31) & 1
    let exp = (bits >> 23) & 0xFF
    let mant = bits & 0x7FFFFF
    if exp == 0 { return UInt16(truncatingIfNeeded: sign << 15) }
    if exp == 0xFF {
        let m: UInt32 = mant != 0 ? 0x200 : 0
        return UInt16(truncatingIfNeeded: (sign << 15) | (0x1F << 10) | m)
    }
    let unbiased = Int(exp) - 127
    if unbiased > 15 { return UInt16(truncatingIfNeeded: (sign << 15) | (0x1F << 10)) }
    if unbiased < -14 {
        let shift = -14 - unbiased + 13
        if shift > 24 { return UInt16(truncatingIfNeeded: sign << 15) }
        let full = (mant | 0x800000) >> (shift - 1)
        let halfMant = (full + 1) >> 1
        return UInt16(truncatingIfNeeded: (sign << 15) | halfMant)
    }
    let halfExp = UInt32(unbiased + 15)
    let halfMant = (mant + 0x1000) >> 13
    if halfMant >= 0x400 {
        if halfExp + 1 >= 0x1F {
            return UInt16(truncatingIfNeeded: (sign << 15) | (0x1F << 10))
        }
        return UInt16(truncatingIfNeeded: (sign << 15) | ((halfExp + 1) << 10))
    }
    return UInt16(truncatingIfNeeded: (sign << 15) | (halfExp << 10) | halfMant)
}

/// BF16 → F32 zero-extending (BF16 è semplicemente i 16 bit alti
/// di un F32). Costruisce il pattern bit-per-bit, senza arrotondamenti.
@inline(__always)
internal func bf16ToFloat(_ b: UInt16) -> Float {
    return Float(bitPattern: UInt32(b) << 16)
}
