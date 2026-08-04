import SwiftUI

/// Editable context-size field with the native Stepper arrows preserved.
/// Direct keyboard input is clamped to the same limits used by the arrows.
struct ContextSizeControl: View {
    @Binding var value: Int

    private let limits = 1_024...1_000_000
    private let step = 1_024

    private var clampedValue: Binding<Int> {
        Binding(
            get: { min(max(value, limits.lowerBound), limits.upperBound) },
            set: { value = min(max($0, limits.lowerBound), limits.upperBound) }
        )
    }

    var body: some View {
        Stepper(value: clampedValue, in: limits, step: step) {
            HStack(spacing: 6) {
                Text("Context:")
                Spacer()
                TextField("Context tokens", value: clampedValue,
                          format: .number.grouping(.never))
                    .frame(width: 104)
                    .multilineTextAlignment(.trailing)
                    .textFieldStyle(.roundedBorder)
                Text("tokens")
                    .foregroundStyle(.secondary)
            }
        }
        .accessibilityLabel("Context size")
        .accessibilityValue("\(clampedValue.wrappedValue) tokens")
    }
}
