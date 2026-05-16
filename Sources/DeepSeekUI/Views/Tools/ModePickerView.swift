import SwiftUI
import DeepSeekTools

/// Segmented control pinned next to the model/agent pickers in the
/// chat toolbar. Lets the user flip the current chat between Plan
/// and Build without going through Settings.
///
/// The selection is owned by the host (typically `Conversation` or
/// the active `AgentConfig`); this view is intentionally stateless
/// so undo/redo can be threaded through whoever owns the binding.
struct ModePickerView: View {
    @Binding var mode: AgentMode

    var body: some View {
        Picker("Mode", selection: $mode) {
            ForEach(AgentMode.allCases, id: \.self) { m in
                Label(m.displayName, systemImage: icon(for: m))
                    .tag(m)
            }
        }
        .pickerStyle(.segmented)
        .help(mode.summary)
    }

    private func icon(for mode: AgentMode) -> String {
        switch mode {
        case .build: return "hammer"
        case .plan:  return "map"
        }
    }
}
