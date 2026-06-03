import Foundation

/// Stable marker the compaction pipeline prepends to the synthetic
/// `.system` message that replaces the dropped prefix. Lives in its
/// own minimal file so it stays compilable even when other parts of
/// the compaction code path are broken — call sites in `ChatView`
/// (the banner that detects "already compacted" chats) won't be
/// dragged into a Cannot-find-in-scope error just because the
/// streaming logic next door fails to type-check.
enum ChatCompactionConstants {
    static let marker = "[compacted summary of older turns]"
}
