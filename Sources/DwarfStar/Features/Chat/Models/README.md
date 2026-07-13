# Chat Models

`ChatModels.swift` defines presentation models such as `UIMessage` and
`ChatAttachment`. They bridge engine events to stable, identifiable SwiftUI
state and intentionally contain no persistence or generation behavior.

When adding fields, update the persistence mappings when the value must survive
an app restart. Keep engine-owned conversation DTOs in `DS4Core` or
`DS4Engine` instead of duplicating them here.

