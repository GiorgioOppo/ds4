# Distributed Views

`DistributedView.swift` contains the worker configuration panel,
`CoordinatorChatView`, and distributed log presentation.

Views bind to `DistributedController` and shared settings. Do not implement
transport or protocol decisions in SwiftUI. New role-specific panels should be
split into focused files when they grow beyond simple presentation.

