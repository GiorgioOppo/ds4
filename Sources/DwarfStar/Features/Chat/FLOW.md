# Chat Execution Flow

This document describes the ownership boundaries of the most stateful app
feature.

## New message

1. `ChatView` sends composer text and selected attachments to `ChatStore`.
2. `ChatStore+Generation` builds the request and invokes the shared
   `InferenceService`.
3. Engine events are converted to `UIMessage` state and rendered immediately.
4. Tool calls are validated and dispatched by `ChatStore+ToolLoop`; sub-agent
   work is coordinated by `ChatStore+Agents`.
5. The completed transcript is mapped to `StoredMessage` and saved by
   `ChatSessionStore`.

The stream and every continuation carry the turn's `conversationEpoch`. Stop,
New Chat, or a session switch invalidates it before changing the transcript, so
a late token/tool result cannot execute another tool, write through a reused
message index, or clear the status of newer work. Mixed tool batches also retain
one ordered output slot per call while waiting for any manual results.

## Reopened session

Opening a chat restores UI history, not the model's in-memory KV state. On the
next send, `sendWithHistory` renders the visible history again. A compatible
disk-KV prefix may accelerate that rebuild; following turns return to
incremental generation.

## Shared-engine invariant

Chat, local Benchmark, and Server share one `InferenceService`. Chat generation
and HTTP requests are serialized, while Benchmark may run only when Chat is
idle because it rewrites KV state. Changes must preserve this single-engine
ownership rule to avoid duplicate multi-gigabyte model allocations.
