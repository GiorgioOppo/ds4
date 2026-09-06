**English** | [Italiano](DEEPSEEK-VISION.it.md)

# DeepSeek Vision Experimental in local chat

The SwiftUI chat integrates **DeepSeek V4 Flash Vision-Exp** and its separate
encoder through native Swift/Metal inference. Images stay on the Mac. Ordinary
Flash 0731 has different language weights; adding the encoder alone does not
make it a vision model.

## Setup

1. Open **Modelli e impostazioni → Scarica…**, then the **Vision** filter.
2. Download Vision IQ2XXS (86.72 GB) or mixed IQ2XXS/Q4_K (97.59 GB), plus the
   separate **encoder immagini** (932.86 MB). Existing files can also be selected.
3. Select the main model and its encoder in **DeepSeek Vision**, or use
   **Usa encoder** on the installed encoder's catalog row.
4. Load the model. Use **Allega** or drop files into the composer. Images can
   accompany text documents or be sent without typed text.

Chat accepts at most four images per message, each at most 20 MiB and 40
megapixels. The encoder accepts at most 16,384 pixels on either axis. PNG and
JPEG are recommended; ImageIO applies EXIF orientation. Each image uses up to
384 context tokens. The [publisher's catalog](https://huggingface.co/antirez/deepseek-v4-gguf/tree/main)
is pinned to `f71f23d552d664e523b422157b2befbf74040380`, including exact byte
counts and SHA-256 digests. Building does not download weights.

## Behavior and limitations

- Original images and attached document text persist with chat JSON under
  Application Support/DwarfStar/chats and are replayed when reopening a chat.
  Legacy sessions remain readable; text attachments discarded by older builds
  cannot be recovered automatically.
- Switching to a text-only model cannot silently discard the images of an
  existing conversation. A message explains the required configuration.
- In-memory KV is reused. Image conversations bypass disk KV because its
  existing token-only keys cannot distinguish different image embeddings.
- Images are supported in local GUI chat and `InferenceService.sendWithImages`.
  HTTP request parsers and distributed transport do not accept image payloads.
- MXFP4 and Vision DSpark files remain download-only. DSpark speculation and
  auto-tuning that reloads the engine are disabled for Vision.
- Visual prefill uses a conservative complete-block path and is less optimized
  than batched text prefill.

## Implementation and validation

See the [encoder README](../Sources/DS4Metal/Backends/DeepSeekV4/Vision/README.md)
for preprocessing, BF16/MPS math, 2D RoPE and the 3×3 aligner. The language
decoder uses the checkpoint's `1e-20` RMS epsilon, synthetic visual embeddings,
`bias_vl` routing and bidirectional raw-image attention with causal compressed keys.

On September 6, 2026, SwiftPM build and 25 standalone checks passed: 10 encoder,
8 decoder/catalog and 7 persistence/attachment checks. GPU checks cover BF16,
MPS, attention, alignment and prefill parity between resident and gathered
synthetic weights (logits within `1e-4`). A 17×9 RGB fixture produced 446,880
Float32 values identical to upstream C.
Corresponding XCTest tests are included, but this host's Command Line Tools
lack XCTest, preventing `swift test`. End-to-end logits parity and answer quality
with complete real encoder/language weights remain unvalidated; large model
weights were not downloaded during implementation.

On a Mac with full Xcode, run the corresponding tests with:

```sh
swift test --filter 'DeepSeekV4Vision|DeepSeekV4ConfigurationTests|ModelDownloaderTests|ChatPersistenceTests|ChatAttachmentTests|PrefillBatchAttnParityTests'
```
