# DwarfStar application contracts

These tests cover saved-conversation compatibility, preservation of original text
attachments and image payloads, reconstruction of vision history, searchable titles
for attachment-only conversations, and rejection of binary files by the text decoder.
They use only value models and static helpers;
they do not construct `ChatStore`, load a GGUF, open a file picker, or read/write the
user's saved conversations. File fixtures live in temporary files and are removed
after each test.

Run `swift test --filter 'ChatPersistenceTests|ChatAttachmentTests'`. The generated
Xcode project includes the same cases in the `DwarfStarTests` application-hosted
bundle (the SwiftPM run does not launch the GUI).
