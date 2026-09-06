# DwarfStar application contracts

These tests cover saved-conversation compatibility, preservation of original text
attachments and image payloads, reconstruction of vision history, searchable titles
for attachment-only conversations, and rejection of binary files by the text decoder.
They use only value models and static helpers;
they do not construct `ChatStore`, load a GGUF, open a file picker, or read/write the
user's saved conversations. File fixtures live in temporary files and are removed
after each test.

`MiniToolBenchTests` validates the pinned official runner commands and normalized
result import, including suite/profile identity and artifact references.
`LocalServerModelTests` uses a mock backend to verify that model discovery exposes
the loaded context capacity separately from the per-response token default.
Neither suite starts inference, containers or an external benchmark.
`MiniToolBenchRunnerTests` validates VM selection and the quoted SSH bootstrap
protocol. The bootstrap's subprocess/cancellation tests are independent Python
tests: `python3 -m unittest discover -s Tests/BootstrapTests -v`.

Run `swift test --filter DwarfStarTests`. The generated
Xcode project includes the same cases in the `DwarfStarTests` application-hosted
bundle (the SwiftPM run does not launch the GUI).
