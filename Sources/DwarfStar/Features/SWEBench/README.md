# Swift-bench locale

Swift-bench evaluates patches against Swift Package Manager repositories using
the native `DS4SWEBenchLocal` runner. It has no Python, cloud, Docker, pip,
virtual-environment or pytest path.

The task JSON contains `instance_id`, `repo`, `base_commit`, optional `patch`
and `test_patch`, `problem_statement`, plus `FAIL_TO_PASS` and `PASS_TO_PASS`
arrays containing Swift test identifiers. Gold mode turns the optional `patch`
field into a one-line prediction automatically; manual prediction JSONL remains
available for model-generated patches.

The default `Generata dal modello` mode sends the selected task's official
`problem_statement`, repository and base commit to the already-loaded local
DwarfStar model. The response must be a git unified diff; the GUI saves it as
the prediction `model_patch` before the native runner applies it. Gold mode is
kept as a separate reference path and never invokes the model.

Every model attempt exposes the exact rendered prompt, streamed reasoning,
raw response and extracted patch in the SWE-bench view. The same exchange is
persisted as a timestamped JSON transcript in the configured results directory,
so benchmark generations can be inspected after the app exits.

The GUI can run the whole downloaded Lite catalog sequentially. It exposes
completed/total progress and cancellation, and writes a distinct prediction and
transcript for every instance rather than overwriting the previous task.

The runner clones and resets the repository, applies the prediction and test
patches, invokes `/usr/bin/swift test` with a timeout, parses XCTest and Swift
Testing output, and grades both test groups into the JSON report shown by the
GUI.

Direct host execution is intentionally opt-in because package tests and build
plugins execute third-party code on the Mac.

The separate Gold verification mode downloads the 300 official SWE-bench Lite
rows in Swift, converts a selected row's `patch` field to prediction JSONL, and
checks only whether `git apply` succeeds at `base_commit`. It invokes no Python
and no tests, and therefore reports `verification_only = true` with
`resolved = false` even when the patch is applicable. In this mode the process
exit status and the GUI result are based on `patch_applied`; `resolved` is kept
only for report compatibility and must be treated as not applicable.

## External test-result contract

For repositories whose runtime is not embedded in DwarfStar, a manually run
extractor writes both the known-good and patched outputs using this format:

```json
{
  "instance_id": "astropy__astropy-12907",
  "cases": [
    { "id": "case-1", "value": { "answer": 42 }, "error": null }
  ]
}
```

The native runner compares arbitrary JSON values without invoking Python:

```text
ds4-swe-local --expected-results expected.json --actual-results actual.json \
  --report comparison.json
```

Case IDs must be unique. The comparison passes only when every expected case
has the same JSON value and error, with no missing or unexpected cases.
