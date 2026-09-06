# Mini Tool Bench

This panel integrates [kyuz0/terminal-bench-mini](https://github.com/kyuz0/terminal-bench-mini)
at revision `4a84b3dad49750a2db9f2e96d23a9bd8dafe7b66`. Upstream now calls the
project **Terminal-Bench-Local**: Core-19 is the default for new comparisons,
while `legacy-mini20` reproduces the historical 20-task suite. Both smoke tiers
contain the single `git-leak-recovery` task.

The panel replaces the former SWE-bench patch-generation UI and removes the
`ds4-swe-local` helper. Its primary action starts the official benchmark in a
Podman Linux VM and imports verified results. Task code runs in Linux containers.

## Start with one button

1. Install Podman Desktop/CLI on the Mac and load a model in DwarfStar.
2. Open **Mini Tool Bench**, enter a stable model family/revision name and choose
   Core-19 or historical Mini-20, full or smoke, and one or two attempts.
3. Press **Avvia test con Podman**. DwarfStar starts its API server on the already
   loaded engine, reuses an available VM or creates `dwarfstar-bench`, and starts
   the VM when needed. A new VM uses four CPUs, 4 GiB RAM and a 60 GiB disk.
4. The first run prepares uv 0.12.10, managed Python 3.12, Harbor 0.20.0,
   podman-compose 1.6.0 and the pinned upstream checkout. Later runs reuse them.
   The button runs doctor and starts an inert container to check execution and
   endpoint DNS before starting the selected tests. It automatically imports
   the exported summary, task details and transcripts when finished.

Podman 6 removed `slirp4netns`. The managed integration substitutes an external
network overlay: `bridge` for rootful Podman and `pasta:--map-gw` for rootless
Podman. The pinned runner, task files and verifiers remain unchanged. A failed
container preflight stops before any benchmark attempt; its temporary Compose
project is removed, including on cancellation.

The main panel shows phases and live logs. Stop sends a durable cancellation
request into Linux, including during setup; it targets only this run's process
group. The VM remains available, and an already running API server remains
available. If cancellation occurs while this run is starting its server, that
startup is stopped. Test/container images are downloaded when the benchmark
requires them; a full run can take many hours.

The managed runner reaches the Mac through `host.containers.internal` using the
actual API server port, including when the server binds to `127.0.0.1`. Manual
endpoint, Python and checkout fields apply only to the optional manual commands.
The benchmark never loads a second copy of the model.

VM dependencies, jobs and container bind mounts live in Linux under
`~/.local/share/dwarfstar-bench`, avoiding SELinux operations on Mac-shared
files. Exported reports and bounded application logs live in the Mac application
support directory at `DwarfStar/MiniToolBench/runs`. Keys travel over SSH stdin;
the app redacts them from its logs and does not persist the request. Upstream
Harbor can retain connection settings in its private VM job configuration;
those configuration files are not exported to the Mac.

Use the optimized **Local** Xcode Run configuration or the `make app` bundle for
Podman management. Distribution archives retain the sandboxed Release build,
which supports manual commands and report import. Creating/managing an external
Podman VM requires the local build's process and filesystem access.

## Optional manual runner

1. Load the intended model and start **Server API**. Keep the app and its server
   running for the duration of the benchmark. The server advertises the loaded
   backend's actual context capacity in `/v1/models`; the output-token limit is
   a separate setting.
2. Prepare a Linux host or VM with Python 3.11+, Docker Engine with Compose v2
   (or supported Podman setup), and `uv` or Harbor 0.20.0. The runner can resolve
   the pinned Harbor package on first use. Container images require space and
   network access. The model stays on the Mac and is reached over HTTP.
3. Copy the setup commands from **Mini Tool Bench** and execute them on Linux.
   They clone a separate checkout under the Linux user's home directory and
   pin the integration revision. Existing checkouts are not reset or replaced.
4. Configure a URL reachable **from Linux**, ending in `/v1`. `127.0.0.1`
   refers to Linux, not the Mac, unless a tunnel is configured. For LAN access,
   select a suitable bind address in Server API and enter the Mac's LAN address
   in the benchmark endpoint. `0.0.0.0` is a bind address, not an endpoint.
5. Enter the actual hardware platform and a stable model family/revision name.
   Keep quantization and inference profile in their separate fields. The exact
   served model ID and context capacity are discovered automatically; override
   only when needed. Defaults `DwarfStar` / `metal` describe the native Mac
   engine and can be changed for another server.
6. Choose full or smoke and one or two attempts. Copy or export the generated
   command and execute it in a persistent Linux terminal, preferably `tmux`.
   Under advanced options, select the Python executable on the Linux runner
   (for example `python3.12` or an absolute virtual-environment path). This
   setting is an executable path, not a shell command. Existing preferences
   keep their values and default this new field to `python3`.
   The generated command checks the pinned checkout, Python 3.11+ and the Linux
   platform before running `doctor` and then `run`. Doctor performs
   no inference but may download Harbor. Runs are serial; a second attempt is
   conditional on failure. The official runner retains its own agent timeout
   and context-management behavior.

For authenticated endpoints set `TBENCH_API_KEY` in the Linux session. The GUI
does not persist keys or embed them in exported scripts. Runtime prerequisites
are checked by upstream; preparing a command is not represented as a completed
benchmark or a successful doctor check. Stop a run with Ctrl-C in its terminal.
Use upstream `resume jobs/<job-name>` to continue an interrupted job.

### A Podman VM on a Mac

`podman machine ssh <machine-name>` opens a Linux shell in an existing Podman
machine. Check `python3 --version`, `docker compose version` and the Harbor/uv
installation **inside that shell**: tools installed on macOS are not necessarily
installed in the VM. The default Apple command-line-tools Python 3.9.6 is too
old for Harbor and runs on the wrong host for this workflow.

When a checkout is shared into the VM, set the runner directory to its mounted
absolute path, not `~/terminal-bench-mini` (the VM has a different home). Podman
can expose the Mac as `host.containers.internal`; verify resolution and server
reachability in the VM before using `http://host.containers.internal:8080/v1`.
Keep the model's API server running and configured to accept those connections.

## Results

Copy the exported result directory from Linux to the Mac and import that
directory containing `summary.json`. Keep `run-meta.json`, `results-*.json` and
`transcript-*.json` together. The app validates summary counts, task identities,
suite manifest and evaluation-profile identity, and confines references to the
selected directory. Missing task files are reported explicitly. Transcript
links open the original normalized JSON without executing it.

`total_tasks` is the denominator of the exported aggregate, not necessarily 19
or 20 and not proof that a job finished. Failed attempts with exceptions remain
in the denominator. The attempt budget comes from matching `run-meta.json`;
the app never labels an imported summary pass@2 merely because the current UI
configuration selects two attempts. Different suites and profiles stay separate.

Attempts with exceptions include an **Errore** details button and can be shown
with the **Con errori** filter. The app distinguishes container setup failures
from verifier results. When all attempts have a recognized startup failure and
the task and summary counters record zero tokens, a banner explains that the
run did not measure model quality. Zero usage alone does not establish that
inference never started. A suite that finishes each task in a few seconds can therefore
indicate an environment failure rather than fast inference.

The upstream context recommendation is separate from container startup. With
Harbor's 8,000-token summarization reserve, a real 8,192-token context can require
frequent summaries after the initial turn. Only advertise the loaded backend's
actual capacity. The API returns `context_length_exceeded` for overflow so the
agent can recognize it and attempt its normal context recovery.

## Validation

`MiniToolBenchTests` covers command construction, CLI option preservation,
official result contracts, missing files and unsafe/mixed artifact references.
`LocalServerModelTests` covers real context advertisement separately from the
completion limit. `MiniToolBenchRunnerTests` checks VM selection and the SSH
protocol. `python3 -m unittest discover -s Tests/BootstrapTests -v` checks setup,
doctor and container preflight before run, Podman network compatibility, export
and cancellation with fake subprocesses. These tests
do not run model inference or task containers.
An end-to-end score requires the Linux runtime and a loaded model.
