import Foundation

/// Sent to the Linux VM with `python3 -c`; its request arrives separately on
/// stdin. No credentials or mutable runner installation are embedded here.
enum MiniToolBenchBootstrap {
    static let script = #"""
from __future__ import annotations

import collections
import fcntl
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import time
from urllib.parse import urlsplit
import uuid

REVISION = "4a84b3dad49750a2db9f2e96d23a9bd8dafe7b66"
REPOSITORY = "https://github.com/kyuz0/terminal-bench-mini.git"
UV_VERSION = "0.12.10"
HARBOR_VERSION = "0.20.0"
COMPOSE_VERSION = "1.6.0"
OWNER = "DwarfStar.MiniToolBench.VM.v1"
MAX_REQUEST = 128 * 1024
_secret = ""
_log = None
_control = None
_child = None
_cancelled = False
_cancel_time = None
_cancel_marker = None


class BootstrapError(Exception):
    pass


class Cancelled(Exception):
    pass


def redact(value):
    if isinstance(value, str):
        return value.replace(_secret, "[redacted]") if _secret else value
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def emit(event, **fields):
    line = json.dumps(redact({"event": event, **fields}), ensure_ascii=False, allow_nan=False)
    if _log:
        _log.write(line + "\n")
        _log.flush()
    try:
        print(line, flush=True)
    except BrokenPipeError:
        request_cancel(signal.SIGHUP, None)
        raise Cancelled()


def process_start(pid):
    try:
        # comm may contain spaces and parentheses; fields after its final ')'
        # start at field 3, and starttime is field 22.
        text = Path(f"/proc/{int(pid)}/stat").read_text()
        return text.rsplit(")", 1)[1].split()[19]
    except (OSError, ValueError, IndexError):
        return None


def write_json(path, payload):
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle)
    os.replace(temporary, path)


def update_control(status="active"):
    if _control is None:
        return
    write_json(_control, {
        "owner": OWNER, "run_id": _control.parent.name, "status": status,
        "pid": os.getpid(), "start": process_start(os.getpid()),
        "child_pid": _child.pid if _child else None,
        "child_start": process_start(_child.pid) if _child else None,
    })


def signal_child(sig):
    if _child is not None:
        try:
            os.killpg(_child.pid, sig)
        except ProcessLookupError:
            pass


def request_cancel(signum, frame):
    global _cancelled, _cancel_time
    if not _cancelled:
        _cancel_time = time.monotonic()
    _cancelled = True
    signal_child(signal.SIGINT)


def check_cancelled():
    notice_cancellation()
    if _cancelled:
        raise Cancelled()


def strip_ansi(text):
    """Remove CSI display controls emitted by the Compose provider in pipes."""
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def notice_cancellation():
    if not _cancelled and _cancel_marker is not None and _cancel_marker.exists():
        request_cancel(signal.SIGINT, None)


def run_process(arguments, *, cwd=None, env=None, timeout=600, check=True, cancellable=True):
    global _child
    if cancellable:
        check_cancelled()
    tail = collections.deque(maxlen=2000)
    pending = b""
    started = time.monotonic()
    timed_out = False
    escalation = 0
    process = subprocess.Popen(
        [str(value) for value in arguments], cwd=cwd, env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, start_new_session=True,
    )
    _child = process
    update_control()
    if cancellable:
        notice_cancellation()
    if cancellable and _cancelled:
        signal_child(signal.SIGINT)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        while selector.get_map() or process.poll() is None:
            if cancellable:
                notice_cancellation()
            now = time.monotonic()
            if timeout is not None and now - started > timeout and not timed_out:
                timed_out = True
                signal_child(signal.SIGTERM)
                emit("log", message="Tempo massimo della fase di preparazione raggiunto.")
            if timed_out and now - started > timeout + 10:
                signal_child(signal.SIGKILL)
            if cancellable and _cancelled and _cancel_time is not None:
                elapsed = now - _cancel_time
                if elapsed >= 20 and escalation < 1:
                    signal_child(signal.SIGTERM)
                    escalation = 1
                if elapsed >= 25 and escalation < 2:
                    signal_child(signal.SIGKILL)
                    escalation = 2
            for key, _ in selector.select(timeout=0.2):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                pending += chunk
                while b"\n" in pending or len(pending) > 32768:
                    if b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                    else:
                        line, pending = pending[:32768], pending[32768:]
                    text = strip_ansi(line.decode("utf-8", errors="replace").rstrip("\r"))
                    tail.append(text)
                    emit("log", message=text)
        if pending:
            text = strip_ansi(pending.decode("utf-8", errors="replace"))
            tail.append(text)
            emit("log", message=text)
        code = process.wait()
    finally:
        selector.close()
        process.stdout.close()
        if process.poll() is None:
            signal_child(signal.SIGINT)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                signal_child(signal.SIGKILL)
                process.wait()
        _child = None
        update_control()
    if cancellable:
        check_cancelled()
    output = "\n".join(tail)
    if timed_out:
        raise BootstrapError("Preparazione interrotta per timeout; puoi riprovare.")
    if code != 0 and check:
        raise BootstrapError(f"{Path(str(arguments[0])).name} è terminato con codice {code}: {output[-2000:]}")
    return code, output


def field(configuration, snake, camel=None, *, required=False):
    value = configuration.get(snake, configuration.get(camel, "") if camel else "")
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise BootstrapError(f"Campo non valido: {snake}")
    value = str(value).strip()
    if len(value) > 8192 or any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise BootstrapError(f"Campo non valido: {snake}")
    if required and not value:
        raise BootstrapError(f"Campo obbligatorio: {snake}")
    return value


def connection_arguments(configuration):
    suite = field(configuration, "suite", required=True)
    tier = field(configuration, "tier", required=True)
    if suite not in {"core19", "legacy-mini20"} or tier not in {"full", "smoke"}:
        raise BootstrapError("Suite o sottoinsieme non supportato.")
    endpoint = field(configuration, "endpoint", required=True)
    try:
        parsed = urlsplit(endpoint)
        valid = (parsed.scheme in {"http", "https"} and parsed.hostname
                 and parsed.hostname not in {"0.0.0.0", "::"}
                 and parsed.username is None and parsed.password is None
                 and not parsed.query and not parsed.fragment
                 and parsed.path.strip("/") == "v1")
        _ = parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise BootstrapError("Endpoint non valido: usa un URL HTTP(S) raggiungibile dalla VM, terminante in /v1.")
    arguments = [f"--suite={suite}", f"--tier={tier}", f"--endpoint={endpoint}"]
    model = field(configuration, "model_id", "modelID")
    context = field(configuration, "context_length", "contextLength")
    if model:
        arguments.append(f"--model={model}")
    if context:
        if not context.isdecimal() or int(context) < 1:
            raise BootstrapError("La capacità del contesto deve essere un intero positivo.")
        arguments.append(f"--context-length={context}")
    return arguments


def run_arguments(configuration, run_id, results):
    arguments = connection_arguments(configuration)
    for snake, camel, option in [
        ("platform", None, "platform"), ("model_name", "modelName", "model-name"),
        ("engine", None, "engine"), ("backend", None, "backend"),
    ]:
        arguments.append(f"--{option}={field(configuration, snake, camel, required=True)}")
    attempts = configuration.get("attempts", 2)
    if type(attempts) is not int or attempts not in {1, 2}:
        raise BootstrapError("Sono supportati uno o due tentativi per task.")
    arguments.extend([f"--attempts={attempts}", "--concurrency=1", f"--job-name={run_id}", f"--results-dir={results}"])
    for snake, camel, option in [("quant", None, "quant"), ("inference_profile", "inferenceProfile", "inference-profile")]:
        value = field(configuration, snake, camel)
        if value:
            arguments.append(f"--{option}={value}")
    return arguments


def owned_workspace(request):
    supplied = request.get("workspace_root")
    root = Path(supplied).expanduser() if supplied else Path.home() / ".local/share/dwarfstar-bench"
    if not root.is_absolute():
        raise BootstrapError("La directory della VM deve essere assoluta.")
    root = root.resolve()
    if str(root).startswith(("/Users/", "/mnt/")) or root in {Path("/"), Path.home().resolve()}:
        raise BootstrapError("L’ambiente deve stare nel filesystem Linux della VM, non nella cartella condivisa del Mac.")
    marker = root / ".dwarfstar-workspace.json"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    # A simultaneous early cancel can initialize this directory too. Lock the
    # directory itself so no lock file is written into an unowned installation.
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if marker.is_file():
            if json.loads(marker.read_text()).get("owner") != OWNER:
                raise BootstrapError("La directory della VM appartiene a un’altra installazione.")
        elif any(root.iterdir()):
            raise BootstrapError("La directory della VM contiene dati non gestiti da DwarfStar; non verrà modificata.")
        else:
            write_json(marker, {"owner": OWNER})
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    return root


def install_runtime_compatibility(root, repository, python, podman, env):
    _, description = run_process([podman, "--version"], env=env, timeout=30)
    match = re.search(r"\bpodman version (\d+)\.", description, re.IGNORECASE)
    if not match:
        raise BootstrapError("Impossibile determinare la versione di Podman per configurare la rete.")
    major = int(match.group(1))
    # Podman 6 removed slirp4netns and only supports pasta in rootless mode.
    # Rootful bridge retains the VM's gvproxy route to the Mac through
    # host.containers.internal; rootless pasta explicitly allows host access.
    network = "slirp4netns:allow_host_loopback=true"
    if major >= 6:
        _, rootless = run_process([podman, "info", "--format={{.Host.Security.Rootless}}"], env=env, timeout=30)
        rootless = rootless.strip()
        if rootless not in {"true", "false"}:
            raise BootstrapError("Impossibile determinare se Podman è rootless; configurazione della rete interrotta.")
        network = "pasta:--map-gw" if rootless == "true" else "bridge"
    compatibility = root / "compat"
    compatibility.mkdir(exist_ok=True, mode=0o700)
    overlay = compatibility / "terminal-network-loopback.json"
    write_json(overlay, {"services": {"main": {"network_mode": network}}})
    original = repository / "compat/podman/terminal-network-podman-loopback.json"
    adapter = compatibility / "docker-runtime.py"
    # The official shim still handles project directories, image names, bind
    # labels and compose cp. Replace only its exact known network-overlay path,
    # after it delegates to TBENCH_REAL_DOCKER; never edit the pinned checkout.
    adapter.write_text("import os, sys\nfrom pathlib import Path\n"
        + "original = " + repr(str(original.resolve())) + "\n"
        + "replacement = " + repr(str(overlay)) + "\n"
        + "runtime = " + repr(str(podman)) + "\n"
        + "arguments = sys.argv[1:]\n"
        + "if arguments and arguments[0] == 'compose':\n"
        + "    for index, value in enumerate(arguments):\n"
        + "        if index and arguments[index - 1] in {'-f', '--file'} and str(Path(value).resolve()) == original:\n"
        + "            arguments[index] = replacement\n"
        + "        elif value.startswith(('-f=', '--file=')) and str(Path(value.split('=', 1)[1]).resolve()) == original:\n"
        + "            arguments[index] = value.split('=', 1)[0] + '=' + replacement\n"
        + "os.execv(runtime, [runtime, *arguments])\n")
    adapter.chmod(0o600)
    docker = root / "bin/docker"
    docker.write_text("#!/bin/sh\nexec " + shlex.quote(str(python)) + " " + shlex.quote(str(adapter)) + " \"$@\"\n")
    docker.chmod(0o700)
    emit("log", message=f"Compatibilità Podman {major}: rete {network}; checkout ufficiale invariato.")


def container_preflight(root, repository, env, run_directory, configuration):
    emit("phase", message="Verifica reale di avvio ed esecuzione del container, senza richieste al modello…")
    project = "dwarfstar-preflight-" + uuid.uuid4().hex
    directory = run_directory / "preflight"
    directory.mkdir(exist_ok=True, mode=0o700)
    compose = directory / "compose.json"
    # Reuse the official smoke task's cached image and instruction set. An
    # inert shell tests the same image/runtime compatibility without executing
    # its task, agent or verifier and without mounting host directories.
    image = "docker.io/alexgshaw/git-leak-recovery:20251031"
    write_json(compose, {"services": {"main": {
        "image": image, "entrypoint": ["/bin/sh", "-c", 'trap "exit 0" TERM INT; sleep 120 & wait'],
        "network_mode": "none", "labels": {"org.dwarfstar.preflight": project},
    }}})
    probe_env = {**env,
        "TBENCH_REAL_DOCKER": str(root / "bin/docker"),
        "TBENCH_CONTAINER_RUNTIME": "podman", "TBENCH_CONTAINER_NETWORK_MODE": "podman-loopback",
        "TBENCH_JOBS_DIR": str(repository / "jobs"),
        "PATH": str(repository / "compat/podman") + os.pathsep + env.get("PATH", ""),
    }
    command = [repository / "compat/podman/docker", "compose", "--project-directory", directory,
               "-f", compose, "--project-name", project]
    cleanup_code = 0
    try:
        run_process([*command, "up", "--detach"], cwd=directory, env=probe_env, timeout=900)
        _, output = run_process([*command, "exec", "-T", "main", "/bin/sh", "-c",
                                 "printf 'DWARFSTAR_PREFLIGHT_OK\\n'"],
                                cwd=directory, env=probe_env, timeout=60)
        # podman-compose may prefix even non-TTY exec output with an ANSI reset.
        # Remove CSI display controls, then require a whole acknowledgement
        # line so a command echo or an error quoting the token cannot pass.
        plain_output = strip_ansi(output)
        if "DWARFSTAR_PREFLIGHT_OK" not in plain_output.splitlines():
            raise BootstrapError("Il container non ha confermato l’esecuzione del comando di verifica.")
        endpoint = urlsplit(field(configuration, "endpoint", required=True))
        run_process([*command, "exec", "-T", "main", "/usr/bin/getent", "--", "ahosts", endpoint.hostname],
                    cwd=directory, env=probe_env, timeout=30)
        emit("log", message="DNS del container verificato; disponibilità e autenticazione dell’endpoint verificate dal doctor nella VM.")
    except BootstrapError as error:
        raise BootstrapError("Il controllo del container è fallito prima dei task; nessun tentativo è stato consumato. " + str(error)) from error
    finally:
        # A cancellation must still tear down this one owned Compose project.
        cleanup_code, _ = run_process([*command, "down"], cwd=directory, env=probe_env,
                                      timeout=45, check=False, cancellable=False)
        if cleanup_code:
            emit("log", message=f"Pulizia del container di verifica non riuscita: progetto {project}.")
    check_cancelled()
    if cleanup_code:
        raise BootstrapError("Il controllo è terminato ma la pulizia del suo container non è riuscita; prova interrotta.")
    emit("log", message="Container avviato ed eseguito correttamente; progetto di verifica rimosso.")


def prepare(root):
    check_cancelled()
    podman = shutil.which("podman")
    if not podman:
        raise BootstrapError("Podman non è disponibile nella VM.")
    binaries = root / "bin"
    binaries.mkdir(exist_ok=True, mode=0o700)
    env = os.environ.copy()
    env.update({
        "UV_CACHE_DIR": str(root / "cache"), "UV_PYTHON_INSTALL_DIR": str(root / "python"),
        "UV_HTTP_TIMEOUT": "120", "PYTHONUNBUFFERED": "1", "GIT_TERMINAL_PROMPT": "0",
    })
    uv = binaries / "uv"
    emit("phase", message="Preparazione di uv e Python 3.12 nella VM…")
    uv_ok = False
    if uv.is_file():
        code, version = run_process([uv, "--version"], env=env, timeout=30, check=False)
        uv_ok = code == 0 and version.split()[:2] == ["uv", UV_VERSION]
    if not uv_ok:
        installer = root / f"install-uv-{UV_VERSION}.sh"
        run_process(["curl", "-LsSf", "--connect-timeout", "20", "--max-time", "180", "--retry", "2",
                     f"https://astral.sh/uv/{UV_VERSION}/install.sh", "-o", installer], env=env, timeout=600)
        install_env = {**env, "UV_UNMANAGED_INSTALL": str(binaries)}
        run_process(["sh", installer], env=install_env, timeout=600)
        _, version = run_process([uv, "--version"], env=env, timeout=30)
        if version.split()[:2] != ["uv", UV_VERSION]:
            raise BootstrapError("La versione di uv installata non corrisponde alla versione richiesta.")
    venv = root / "venv-py312-harbor020"
    python = venv / "bin/python"
    if not python.is_file():
        if venv.exists():
            # Preserve interrupted installation data and create a fresh private
            # environment. Never remove a supplied directory or reset sources.
            os.rename(venv, root / ("incomplete-venv-" + uuid.uuid4().hex))
        run_process([uv, "--no-config", "venv", "--managed-python", "--python", "3.12", venv], env=env, timeout=900)
    probe = ("import importlib.metadata as m,sys; "
             "assert sys.version_info[:2] == (3,12); "
             f"assert m.version('harbor') == '{HARBOR_VERSION}'; "
             f"assert m.version('podman-compose') == '{COMPOSE_VERSION}'")
    code, _ = run_process([python, "-c", probe], env=env, timeout=30, check=False)
    if code:
        emit("phase", message="Installazione di Harbor 0.20.0 e podman-compose 1.6.0…")
        run_process([uv, "--no-config", "pip", "install", "--python", python,
                     f"harbor=={HARBOR_VERSION}", f"podman-compose=={COMPOSE_VERSION}"], env=env, timeout=1200)
        run_process([python, "-c", probe], env=env, timeout=30)
    docker = binaries / "docker"
    docker.write_text("#!/bin/sh\nexec " + shlex.quote(podman) + " \"$@\"\n")
    docker.chmod(0o700)
    env["PATH"] = f"{venv / 'bin'}:{binaries}:{env.get('PATH', '')}"
    env["PODMAN_COMPOSE_PROVIDER"] = str(venv / "bin/podman-compose")
    env["TBENCH_API_KEY"] = _secret or "local"
    run_process(["docker", "compose", "version"], env=env, timeout=30)
    _, version = run_process([venv / "bin/harbor", "--version"], env=env, timeout=90)
    if version.strip() != HARBOR_VERSION:
        raise BootstrapError("Harbor non corrisponde alla versione 0.20.0 richiesta dal benchmark.")
    emit("phase", message="Preparazione del runner ufficiale alla revisione fissata…")
    repository = root / "runner"
    if not repository.exists():
        staging = root / ("runner-download-" + uuid.uuid4().hex)
        run_process(["git", "init", staging], env=env, timeout=30)
        run_process(["git", "-C", staging, "remote", "add", "origin", REPOSITORY], env=env, timeout=30)
        run_process(["git", "-C", staging, "fetch", "--depth=1", "origin", REVISION], env=env, timeout=900)
        run_process(["git", "-C", staging, "checkout", "--detach", REVISION], env=env, timeout=300)
        os.rename(staging, repository)
    _, revision = run_process(["git", "-C", repository, "rev-parse", "HEAD"], env=env, timeout=30)
    _, remote = run_process(["git", "-C", repository, "remote", "get-url", "origin"], env=env, timeout=30)
    _, dirty = run_process(["git", "-C", repository, "status", "--porcelain", "--untracked-files=no"], env=env, timeout=30)
    if revision.strip() != REVISION or remote.strip().removesuffix(".git") != REPOSITORY.removesuffix(".git") or dirty.strip():
        raise BootstrapError("Il checkout del runner è stato modificato o ha una revisione diversa; DwarfStar non lo resetta.")
    install_runtime_compatibility(root, repository, python, podman, env)
    emit("prepared", workspace=str(root), repository=str(repository), revision=REVISION)
    return python, repository, env


def inside(path, parent):
    return path.resolve().is_relative_to(parent.resolve())


def export_artifacts(run_directory, export_root, run_id):
    destination = export_root / run_id
    destination.mkdir(mode=0o700)
    results = run_directory / "results"
    summaries = [path for path in results.rglob("summary.json")
                 if path.is_file() and not path.is_symlink() and inside(path, results)] if results.exists() else []
    if len(summaries) > 1:
        raise BootstrapError("Il runner ha prodotto più profili: i risultati originali rimangono nella VM.")
    if _log:
        _log.flush()
    log = run_directory / "bootstrap.jsonl"
    if log.is_file():
        shutil.copyfile(log, destination / "bootstrap.jsonl")
    if not summaries:
        return destination, None
    source = summaries[0].parent
    summary = json.loads(summaries[0].read_text())
    for path in source.iterdir():
        if (path.is_file() and not path.is_symlink() and path.suffix.lower() == ".json"
                and (path.name in {"summary.json", "run-meta.json"}
                     or path.name.startswith(("results-", "transcript-")))):
            shutil.copyfile(path, destination / path.name)
    return destination, summary


def cancel(root, run_id):
    directory = root / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Persist the request even if the original SSH command has not started yet.
    # The run checks it before preparing anything and while each child runs.
    write_json(directory / "cancel-requested.json", {"owner": OWNER, "run_id": run_id})
    state_file = directory / "control.json"
    if not state_file.is_file():
        emit("completed", action="cancel", status="requested", run_id=run_id)
        return 0
    state = json.loads(state_file.read_text())
    if state.get("owner") != OWNER or state.get("run_id") != run_id:
        raise BootstrapError("Identità del processo da interrompere non valida.")
    if state.get("status") != "active":
        emit("completed", action="cancel", status="not_running", run_id=run_id)
        return 0
    pid = state.get("pid")
    try:
        if type(pid) is int and pid > 1 and state.get("start") and process_start(pid) == state["start"]:
            os.kill(pid, signal.SIGINT)
            emit("completed", action="cancel", status="requested", run_id=run_id)
            return 0
        child = state.get("child_pid")
        if type(child) is int and child > 1 and state.get("child_start") and process_start(child) == state["child_start"]:
            if os.getpgid(child) == child:
                os.killpg(child, signal.SIGINT)
                emit("completed", action="cancel", status="requested", run_id=run_id)
                return 0
    except ProcessLookupError:
        pass
    emit("completed", action="cancel", status="not_running", run_id=run_id)
    return 0


def main():
    global _secret, _log, _control, _cancel_marker
    if sys.platform != "linux" or sys.version_info < (3, 11):
        raise BootstrapError("Il bootstrap deve essere eseguito dentro la VM Linux con Python 3.11 o successivo.")
    raw = sys.stdin.buffer.read(MAX_REQUEST + 1)
    if len(raw) > MAX_REQUEST:
        raise BootstrapError("Configurazione troppo grande.")
    request = json.loads(raw)
    if not isinstance(request, dict):
        raise BootstrapError("Configurazione non valida.")
    action = request.get("action", "run")
    run_id = request.get("run_id", "")
    if action not in {"prepare", "doctor", "run", "cancel"} or not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise BootstrapError("Azione o identificativo della prova non valido.")
    _secret = request.get("api_key") or ""
    if not isinstance(_secret, str) or any(ord(character) < 32 for character in _secret):
        _secret = ""
        raise BootstrapError("API key non valida.")
    root = owned_workspace(request)
    if action == "cancel":
        return cancel(root, run_id)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, request_cancel)
    run_directory = root / "runs" / run_id
    run_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    _cancel_marker = run_directory / "cancel-requested.json"
    _control = run_directory / "control.json"
    lock = (root / "execution.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise BootstrapError("La VM sta già preparando o eseguendo una prova DwarfStar.")
    update_control()
    _log = (run_directory / "bootstrap.jsonl").open("a", buffering=1)
    os.chmod(run_directory / "bootstrap.jsonl", 0o600)
    destination = None
    export_root = None
    try:
        check_cancelled()
        configuration = request.get("configuration", {})
        if not isinstance(configuration, dict):
            raise BootstrapError("Configurazione della prova non valida.")
        if action in {"doctor", "run"}:
            connection_arguments(configuration)
        if action == "run":
            run_arguments(configuration, run_id, run_directory / "results")
            supplied = request.get("export_directory", "")
            if not isinstance(supplied, str) or not supplied or not Path(supplied).is_absolute():
                raise BootstrapError("La cartella condivisa per i risultati non è valida.")
            export_root = Path(supplied)
            if not export_root.is_dir() or (export_root / run_id).exists():
                raise BootstrapError("La cartella condivisa non è disponibile oppure contiene già questa prova.")
            marker = run_directory / "run-requested.json"
            if marker.exists():
                raise BootstrapError("Questa prova esiste già nella VM. I suoi dati sono conservati; avvia una nuova prova con un nuovo identificativo.")
        python, repository, env = prepare(root)
        if action in {"doctor", "run"}:
            emit("phase", message="Verifica di endpoint, modello e prerequisiti…")
            run_process([python, repository / "terminal_bench.py", "doctor", *connection_arguments(configuration)],
                        cwd=repository, env=env, timeout=180)
            container_preflight(root, repository, env, run_directory, configuration)
        if action == "run":
            write_json(run_directory / "run-requested.json", {"run_id": run_id, "revision": REVISION})
            emit("phase", message="Esecuzione dei task e dei verificatori nei container…")
            code, _ = run_process([python, repository / "terminal_bench.py", "run",
                                   *run_arguments(configuration, run_id, run_directory / "results")],
                                  cwd=repository, env=env, timeout=None, check=False)
            destination, summary = export_artifacts(run_directory, export_root, run_id)
            if summary is not None:
                emit("result", directory=str(destination), relative_directory=run_id, summary=summary, partial=code != 0)
            if code != 0:
                raise BootstrapError(f"Il runner è terminato con codice {code}. Log e stato dei job rimangono nella VM: {repository / 'jobs' / run_id}")
            if summary is None:
                raise BootstrapError("Il runner non ha prodotto un riepilogo verificato; nessun punteggio viene dedotto.")
        update_control("completed")
        emit("completed", action=action, status="success", run_id=run_id,
             artifacts_directory=str(destination) if destination else None)
        return 0
    except Cancelled:
        update_control("cancelled")
        emit("completed", action=action, status="cancelled", run_id=run_id)
        return 130
    except Exception as error:
        update_control("failed")
        emit("log", message=str(error))
        if export_root is not None and not (export_root / run_id).exists():
            try:
                destination, summary = export_artifacts(run_directory, export_root, run_id)
                if summary is not None:
                    emit("result", directory=str(destination), relative_directory=run_id, summary=summary, partial=True)
            except Exception as export_error:
                emit("log", message=f"I log originali restano nella VM; esportazione non riuscita: {export_error}")
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    try:
        exit_code = main()
    except (BootstrapError, OSError, ValueError, TypeError) as error:
        emit("error", message=str(error))
        exit_code = 1
    except Cancelled:
        emit("completed", status="cancelled")
        exit_code = 130
    sys.exit(exit_code)
"""#
}
