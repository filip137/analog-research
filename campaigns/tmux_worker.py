"""A receipt-driven tmux worker for the persistent run-watch dispatcher.

``bridge --spec PATH`` is the foreground command given to queue_remote. The
attempt token comes from RUN_WATCH_ATTEMPT (or --attempt). Its independent tmux
supervisor holds inherited reservation locks, enforces a remote-monotonic lease,
and retains the pane after completion. Session existence is never completion.
Only the controller calls ``renew``; a worker cannot renew its own supervision.

Example spec (all paths absolute, argv uses the native campaign runner)::

    {"case_id": "exp1e-linear-1em4", "state_root": "/runs/workers",
     "cwd": "/source/hwa", "timeout_seconds": 7200, "lease_seconds": 900,
     "env": {"EBL_CAMPAIGN_STREAM": "1"},
     "argv": ["/env/bin/python", "-m", "ebl", "campaign", "run",
              "--manifest", "/runs/slice.json", "--output-dir", "/runs/native",
              "--resume", "--fail-fast"]}

Use ``python -m campaigns.tmux_worker bridge --spec SPEC`` as the queue argv.
Observe/renew using ``status``/``renew --spec SPEC --attempt TOKEN``. Attach
read-only with ``tmux attach -r -t SESSION`` from the receipt. Streaming keeps
native stdout.log/stderr.log as well as the combined pane/run.log output.
"""
from __future__ import annotations

import argparse
import array
import contextlib
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import selectors
import shlex
import signal
import socket
import stat
import subprocess
import sys
import time

TERMINAL = {"complete", "failed", "cancelled", "timeout", "supervision_lost", "rejected"}


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def identity(pid: int) -> dict | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": int(fields[19]), "pgid": int(fields[2]),
                "sid": int(fields[3]), "boot_id": boot_id()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(value: dict | None) -> bool:
    return bool(value and identity(value["pid"]) == value)


def group_members(pgid: int) -> list[int]:
    result = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                current = identity(int(entry.name))
                if current and current["pgid"] == pgid:
                    result.append(current["pid"])
            except (OSError, ValueError, IndexError):
                continue
    return result


def load_spec(path: Path, attempt: str | None = None) -> tuple[dict, Path]:
    value = json.loads(path.read_text())
    required = {"case_id", "state_root", "cwd", "argv", "timeout_seconds"}
    optional = {"env", "lease_seconds", "session_prefix", "tmux_socket", "poll_seconds",
                "termination_grace_seconds"}
    if not required <= value.keys() or value.keys() - required - optional:
        raise ValueError("Invalid tmux worker spec keys")
    for key in ("case_id", "session_prefix"):
        if key in value and not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value[key]):
            raise ValueError(f"Invalid {key}")
    if not isinstance(value["argv"], list) or not value["argv"] or any(
            not isinstance(item, str) for item in value["argv"]):
        raise ValueError("argv must be a nonempty list of strings")
    if not isinstance(value.get("env", {}), dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in value.get("env", {}).items()):
        raise ValueError("env must map strings to strings")
    for key, default in (("lease_seconds", 900), ("poll_seconds", 0.2),
                         ("termination_grace_seconds", 2)):
        value.setdefault(key, default)
    for key in ("timeout_seconds", "lease_seconds", "poll_seconds", "termination_grace_seconds"):
        if isinstance(value[key], bool) or not isinstance(value[key], (int, float)) or not (
                math.isfinite(value[key]) and value[key] > 0):
            raise ValueError(f"{key} must be finite and positive")
    for key in ("state_root", "cwd"):
        value[key] = str(Path(value[key]).expanduser().resolve())
    token = attempt or os.environ.get("RUN_WATCH_ATTEMPT")
    if not token or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", token):
        raise ValueError("An explicit --attempt or RUN_WATCH_ATTEMPT token is required")
    value["attempt_id"] = token
    value["session"] = f"{value.get('session_prefix', value['case_id'])}-{token}"
    value["spec_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value, Path(value["state_root"]) / token


def tmux(spec: dict, *argv: str, check: bool = True) -> subprocess.CompletedProcess:
    command = ["tmux"]
    if spec.get("tmux_socket"):
        command += ["-S", spec["tmux_socket"]]
    return subprocess.run(command + list(argv), capture_output=True, text=True, check=check)


def observe(spec: dict, directory: Path) -> dict:
    receipt = read(directory / "receipt.json")
    if receipt is None:
        return {"attempt_id": spec["attempt_id"], "missing": True}
    if receipt["spec_sha256"] != spec["spec_sha256"]:
        raise ValueError("Attempt token already belongs to a different immutable spec")
    worker = receipt.get("worker")
    live = alive(worker)
    # A group can outlive its initial process. Only inspect it on the same boot.
    if worker and worker["boot_id"] == boot_id():
        live = live or bool(group_members(worker["pgid"]))
    return {"attempt_id": spec["attempt_id"], "receipt": receipt,
            "worker_alive": live, "supervisor_alive": alive(receipt.get("supervisor")),
            "session": receipt["session"], "directory": str(directory)}


def renew(spec: dict, directory: Path) -> dict:
    observation = observe(spec, directory)
    if observation.get("missing"):
        raise ValueError("Cannot renew an unknown attempt")
    if observation["receipt"]["state"] in TERMINAL:
        return {"renewed": False, "terminal": True}
    value = {"attempt_id": spec["attempt_id"], "boot_id": boot_id(),
             "renewed_monotonic": time.monotonic()}
    atomic(directory / "lease.json", value)
    return {"renewed": True, **value}


def lease_valid(spec: dict, directory: Path) -> bool:
    try:
        value = read(directory / "lease.json")
        age = time.monotonic() - value["renewed_monotonic"]
        return (value["attempt_id"] == spec["attempt_id"] and value["boot_id"] == boot_id()
                and 0 <= age < spec["lease_seconds"])
    except (OSError, ValueError, TypeError, KeyError):
        return False


def cancel(spec: dict, directory: Path) -> dict:
    if not directory.is_dir():
        raise ValueError("Cannot cancel an unknown attempt")
    # A service restart and an outer queue timeout can request cancellation at
    # once. Serialize orphan recovery so neither can overwrite a final receipt.
    with (directory / "cancel.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _cancel_locked(spec, directory)


def _cancel_locked(spec: dict, directory: Path) -> dict:
    observation = observe(spec, directory)
    if observation.get("missing"):
        raise ValueError("Cannot cancel an unknown attempt")
    atomic(directory / "cancel.json", {"attempt_id": spec["attempt_id"], "at": time.time()})
    if observation["worker_alive"] and not observation["supervisor_alive"]:
        receipt = observation["receipt"]
        worker = receipt.get("worker")
        # A recovered controller may act only on the exact original session
        # leader. An old boot, recycled PID, or leaderless group needs manual
        # reconciliation; never turn a stale numeric PGID into a kill target.
        if (not alive(worker) or worker["pid"] != worker["pgid"]
                or worker["sid"] != worker["pid"]):
            raise RuntimeError("Orphan group lacks a live verified owner; refusing cancellation")
        receipt["orphan_cancellation"] = {"owner": identity(os.getpid()), "requested_at": time.time(),
                                          "reason": "supervisor_missing_with_live_worker", "complete": False}
        atomic(directory / "receipt.json", receipt)
        try:
            stop_orphan_group(worker, spec["termination_grace_seconds"])
        except BaseException as exc:
            receipt["orphan_cancellation"]["error"] = f"{type(exc).__name__}: {exc}"
            atomic(directory / "receipt.json", receipt)
            raise
        receipt["orphan_cancellation"]["complete"] = True
        elapsed = (time.monotonic() - receipt["started_monotonic"] if "started_monotonic" in receipt
                   else time.time() - receipt["started_at"])
        receipt.update(state="supervision_lost", finished_at=time.time(), elapsed_seconds=max(0, elapsed),
                       exit_code=None)
        atomic(directory / "receipt.json", receipt)
        return {"cancel_requested": True, "orphan_stopped": True, "state": receipt["state"]}
    return {"cancel_requested": True}


def stop_orphan_group(owner: dict, grace: float) -> None:
    """Stop only verified members of a continuously identified owned session.

    pidfds prevent individual PID reuse between verification and signalling.
    Newly observed group members are accepted only while a previously verified
    member still holds that group identity; otherwise stop rather than guess.
    """
    if not alive(owner) or owner["pid"] != owner["pgid"] or owner["sid"] != owner["pid"]:
        raise RuntimeError("Orphan process identity changed before cancellation")
    known = {owner["pid"]: owner}

    def members():
        current = {}
        for pid in group_members(owner["pgid"]):
            value = identity(pid)
            if value is not None:
                if value["boot_id"] != owner["boot_id"] or value["sid"] != owner["sid"]:
                    raise RuntimeError("Orphan group identity changed")
                current[pid] = value
        new = {pid: value for pid, value in current.items() if known.get(pid) != value}
        if new and not any(known.get(pid) == value for pid, value in current.items()):
            raise RuntimeError("Orphan group ownership became ambiguous")
        known.update(new)
        return current

    def send(value, sig):
        try:
            descriptor = _pidfd_open(value["pid"])
        except ProcessLookupError:
            return
        try:
            if identity(value["pid"]) == value:
                with contextlib.suppress(ProcessLookupError):
                    _pidfd_send_signal(descriptor, sig)
        finally:
            os.close(descriptor)

    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 2)):
        deadline = time.monotonic() + wait
        signalled = set()
        while True:
            current = members()
            if not current:
                return
            for value in current.values():
                key = (value["pid"], value["start_ticks"])
                if key not in signalled:
                    send(value, sig)
                    signalled.add(key)
            if time.monotonic() >= deadline:
                break
            time.sleep(.02)
    if members():
        raise RuntimeError("Verified orphan group did not stop")


def _pidfd_syscall(number, *arguments):
    # Older Conda builds omit Python/glibc pidfd wrappers even on modern lab
    # kernels. Linux x86-64 and arm64 assign these two syscall numbers equally.
    # Unsupported platforms fail closed; never fall back to unchecked PIDs.
    if sys.platform != "linux" or platform.machine() not in {"x86_64", "aarch64"}:
        raise RuntimeError("Safe orphan cancellation requires Linux pidfd support")
    library = ctypes.CDLL(None, use_errno=True)
    library.syscall.restype = ctypes.c_long
    result = library.syscall(ctypes.c_long(number), *arguments)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def _pidfd_open(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    return _pidfd_syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def _pidfd_send_signal(descriptor, sig):
    if hasattr(signal, "pidfd_send_signal"):
        return signal.pidfd_send_signal(descriptor, sig)
    return _pidfd_syscall(424, ctypes.c_int(descriptor), ctypes.c_int(sig), ctypes.c_void_p(), ctypes.c_uint(0))


def inherited_fds() -> list[int]:
    """queue_remote passes flock descriptors; carry them across the tmux server."""
    result = []
    for name in os.listdir("/proc/self/fd"):
        fd = int(name)
        try:
            if fd > 2 and os.get_inheritable(fd) and stat.S_ISREG(os.fstat(fd).st_mode):
                result.append(fd)
        except OSError:
            pass
    return result


def socket_path(directory: Path) -> str:
    # Linux's pathname socket limit is 108 bytes. Abstract sockets are likewise
    # bounded but need no global file cleanup after a failed launch.
    digest = hashlib.sha256(str(directory).encode()).hexdigest()
    return "\0hwa-tmux-" + str(os.getuid()) + "-" + digest


def launch(spec: dict, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / "launch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = read(directory / "receipt.json")
        if previous:
            return {"existing": True, **observe(spec, directory)}
        if tmux(spec, "has-session", "-t", "=" + spec["session"], check=False).returncode == 0:
            raise ValueError("Refusing to adopt an existing unrelated tmux session")
        environment = dict(spec.get("env", {}))
        for key in ("CUDA_VISIBLE_DEVICES", "RUN_WATCH_ATTEMPT"):
            if key in os.environ:
                environment[key] = os.environ[key]
        environment["PYTHONUNBUFFERED"] = "1"
        saved = dict(spec, effective_env=environment)
        atomic(directory / "spec.json", saved)
        receipt = {"schema": "campaign.tmux_worker.v1", "case_id": spec["case_id"],
                   "attempt_id": spec["attempt_id"], "spec_sha256": spec["spec_sha256"],
                   "session": spec["session"], "state": "starting", "started_at": time.time(),
                   "supervisor": None, "worker": None, "exit_code": None,
                   "log_path": str(directory / "run.log")}
        atomic(directory / "receipt.json", receipt)
        renew(spec, directory)
        command = shlex.join([sys.executable, str(Path(__file__).resolve()), "supervise",
                              "--directory", str(directory)])
        try:
            tmux(spec, "new-session", "-d", "-s", spec["session"], "-c", spec["cwd"], command)
            tmux(spec, "set-option", "-w", "-t", "=" + spec["session"] + ":", "remain-on-exit", "on")
            deadline = time.monotonic() + 10
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
                while True:
                    try:
                        stream.connect(socket_path(directory))
                        break
                    except (ConnectionRefusedError, FileNotFoundError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("tmux supervisor did not accept reservation handoff")
                        time.sleep(0.05)
                descriptors = array.array("i", inherited_fds())
                ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, descriptors)] if descriptors else []
                stream.sendmsg([b"start"], ancillary)
                stream.settimeout(10)
                if stream.recv(16) != b"accepted":
                    raise RuntimeError("tmux supervisor rejected reservation handoff")
        except BaseException:
            # Never erase an ambiguous receipt or relaunch its token. The worker
            # has not started before the handoff; its startup wait is bounded.
            atomic(directory / "cancel.json", {"attempt_id": spec["attempt_id"], "at": time.time()})
            raise
    return {"existing": False, **observe(spec, directory)}


def stop_group(child: subprocess.Popen, grace: float) -> None:
    # Only called by the supervisor which created this session and held its
    # identity continuously. No pid discovered from a stale receipt is killed.
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 2)):
        if not group_members(child.pid):
            break
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, sig)
        deadline = time.monotonic() + wait
        while group_members(child.pid) and time.monotonic() < deadline:
            child.poll()
            time.sleep(0.02)
    child.wait(timeout=3)
    if group_members(child.pid):
        raise RuntimeError("Owned process group did not stop")


def supervise(directory: Path) -> int:
    spec = read(directory / "spec.json")
    receipt = read(directory / "receipt.json")
    descriptors: list[int] = []
    child = None
    started = time.monotonic()
    requested_stop = False

    def interrupted(_signum, _frame):
        nonlocal requested_stop
        requested_stop = True

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    with (directory / "worker.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 75
        if receipt["state"] != "starting":
            return 75
        receipt["supervisor"] = identity(os.getpid())
        receipt["started_monotonic"] = started
        atomic(directory / "receipt.json", receipt)
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(socket_path(directory))
                server.listen(1)
                server.settimeout(15)
                connection, _ = server.accept()
                with connection:
                    message, ancillary, flags, _ = connection.recvmsg(16, socket.CMSG_SPACE(64 * 4))
                    if message != b"start" or flags & socket.MSG_CTRUNC:
                        raise RuntimeError("Invalid reservation handoff")
                    for level, kind, payload in ancillary:
                        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                            values = array.array("i")
                            values.frombytes(payload[:len(payload) - len(payload) % values.itemsize])
                            descriptors.extend(values)
                    receipt["reservation_fds"] = len(descriptors)
                    atomic(directory / "receipt.json", receipt)
                    connection.sendall(b"accepted")
            if not lease_valid(spec, directory):
                receipt["state"] = "supervision_lost"
                return 125
            if requested_stop or (directory / "cancel.json").exists():
                receipt["state"] = "cancelled"
                return 130
            with (directory / "run.log").open("ab", buffering=0) as log:
                child = subprocess.Popen(spec["argv"], cwd=spec["cwd"],
                                         env=dict(os.environ, **spec["effective_env"]),
                                         stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, start_new_session=True,
                                         pass_fds=tuple(descriptors) + (lock.fileno(),))
                receipt.update(state="running", worker=identity(child.pid))
                atomic(directory / "receipt.json", receipt)
                selector = selectors.DefaultSelector()
                selector.register(child.stdout, selectors.EVENT_READ)
                reason = None
                while child.poll() is None or group_members(child.pid):
                    if requested_stop or (directory / "cancel.json").exists():
                        reason = "cancelled"
                    elif time.monotonic() - started >= spec["timeout_seconds"]:
                        reason = "timeout"
                    elif not lease_valid(spec, directory):
                        reason = "supervision_lost"
                    if reason:
                        stop_group(child, spec["termination_grace_seconds"])
                        break
                    for key, _ in selector.select(spec["poll_seconds"]):
                        data = os.read(key.fd, 65536)
                        if data:
                            log.write(data)
                            with contextlib.suppress(BrokenPipeError, OSError):
                                os.write(sys.stdout.fileno(), data)
                        else:
                            selector.unregister(key.fileobj)
                    if not selector.get_map():
                        time.sleep(spec["poll_seconds"])
                tail = child.stdout.read()
                if tail:
                    log.write(tail)
                    with contextlib.suppress(BrokenPipeError, OSError):
                        os.write(sys.stdout.fileno(), tail)
                selector.close()
                child.stdout.close()
                receipt.update(state=reason or ("complete" if child.returncode == 0 else "failed"),
                               exit_code=child.returncode)
        except BaseException as exc:
            if child is not None:
                stop_group(child, spec["termination_grace_seconds"])
            receipt.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            receipt.update(finished_at=time.time(), elapsed_seconds=time.monotonic() - started)
            atomic(directory / "receipt.json", receipt)
            for descriptor in descriptors:
                os.close(descriptor)
    return 0 if receipt["state"] == "complete" else (receipt.get("exit_code") or 1)


def bridge(spec: dict, directory: Path) -> int:
    stopped = False

    def interrupted(_signum, _frame):
        nonlocal stopped
        stopped = True

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    launch(spec, directory)
    offset = 0
    while True:
        if stopped:
            cancel(spec, directory)
        log = directory / "run.log"
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(offset)
                data = stream.read(65536)
                offset = stream.tell()
                if data:
                    os.write(sys.stdout.fileno(), data)
        observation = observe(spec, directory)
        receipt = observation["receipt"]
        if receipt["state"] in TERMINAL and not observation["worker_alive"]:
            return 0 if receipt["state"] == "complete" else (receipt.get("exit_code") or 1)
        if receipt["state"] != "starting" and not observation["supervisor_alive"]:
            # Returning would release the outer reservation if no descendant
            # holds it. Refuse to assert completion while a worker remains.
            if not observation["worker_alive"]:
                # The supervisor publishes its final receipt before exiting.
                # Our first read can precede that rename while the process
                # checks follow it; reread before diagnosing a lost worker.
                final = read(directory / "receipt.json")
                if final["state"] in TERMINAL:
                    return 0 if final["state"] == "complete" else (final.get("exit_code") or 1)
                return 125
        time.sleep(spec["poll_seconds"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "bridge", "status", "renew", "cancel", "supervise"))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--attempt")
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.action == "supervise":
        if not args.directory:
            parser.error("supervise requires --directory")
        return supervise(args.directory)
    if not args.spec:
        parser.error("--spec is required")
    spec, directory = load_spec(args.spec, args.attempt)
    if args.action == "bridge":
        return bridge(spec, directory)
    result = {"launch": launch, "status": observe, "renew": renew, "cancel": cancel}[args.action](spec, directory)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
