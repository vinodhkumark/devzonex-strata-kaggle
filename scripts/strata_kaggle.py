"""Guided Kaggle helpers for running Strata-supported models safely."""
from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import json
import os
import platform
import posixpath
import re
import shutil
import signal
import queue
import tempfile
import threading
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable

WORKING = Path("/kaggle/working")
INPUT_ROOT = Path("/kaggle/input/datasets")
STRATA_ROOT = Path("/root/.local/share/strata")
STRATA_BIN = Path("/root/.local/bin/strata")
CLOUDFLARED_BIN = Path("/root/.local/bin/cloudflared")


def log(message: str) -> None:
    print("[DevZoneX Strata] " + message, flush=True)


def redact(text: str) -> str:
    text = re.sub(r"sk-strata-[A-Za-z0-9._~-]+", "sk-strata-<redacted>", text)
    text = re.sub(r"(?i)(authorization\s*:\s*bearer\s+)\S+", r"\1<redacted>", text)
    text = re.sub(r"(?i)(KAGGLE_API_TOKEN\s*=\s*)\S+", r"\1<redacted>", text)
    return text


def run_command(command: list[str], *, check: bool = True, timeout: int | None = None,
                env: dict[str, str] | None = None, capture: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command, text=True, stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None, env=env,
        timeout=timeout, check=False,
    )
    output = (result.stdout or "") if capture else ""
    safe_output = redact(output)
    if safe_output.strip():
        print(safe_output, end="" if safe_output.endswith("\n") else "\n", flush=True)
    if check and result.returncode != 0:
        raise RuntimeError("Command failed (" + str(result.returncode) + "): " +
                           " ".join(command) + "\n" + safe_output[-5000:])
    return result


def kaggle_env(username: str, secret_name: str = "KAGGLE_API_TOKEN") -> dict[str, str]:
    if not username or username.lower().startswith("your-"):
        raise ValueError("Set KAGGLE_USERNAME in the notebook configuration cell.")
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret(secret_name).strip()
    except Exception as exc:
        raise RuntimeError(
            "Kaggle Secret " + secret_name + " was not available. Add it in Add-ons > Secrets, "
            "enable notebook access, and rerun. Do not paste the token into notebook code."
        ) from exc
    if not token:
        raise RuntimeError("Kaggle Secret " + secret_name + " is empty.")
    # Keep credentials outside the saved notebook workspace and never print the token.
    home = Path("/tmp/devzonex-kaggle-home")
    config = home / ".kaggle"
    config.mkdir(parents=True, exist_ok=True)
    token_path = config / "access_token"
    token_path.write_text(token + "\n", encoding="utf-8")
    token_path.chmod(0o600)
    env = os.environ.copy()
    env.update({
        "HOME": str(home),
        "TMPDIR": "/kaggle/working/tmp",
        "XDG_CACHE_HOME": str(home / ".cache"),
        "KAGGLE_CONFIG_DIR": str(config),
        "KAGGLE_USERNAME": username,
    })
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLE_CONFIG_DIR"] = str(config)
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["TMPDIR"] = env["TMPDIR"]
    os.environ["XDG_CACHE_HOME"] = env["XDG_CACHE_HOME"]
    log("Kaggle Secret loaded into a temporary credential directory (value hidden).")
    return env


def install_kaggle_cli(env: dict[str, str] | None = None) -> None:
    run_command([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "kaggle"],
                timeout=900, env=env)
    run_command(["kaggle", "--version"], timeout=60, env=env)


def _directory_snapshot(path: Path) -> tuple[int, int]:
    """Return file count and total bytes without following symlinks."""
    count = 0
    size = 0
    if not path.exists():
        return count, size
    for root, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(root) / d).is_symlink()]
        for name in files:
            file_path = Path(root) / name
            try:
                if file_path.is_symlink():
                    continue
                stat = file_path.stat()
                count += 1
                size += stat.st_size
            except OSError:
                continue
    return count, size


def run_streaming(command: list[str], *, check: bool = True, timeout: int | None = None,
                  env: dict[str, str] | None = None,
                  heartbeat_interval: float | None = None,
                  heartbeat_label: str = "Command",
                  progress_paths: list[Path] | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command while streaming output and reporting real directory-size heartbeats."""
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    child_env.setdefault("PYTHONIOENCODING", "utf-8")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=child_env,
        bufsize=8192, start_new_session=True,
    )
    if process.stdout is None:
        raise RuntimeError("Could not capture process output.")
    output_queue: queue.Queue[bytes | None] = queue.Queue()
    captured: list[str] = []

    def pump() -> None:
        pending = bytearray()
        try:
            while True:
                chunk = process.stdout.read1(4096)
                if not chunk:
                    break
                pending.extend(chunk)
                while True:
                    split_at = next((i for i, byte in enumerate(pending) if byte in (10, 13)), None)
                    if split_at is None:
                        break
                    separator = pending[split_at]
                    piece = bytes(pending[:split_at + 1])
                    del pending[:split_at + 1]
                    # Combine CRLF into one newline event.
                    if separator == 13 and pending[:1] == b"\n":
                        piece += b"\n"
                        del pending[:1]
                    output_queue.put(piece)
            if pending:
                output_queue.put(bytes(pending))
        finally:
            output_queue.put(None)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout if timeout is not None else None
    last_heartbeat = time.monotonic()
    previous_snapshots: dict[str, tuple[int, int]] = {}
    reached_eof = False
    try:
        while not reached_eof:
            if deadline is not None and time.monotonic() >= deadline and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise TimeoutError(
                    f"Command timed out after {timeout}s: {' '.join(command)}. "
                    f"Recent output: {redact(''.join(captured)[-3000:])}"
                )
            try:
                item = output_queue.get(timeout=0.5)
            except queue.Empty:
                now = time.monotonic()
                if heartbeat_interval and now - last_heartbeat >= heartbeat_interval and process.poll() is None:
                    details = []
                    for path in progress_paths or []:
                        count, size = _directory_snapshot(path)
                        label = str(path)
                        current = (count, size)
                        previous = previous_snapshots.get(label)
                        change = ""
                        if previous is not None:
                            delta = size - previous[1]
                            if delta > 0:
                                change = f", +{delta / (1024**2):.1f} MiB since last check"
                        details.append(f"{path}: {count} files, {size / (1024**3):.2f} GiB{change}")
                        previous_snapshots[label] = current
                    print(f"[DevZoneX Strata] {heartbeat_label} still running ({int(now - (deadline - timeout)) if deadline and timeout else int(now - last_heartbeat)}s elapsed)."
                          + (("\n  " + "\n  ".join(details)) if details else ""), flush=True)
                    last_heartbeat = now
                continue
            if item is None:
                reached_eof = True
                continue
            text_piece = item.decode("utf-8", errors="replace")
            safe_piece = redact(text_piece)
            captured.append(text_piece)
            if text_piece.endswith("\r") and not text_piece.endswith("\r\n"):
                print("\r" + safe_piece.rstrip("\r"), end="", flush=True)
            else:
                print(safe_piece, end="" if safe_piece.endswith("\n") else "\n", flush=True)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        reader.join(timeout=2)
        process.stdout.close()

    output = redact("".join(captured))
    completed = subprocess.CompletedProcess(command, process.returncode, output, None)
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            + output[-5000:]
        )
    return completed


def _gpu_names() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=20, check=True,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def hardware_summary() -> dict[str, Any]:
    return {"platform": platform.platform(), "machine": platform.machine(), "gpu_names": _gpu_names()}


def default_gpu_list() -> str:
    names = _gpu_names()
    return ",".join(str(i) for i in range(len(names))) if names else "0"


def _dataset_roots() -> list[Path]:
    """Every root Kaggle may mount an attached dataset under.

    Datasets usually appear as <root>/<owner>/<slug>, and older or competition
    style mounts use <root>/<slug>. Kaggle has used both /kaggle/input and
    /kaggle/input/datasets as the root, so search both instead of assuming one.
    """
    roots = [INPUT_ROOT]
    parent = INPUT_ROOT.parent
    if parent != INPUT_ROOT and parent.is_dir():
        roots.append(parent)
    return roots


def find_attached_dataset(owner: str, slug: str, marker: str | None = None) -> Path | None:
    for root in _dataset_roots():
        for candidate in (root / owner / slug, root / slug):
            if candidate.is_dir() and (marker is None or (candidate / marker).is_file()):
                return candidate
    marker_name = marker or "dataset-metadata.json"
    for root in _dataset_roots():
        if not root.is_dir():
            continue
        for candidate in root.rglob(marker_name):
            parent = candidate.parent
            if parent.name == slug and (parent.parent.name == owner or owner in parent.parts):
                return parent
    return None


def attached_datasets() -> list[Path]:
    """Dataset directories mounted right now, without walking their files.

    Kaggle writes dataset-metadata.json into every mounted dataset, so checking
    the two mount layouts is enough. Walking the mount instead would read tens of
    gigabytes of model files just to answer "is it attached?".
    """
    found: list[Path] = []
    for root in _dataset_roots():
        if not root.is_dir():
            continue
        try:
            first_level = sorted(p for p in root.iterdir() if p.is_dir())
        except OSError:
            continue
        for first in first_level:
            candidates = [first]
            try:
                candidates += sorted(p for p in first.iterdir() if p.is_dir())
            except OSError:
                pass
            for path in candidates:
                if path not in found and (path / "dataset-metadata.json").is_file():
                    found.append(path)
    return found


def dataset_mount_report() -> dict[str, Any]:
    """Summarize the current input mounts so a missing dataset is diagnosable."""
    roots = [str(root) for root in _dataset_roots()]
    attached = [str(path) for path in attached_datasets()]
    return {"roots": roots, "attached": attached}


def log_mount_state() -> dict[str, Any]:
    """Print what is mounted, so "not attached" comes with evidence."""
    report = dataset_mount_report()
    if report["attached"]:
        log("Attached datasets: " + ", ".join(report["attached"]))
    else:
        log("No dataset is attached to this session (searched " + ", ".join(report["roots"])
            + "). Add it under Add Input, then restart the session so it mounts.")
    return report


def safe_extract_runtime(archive: Path, destination: Path) -> None:
    """Safely extract runtime paths, normalizing internal absolute symlinks.

    Runtime archives created from STRATA_ROOT can contain absolute symlinks
    such as terminal/current -> /root/.local/share/strata/terminal/<version>.
    Rewrite those to archive-relative links, but only when the target remains
    inside STRATA_ROOT and within the same allowed top-level runtime tree.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    runtime_root = Path(STRATA_ROOT).absolute()

    with tarfile.open(archive, "r:") as tf:
        members = tf.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise RuntimeError("Unsafe path in runtime archive: " + member.name)
            if not name.parts or name.parts[0] not in {"terminal", "components", "tools"}:
                raise RuntimeError("Unexpected runtime archive path: " + member.name)

            if not (member.issym() or member.islnk()):
                continue

            link_text = member.linkname
            if os.path.isabs(link_text):
                # Absolute links are only accepted when their target is inside
                # the known Strata runtime root. Convert that target back to
                # a path relative to the archive's top-level tree.
                absolute_target = Path(os.path.normpath(link_text))
                try:
                    archive_target = absolute_target.relative_to(runtime_root)
                except ValueError as exc:
                    raise RuntimeError(
                        "Unsafe absolute link in runtime archive: " + member.name
                    ) from exc

                resolved = PurePosixPath(archive_target.as_posix())
                if (
                    not resolved.parts
                    or resolved.parts[0] != name.parts[0]
                    or ".." in resolved.parts
                ):
                    raise RuntimeError(
                        "Unsafe absolute/cross-tree link in runtime archive: "
                        + member.name
                    )

                if member.issym():
                    # Symlinks are relative to the link's containing directory.
                    member.linkname = posixpath.relpath(
                        resolved.as_posix(),
                        start=name.parent.as_posix(),
                    )
                else:
                    # Tar hard-link targets are relative to archive root.
                    member.linkname = resolved.as_posix()
                continue

            link = PurePosixPath(link_text)
            if member.issym():
                resolved = PurePosixPath(
                    posixpath.normpath(str(name.parent / link))
                )
            else:
                resolved = PurePosixPath(posixpath.normpath(str(link)))
            if (
                not resolved.parts
                or resolved.parts[0] != name.parts[0]
                or ".." in resolved.parts
            ):
                raise RuntimeError(
                    "Unsafe cross-tree link in runtime archive: " + member.name
                )

        tf.extractall(destination, members=members, filter="data")



def _remove_existing_path(path: Path) -> None:
    """Remove a directory/file/symlink safely before atomically installing staged runtime files."""
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)

def restore_runtime_cache(owner: str, slug: str) -> bool:
    ds = find_attached_dataset(owner, slug, "strata-runtime-cache.bin")
    if not ds:
        log("Runtime cache not attached; Strata will be installed if needed.")
        log_mount_state()
        return False
    archive = ds / "strata-runtime-cache.bin"
    if not archive.is_file():
        log("Runtime archive missing; skipping runtime restore.")
        return False
    checksum = ds / "strata-runtime-cache.bin.sha256"
    if checksum.is_file():
        expected = checksum.read_text(encoding="utf-8").strip().split()[0]
        if _sha256(archive) != expected:
            raise RuntimeError("Runtime cache checksum mismatch; refusing extraction.")

    manifest = ds / "runtime-manifest.json"
    meta: dict[str, Any] = {}
    if manifest.is_file():
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        if meta.get("machine") and meta["machine"] != platform.machine():
            log("Runtime machine differs from cached build; skipping cache restore.")
            return False
        saved_gpus = meta.get("gpu_names") or []
        current_gpus = _gpu_names()
        if saved_gpus and current_gpus and saved_gpus != current_gpus:
            log("GPU differs from cache; restoring terminal but Strata may need a compatible engine.")

    # Avoid overwriting an executable currently being used by a Strata service/CLI.
    # If the installed runtime matches this cache, reuse it without extracting over it.
    os.environ["PATH"] = "/root/.local/bin:" + os.environ.get("PATH", "")
    existing_version = None
    if shutil.which("strata"):
        probe = subprocess.run(["strata", "--version"], text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=60, check=False)
        existing_version = (probe.stdout or "").strip().splitlines()[0] if probe.stdout else None
    cached_version = str(meta.get("strata_version", "")).strip()
    engine_dirs = [str(x) for x in meta.get("engine_dirs", [])]
    engine_cache_ok = bool(engine_dirs) and all(
        (STRATA_ROOT / "components" / "engine" / name).is_dir() for name in engine_dirs
    )
    if (existing_version and cached_version and existing_version == cached_version
            and STRATA_BIN.exists() and engine_cache_ok):
        log("A matching Strata terminal and engine are already installed; reusing them without extraction to avoid overwriting a busy Python executable.")
        return True

    if shutil.which("strata"):
        log("Stopping any existing Strata server before replacing its runtime files…")
        stopped = subprocess.run(["strata", "launch", "--stop"], text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 timeout=90, check=False)
        if stopped.stdout and stopped.stdout.strip():
            print(redact(stopped.stdout), flush=True)
        # The CLI can report that nothing was running; that is not a failure.
        time.sleep(1)

    log("Verifying and extracting Strata runtime cache into a staging directory…")
    staging_root = Path("/root/.local/share/strata-restore-staging")
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    safe_extract_runtime(archive, staging_root)

    staged_terminal = staging_root / "terminal"
    staged_components = staging_root / "components"
    if not staged_terminal.is_dir() or not staged_components.is_dir():
        shutil.rmtree(staging_root, ignore_errors=True)
        raise RuntimeError("Runtime archive must contain terminal/ and components/ directories.")
    staged_bin = staged_terminal / "current" / "bin" / "strata"
    if not staged_bin.exists():
        matches = sorted(staged_terminal.glob("*/bin/strata"))
        if not matches:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise RuntimeError("No strata executable found in runtime archive.")

    STRATA_ROOT.mkdir(parents=True, exist_ok=True)
    # Replace only after staging has extracted and validated successfully.
    for part in ("terminal", "components"):
        old_path = STRATA_ROOT / part
        _remove_existing_path(old_path)
        os.replace(staging_root / part, old_path)
    staged_tools = staging_root / "tools"
    if staged_tools.is_dir():
        tools_root = STRATA_ROOT / "tools"
        tools_root.mkdir(parents=True, exist_ok=True)
        for tool in staged_tools.iterdir():
            target = tools_root / tool.name
            if target.exists() or target.is_symlink():
                target.unlink()
            os.replace(tool, target)
    shutil.rmtree(staging_root, ignore_errors=True)

    STRATA_BIN.parent.mkdir(parents=True, exist_ok=True)
    target = STRATA_ROOT / "terminal" / "current" / "bin" / "strata"
    if not target.exists():
        matches = sorted((STRATA_ROOT / "terminal").glob("*/bin/strata"))
        if not matches:
            raise RuntimeError("No strata executable found in restored runtime cache.")
        target = matches[-1]
    STRATA_BIN.unlink(missing_ok=True)
    STRATA_BIN.symlink_to(target)
    os.environ["PATH"] = "/root/.local/bin:" + os.environ.get("PATH", "")
    run_command([str(STRATA_BIN), "--version"], timeout=60)
    saved_cloudflared = STRATA_ROOT / "tools" / "cloudflared"
    if saved_cloudflared.is_file() and not CLOUDFLARED_BIN.exists():
        CLOUDFLARED_BIN.parent.mkdir(parents=True, exist_ok=True)
        saved_cloudflared.chmod(0o755)
        CLOUDFLARED_BIN.symlink_to(saved_cloudflared)
    log("Runtime cache restore completed.")
    return True


def install_strata_if_missing() -> None:
    os.environ["PATH"] = "/root/.local/bin:" + os.environ.get("PATH", "")
    if shutil.which("strata"):
        run_command(["strata", "--version"], timeout=60)
        return
    installer = Path("/tmp/strata-install.sh")
    run_command(["curl", "-fsSL",
                 "https://github.com/Niko1221/Strata/releases/latest/download/install.sh",
                 "-o", str(installer)], timeout=180)
    run_command(["sh", str(installer)], timeout=900)
    os.environ["PATH"] = "/root/.local/bin:" + os.environ.get("PATH", "")
    if not shutil.which("strata"):
        raise RuntimeError("Strata installer completed but the command is not on PATH.")
    run_command(["strata", "--version"], timeout=60)


def _copy_verified(
    src: Path,
    dst: Path,
    progress_callback: Callable[[int, Path, str], None] | None = None,
) -> tuple[int, bool]:
    """Copy one verified file and optionally report byte-level progress events."""
    expected = src.stat().st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    if progress_callback:
        progress_callback(0, dst, "start")

    # Copy mode must materialize symlinked input files into local storage.
    if dst.is_symlink():
        dst.unlink()
    elif dst.is_file() and dst.stat().st_size == expected:
        if progress_callback:
            progress_callback(0, dst, "complete")
        return expected, False

    temp = dst.with_name(dst.name + ".part")
    temp.unlink(missing_ok=True)
    try:
        with src.open("rb") as inp, temp.open("wb") as out:
            while True:
                chunk = inp.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                if progress_callback:
                    progress_callback(len(chunk), dst, "progress")
            out.flush()
            os.fsync(out.fileno())
        if temp.stat().st_size != expected:
            raise RuntimeError("Size verification failed: " + src.name)
        temp.replace(dst)
    except Exception:
        temp.unlink(missing_ok=True)
        raise

    if progress_callback:
        progress_callback(0, dst, "complete")
    return expected, True


def restore_model_cache(owner: str, slug: str, data_dir: Path, workers: int = 8,
                        family: str | None = None, variant: str | None = None,
                        restore_mode: str = "copy") -> bool:
    """Restore a private cache by copying files locally or symlinking to the Kaggle input mount.

    'copy' duplicates bytes into local storage and reports live byte-level progress.
    'symlink' avoids the extra local copy but may be slower because the engine reads the dataset mount.
    Kaggle still has to make attached input dataset files available to the runtime.
    """
    if restore_mode not in {"copy", "symlink"}:
        raise ValueError("restore_mode must be 'copy' or 'symlink'.")
    ds = find_attached_dataset(owner, slug, "manifest.json")
    if not ds:
        log("Model cache not attached; Strata will download the selected model on first launch.")
        log_mount_state()
        return False
    try:
        manifest = json.loads((ds / "manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("Cannot read model-cache manifest.") from exc
    if manifest.get("format") != "devzonex.strata-model-cache.v1":
        log("Unrecognized model cache manifest; not restoring it.")
        return False
    saved = manifest.get("model", {})
    if family and saved.get("family") != family:
        raise RuntimeError("Attached model cache family does not match the selected family.")
    if variant and saved.get("variant") != variant:
        raise RuntimeError("Attached model cache variant does not match the selected variant.")
    entries = manifest.get("files", [])
    if not entries:
        raise RuntimeError("Model cache manifest is empty.")
    total = sum(int(e["size"]) for e in entries)
    log("Restoring " + str(len(entries)) + " model files (" + f"{total / (1024**3):.2f} GiB" + ") using " + restore_mode + " mode.")
    data_dir.mkdir(parents=True, exist_ok=True)

    validated: list[tuple[Path, Path, int]] = []
    for item in entries:
        rel = PurePosixPath(item["relative_path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError("Unsafe path in model manifest: " + str(rel))
        dataset_name = PurePosixPath(str(item.get("dataset_name") or ""))
        if not dataset_name.parts or dataset_name.is_absolute() or ".." in dataset_name.parts:
            raise RuntimeError("Unsafe dataset file name in model manifest: "
                               + str(item.get("dataset_name")))
        src = ds.joinpath(*dataset_name.parts)
        dst = data_dir.joinpath(*rel.parts)
        size = int(item["size"])
        if not src.is_file() or src.stat().st_size != size:
            raise RuntimeError("Model cache file missing or size mismatch: " + item["dataset_name"])
        validated.append((src, dst, size))

    if restore_mode == "symlink":
        created = 0
        already_linked = 0
        for src, dst, size in validated:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink():
                try:
                    if dst.resolve(strict=True) == src.resolve(strict=True) and dst.stat().st_size == size:
                        already_linked += 1
                        continue
                except OSError:
                    pass
                # Replace a stale/dangling symlink from a previous mount/session.
                dst.unlink()
            elif dst.exists():
                if dst.is_file() and dst.stat().st_size == size:
                    # Keep a valid existing local copy rather than replacing it with a link.
                    continue
                raise RuntimeError("Restore target already exists with unexpected content: " + str(dst))
            dst.symlink_to(src.resolve())
            created += 1
        log(f"Model cache linked without copying model bytes: {created} links created, {already_linked} already linked.")
        log("Note: Kaggle still stages the attached input dataset; symlink mode avoids only the extra copy to local storage.")
        return True

    # Files already present as regular local files count as restored without re-copying.
    existing = {
        str(dst): (dst, size)
        for _src, dst, size in validated
        if not dst.is_symlink() and dst.is_file() and dst.stat().st_size == size
    }
    state: dict[str, Any] = {
        "progress_bytes": sum(size for _dst, size in existing.values()),
        "copied_bytes": 0,
        "completed": set(existing),
        "active": set(),
        "last_report_time": 0.0,
        "last_report_bytes": 0,
        "printed": False,
    }
    state_lock = threading.Lock()
    started_at = time.monotonic()
    report_interval = 1.5
    report_byte_interval = 512 * 1024 * 1024
    total_files = len(validated)

    def progress_callback(delta: int, dst: Path, event: str) -> None:
        now = time.monotonic()
        with state_lock:
            key = str(dst)
            if event == "start":
                state["active"].add(key)
            elif event == "progress":
                state["progress_bytes"] += delta
                state["copied_bytes"] += delta
            elif event == "complete":
                state["active"].discard(key)
                state["completed"].add(key)

            is_finished = (
                state["progress_bytes"] >= total
                and len(state["completed"]) >= total_files
                and not state["active"]
            )
            should_report = (
                not state["printed"]
                or now - state["last_report_time"] >= report_interval
                or state["progress_bytes"] - state["last_report_bytes"] >= report_byte_interval
                or is_finished
            )
            if not should_report:
                return

            elapsed_seconds = max(now - started_at, 0.001)
            elapsed = int(elapsed_seconds)
            elapsed_text = f"{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}"
            percent = 100.0 if total == 0 else min(100.0, 100 * state["progress_bytes"] / total)
            speed_mib_s = state["copied_bytes"] / (1024 ** 2) / elapsed_seconds
            progress_text = (
                f"[DevZoneX Strata] Model restore: "
                f"{state['progress_bytes'] / (1024 ** 3):.2f}/{total / (1024 ** 3):.2f} GiB "
                f"({percent:.1f}%) | {len(state['completed'])}/{total_files} files complete "
                f"| {len(state['active'])} copying | {speed_mib_s:.1f} MiB/s "
                f"| {elapsed_text} elapsed"
            )
            print("\r" + progress_text, end="", flush=True)
            state["last_report_time"] = now
            state["last_report_bytes"] = state["progress_bytes"]
            state["printed"] = True

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_copy_verified, src, dst, progress_callback)
                for src, dst, _size in validated
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    finally:
        if state["printed"]:
            print(flush=True)

    log("Model cache copied into local storage and size-verified.")
    return True



def _cached_model_path(data_dir: Path, family: str, variant: str) -> Path | None:
    """Find a compatible-looking local GGUF to prevent Strata redownloading restored models."""
    models_dir = data_dir / "models"
    if not models_dir.is_dir():
        return None
    files = [p for p in models_dir.rglob("*.gguf") if p.is_file() and not p.name.endswith(".part")]
    if not files:
        return None
    expected_folder = (family + "-" + variant).lower().replace("_", "-")
    matching = [p for p in files if expected_folder in p.parent.as_posix().lower().replace("_", "-")]
    if matching:
        return max(matching, key=lambda p: p.stat().st_size).parent
    parent_dirs = {p.parent for p in files}
    if len(parent_dirs) == 1:
        return next(iter(parent_dirs))
    raise RuntimeError(
        "Multiple cached GGUF model folders were found under " + str(models_dir) +
        ". Keep one selected model in this cache or choose a dedicated MODEL_DATA_DIR."
    )


def _model_source_args(data_dir: Path, family: str, variant: str) -> list[str]:
    cached = _cached_model_path(data_dir, family, variant)
    if cached:
        log("Using local model files at " + str(cached) + " (avoiding a new model download).")
        return ["--model-path", str(cached)]
    return ["--data-dir", str(data_dir)]


DRAFTING_MODES = ("auto", "mtp", "dflash2", "off")
MTP_FETCH_GB = 5.2


def _drafting_args(drafting: str) -> list[str]:
    """Map the notebook's drafting choice to Strata launch flags.

    "auto" and "mtp" load the model's own draft layer (about 5.2 GB on disk and
    1.08 GiB of VRAM) and are the upstream default. "dflash2" selects the
    separate Qwen3.8-Flash-Next drafter instead, so the MTP layer is not needed.
    "off" disables the drafter; prompt-lookup/suffix speculation still runs.
    """
    mode = (drafting or "auto").strip().lower()
    if mode not in DRAFTING_MODES:
        raise ValueError("drafting must be one of " + ", ".join(DRAFTING_MODES))
    if mode == "off":
        return ["--mtp", "off", "--drafting", "off"]
    if mode == "dflash2":
        return ["--mtp", "off", "--drafting", "dflash2"]
    return ["--mtp", "on", "--drafting", mode]


def mtp_layer_path(data_dir: Path) -> Path | None:
    """The fetched MTP draft layer (data_dir/mtp/rt), when it is complete."""
    layer = Path(data_dir) / "mtp" / "rt"
    return layer if (layer / "experts.bin").is_file() else None


def drafting_status(data_dir: Path, drafting: str = "auto") -> dict[str, Any]:
    """Report whether the requested drafter is ready before a long launch/upload."""
    mode = (drafting or "auto").strip().lower()
    if mode not in DRAFTING_MODES:
        raise ValueError("drafting must be one of " + ", ".join(DRAFTING_MODES))
    layer = mtp_layer_path(data_dir)
    needs_mtp = mode in {"auto", "mtp"}
    ready = bool(layer) if needs_mtp else True
    status = {
        "drafting": mode,
        "needs_mtp_layer": needs_mtp,
        "mtp_layer": str(layer) if layer else None,
        "ready": ready,
        "mtp_fetch_gb": 0.0 if ready or not needs_mtp else MTP_FETCH_GB,
    }
    if needs_mtp and not layer:
        log(f"Draft layer not cached yet: the first launch downloads about {MTP_FETCH_GB} GB. "
            "Keep it under MODEL_DATA_DIR so the private model cache includes it.")
    elif needs_mtp:
        log("Draft layer found at " + str(layer) + "; it will be reused from the cache.")
    else:
        log("Draft layer not used by drafting=" + mode + ".")
    return status


def stop_strata() -> None:
    """Stop a running Strata server without failing when none is running."""
    run_command(["strata", "launch", "--stop"], check=False, timeout=180)


SESSION_MARKER = "STRATA_SESSION="
SHUTDOWN_MARKER = "STRATA_SHUTDOWN="
API_BASE = "http://127.0.0.1:8080/v1"


def session_identity() -> dict[str, Any]:
    """Identify the Kaggle kernel session that currently holds the GPU.

    Kaggle names the container "<kernel>-<session>-webtier", so the number in the
    middle is the session a launcher must cancel. Knowing it means a stop request
    cannot hit a newer run that started after this one ended.
    """
    container = os.environ.get("KAGGLE_CONTAINER_NAME", "")
    match = re.search(r"-(\d+)-webtier$", container)
    return {"session_id": int(match.group(1)) if match else 0, "container": container}


def announce_session(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Print one machine-readable session record so a local launcher can find it."""
    info = session_identity()
    if extra:
        info.update(extra)
    print(SESSION_MARKER + json.dumps(info), flush=True)
    return info


def server_activity(key: str | None = None, base: str = API_BASE) -> dict[str, Any] | None:
    """Return the server's request activity, or None when the endpoint is unavailable."""
    import requests
    try:
        response = requests.get(base.rstrip("/") + "/status",
                                headers={"Authorization": "Bearer " + _api_key(key)}, timeout=10)
        if not response.ok:
            return None
        activity = response.json().get("activity")
    except Exception:
        return None
    return activity if isinstance(activity, dict) else None


def serve_until_idle(idle_seconds: int = 900, serve_seconds: int = 0,
                     poll_seconds: float = 10.0, key: str | None = None,
                     stop: Callable[[], None] | None = None) -> dict[str, Any]:
    """Supervise the server, then stop it once the session is no longer in use.

    Returning from this function is what lets the notebook finish, and a finished
    notebook releases the GPU allocation — so call it from the last cell. Stopping
    Strata first avoids leaving the engine holding VRAM while the session closes.

    idle_seconds: finish after this long without a request (0 disables the check).
    serve_seconds: hard limit for the whole wait (0 disables the limit).
    """
    if idle_seconds < 0 or serve_seconds < 0:
        raise ValueError("idle_seconds and serve_seconds must not be negative.")
    if idle_seconds == 0 and serve_seconds == 0:
        raise ValueError("Set idle_seconds, serve_seconds, or both; refusing to wait forever.")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive.")
    started = time.monotonic()
    deadline = started + serve_seconds if serve_seconds else None
    stopper = stop or stop_strata
    if server_activity(key) is None and deadline is None:
        stopper()
        raise RuntimeError(
            "The server does not report /v1/status activity, so an idle timeout cannot be "
            "detected. Set SERVE_MINUTES to bound the wait, or stop the session yourself.")
    idle_since = started
    signature: tuple[Any, Any] | None = None
    requests_served = 0
    reason = "serve_deadline"
    while True:
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            reason = "serve_deadline"
            break
        activity = server_activity(key)
        if activity is not None:
            current = (activity.get("requests"), activity.get("last_request_at"))
            if activity.get("in_flight") or (signature is not None and current != signature):
                idle_since = now
            signature = current
            requests_served = int(activity.get("requests") or 0)
            if idle_seconds and now - idle_since >= idle_seconds:
                reason = "idle_timeout"
                break
        pause = poll_seconds if deadline is None else min(poll_seconds, max(0.0, deadline - now))
        time.sleep(pause)
    elapsed = round(time.monotonic() - started, 1)
    stopper()
    summary = {"reason": reason, "elapsed_seconds": elapsed, "idle_seconds": idle_seconds,
               "requests_served": requests_served, **session_identity()}
    print(SHUTDOWN_MARKER + json.dumps(summary), flush=True)
    log("Stopping the session after " + reason.replace("_", " ") + " (" + str(elapsed) + "s).")
    return summary


CONTEXT_CHOICES = {"8k": 8192, "16k": 16384, "32k": 32768, "65k": 65536, "128k": 131072}
VISION_MODES = ("on", "off")
RESTORE_MODES = ("copy", "symlink")


def validate_config(family: str, variant: str, context: str, drafting: str = "auto",
                    vision: str = "off", restore_mode: str = "copy",
                    gpu_list: str = "") -> dict[str, Any]:
    """Check the notebook configuration before a long launch spends GPU time.

    Pure and side-effect free so it can be tested without Kaggle. Returns the
    normalized values, including the context size in tokens.
    """
    family = (family or "").strip().lower()
    variant = (variant or "").strip()
    if not family:
        raise ValueError("MODEL_FAMILY is empty; set the Strata family to run.")
    if not variant:
        raise ValueError("MODEL_VARIANT is empty; set the Strata model variant to run.")
    context_key = (context or "").strip().lower()
    if context_key not in CONTEXT_CHOICES:
        raise ValueError("CONTEXT must be one of " + ", ".join(CONTEXT_CHOICES))
    drafting = (drafting or "auto").strip().lower()
    if drafting not in DRAFTING_MODES:
        raise ValueError("DRAFTING must be one of " + ", ".join(DRAFTING_MODES))
    vision = (vision or "off").strip().lower()
    if vision not in VISION_MODES:
        raise ValueError("VISION must be 'on' or 'off'.")
    restore_mode = (restore_mode or "copy").strip().lower()
    if restore_mode not in RESTORE_MODES:
        raise ValueError("MODEL_RESTORE_MODE must be 'copy' or 'symlink'.")
    gpu_list = (gpu_list or "").strip()
    if gpu_list and not re.fullmatch(r"\d+(,\d+)*", gpu_list):
        raise ValueError("GPU_IDS must be blank or a comma-separated list such as 0,1.")
    return {"family": family, "variant": variant, "context": context_key,
            "context_tokens": CONTEXT_CHOICES[context_key], "drafting": drafting,
            "vision": vision, "restore_mode": restore_mode, "gpu_ids": gpu_list}


def environment_report(require_cuda: bool = True) -> dict[str, Any]:
    """Collect hardware facts and fail early when the session cannot run the model."""
    report: dict[str, Any] = {"gpus": [], "ram_gb": None, "disk_free_gb": None,
                              "cpu_count": os.cpu_count()}
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap,driver_version",
             "--format=csv,noheader,nounits"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        for line in (result.stdout or "").strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                report["gpus"].append({"index": int(parts[0]), "name": parts[1],
                                       "memory_mib": int(float(parts[2])),
                                       "compute_capability": float(parts[3]),
                                       "driver": parts[4]})
            except ValueError:
                continue
    if require_cuda and not report["gpus"]:
        raise RuntimeError("No NVIDIA GPU detected. Select a GPU accelerator in Kaggle settings.")
    slow = [g for g in report["gpus"] if g["compute_capability"] < 7.0]
    if slow:
        raise RuntimeError("Compute capability below 7.0 is not supported: "
                           + ", ".join(g["name"] for g in slow))
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        total = next(line for line in meminfo if line.startswith("MemTotal:"))
        report["ram_gb"] = round(int(total.split()[1]) / (1024 ** 2), 1)
    except (OSError, StopIteration, ValueError, IndexError):
        pass
    try:
        anchor = WORKING if WORKING.is_dir() else Path("/")
        report["disk_free_gb"] = round(shutil.disk_usage(anchor).free / (1024 ** 3), 1)
    except OSError:
        pass
    log("GPUs: " + (", ".join(g["name"] + " (" + str(g["memory_mib"]) + " MiB)" for g in report["gpus"])
                    or "none") + "; RAM: " + str(report["ram_gb"]) + " GiB; free disk: "
        + str(report["disk_free_gb"]) + " GiB")
    return report


def _preset_args(preset: str | None = None, save_preset: str | None = None) -> list[str]:
    """Map notebook preset choices to Strata launch flags.

    A preset keeps tuning, engine settings and context between sessions. Flags
    next to --preset change that start only, so the notebook's family/model and
    data directory still win over whatever the preset remembers.
    """
    args: list[str] = []
    name = (preset or "").strip()
    if name:
        args += ["--preset", name]
    saved = (save_preset or "").strip()
    if saved:
        args += ["--save-preset", saved]
    return args


def list_presets() -> str:
    """Print the presets Strata has saved for this machine."""
    result = run_command(["strata", "launch", "--list-presets"], check=False, timeout=120)
    return result.stdout or ""


#: Flattened file prefix used by public mirrors of a Strata data folder.
DRAFT_LAYER_PREFIX = "mtp_rt__"
#: Sizes of the packed draft layer in the build this helper was written against.
#: A different Strata version can pack these differently, so a mismatch warns
#: instead of failing: the engine checks the layer itself at start.
KNOWN_DRAFT_LAYER_SIZES = {
    "experts.bin": 707_788_800,
    "dense.bin": 116_099_072,
    "dense.txt": 1_851,
    "draft_vocab.bin": 425_196,
}
#: The engine refuses a layer whose experts file is missing or truncated.
MIN_DRAFT_LAYER_BYTES = 64 * 1024 * 1024


def _dataset_dirs(owner: str, slug: str) -> list[Path]:
    """Every place an attached Kaggle dataset for owner/slug may be mounted."""
    candidates: list[Path] = []
    for root in _dataset_roots():
        for path in (root / owner / slug, root / slug):
            if path.is_dir() and path not in candidates:
                candidates.append(path)
        if root.is_dir():
            for path in sorted(root.rglob(slug)):
                if path.is_dir() and path not in candidates:
                    candidates.append(path)
    return candidates


def _draft_layer_entries(dataset_root: Path) -> dict[str, Path]:
    """Map a dataset's draft-layer files to their names inside mtp/rt/.

    Accepts a real ``mtp/rt/`` folder as well as the flattened
    ``mtp_rt__<name>`` layout some public mirrors use.
    """
    found: dict[str, Path] = {}
    real = dataset_root / "mtp" / "rt"
    if real.is_dir():
        for path in sorted(real.iterdir()):
            if path.is_file():
                found[path.name] = path
    for path in sorted(dataset_root.rglob(DRAFT_LAYER_PREFIX + "*")):
        if path.is_file():
            found.setdefault(path.name[len(DRAFT_LAYER_PREFIX):], path)
    return found


def restore_draft_layer(owner: str, slug: str, data_dir: Path,
                        restore_mode: str = "copy", force: bool = False,
                        min_experts_bytes: int = MIN_DRAFT_LAYER_BYTES) -> bool:
    """Install the MTP draft layer (mtp/rt) from an attached public dataset.

    The MTP layer is packed from the original BF16 checkpoint, not from a
    quantization, so one copy serves every Qwen3.8-Flash-Next variant (Gyro,
    Unsloth, Q2_0, IQ2_XS...). Restoring it avoids the ~5.2 GB fetch Strata
    would otherwise repeat on every fresh Kaggle session.

    Nothing but the draft layer is taken from the dataset: the model weights,
    the pack and the tokenizer are variant-specific and must come from a
    matching source.
    """
    if restore_mode not in {"copy", "symlink"}:
        raise ValueError("restore_mode must be 'copy' or 'symlink'.")
    if not force:
        existing = mtp_layer_path(data_dir)
        if existing is not None:
            log("MTP draft layer already present at " + str(existing) + "; not restoring it again.")
            return False
    dataset_root: Path | None = None
    entries: dict[str, Path] = {}
    for candidate in _dataset_dirs(owner, slug):
        entries = _draft_layer_entries(candidate)
        if "experts.bin" in entries:
            dataset_root = candidate
            break
    if dataset_root is None:
        if not _dataset_dirs(owner, slug):
            log("Draft-layer dataset " + owner + "/" + slug +
                " is not attached; Strata will fetch the draft layer itself.")
            log_mount_state()
            return False
        raise RuntimeError(
            "No MTP draft layer found in " + owner + "/" + slug +
            ": expected mtp/rt/experts.bin or " + DRAFT_LAYER_PREFIX + "experts.bin."
        )

    sizes = {name: src.stat().st_size for name, src in entries.items()}
    if sizes["experts.bin"] < min_experts_bytes:
        raise RuntimeError(
            "The draft layer's experts.bin is only " + str(sizes["experts.bin"]) +
            " bytes; at least " + str(min_experts_bytes) + " bytes are required for a real MTP layer."
        )
    unexpected = {n: s for n, s in sizes.items() if KNOWN_DRAFT_LAYER_SIZES.get(n, s) != s}
    if unexpected:
        log("Note: the draft layer came from a different Strata build than this helper knows "
            + str(unexpected) + "; the engine will validate it at start.")

    target = Path(data_dir) / "mtp" / "rt"
    target.mkdir(parents=True, exist_ok=True)
    total = sum(sizes.values())
    log(f"Restoring the MTP draft layer ({len(entries)} files, {total / (1024**3):.2f} GiB) "
        f"from {owner}/{slug} using {restore_mode} mode.")
    for name, src in entries.items():
        dst = target / name
        size = sizes[name]
        if restore_mode == "symlink":
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink():
                try:
                    if dst.resolve(strict=True) == src.resolve(strict=True) and dst.stat().st_size == size:
                        continue
                except OSError:
                    pass
                dst.unlink()
            elif dst.exists():
                if dst.is_file() and dst.stat().st_size == size:
                    continue
                raise RuntimeError("Draft-layer target already exists with unexpected content: " + str(dst))
            dst.symlink_to(src.resolve())
        else:
            _copy_verified(src, dst)
    log("MTP draft layer installed at " + str(target) + ".")
    return True


def run_plan(family: str, variant: str, context: str, vision: str,
             gpu_list: str, data_dir: Path, drafting: str = "auto",
             preset: str | None = None) -> str:
    cmd = ["strata", "launch", "--plan", "--family", family, "--model", variant,
           *_model_source_args(data_dir, family, variant), "--context", context, "--vision", vision,
           *_drafting_args(drafting), *_preset_args(preset), "--tune", "skip", "--access", "cloudflare",
           "--port", "8080", "--gpu", gpu_list]
    result = run_streaming(cmd, timeout=300, check=False,
                           heartbeat_interval=15, heartbeat_label="Strata launch plan",
                           progress_paths=[data_dir, STRATA_ROOT / "components"])
    output = result.stdout or ""
    if result.returncode:
        raise RuntimeError("Strata plan failed. Choose a supported family/model variant.")
    return output


def launch_strata(family: str, variant: str, context: str, vision: str,
                  gpu_list: str, data_dir: Path, drafting: str = "auto",
                  access: str = "cloudflare", preset: str | None = None,
                  save_preset: str | None = None) -> str:
    cmd = ["strata", "launch", "--family", family, "--model", variant,
           *_model_source_args(data_dir, family, variant), "--context", context, "--vision", vision,
           *_drafting_args(drafting), *_preset_args(preset, save_preset), "--tune", "skip", "--access", access,
           "--port", "8080", "--gpu", gpu_list, "--yes", "--no-tui"]
    log("Launching Strata. First-time model/engine downloads may take a while.")
    result = run_streaming(cmd, timeout=3600, check=False,
                           heartbeat_interval=20, heartbeat_label="Strata model/engine launch",
                           progress_paths=[data_dir, STRATA_ROOT / "components"])
    output = result.stdout or ""
    if result.returncode:
        raise RuntimeError("Strata launch failed. Inspect the redacted output above.")
    return output


def _local_model_id() -> str:
    """The model ID the running local server advertises."""
    import requests
    key = _read_key()
    response = requests.get("http://127.0.0.1:8080/v1/models",
                            headers={"Authorization": "Bearer " + key}, timeout=30)
    if not response.ok:
        raise RuntimeError("Local /v1/models failed: HTTP " + str(response.status_code))
    rows = response.json().get("data", [])
    available = [x.get("id") for x in rows if isinstance(x, dict) and x.get("id")]
    if not available:
        raise RuntimeError("Strata did not advertise a model ID from /v1/models.")
    return available[0]


BENCH_PROMPT = (
    "Write a short Python function that returns the sum of the squares of the "
    "first n positive integers. Include a docstring."
)


def measure_generation(prompt: str = BENCH_PROMPT, max_tokens: int = 128,
                       timeout: int = 900) -> dict[str, Any]:
    """Send one fixed request to the local server and measure decode speed.

    Wall time includes the first token, so longer answers give a steadier
    number. This is a like-for-like comparison, not a benchmark of Strata.
    """
    import requests
    key = _read_key()
    payload = {
        "model": _local_model_id(),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "reasoning_effort": "none",
        "stream": False,
    }
    started = time.monotonic()
    response = requests.post(
        "http://127.0.0.1:8080/v1/chat/completions",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        json=payload, timeout=timeout,
    )
    elapsed = time.monotonic() - started
    if not response.ok:
        raise RuntimeError("Benchmark request failed: HTTP " + str(response.status_code) + ": "
                           + redact(response.text[:800]))
    usage = response.json().get("usage") or {}
    tokens = usage.get("completion_tokens")
    if not isinstance(tokens, int) or tokens <= 0:
        raise RuntimeError("The server did not report completion_tokens; cannot measure speed.")
    return {"tokens": tokens, "seconds": round(elapsed, 3),
            "tokens_per_second": round(tokens / elapsed, 2) if elapsed > 0 else 0.0}


def compare_drafting(family: str, variant: str, context: str, vision: str,
                     gpu_list: str, data_dir: Path,
                     modes: tuple[str, ...] = ("off", "auto"),
                     max_tokens: int = 128) -> dict[str, Any]:
    """Measure the same prompt with and without the drafter.

    Opt-in and slow: each mode reloads the model. No tunnel is created, and
    the server is stopped at the end so the normal launch cell starts clean.
    """
    results: dict[str, Any] = {}
    try:
        for mode in modes:
            log("Drafting A/B: stopping the server before the '" + mode + "' run.")
            stop_strata()
            launch_strata(family, variant, context, vision, gpu_list, data_dir,
                          drafting=mode, access="local")
            wait_health(timeout_seconds=1800)
            measure_generation("Reply with one short sentence.", max_tokens=8)
            results[mode] = measure_generation(max_tokens=max_tokens)
            log("Drafting " + mode + ": " + str(results[mode]["tokens_per_second"]) + " tok/s")
    finally:
        stop_strata()
    off = results.get("off", {}).get("tokens_per_second")
    auto = results.get("auto", {}).get("tokens_per_second")
    if off and auto:
        results["auto_vs_off_percent"] = round((auto / off - 1) * 100, 1)
    return results


QUALITY_SIZES = (128, 512, 2048)


def _number(value: Any) -> float | None:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def discover_model(base: str = API_BASE, key: str | None = None) -> str:
    """The model ID the server advertises at /v1/models."""
    import requests
    response = requests.get(base.rstrip("/") + "/models",
                            headers={"Authorization": "Bearer " + _api_key(key)}, timeout=30)
    if not response.ok:
        raise RuntimeError("GET /v1/models failed: HTTP " + str(response.status_code))
    rows = [row for row in response.json().get("data", [])
            if isinstance(row, dict) and row.get("id")]
    if not rows:
        raise RuntimeError("Strata advertised no model ID from /v1/models.")
    return rows[0]["id"]


def stream_chat(messages: list[dict[str, Any]], *, model: str | None = None,
                max_tokens: int = 128, temperature: float = 0.0,
                tools: list[dict[str, Any]] | None = None,
                key: str | None = None, base: str = API_BASE,
                timeout: int = 900) -> dict[str, Any]:
    """Send one streaming chat request and collect the whole answer.

    Streamed responses are what agent clients use, so measuring them exercises
    the same path: token timings, tool calls split across deltas, and TTFT.
    """
    import requests
    key_value = _api_key(key)
    base_url = base.rstrip("/")
    payload: dict[str, Any] = {
        "model": model or discover_model(base_url, key_value),
        "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
        "reasoning_effort": "none", "stream": True,
    }
    if tools:
        payload["tools"] = tools
    started = time.monotonic()
    first_token: float | None = None
    content: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    finish_reason = None
    done = False
    with requests.post(base_url + "/chat/completions",
                       headers={"Authorization": "Bearer " + key_value,
                                "Content-Type": "application/json"},
                       json=payload, stream=True, timeout=timeout) as response:
        if not response.ok:
            raise RuntimeError("Chat request failed: HTTP " + str(response.status_code) + ": "
                               + redact(response.text[:800]))
        for raw in response.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError("The server returned an error while streaming: "
                                   + redact(str(event["error"])[:800]))
            if event.get("usage"):
                usage = event["usage"]
            if event.get("timings"):
                timings = event["timings"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if first_token is None and any(delta.get(name) for name in
                                               ("content", "reasoning_content", "tool_calls")):
                    first_token = time.monotonic() - started
                if delta.get("content"):
                    content.append(delta["content"])
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                for part in delta.get("tool_calls") or []:
                    index = int(part.get("index") or 0)
                    call = calls.setdefault(index, {"id": "", "type": "function",
                                                    "function": {"name": "", "arguments": ""}})
                    call["id"] += part.get("id") or ""
                    function = part.get("function") or {}
                    call["function"]["name"] += function.get("name") or ""
                    call["function"]["arguments"] += function.get("arguments") or ""
    if finish_reason is None and not done:
        raise RuntimeError("The streamed response ended before completion.")
    message: dict[str, Any] = {"role": "assistant", "content": "".join(content) or None}
    if calls:
        message["tool_calls"] = [calls[index] for index in sorted(calls)]
    return {"message": message, "tool_calls": message.get("tool_calls", []),
            "usage": usage, "timings": timings, "finish_reason": finish_reason,
            "ttft_s": round(first_token, 3) if first_token is not None else None,
            "elapsed_s": round(time.monotonic() - started, 3)}


def benchmark_api(words: tuple[int, ...] = QUALITY_SIZES, repeats: int = 1,
                  max_tokens: int = 128) -> dict[str, Any]:
    """Measure prompt processing and decode speed through the real chat API.

    Rates come from the server's own timings when it reports them, and fall back
    to token counts divided by wall time otherwise. Prompts are synthetic and the
    answers are short, so treat this as a like-for-like comparison, not a
    benchmark of the engine.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if any(size < 1 for size in words):
        raise ValueError("word counts must be positive.")
    runs: list[dict[str, Any]] = []
    for size in words:
        for _ in range(repeats):
            prompt = ("Request " + str(time.time_ns()) + ".\n" + " ".join(["item"] * size)
                      + "\nWrite a Python function that counts the listed items. Explain briefly.")
            result = stream_chat([{"role": "user", "content": prompt}], max_tokens=max_tokens)
            usage, timings = result["usage"], result["timings"]
            prompt_tokens = usage.get("prompt_tokens")
            generated = usage.get("completion_tokens")
            row: dict[str, Any] = {
                "prompt_words": size,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated,
                "ttft_s": result["ttft_s"],
                "elapsed_s": result["elapsed_s"],
                "prompt_tokens_per_second": _number(timings.get("prompt_per_second")),
                "generation_tokens_per_second": _number(timings.get("predicted_per_second")),
            }
            if row["prompt_tokens_per_second"] is None and prompt_tokens:
                prefill = result["ttft_s"] or result["elapsed_s"]
                if prefill:
                    row["prompt_tokens_per_second"] = round(prompt_tokens / prefill, 2)
            if row["generation_tokens_per_second"] is None and generated:
                decode = result["elapsed_s"] - (result["ttft_s"] or 0)
                if decode > 0:
                    row["generation_tokens_per_second"] = round(generated / decode, 2)
            runs.append(row)
            log(str(size) + " words: prompt " + str(row["prompt_tokens_per_second"]) + " tok/s, "
                "decode " + str(row["generation_tokens_per_second"]) + " tok/s, TTFT "
                + str(row["ttft_s"]) + "s")
    return {"runs": runs, "summary": _benchmark_summary(runs)}


def _benchmark_summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for size in sorted({row["prompt_words"] for row in runs}):
        group = [row for row in runs if row["prompt_words"] == size]
        entry: dict[str, Any] = {"prompt_words": size, "samples": len(group)}
        for field in ("prompt_tokens_per_second", "generation_tokens_per_second", "ttft_s"):
            values = [row[field] for row in group if isinstance(row.get(field), (int, float))]
            entry[field] = round(sum(values) / len(values), 2) if values else None
        summary.append(entry)
    return summary


SMOKE_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read the local file answer.py.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Replace the local file answer.py with the supplied Python source.",
        "parameters": {"type": "object", "properties": {"content": {"type": "string"}},
                       "required": ["content"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "run_test", "description": "Run the local command that checks answer.py.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
]


def agent_smoke(rounds: int = 12, repeats: int = 1) -> dict[str, Any]:
    """Check that the model can read, edit and test a local file through tools.

    Adapted from Ninnix/sok (MIT). This is the quality gate that matters for an
    agent: a model can generate fluent tokens and still fail to emit a valid tool
    call, edit before reading, or claim success without running the test.
    """
    if rounds < 1 or repeats < 1:
        raise ValueError("rounds and repeats must be at least 1.")
    runs: list[dict[str, Any]] = []
    for _ in range(repeats):
        started = time.monotonic()
        runs.append(_agent_task(rounds))
        runs[-1]["elapsed_s"] = round(time.monotonic() - started, 2)
        log("Agent smoke: completed in " + str(runs[-1]["turns"]) + " turns, "
            + str(runs[-1]["elapsed_s"]) + "s")
    return {"runs": runs, "ok": all(run["ok"] for run in runs)}


def _agent_task(rounds: int) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a coding agent. Use the available local tools to "
                                      "complete the task. Read the file before editing it, then run "
                                      "the test. Do not claim success until the test passes."},
        {"role": "user", "content": "Fix answer.py so that answer() returns 42. Read it, edit it, "
                                    "run the test, then briefly report the result."},
    ]
    seen: list[str] = []
    passed = False
    with tempfile.TemporaryDirectory(prefix="strata-agent-") as folder:
        path = Path(folder) / "answer.py"
        path.write_text("def answer():\n    return 41\n", encoding="utf-8")
        for turn in range(rounds):
            result = stream_chat(messages, max_tokens=512, tools=SMOKE_TOOLS)
            messages.append(result["message"])
            calls = result["tool_calls"]
            if not calls:
                if passed and all(name in seen for name in ("read_file", "write_file", "run_test")):
                    return {"ok": True, "turns": turn + 1, "local_tools": seen}
                messages.append({"role": "user",
                                 "content": "Use read_file, write_file and run_test before finishing."})
                continue
            for call in calls:
                name = (call.get("function") or {}).get("name")
                try:
                    arguments = json.loads((call.get("function") or {}).get("arguments") or "{}")
                except ValueError:
                    raise RuntimeError("The model emitted arguments that are not valid JSON.")
                if name == "read_file" and not arguments:
                    output = path.read_text(encoding="utf-8")
                elif name == "write_file" and set(arguments) == {"content"} and "read_file" in seen:
                    path.write_text(str(arguments["content"]), encoding="utf-8")
                    passed = False
                    output = "answer.py updated"
                elif name == "run_test" and not arguments:
                    passed, output = _run_smoke_test(folder, path)
                else:
                    raise RuntimeError("Invalid tool call, or an edit attempted before reading the file.")
                seen.append(name)
                messages.append({"role": "tool", "tool_call_id": call.get("id") or "",
                                 "content": output})
    raise RuntimeError("The agent did not finish the task within " + str(rounds) + " turns.")


def _run_smoke_test(folder: str, path: Path) -> tuple[bool, str]:
    """Run the fixed check with credentials stripped from the child environment."""
    env = {name: value for name, value in os.environ.items()
           if not any(word in name.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
    result = subprocess.run(
        [sys.executable, "-I", "-c",
         "import runpy; assert runpy.run_path('answer.py')['answer']() == 42; print('PASS')"],
        cwd=folder, env=env, capture_output=True, text=True, timeout=10, check=False)
    ok = result.returncode == 0 and path.read_text(encoding="utf-8").find("42") >= 0
    return ok, json.dumps({"exit_code": result.returncode, "stdout": result.stdout,
                           "stderr": result.stderr})


def run_quality_gate(words: tuple[int, ...] = QUALITY_SIZES, repeats: int = 1,
                     agent_repeats: int = 1) -> dict[str, Any]:
    """Benchmark the API and check tool calling on the running server."""
    log("Quality gate: benchmarking the chat API…")
    benchmark = benchmark_api(words, repeats)
    log("Quality gate: checking local tool calling…")
    smoke = agent_smoke(repeats=agent_repeats)
    for row in benchmark["summary"]:
        print(str(row["prompt_words"]).rjust(5) + " words  prompt "
              + str(row["prompt_tokens_per_second"]).rjust(7) + " tok/s  decode "
              + str(row["generation_tokens_per_second"]).rjust(6) + " tok/s  TTFT "
              + str(row["ttft_s"]).rjust(5) + "s  (" + str(row["samples"]) + " samples)",
              flush=True)
    print("Tool calling: " + ("PASS" if smoke["ok"] else "FAIL")
          + " in " + str(smoke["runs"][0]["turns"]) + " turns "
          + "(" + str(smoke["runs"][0]["elapsed_s"]) + "s)", flush=True)
    return {"benchmark": benchmark, "agent_smoke": smoke}


def install_cloudflared_if_missing() -> None:
    os.environ["PATH"] = "/root/.local/bin:" + os.environ.get("PATH", "")
    if shutil.which("cloudflared"):
        return
    cached = STRATA_ROOT / "tools" / "cloudflared"
    CLOUDFLARED_BIN.parent.mkdir(parents=True, exist_ok=True)
    if cached.is_file():
        cached.chmod(0o755)
        CLOUDFLARED_BIN.unlink(missing_ok=True)
        CLOUDFLARED_BIN.symlink_to(cached)
        return
    machine = platform.machine().lower()
    asset = {"x86_64": "cloudflared-linux-amd64", "amd64": "cloudflared-linux-amd64",
             "aarch64": "cloudflared-linux-arm64", "arm64": "cloudflared-linux-arm64"}.get(machine)
    if not asset:
        raise RuntimeError("Unsupported cloudflared architecture: " + machine)
    run_command(["curl", "-fL", "--retry", "3",
                 "https://github.com/cloudflare/cloudflared/releases/latest/download/" + asset,
                 "-o", str(CLOUDFLARED_BIN)], timeout=600)
    CLOUDFLARED_BIN.chmod(0o755)
    run_command([str(CLOUDFLARED_BIN), "--version"], timeout=60)


def _read_key() -> str:
    result = subprocess.run(["strata", "launch", "--show-key"], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=60, check=True)
    match = re.search(r"sk-strata-[A-Za-z0-9._~-]+", result.stdout or "")
    if not match:
        raise RuntimeError("Could not safely identify the Strata API key.")
    return match.group(0)


def _api_key(key: str | None = None) -> str:
    """Use a caller-supplied key when given, otherwise ask Strata for the running one."""
    return key or _read_key()


def wait_health(timeout_seconds: int = 900) -> dict[str, Any]:
    import requests
    deadline, last = time.time() + timeout_seconds, "no response"
    while time.time() < deadline:
        try:
            response = requests.get("http://127.0.0.1:8080/health", timeout=5)
            if response.ok:
                body = response.json()
                if body.get("loaded") is True:
                    if not body.get("api_key"):
                        raise RuntimeError("Strata has no API-key protection; refusing a public tunnel.")
                    log("Strata health is OK; model=" + str(body.get("model")))
                    return body
                last = "model not loaded yet"
            else:
                last = "HTTP " + str(response.status_code)
        except RuntimeError:
            raise
        except Exception as exc:
            last = str(exc)
        time.sleep(5)
    raise TimeoutError("Strata did not become healthy; last result: " + last)


def _extract_tunnel_url(text: str) -> str | None:
    match = re.search(r"https://[A-Za-z0-9-]+\.trycloudflare\.com", text)
    return match.group(0) if match else None


def start_quick_tunnel(existing_url: str | None = None, log_path: Path | None = None) -> str:
    if existing_url:
        log("Using HTTPS URL printed by Strata.")
        return existing_url
    install_cloudflared_if_missing()
    log_path = log_path or (WORKING / "cloudflared.log")
    if log_path.exists():
        previous = _extract_tunnel_url(log_path.read_text(encoding="utf-8", errors="replace"))
        try:
            import requests
            if previous and requests.get(previous + "/health", timeout=8).status_code in (200, 401, 403):
                log("Reusing URL from the existing Cloudflare log.")
                return previous
        except Exception:
            pass
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            [str(CLOUDFLARED_BIN), "tunnel", "--no-autoupdate",
             "--url", "http://127.0.0.1:8080"],
            stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, text=True,
        )
    for _ in range(60):
        time.sleep(2)
        if process.poll() is not None:
            raise RuntimeError("cloudflared exited early; inspect " + str(log_path))
        url = _extract_tunnel_url(log_path.read_text(encoding="utf-8", errors="replace"))
        if url:
            log("Cloudflare Quick Tunnel started. URL: " + url)
            return url
    raise RuntimeError("No trycloudflare.com URL appeared. Inspect " + str(log_path))


def verify_public_endpoint(public_url: str, expected_model: str | None = None) -> dict[str, Any]:
    import requests
    base, key = public_url.rstrip("/"), _read_key()
    unauth = requests.get(base + "/v1/models", timeout=20)
    if unauth.status_code not in (401, 403):
        raise RuntimeError("Unauthenticated /v1/models returned HTTP " + str(unauth.status_code) +
                           "; expected 401/403. Stop the tunnel and secure the server.")
    headers = {"Authorization": "Bearer " + key}
    response = requests.get(base + "/v1/models", headers=headers, timeout=30)
    if not response.ok:
        raise RuntimeError("Authenticated /v1/models failed: HTTP " + str(response.status_code) +
                           ": " + redact(response.text[:1000]))
    rows = response.json().get("data", [])
    available = [x.get("id") for x in rows if isinstance(x, dict) and x.get("id")]
    model_id = expected_model if expected_model in available else (available[0] if available else expected_model)
    if not model_id:
        raise RuntimeError("Strata did not advertise a model ID from /v1/models.")
    log("Unauthenticated request rejected with HTTP " + str(unauth.status_code) + ".")
    log("Sending short authenticated generation test…")
    chat = requests.post(
        base + "/v1/chat/completions", headers={**headers, "Content-Type": "application/json"},
        json={"model": model_id, "messages": [{"role": "user", "content": "Reply with exactly: Strata connection working."}],
              "max_tokens": 32, "reasoning_effort": "none", "stream": False},
        timeout=240,
    )
    if not chat.ok:
        raise RuntimeError("Authenticated chat test failed: HTTP " + str(chat.status_code) + ": " + redact(chat.text[:1500]))
    body = chat.json()
    choices = body.get("choices", [])
    answer = choices[0].get("message", {}).get("content", "") if choices else ""
    if isinstance(answer, list):
        answer = "".join(part.get("text", "") for part in answer if isinstance(part, dict))
    print("Authenticated test response: " + str(answer))
    print("OpenCode base URL: " + base + "/v1")
    print("API key was not printed. Retrieve it privately using: strata launch --show-key")
    return {"model_id": model_id, "url": base, "unauthenticated_status": unauth.status_code,
            "authenticated_status": chat.status_code}


CONNECTION_MARKER = "STRATA_CONNECTION="


def publish_connection(public_url: str, *, client_public_key: str | None = None,
                       api_key: str | None = None,
                       model: str | None = None) -> dict[str, Any]:
    """Print a machine-readable connection record for a local launcher.

    With a client public key the API key is encrypted with RSA-OAEP/SHA-256, so
    the key never appears in the notebook output or in Kaggle Secrets. Without
    one, only the URL is published and the key stays where Strata keeps it.
    """
    connection: dict[str, Any] = {"base_url": public_url.rstrip("/"),
                                  "model": model or _local_model_id(),
                                  **session_identity()}
    if client_public_key:
        if not client_public_key.strip().startswith("-----BEGIN PUBLIC KEY-----"):
            raise ValueError("client_public_key must be a PEM RSA public key.")
        openssl = shutil.which("openssl")
        if not openssl:
            raise RuntimeError("OpenSSL is required to encrypt the API key for the client.")
        with tempfile.NamedTemporaryFile("w", suffix=".pem", delete=False) as handle:
            handle.write(client_public_key)
            pem = Path(handle.name)
        try:
            result = subprocess.run(
                [openssl, "pkeyutl", "-encrypt", "-pubin", "-inkey", str(pem),
                 "-pkeyopt", "rsa_padding_mode:oaep",
                 "-pkeyopt", "rsa_oaep_md:sha256",
                 "-pkeyopt", "rsa_mgf1_md:sha256"],
                input=_api_key(api_key).encode(), stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=False)
        finally:
            pem.unlink(missing_ok=True)
        if result.returncode != 0:
            raise RuntimeError("OpenSSL could not encrypt the API key: "
                               + redact(result.stderr.decode("utf-8", "replace")[:400]))
        connection["encrypted_api_key"] = base64.b64encode(result.stdout).decode()
    else:
        connection["api_key"] = "not published; retrieve it with: strata launch --show-key"
    print(CONNECTION_MARKER + json.dumps(connection), flush=True)
    return connection


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_dataset_metadata(stage: Path, username: str, slug: str, title: str, description: str) -> None:
    metadata = {"title": title, "id": username + "/" + slug,
                "licenses": [{"name": "other"}],
                "description": description + " Components/model files retain upstream terms. Private by default."}
    (stage / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _upload_create_or_version(stage: Path, username: str, slug: str,
                              env: dict[str, str], message: str) -> None:
    cmd = ["kaggle", "datasets", "create", "-p", str(stage), "--dir-mode", "skip"]
    log("Uploading dataset files to Kaggle. Progress output will stream below; large uploads may take several minutes.")
    result = run_streaming(
        cmd, timeout=7200, env=env, check=False,
        heartbeat_interval=30, heartbeat_label="Kaggle dataset upload",
    )
    output = redact(result.stdout or "")
    lower = output.lower()

    # Some Kaggle CLI versions can print a slug-conflict error but still exit zero.
    slug_conflict = (
        "already in use by a dataset" in lower
        or ("requested slug" in lower and "already" in lower)
        or "duplicate slug" in lower
        or "slug is already taken" in lower
        or "dataset already exists" in lower
    )
    if result.returncode == 0 and not slug_conflict:
        log("Upload submitted for " + username + "/" + slug + ". Check Kaggle's processing status.")
        return

    if not slug_conflict:
        raise RuntimeError(
            "Kaggle dataset creation failed (exit " + str(result.returncode) + "): "
            + output[-4000:]
        )

    log("Dataset slug already exists; uploading a new private dataset version instead.")
    version = run_streaming(
        ["kaggle", "datasets", "version", "-p", str(stage), "-m", message, "--dir-mode", "skip"],
        timeout=7200, env=env, check=False,
        heartbeat_interval=30, heartbeat_label="Kaggle dataset version upload",
    )
    version_output = redact(version.stdout or "")
    if version.returncode:
        raise RuntimeError("Kaggle dataset version failed: " + version_output[-4000:])
    version_lower = version_output.lower()
    if "error" in version_lower and not any(
        success in version_lower for success in ("upload successful", "version created", "version updated")
    ):
        raise RuntimeError("Kaggle reported an error while creating the dataset version: " + version_output[-4000:])
    log("New version upload submitted for " + username + "/" + slug + ". Wait for Kaggle processing to finish.")

def _clean_stage(stage: Path) -> None:
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)


def upload_model_cache(data_dir: Path, username: str, slug: str, family: str, variant: str,
                       env: dict[str, str], license_confirmed: bool, force: bool = False,
                       require_mtp: bool = False) -> bool:
    if not license_confirmed:
        log("Skipped model upload. Confirm the model terms and enable the explicit license flag first.")
        return False
    layer = mtp_layer_path(data_dir)
    if require_mtp and not layer:
        raise RuntimeError(
            "Drafting is enabled but the MTP draft layer is not under " + str(Path(data_dir) / "mtp" / "rt")
            + ". Launch once with drafting='auto' (about " + str(MTP_FETCH_GB) + " GB) before uploading, "
            "or set DRAFTING='off' to cache the model without it."
        )
    attached = find_attached_dataset(username, slug, "manifest.json")
    if attached and not force:
        try:
            old = json.loads((attached / "manifest.json").read_text(encoding="utf-8"))
            old_entries = old.get("files", [])
            cache_has_mtp = any(
                str(PurePosixPath(item.get("relative_path", ""))).startswith("mtp/")
                for item in old_entries
            )
            if (old.get("format") == "devzonex.strata-model-cache.v1"
                    and old.get("model") == {"family": family, "variant": variant}
                    and (cache_has_mtp or not require_mtp)):
                cache_matches = bool(old_entries)
                for item in old_entries:
                    rel = PurePosixPath(item["relative_path"])
                    if rel.is_absolute() or ".." in rel.parts:
                        cache_matches = False
                        break
                    local_file = data_dir.joinpath(*rel.parts)
                    if not local_file.is_file() or local_file.stat().st_size != int(item["size"]):
                        cache_matches = False
                        break
                if cache_matches:
                    log("Attached private model cache already matches the selected model; skipping duplicate upload.")
                    return False
        except Exception:
            pass

    allowed_roots = {"models", "packs", "mtp"}
    files = sorted(
        p for p in data_dir.rglob("*")
        if p.is_file() and not p.name.endswith(".part")
        and p.relative_to(data_dir).parts and p.relative_to(data_dir).parts[0] in allowed_roots
    )
    if not files:
        raise RuntimeError("No model/packs/mtp files found under " + str(data_dir) + ".")
    if layer:
        log("Including the MTP draft layer in the cache (" + str(layer) + ").")
    else:
        log("No MTP draft layer to cache; the next session will fetch it again if drafting is enabled.")
    sizes = {p.relative_to(data_dir).as_posix(): p.stat().st_size for p in files}
    stage = WORKING / "strata-model-dataset-stage"
    _clean_stage(stage)
    entries = []
    for number, path in enumerate(files, 1):
        relative = path.relative_to(data_dir).as_posix()
        name = f"{number:04d}__" + re.sub(r"[^A-Za-z0-9._-]+", "__", relative.replace("/", "__"))
        # Resolve input-mount symlinks so the Kaggle CLI sees the underlying file.
        (stage / name).symlink_to(path.resolve())
        entries.append({"dataset_name": name, "relative_path": relative, "size": sizes[relative]})
    manifest = {"format": "devzonex.strata-model-cache.v1",
                "model": {"family": family, "variant": variant}, "files": entries}
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_dataset_metadata(stage, username, slug, "DevZoneX Strata Model Cache",
                            f"Personal private cache for Strata-supported {family}/{variant} model data.")
    total = sum(item["size"] for item in entries)
    log(f"Preparing {len(entries)} model files ({total / (1024**3):.2f} GiB) for private upload.")
    _upload_create_or_version(stage, username, slug, env, f"Update Strata model cache: {family}/{variant}")
    return True



def upload_runtime_cache(username: str, slug: str, env: dict[str, str], force: bool = False) -> bool:
    if not (STRATA_ROOT / "terminal").exists() or not (STRATA_ROOT / "components").exists():
        log("Runtime upload skipped: terminal/components directory missing.")
        return False
    attached = find_attached_dataset(username, slug, "strata-runtime-cache.bin")
    version = subprocess.run(["strata", "--version"], text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=60, check=False).stdout.strip()
    hw = hardware_summary()
    engine_dirs = sorted(p.name for p in (STRATA_ROOT / "components" / "engine").glob("*") if p.is_dir())
    cloudflared_cmd = shutil.which("cloudflared")
    cloudflared_version = None
    if cloudflared_cmd:
        v = subprocess.run([cloudflared_cmd, "--version"], text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=60, check=False)
        cloudflared_version = (v.stdout or "").strip()
    if attached and not force:
        try:
            old = json.loads((attached / "runtime-manifest.json").read_text(encoding="utf-8"))
            if (old.get("strata_version") == version and old.get("machine") == hw["machine"]
                    and old.get("gpu_names") == hw["gpu_names"] and old.get("engine_dirs", []) == engine_dirs
                    and old.get("cloudflared_version") == cloudflared_version):
                log("Runtime dataset matches installed terminal/engine; skipping upload.")
                return False
        except Exception:
            pass
    stage = WORKING / "strata-runtime-dataset-stage"
    _clean_stage(stage)
    archive = stage / "strata-runtime-cache.bin"
    log("Packaging Strata terminal and engine cache (model data remains separate)…")
    with tarfile.open(archive, "w:") as tf:
        tf.add(STRATA_ROOT / "terminal", arcname="terminal", recursive=True)
        tf.add(STRATA_ROOT / "components", arcname="components", recursive=True)
        if cloudflared_cmd:
            tf.add(Path(cloudflared_cmd).resolve(), arcname="tools/cloudflared", recursive=False)
    (stage / "strata-runtime-cache.bin.sha256").write_text(
        _sha256(archive) + "  strata-runtime-cache.bin\n", encoding="utf-8")
    meta = {"format": "devzonex.strata-runtime-cache.v1", "strata_version": version,
            **hw, "engine_dirs": engine_dirs, "cloudflared_version": cloudflared_version,
            "contents": ["terminal", "components", "tools/cloudflared (if present)"]}
    (stage / "runtime-manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    _write_dataset_metadata(stage, username, slug, "DevZoneX Strata Runtime Cache",
                            "Private runtime cache for Strata terminal and engine components.")
    log(f"Runtime archive size: {archive.stat().st_size / (1024**3):.2f} GiB.")
    _upload_create_or_version(stage, username, slug, env, "Update Strata terminal and engine cache")
    return True


def opencode_config_snippet(public_url: str, model_id: str, context_tokens: int = 32768,
                            output_tokens: int = 8192,
                            provider_package: str = "@opencode/ai/providers/openai-compatible",
                            key_path: str = "~/.config/opencode/strata-key.txt") -> str:
    """Build the provider entry to merge into OpenCode's providers map.

    A T4 can take minutes to the first token and pause mid-answer while the
    engine pages experts, so the timeouts are 15 minutes rather than OpenCode's
    shorter defaults. `compatibility.reasoningField` keeps Strata's separate
    reasoning channel out of the answer text.
    """
    provider = {
        "name": "Strata Kaggle", "package": provider_package,
        "settings": {"baseURL": public_url.rstrip("/") + "/v1",
                     "apiKey": "{file:" + key_path + "}",
                     "encoding": "none", "headerTimeout": 900000, "chunkTimeout": 900000},
        "models": {model_id: {
            "name": model_id, "modelID": model_id,
            "capabilities": {"tools": True, "input": ["text"], "output": ["text"]},
            "compatibility": {"reasoningField": "reasoning_content"},
            "limit": {"context": int(context_tokens), "output": int(output_tokens)}}},
    }
    return json.dumps({"strata": provider}, indent=2)
