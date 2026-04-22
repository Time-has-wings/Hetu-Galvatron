# tests/conftest.py
"""Pytest hooks and fixtures. Ensures vendored ``megatron`` under ``galvatron/site_package`` is importable."""
import os
import sys
import json
import signal
import socket
import subprocess
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from typing import Dict, Callable, List, Tuple
import tempfile


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])

@pytest.fixture
def small_model_config():
    """Provide a small model config for testing"""
    return {
        "hidden_size": 128,
        "num_layers": 2,
        "num_attention_heads": 4,
        "seq_length": 32,
        "vocab_size": 1000,
    }

@pytest.fixture
def device():
    """Provide device for testing"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def seed():
    """Return a fixed seed for reproducibility"""
    return 42

def _terminate_process(p: subprocess.Popen, grace: float = 5.0) -> None:
    """Terminate a process (and its whole session/group), escalating to SIGKILL."""
    if p.poll() is not None:
        return
    try:
        if os.name == "posix":
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except ProcessLookupError:
                return
        else:
            p.terminate()
    except Exception:
        pass
    try:
        p.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                return
        else:
            p.kill()
    except Exception:
        pass
    try:
        p.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass


@pytest.fixture
def run_distributed():
    """Fixture that provides a robust distributed test runner.

    Spawns ``world_size`` subprocesses. If any rank exits non-zero (or the
    whole run exceeds ``timeout`` seconds), all remaining processes are
    terminated and the test is failed with the collected output of every
    rank.
    """
    def _run_distributed(
        func_name: str,
        world_size: int,
        args: Dict,
        script: str,
        timeout: float = 600.0,
        poll_interval: float = 0.5,
    ):
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need at least {world_size} GPUs, but got {torch.cuda.device_count()}")

        master_port = str(_pick_free_port())

        processes: List[subprocess.Popen] = []
        log_files: List[Tuple[tempfile._TemporaryFileWrapper, tempfile._TemporaryFileWrapper]] = []

        def _collect_outputs() -> str:
            parts = []
            for rank, p in enumerate(processes):
                stdout_f, stderr_f = log_files[rank]
                try:
                    stdout_f.flush(); stderr_f.flush()
                    stdout_f.seek(0); stderr_f.seek(0)
                    out = stdout_f.read().decode(errors="replace")
                    err = stderr_f.read().decode(errors="replace")
                except Exception as e:
                    out, err = "", f"<failed to read output: {e}>"
                rc = p.returncode if p.returncode is not None else "running"
                parts.append(
                    f"--- rank {rank} (exit={rc}) ---\n"
                    f"[stdout]\n{out}\n[stderr]\n{err}"
                )
            return "\n".join(parts)

        try:
            for rank in range(world_size):
                env = os.environ.copy()
                env["MASTER_ADDR"] = "127.0.0.1"
                env["MASTER_PORT"] = master_port
                env["WORLD_SIZE"] = str(world_size)
                env["RANK"] = str(rank)
                env["LOCAL_RANK"] = str(rank)

                stdout_f = tempfile.TemporaryFile(mode="w+b")
                stderr_f = tempfile.TemporaryFile(mode="w+b")
                log_files.append((stdout_f, stderr_f))

                cmd = [sys.executable, script, func_name, json.dumps(args)]
                p = subprocess.Popen(
                    cmd,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    env=env,
                    start_new_session=True,
                )
                processes.append(p)

            deadline = time.monotonic() + timeout
            failed_rank = None
            timed_out = False

            while True:
                all_done = True
                for rank, p in enumerate(processes):
                    rc = p.poll()
                    if rc is None:
                        all_done = False
                    elif rc != 0:
                        failed_rank = rank
                        break
                if failed_rank is not None or all_done:
                    break
                if time.monotonic() > deadline:
                    timed_out = True
                    break
                time.sleep(poll_interval)

            if failed_rank is not None or timed_out:
                for p in processes:
                    _terminate_process(p)

                details = _collect_outputs()
                if timed_out:
                    pytest.fail(
                        f"Distributed test timed out after {timeout:.1f}s\n{details}"
                    )
                else:
                    rc = processes[failed_rank].returncode
                    pytest.fail(
                        f"Distributed test failed: rank {failed_rank} exited with code {rc}\n{details}"
                    )
        finally:
            for p in processes:
                if p.poll() is None:
                    _terminate_process(p, grace=2.0)
            for stdout_f, stderr_f in log_files:
                for f in (stdout_f, stderr_f):
                    try:
                        f.close()
                    except Exception:
                        pass

    return _run_distributed

@pytest.fixture
def checkpoint_dir():
    with tempfile.TemporaryDirectory() as baseline_dir, \
         tempfile.TemporaryDirectory() as converted_dir:
        yield {
            "baseline": baseline_dir,
            "converted": converted_dir
        }

@pytest.fixture
def base_config_dirs(tmp_path: Path) -> Tuple[Path, Path, Path]:
    """Create and return config directories"""
    configs_dir = tmp_path / "configs"
    hardware_dir = tmp_path / "hardware_configs"
    output_dir = tmp_path / "output"
    return configs_dir, hardware_dir, output_dir

@pytest.fixture
def profiler_model_configs_dir(tmp_path: Path) -> Path:
    """Create and return profiler config directories"""
    configs_dir = tmp_path / "configs"
    os.makedirs(configs_dir, exist_ok=True)
    return configs_dir

@pytest.fixture
def profiler_hardware_configs_dir(tmp_path: Path) -> Path:
    """Create and return profiler config directories"""
    hardware_configs_dir = tmp_path / "hardware_configs"
    scripts_dir = tmp_path / "scripts"
    os.makedirs(hardware_configs_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)
    return tmp_path

@pytest.fixture
def base_log_dirs(tmp_path: Path) -> str:
    """Create and return log directories"""
    log_dir = tmp_path / "logs"
    os.makedirs(log_dir, exist_ok=True)
    return str(log_dir)

