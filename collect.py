#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def wait_port(host: str, port: int, process: subprocess.Popen, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"OpenWAM server exited with {process.returncode}")
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"OpenWAM server did not listen on {host}:{port}")


def run(command: list[str], *, cwd: Path, env: dict, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f"command failed ({code}); see {log}")


def manifest_hash(path: Path) -> str:
    return load(path)["manifest_hash"]


def policy_sample_seed(task: str, state: int, rollout: int) -> int:
    key = f"experiment2A:{task}:{state}:{rollout}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") & 0x7FFFFFFF


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--states", type=int)
    parser.add_argument("--rollouts", type=int)
    parser.add_argument("--tasks", help="comma-separated task subset")
    parser.add_argument("--methods", default="no_wm,wm")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load(args.config.resolve())
    states = args.states or int(config["states_per_task"])
    rollouts = args.rollouts or int(config["rollouts_per_state"])
    tasks = config["tasks"] if not args.tasks else [x.strip() for x in args.tasks.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)

    openwam = Path(config["paths"]["openwam_repo"])
    robotwin = Path(config["paths"]["robotwin_repo"])
    wrapper = openwam / "benchmarks/robotwin/eval_policy_wrapper.py"
    policy = openwam / "benchmarks/robotwin/policy_config.yml"
    manifest_root = Path(config["paths"]["manifest_root"])
    robotwin_python = config["paths"]["robotwin_python"]
    hw = config["hardware"]
    model_gpu = int(os.environ.get("EXP2A_MODEL_GPU", hw["model_gpu"]))
    sim_gpu = int(os.environ.get("EXP2A_SIM_GPU", hw["sim_gpu"]))
    port = int(os.environ.get("EXP2A_PORT", hw["port"]))
    config.update({"tasks": tasks, "states_per_task": states, "rollouts_per_state": rollouts,
                   "collection_methods": methods})
    config["hardware"].update({"model_gpu": model_gpu, "sim_gpu": sim_gpu, "port": port})
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    base_env = os.environ.copy()
    base_env.update({
        "ROBOTWIN_PATH": str(robotwin),
        "PYTHONPATH": os.pathsep.join((str(openwam), str(openwam / "benchmarks/robotwin"), base_env.get("PYTHONPATH", ""))),
        "PYTHONUNBUFFERED": "1",
        "ROBOTWIN_DISABLE_EVAL_VIDEO": "1",
    })

    for task in tasks:
        path = manifest_root / task / config["mode"] / "manifest.json"
        if path.is_file() and len(load(path).get("entries", [])) >= states:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        env = base_env | {"CUDA_VISIBLE_DEVICES": str(sim_gpu),
                          "ROBOTWIN_RUNTIME_ROOT": str(manifest_root / task / config["mode"] / "runtime")}
        run([robotwin_python, str(wrapper), "labtasker", "--operation", "build_manifest",
             "--task", task, "--mode", config["mode"], "--policy-config", str(policy),
             "--result-file", str(path), "--progress-file", str(path.with_name("progress.json")),
             "--total-episodes", str(states)], cwd=openwam, env=env,
            log=output / "logs" / f"manifest_{task}.log")

    total = len(tasks) * states * rollouts * len(methods)
    completed = 0
    started = time.monotonic()
    for method in methods:
        server_log_path = output / "logs" / f"server_{method}.log"
        server_log = server_log_path.open("a", encoding="utf-8")
        server = subprocess.Popen(
            ["conda", "run", "--no-capture-output", "-n", "openwam", "python", "scripts/deploy.py",
             "--ckpt-dir", config["models"][method], "--device", f"cuda:{model_gpu}",
             "--host", hw["host"], "--port", str(port), "--compile-enabled", "false"],
            cwd=openwam, env=base_env, stdout=server_log, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        try:
            wait_port(hw["host"], port, server, int(hw["startup_timeout_seconds"]))
            for task in tasks:
                manifest = manifest_root / task / config["mode"] / "manifest.json"
                mh = manifest_hash(manifest)
                for state in range(states):
                    for rollout in range(rollouts):
                        stem = output / "trajectories" / task / method / f"state_{state:02d}" / f"rollout_{rollout:03d}"
                        trace = stem.with_suffix(".trace.json")
                        result = stem.with_suffix(".result.json")
                        if args.resume and trace.is_file() and result.is_file():
                            completed += 1
                            continue
                        runtime = output / "runtime" / task / method / f"s{state:02d}_r{rollout:03d}"
                        runtime.mkdir(parents=True, exist_ok=True)
                        sample_seed = policy_sample_seed(task, state, rollout)
                        env = base_env | {
                            "CUDA_VISIBLE_DEVICES": str(sim_gpu),
                            "ROBOTWIN_RUNTIME_ROOT": str(runtime),
                            "ROBOTWIN_TRACE_FILE": str(trace),
                            "ROBOTWIN_TRACE_METHOD": method,
                            "ROBOTWIN_TRACE_ROLLOUT": str(rollout),
                            "ROBOTWIN_POLICY_SAMPLE_SEED": str(sample_seed),
                        }
                        run([robotwin_python, str(wrapper), "labtasker", "--operation", "run_eval",
                             "--task", task, "--mode", config["mode"], "--policy-config", str(policy),
                             "--host", hw["host"], "--port", str(port), "--result-file", str(result),
                             "--progress-file", str(stem.with_suffix(".progress.json")),
                             "--total-episodes", str(states), "--episode-start", str(state), "--num-episodes", "1",
                             "--manifest-file", str(manifest), "--manifest-hash", mh,
                             "--instruction-type", config["instruction_type"]], cwd=openwam, env=env,
                            log=output / "logs" / f"driver_{task}_{method}.log")
                        completed += 1
                        elapsed = time.monotonic() - started
                        eta = elapsed / completed * (total - completed) if completed else 0
                        print(f"[exp2a] {completed}/{total} task={task} method={method} state={state} "
                              f"rollout={rollout} sample_seed={sample_seed} "
                              f"elapsed={elapsed/3600:.2f}h eta={eta/3600:.2f}h", flush=True)
        finally:
            try:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(server.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            server_log.close()


if __name__ == "__main__":
    main()
