# Pretrained π₀.₅ integration

## Verified scripted baseline (2026-11-02)

Local macOS ARM runs with MuJoCo 3.4.0, NumPy 1.26.4, and SciPy 1.16.3 succeeded for seeds 0, 1, and 42 at exactly 715 ticks (35.75 simulated seconds). Initial-state fingerprints and all saved qpos, Cartesian targets, rotations, and gripper targets matched exactly across seeds. The new evaluation runner also matched the original `scripts/scripted_demo.py` trajectory exactly. These repeats verify determinism on one scene, not generalization. Rendering was disabled.

Artifacts are in `runs/baseline-20261102-124505/` (git-ignored): per-seed `final.json` and `trajectory.npz`, reference log/trajectory, exact comparison reports, dependency versions, and repository revision. Both adapter tests also passed. GPU inference remains unverified.

## Setup and remaining work

This integration controls the left arm using pi05_droid; the existing scripted right-arm stabilization takes 295 ticks. It preserves MuJoCo physics and Cartesian IK control. Joint commands pass through FK, so redundant joint motion is lost. This is hybrid control, not a pretrained bimanual policy.

Install the simulator with `pip install -e '.[dev]'`. In a separate pinned OpenPI checkout, follow its documented `uv sync` setup, then install its lightweight client into the simulator environment with `pip install -e /path/to/openpi/packages/openpi-client`. Record both repository SHAs and `pip freeze` in experiment artifacts. No model weights are bundled.

Follow [the Longleaf runbook](pi05_longleaf.md) to prepare a pinned OpenPI checkout and submit `scripts/pi05/serve_longleaf.sbatch` with OPENPI_ROOT, OPENPI_REV, OPENPI_DATA_HOME, and PI05_RUN_ROOT exported. Confirm partition access and use research storage for caches. Forward local port 8000 to the allocated node through the login host using your permitted SSH topology. The simulator can alternatively run in its own environment on the compute node. Mac rendering normally uses its native backend; Linux headless rendering needs EGL or OSMesa.

Run `python -m scripts.pi05.smoke --out runs/pi05-smoke` for a real-observation inference without robot motion. Then run `python -m integration_pi05.evaluate --controller scripted --seed 0 --out runs/scripted-0`, or select `--controller hybrid`. Output directories must be new. Repeat the scripted run with seeds 1 and 42; the documented reference is 715 ticks. Hybrid calls execute four simulator ticks, corresponding to three DROID action intervals. Simulation pauses during inference.

Load defaults with `--config configs/pi05/default.json` and expand pose/seed conditions with `--episodes configs/pi05/episodes.json`. Each manifest episode gets its own directory. CLI values override configuration defaults; per-condition max_steps in a manifest takes precedence over the global limit. Do not treat seed changes as scene randomization. Record LLM results separately because its cameras, depth sensing, action abstraction, and budget differ.

The evaluator now saves exact inference requests (RGB and robot state), initial state, measured tool poses, mechanism/wrench traces, client software metadata, setup status, and error categories. The client isolates blocking WebSocket calls in a disposable process with separate connection and inference deadlines. Startup defaults to 600 seconds and inference to 60 seconds.

Summarize one or more result directories with `python -m integration_pi05.summarize runs/scripted-0 --out runs/summary.json`. The report includes attempts, errors, completed-task success rate and Wilson interval, episode lengths, and separate LLM scaffold/sensing tracks when existing LLM final.json files are supplied. Infrastructure/control errors are explicitly counted outside the completed-task denominator. Repeated deterministic script runs do not provide independent scene samples.

Integrated verification: all 33 local tests passed with macOS graphics access. A fresh full scripted episode again succeeded at 715 ticks and matched every saved qpos/target/rotation/grip entry from the original demo (maximum error zero); artifacts: `runs/pi05-integrated-baseline-20261102/`. Sandbox-only rendering failed, but the same camera test passed outside the sandbox.

Remaining work: Longleaf environment installation and GPU/tunnel validation, real-observation inference, short motion smoke tests and full hybrid evaluation, server RNG/provenance control, video export, and finer task-failure classification. Server inference RNG is explicitly marked uncontrolled; local seeds do not seed the model. The provided configuration/checkpoint fields describe the required remote policy and do not change a running server.
