# Longleaf pretrained inference runbook

Status: authenticated access, package installation, GPU inference, and forwarding were verified on 2026-11-02. Setup job7876955 completed; GPU job7880140 passed synthetic and real-observation inference and was intentionally cancelled to release its L40S. The account has existing /work/users/a/k/akil storage and visible A100/L40 partitions. No credentials are stored here. Username: `akil@longleaf.unc.edu`. Enter password/MFA in the normal SSH flow, never in chat.

## Scheduled setup for this account

The prepared `scripts/pi05/setup_longleaf.sbatch` uses CPU-only `datamover` for dependency and checkpoint downloads: one task, two CPUs, 16 GB RAM, and a two-hour limit. It purges modules and loads the available python/3.12.4 module to bootstrap uv 0.9.5; OpenPI itself uses managed Python 3.11. It creates no model or inference workload. Keep all caches in `/work/users/a/k/akil/pi05`, outside home storage. UNC reserves login nodes for basic commands and submission and identifies datamover for extensive downloads. [UNC job-submission and partition standards](https://help.rc.unc.edu/getting-started-on-longleaf/#job-submission)

The upstream revision inspected for this setup is `215abfb217dbac7d5f1273282331b9b1866c0479`. Once resource approval is granted, submit from the staged project root:

```bash
cd /work/users/a/k/akil/pi05
export OPENPI_REV=215abfb217dbac7d5f1273282331b9b1866c0479
sbatch scripts/pi05/setup_longleaf.sbatch
```

Both launchers request Slurm END/FAIL notifications to `akil@unc.edu`. CPU setup job `7876955` was submitted with user approval.

Inspect `pi05-setup-JOBID.log` and `setup/JOBID-*.json`. Require status success and downloaded checkpoint params/assets before requesting a GPU. Inference can use `sbatch --time=01:00:00 scripts/pi05/serve_longleaf.sbatch` after exporting the required paths below. Request user approval for resource-intensive jobs before submission; release the GPU with scancel once testing is complete. No GPU research workloads run on login nodes.

## Prepare the environment

Connect with `ssh akil@longleaf.unc.edu` (UNC VPN is required off campus). Check available storage and permissions before choosing a project root. UNC currently documents `/hickory/users/<o>/<n>/<onyen>` for newer accounts; older accounts may have `/work/users/...`. Do not assume either path exists. Install/download in an appropriate scheduled setup job; research workloads must not run on login nodes. [UNC getting started](https://help.rc.unc.edu/getting-started-on-longleaf)

Set these variables in the remote shell, replacing paths:

```bash
export PI05_PROJECT=/absolute/research/storage/pi05
export OPENPI_ROOT="$PI05_PROJECT/openpi"
export OPENPI_DATA_HOME="$PI05_PROJECT/checkpoints"
export PI05_RUN_ROOT="$PI05_PROJECT/runs"
export UV_CACHE_DIR="$PI05_PROJECT/uv-cache"
export HF_HOME="$PI05_PROJECT/hf-cache"
mkdir -p "$PI05_PROJECT" "$PI05_RUN_ROOT"
```

In the scheduled setup environment, clone OpenPI via HTTPS with submodules. Record the exact revision and detach it; subsequent jobs require this SHA. This deliberately does not prescribe an untested commit. Keep the SHA in your experiment records and reuse it when installing the local client.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "$OPENPI_ROOT"
cd "$OPENPI_ROOT"
export OPENPI_REV=$(git rev-parse HEAD)
git checkout --detach "$OPENPI_REV"
printf '%s\n' "$OPENPI_REV" > "$PI05_PROJECT/openpi-revision.txt"
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --python 3.11
```

Use a separately installed `uv` available on your PATH. Follow the pinned checkout's installation instructions if they differ. OpenPI documents NVIDIA inference above 8 GB VRAM and tests Ubuntu 22.04; Longleaf uses RHEL 9, so host compatibility requires validation. Start with one A100/L40 rather than an 8 GB GPU. No training or PyTorch conversion is needed. [OpenPI installation and requirements](https://github.com/Physical-Intelligence/openpi#installation)

## Submit and inspect

Copy this simulator repository to research storage using your normal transfer method. From its root, submit:

```bash
export OPENPI_REV=$(cat "$PI05_PROJECT/openpi-revision.txt")
sbatch --export=ALL scripts/pi05/serve_longleaf.sbatch
squeue -u akil
```

The template requests one GPU in `a100-gpu,l40-gpu` with `gpu_access` QoS. Verify your account can use those partitions; override resources at submission if needed. Four CPUs, 32 GB RAM, and two hours are initial defaults, not measured requirements. [UNC Slurm examples](https://help.rc.unc.edu/longleaf-slurm-examples)

Read `pi05-JOBID.log` and `$PI05_RUN_ROOT/job-JOBID/`. The launcher rejects revision drift, checks JAX sees a GPU, loads `pi05_droid`, runs one synthetic request, checks finite `(15,8)` output, saves warmup metadata, then serves. First startup may download weights and compile. The synthetic call advances the server's inference RNG; NumPy seed zero only fixes the synthetic request.

## Connect and test

The server binds to compute-node loopback. If SSH access to your allocated node is permitted, run locally, replacing COMPUTE_NODE with the hostname recorded by the job:

```bash
ssh -N -o ExitOnForwardFailure=yes -J akil@longleaf.unc.edu \
  -L 8000:127.0.0.1:8000 akil@COMPUTE_NODE
```

This is a proposed topology, not a verified UNC access guarantee. If compute-node SSH is restricted, run the simulator/client on the allocated node in a separate environment, or use a forwarding method approved by UNC. A login-node-only tunnel cannot reach the compute-node loopback listener.

The launcher reuses OpenPI's policy/server APIs with loopback binding instead of the upstream CLI's all-interface binding. Health requests to `/healthz` establish server reachability only. [Upstream server](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/serving/websocket_policy_server.py)

Install the lightweight `packages/openpi-client` from the same OpenPI commit into the simulator environment. Run from the simulator root:

```bash
.venv/bin/python -m scripts.pi05.smoke --out runs/pi05-smoke
# Optional network-only test without rendering:
.venv/bin/python -m scripts.pi05.smoke --synthetic --out runs/pi05-synthetic
```

The smoke script sends actual camera/proprioception observations without advancing robot motion. Require finite `(15,8)` output and save the inputs/output/latencies before trying a short hybrid episode. Upstream also supplies `examples/simple_client/main.py --env droid` for random-observation network tests. [OpenPI simple client](https://github.com/Physical-Intelligence/openpi/blob/main/examples/simple_client/main.py)

Cancel unused allocations with `scancel JOBID`. Keep job logs, revision, dependency list, GPU information, warmup output, and subsequent smoke artifacts together. A successful smoke test establishes inference compatibility, not bottle-opening competence.

## Observed resource usage

Setup job7876955 completed in 7m36s. Slurm reported batch MaxRSS 21,299,112 KiB despite its 16 GB request; use at least 24 GB for a repeat setup/download job and obtain approval for that new allocation. GPU job7880140 used batch MaxRSS 31,598,160 KiB and GPU memory 34,565 MiB after inference, so the tested full L40S has appropriate VRAM. Keep host-memory headroom in mind for longer evaluations. No follow-up allocation is automatically authorized by these measurements.

The working forwarding topology used the authenticated login connection, an SSH local forward on the login host to compute-node loopback, then a local forward through the same login host. The GPU model ran only on the allocated compute node. Both temporary forwards were removed after the test; repeat authentication/forwarding deliberately for a future job.
