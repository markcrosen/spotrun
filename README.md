# spotrun

Burst compute jobs to AWS spot instances. Zero config, one command.

**The problem:** You have a CPU-bound workload (ML training, optimization, simulation) that takes hours on your laptop. A cloud machine with 64 vCPUs would finish in minutes, but setting one up means clicking through the AWS console, configuring SSH keys, syncing files, and remembering to tear it down so you don't get a surprise bill.

**spotrun** automates the entire lifecycle: provision infrastructure, build an AMI, launch a spot instance, sync your files, run your command, and tear everything down. It even picks the cheapest AWS region automatically. Set 2 environment variables, run one command.

## Install

```bash
pip install spotrun
```

Requires Python 3.10+ and an AWS account.

## Prerequisites

- **AWS account** with an access key that has EC2 permissions
- **rsync** and **ssh** installed locally (pre-installed on macOS/Linux)
- No AWS console interaction required -- spotrun creates everything it needs

## Quick Start

### 1. Set your AWS credentials

```bash
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
```

That's it -- just 2 env vars. spotrun automatically finds the cheapest AWS region for your instance type. Set `AWS_REGION` to override if you prefer a specific region.

You can also use a `.env` file or AWS credentials file -- boto3 reads credentials from all standard sources.

### 2. Run your workload

```bash
spotrun launch --workers 8 --sync ./data --sync ./src "python train.py"
```

That's it. spotrun will:

1. **Find the cheapest region** by querying spot prices across 8 AWS regions
2. Create a key pair and security group (first run only, reused after)
3. Build a custom AMI with Python, venv, and common packages (first run only)
4. Launch a spot instance sized for your worker count
5. Rsync your files to the instance
6. Auto-install Python dependencies from `requirements.txt` or `pyproject.toml`
7. Run your command over SSH (with venv activated), streaming output live
8. Terminate the instance when done

You only pay for the minutes you use, at spot prices (typically 60-90% cheaper than on-demand).

## CLI

```bash
# Launch, sync files, run a command, auto-teardown on completion
spotrun launch --workers 8 --sync data/ --sync src/ "python train.py"

# Launch and drop into an interactive SSH session
spotrun launch --workers 4 --ssh

# Check current spot prices for your workload size
spotrun prices --workers 8

# Pre-build the base AMI (saves time on first launch)
spotrun setup --rebuild-ami

# Manually tear down a running instance
spotrun teardown
```

### Options

| Flag | Description |
|------|-------------|
| `--workers, -w` | Number of parallel workers (default: 4). Determines instance size. |
| `--sync, -s` | Local paths to rsync to `/opt/project` on the instance. Repeatable. |
| `--ssh` | Drop into an interactive SSH session instead of running a command. |
| `--arm` | Include ARM/Graviton instances (often 20-40% cheaper, see below). |
| `--no-ht` | Disable hyperthreading (x86 only). Best for CPU-bound workloads. See [CPU Options](#cpu-options). |
| `--no-install` | Skip automatic Python dependency installation after sync. |
| `--tag` | Project tag for AWS resources (default: `spotrun`). |
| `--bootstrap` | Path to a custom bootstrap script (replaces the default). |
| `--requirements, -r` | Path to `requirements.txt` to pre-install in the AMI. |

## Python API

Use spotrun as a library for programmatic control:

```python
from spotrun import Session

# Context manager -- instance auto-terminates on exit
# Dependencies from requirements.txt/pyproject.toml are auto-installed after sync
with Session(workers=8) as s:
    s.sync_project(".")
    s.run("cd /opt/project && python train.py")
```

With more control:

```python
from spotrun import Session, select_instance, estimate_cost

# Check what instance you'd get
instance_type, vcpus = select_instance(workers=8)
print(f"Will use {instance_type} ({vcpus} vCPUs)")

# Manual lifecycle
s = Session(workers=8)
ip = s.launch()
s.sync_project(".", excludes=[".git", "__pycache__", "*.pyc"])
s.sync(["data/model.pt", ".env"])
s.install_deps()  # auto-detects requirements.txt or pyproject.toml
exit_code = s.run("cd /opt/project && python train.py --workers 8")
s.teardown()
```

### Session API

| Method | Description |
|--------|-------------|
| `Session(workers, region, project_tag, include_arm, no_hyperthreading, auto_install)` | Create a session. `workers` determines instance size. `include_arm=True` enables ARM/Graviton. `no_hyperthreading=True` disables HT on x86 (see [CPU Options](#cpu-options)). `auto_install=False` to skip dependency installation. |
| `.launch(idle_timeout=300)` | Provision infra, find/build AMI, launch spot instance. Returns public IP. The idle watchdog auto-terminates after `idle_timeout` seconds of inactivity (see [Idle Watchdog & Heartbeat](#idle-watchdog--heartbeat)). |
| `.sync(paths)` | Rsync individual files/directories to the instance. |
| `.sync_project(root, excludes)` | Rsync an entire project directory with sensible defaults. |
| `.install_deps()` | Detect and install Python dependencies from `requirements.txt` or `pyproject.toml` on the remote. Returns True if deps were installed. |
| `.run(command)` | Run a shell command on the instance via SSH. The project venv is activated automatically. Pass `activate_venv=False` for non-Python commands. Returns exit code. |
| `.ssh()` | Open an interactive SSH session (replaces current process). |
| `.teardown()` | Terminate the instance and clean up state. Raises on failure (logs a warning first). |
| `.is_instance_running()` | Check if the EC2 instance is still running. Returns `True`, `False`, or `None` (API unreachable). Useful for verifying instance state when SSH drops (exit code 255 doesn't always mean the instance died). |
| `.reconnect(timeout=120)` | Re-establish SSH to a running instance after a connection drop. Does not re-sync files. Raises `TimeoutError` if SSH is unreachable within *timeout* seconds. |
| `.get_pricing_info()` | Return pricing info without launching anything. |

## Parallel Jobs Across Machines

spotrun scales your workload *up* (more CPUs per machine). But when you have many independent jobs -- hyperparameter sweeps, config variations, Monte Carlo runs -- you can also scale *out* by launching multiple spotrun instances at once. Each gets its own spot instance and tears itself down when done.

20 configs that each take 12 hours locally and 2 hours on a cloud instance? Run them all in parallel -- 2 hours total, ~$4.

```bash
for i in $(seq 1 20); do
  spotrun launch -w 8 --sync . "python train.py --config configs/run_${i}.yaml" &
done
wait
```

Or with the Python API:

```python
from concurrent.futures import ThreadPoolExecutor
from spotrun import Session

configs = [f"configs/run_{i}.yaml" for i in range(20)]

def run_job(cfg):
    with Session(workers=8, project_tag=f"sweep-{cfg}") as s:
        s.sync_project(".")
        return s.run(f"cd /opt/project && python train.py --config {cfg}")

with ThreadPoolExecutor(max_workers=20) as pool:
    results = list(pool.map(run_job, configs))
```

Anywhere you'd parallelize across CPUs, you can now parallelize across machines.

### Handling SSH Disconnects

When running many parallel sessions, SSH connections can drop due to local network congestion even though the remote instances are healthy. SSH exit code 255 does **not** always mean the instance was terminated -- it can also mean a transient local network issue.

Use `is_instance_running()` and `reconnect()` to recover gracefully instead of tearing down healthy instances:

```python
exit_code = session.run("python train.py")
if exit_code == 255:
    # Don't assume the instance is dead -- check first
    if session.is_instance_running():
        session.reconnect()
        exit_code = session.run("python train.py --resume")
    else:
        # Instance truly terminated (spot reclaim), relaunch
        session.teardown()
```

This is especially important when running 10+ parallel sessions from a single machine, where rsync and SSH traffic can saturate the local network and cause keepalive timeouts.

## Instance Sizing

spotrun picks the smallest (cheapest) instance with enough physical cores for your workers. The sizing formula accounts for the architectural difference between x86 and ARM:

| | x86 (c6a) | ARM/Graviton (c6g, c7g) |
|---|-----------|------------------------|
| **Threads per core** | 2 (hyperthreading) | 1 (no HT) |
| **vCPUs needed** | `workers * 2 + 2` | `workers + 2` |
| **Max workers (64 vCPU)** | 31 | 62 |

The `+2` reserves capacity for the OS and SSH.

**x86_64 instances:**

| Instance | vCPUs | Phys. Cores | Max Workers |
|----------|-------|-------------|-------------|
| c6a.xlarge | 4 | 2 | 1 |
| c6a.2xlarge | 8 | 4 | 3 |
| c6a.4xlarge | 16 | 8 | 7 |
| c6a.8xlarge | 32 | 16 | 15 |
| c6a.12xlarge | 48 | 24 | 23 |
| c6a.16xlarge | 64 | 32 | 31 |

**ARM/Graviton instances** (with `--arm`):

| Instance | vCPUs (= Cores) | Max Workers |
|----------|-----------------|-------------|
| c6g.xlarge | 4 | 2 |
| c6g.2xlarge | 8 | 6 |
| c6g.4xlarge | 16 | 14 |
| c6g.8xlarge / c7g.8xlarge | 32 | 30 |
| c6g.12xlarge | 48 | 46 |
| c6g.16xlarge | 64 | 62 |

When `--arm` is enabled, spotrun considers both architectures and picks the cheapest option that fits. Because ARM instances have 1:1 vCPU-to-core mapping, they often fit your workload at a smaller (cheaper) tier than x86.

Use `spotrun prices --workers N` to see current spot prices, or `spotrun prices --workers N --arm` to compare with ARM instances.

## ARM/Graviton Instances

ARM/Graviton instances (c6g, c7g) are typically **20-40% cheaper** than their x86 equivalents (c6a) at the same vCPU count. However, some software may have compatibility issues with ARM architecture (native extensions, pre-built binaries, etc.).

ARM is **off by default**. To enable it, pass `--arm` to the CLI or `include_arm=True` to the Python API:

```bash
# CLI
spotrun launch --workers 8 --arm "python train.py"

# Python API
Session(workers=8, include_arm=True)
```

When ARM is enabled, spotrun considers both x86 and ARM instances and picks the cheapest option. If your workload is pure Python (no native C extensions), ARM is usually a safe and cheaper choice.

## CPU Options

### Disabling Hyperthreading (`--no-ht`)

x86 instances use hyperthreading by default: each physical CPU core exposes 2 vCPUs (threads). This is great for I/O-bound or multi-threaded workloads, but **CPU-bound single-threaded workloads** (like Python multiprocessing) can't benefit from the second thread. Each Python process is bound by the GIL, so it can only use 1 thread per core -- the second vCPU on that core sits mostly idle.

The result: your instance looks ~50% utilized even when every worker is maxed out.

The `--no-ht` flag tells EC2 to disable hyperthreading via [`CpuOptions`](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-optimize-cpu.html), setting `ThreadsPerCore=1`. Each vCPU then maps to a dedicated physical core, so `os.cpu_count()` accurately reflects usable cores and CPU utilization shows real usage.

```bash
# CLI
spotrun launch --workers 15 --no-ht "python optimize.py"

# Python API
Session(workers=15, no_hyperthreading=True)
```

Same instance, same price -- you just get accurate core counts and eliminate HT contention for CPU-bound work.

### ARM/Graviton and `--no-ht`

ARM/Graviton instances (c6g, c7g) **do not have hyperthreading**. Each vCPU is already a dedicated physical core. AWS does not allow modifying `ThreadsPerCore` on Graviton instances -- the API will reject the request.

spotrun handles this automatically: when `--no-ht` is set but the selected instance is ARM, spotrun skips the `CpuOptions` setting entirely (since Graviton already behaves as if HT is disabled). No error, no wasted cores.

This means you can safely use `--no-ht` together with `--arm` -- spotrun does the right thing regardless of which architecture gets selected:

| Architecture | `--no-ht` behavior |
|-------------|-------------------|
| x86 (c6a) | Sets `ThreadsPerCore=1`, halving visible vCPUs to physical core count |
| ARM (c6g/c7g) | No-op (Graviton already has 1 thread per core) |

## Automatic Region Selection

Spot prices vary significantly across AWS regions -- sometimes 2-3x for the same instance type. When you don't set `AWS_REGION`, spotrun automatically queries spot prices across 8 major regions and launches in the cheapest one:

```
us-east-1, us-east-2, us-west-2,
eu-west-1, eu-central-1,
ap-south-1, ap-southeast-1, ap-northeast-1
```

```bash
# See prices across all regions for your workload
spotrun prices --workers 8

# Output:
# ┌─────────────────────────────────────────────┐
# │ Region           │ $/hr    │                 │
# ├──────────────────┼─────────┼─────────────────┤
# │ ap-south-1       │ $0.0532 │ <-- cheapest    │
# │ us-east-2        │ $0.0618 │                 │
# │ us-east-1        │ $0.0734 │                 │
# │ ...              │         │                 │
# └─────────────────────────────────────────────┘
```

To pin a specific region, set the environment variable:

```bash
export AWS_REGION=us-east-1
```

Infrastructure (key pair, security group, AMI) is created per-region and cached, so switching regions incurs a one-time AMI build.

## Python Dependencies

spotrun automatically handles Python dependencies. After syncing your files, it checks for `requirements.txt` or `pyproject.toml` in the remote project directory and installs dependencies into the instance's venv. The venv is activated automatically when running commands, so `python` and all installed packages just work.

```bash
# Dependencies from requirements.txt or pyproject.toml are auto-installed
spotrun launch --workers 8 --sync . "python train.py"

# Skip auto-install (e.g. if deps are baked into a custom AMI)
spotrun launch --workers 8 --sync . --no-install "python train.py"
```

For faster launches, you can pre-bake dependencies into the AMI with `--requirements` during setup. Auto-install still runs at launch but skips already-installed packages:

```bash
spotrun setup --requirements requirements.txt
```

## Custom Bootstrap

The default bootstrap script installs Python 3, a venv, build-essential, and rsync. You can replace it with your own:

```bash
spotrun setup --bootstrap ./my-bootstrap.sh
```

## How It Works

### Infrastructure (one-time setup)

On first run, spotrun creates:
- **Key pair** (`spotrun-{region}`) -- saved to `~/.spotrun/spotrun-{region}.pem`
- **Security group** (`spotrun-ssh`) -- allows SSH (port 22) inbound

Both are tagged and reused on subsequent runs.

### AMI (one-time build)

spotrun builds a custom AMI from Ubuntu 24.04 LTS:
1. Launches a `t3.medium` builder instance
2. Runs the bootstrap script (installs Python, venv, system packages)
3. Installs your `requirements.txt` if provided
4. Snapshots the instance as an AMI
5. Terminates the builder

The AMI is cached by tag -- subsequent runs skip this step. Use `spotrun setup --rebuild-ami` to force a rebuild.

### State

Active instance info is saved to `~/.spotrun/state.json`. This allows `spotrun teardown` to work without arguments. The state file is automatically cleaned up on teardown.

## Idle Watchdog & Heartbeat

After launch, spotrun installs a background watchdog that auto-terminates the instance after a period of inactivity (default: 5 minutes). This prevents forgotten instances from running up your AWS bill.

The watchdog considers the instance **active** when either:
1. An SSH connection is open, **or**
2. The heartbeat file `/tmp/spotrun-heartbeat` was updated recently

For short commands this just works -- the SSH connection keeps the instance alive while your command runs, and the watchdog shuts it down shortly after.

For **long-running background workloads** where the SSH connection may drop (network timeouts, spot recovery, etc.), have your process touch the heartbeat file periodically:

```bash
# Start a heartbeat alongside your workload
(while true; do touch /tmp/spotrun-heartbeat; sleep 30; done) &
HBPID=$!
trap "kill $HBPID 2>/dev/null" EXIT

# Run your workload
python train.py --epochs 100

# When the workload exits, the trap kills the heartbeat.
# The watchdog will then shut down the instance after the idle timeout.
```

Or from Python:

```python
import subprocess, atexit
hb = subprocess.Popen(
    ["bash", "-c", "while true; do touch /tmp/spotrun-heartbeat; sleep 30; done"]
)
atexit.register(hb.kill)
```

To adjust or disable the idle timeout:

```python
s.launch(idle_timeout=600)   # 10-minute timeout
s.launch(idle_timeout=0)     # disable watchdog entirely (not recommended)
```

## Cost Awareness

Spot instances are billed per-second with a one-minute minimum. Typical c6a spot prices are $0.03-0.30/hr depending on size and region. Use `spotrun prices` to check current rates.

spotrun always terminates instances on:
- Command completion (success or failure)
- Errors during launch
- `spotrun teardown`

The only case where an instance stays running is `spotrun launch` without a command or `--ssh` (for manual use). Always run `spotrun teardown` when you're done.

## Security Notes

- The security group allows SSH from `0.0.0.0/0` (all IPs). For production use, restrict this to your IP.
- Your `.pem` key is stored at `~/.spotrun/` with `400` permissions (read-only, owner only).
- Credentials are passed via environment variables or boto3's standard credential chain.
- spotrun does **not** store or transmit your AWS credentials -- boto3 handles authentication directly.

## License

MIT
