"""Session -- the main user-facing orchestrator."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

from botocore.exceptions import ClientError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from spotrun.ami import AMIManager
from spotrun.ec2 import (
    _AUTH_ERROR_CODES,
    CANDIDATE_REGIONS,
    EC2Manager,
    find_cheapest_region,
    is_capacity_error,
)
from spotrun.exceptions import SpotCapacityError
from spotrun.pricing import (
    all_instance_types,
    instance_arch,
    select_instance,
    select_ranked_instances,
)
from spotrun.sync import DataSync

console = Console()
STATE_FILE = Path.home() / ".spotrun" / "state.json"


class Session:
    """Orchestrates the full lifecycle: launch, sync, run, teardown."""

    def __init__(
        self,
        workers: int = 4,
        region: str | None = None,
        project_tag: str = "spotrun",
        bootstrap_script: str | None = None,
        requirements_file: str | None = None,
        include_arm: bool = False,
        no_hyperthreading: bool = False,
        save_state: bool = True,
        auto_install: bool = True,
        quiet: bool = False,
    ) -> None:
        self.workers = workers
        self.project_tag = project_tag
        self._save_state_enabled = save_state
        self.bootstrap_script = bootstrap_script
        self.requirements_file = requirements_file
        self.include_arm = include_arm
        self.no_hyperthreading = no_hyperthreading
        self.auto_install = auto_install
        self._quiet = quiet
        self.last_output: str = ""

        # Track whether region was explicitly specified (disables region fallback)
        self._region_explicit = region is not None or "AWS_REGION" in os.environ

        # Launch metadata (populated after successful launch)
        self.instance_type: str | None = None
        self.vcpus: int | None = None
        self.spot_price: float | None = None
        self.fallback_log: list[str] = []

        # Auto-select cheapest region if none specified
        if not self._region_explicit:
            it, _ = select_instance(workers, include_arm=include_arm)
            region, _ = find_cheapest_region(it)

        self.ec2 = EC2Manager(region=region)
        self.ami_mgr = AMIManager(self.ec2)
        self._sync: DataSync | None = None
        self._instance_id: str | None = None
        self._ip: str | None = None
        self._pem_path: str | None = None

    def _print(self, *args, **kwargs) -> None:
        """Print only when not in quiet mode."""
        if not self._quiet:
            console.print(*args, **kwargs)

    def launch(
        self,
        bootstrap_script: str | None = None,
        requirements_file: str | None = None,
        idle_timeout: int = 300,
        fallback: bool = True,
    ) -> str:
        """Provision infrastructure and launch a spot instance. Returns public IP.

        Args:
            bootstrap_script: Path to a shell script to run during AMI build.
            requirements_file: Path to requirements.txt to pre-install in AMI.
            idle_timeout: Seconds of inactivity before auto-terminating the
                instance.  Defaults to 300 (5 minutes). Set to 0 to disable.
                Long-running processes can prevent shutdown by touching
                ``/tmp/spotrun-heartbeat`` periodically (see
                :attr:`Session.HEARTBEAT_FILE`).
            fallback: If True (default), try alternative instance types and
                regions on capacity errors. If False, raise SpotCapacityError
                on the first capacity failure.
        """
        bootstrap_script = bootstrap_script or self.bootstrap_script
        requirements_file = requirements_file or self.requirements_file

        if fallback:
            return self._launch_with_fallback(
                bootstrap_script, requirements_file, idle_timeout,
            )
        else:
            return self._launch_single(
                bootstrap_script, requirements_file, idle_timeout,
            )

    def _launch_single(
        self,
        bootstrap_script: str | None,
        requirements_file: str | None,
        idle_timeout: int,
    ) -> str:
        """Launch with no fallback — raise SpotCapacityError on capacity error."""
        key_name, pem_path, sg_id = self.ec2.ensure_infra(self.project_tag)
        self._pem_path = pem_path

        prices = self.ec2.get_spot_prices(all_instance_types(include_arm=self.include_arm))
        itype, vcpus = select_instance(self.workers, prices=prices, include_arm=self.include_arm)
        arch = instance_arch(itype)
        spot_price = prices.get(itype)
        self._show_pricing(itype, vcpus, spot_price, prices)

        try:
            return self._do_launch_instance(
                itype, vcpus, arch, spot_price, key_name, pem_path, sg_id,
                bootstrap_script, requirements_file, idle_timeout,
            )
        except ClientError as e:
            if is_capacity_error(e):
                err_msg = e.response["Error"]["Message"]
                self.fallback_log.append(
                    f"{self.ec2.region}/{itype}: {err_msg}"
                )
                raise SpotCapacityError(
                    f"No spot capacity for {itype} in {self.ec2.region}: {err_msg}",
                    attempts=[(self.ec2.region, itype, err_msg)],
                )
            raise

    def _launch_with_fallback(
        self,
        bootstrap_script: str | None,
        requirements_file: str | None,
        idle_timeout: int,
    ) -> str:
        """Launch with global instance type + region fallback on capacity errors.

        Builds a single flat list of (region, instance_type) candidates sorted
        by spot price across ALL candidate regions, then tries each in order.
        This ensures we always pick the next globally cheapest option rather
        than exhausting all instance types in one region before moving on.
        """
        attempts: list[tuple[str, str, str]] = []
        all_itypes = all_instance_types(include_arm=self.include_arm)

        # Determine regions to query
        if self._region_explicit:
            regions = [self.ec2.region]
        else:
            regions = [self.ec2.region]
            for r in CANDIDATE_REGIONS:
                if r not in regions:
                    regions.append(r)

        # Fetch prices from all regions and build globally ranked list
        # Each entry: (price, region, instance_type, vcpus)
        global_candidates: list[tuple[float, str, str, int]] = []
        region_prices: dict[str, dict[str, float]] = {}

        if len(regions) > 1:
            self._print("[dim]Querying spot prices across regions...[/dim]")

        for region in regions:
            try:
                mgr = self.ec2 if region == self.ec2.region else EC2Manager(region=region)
                prices = mgr.get_spot_prices(all_itypes)
            except ClientError as e:
                if e.response["Error"]["Code"] in _AUTH_ERROR_CODES:
                    raise
                self.fallback_log.append(f"Could not query prices in {region}")
                continue

            region_prices[region] = prices
            try:
                ranked = select_ranked_instances(
                    self.workers, prices=prices, include_arm=self.include_arm,
                )
            except ValueError:
                continue

            for itype, vcpus in ranked:
                price = prices.get(itype)
                if price is not None:
                    global_candidates.append((price, region, itype, vcpus))

        if not global_candidates:
            raise SpotCapacityError(
                "Could not find spot pricing in any region.",
                attempts=[],
            )

        # Sort by price — globally cheapest first
        global_candidates.sort(key=lambda x: x[0])

        # Show pricing table for the cheapest candidate's region
        if not self._quiet:
            price0, region0, itype0, vcpus0 = global_candidates[0]
            best_prices = region_prices.get(region0, {})
            self._show_pricing(itype0, vcpus0, price0, best_prices)

        # Lazy infra setup per region
        region_infra: dict[str, tuple[str, str, str]] = {}

        for i, (price, region, itype, vcpus) in enumerate(global_candidates):
            # Switch region if needed
            if region != self.ec2.region:
                self._switch_region(region)

            if i > 0:
                self.fallback_log.append(
                    f"Trying {itype} in {region} (${price:.4f}/hr)..."
                )
                self._print(
                    f"[yellow]Trying {itype} in {region} "
                    f"(${price:.4f}/hr)...[/yellow]"
                )

            # Ensure infra once per region
            if region not in region_infra:
                key_name, pem_path, sg_id = self.ec2.ensure_infra(self.project_tag)
                self._pem_path = pem_path
                region_infra[region] = (key_name, pem_path, sg_id)
            else:
                key_name, pem_path, sg_id = region_infra[region]
                self._pem_path = pem_path

            arch = instance_arch(itype)
            try:
                return self._do_launch_instance(
                    itype, vcpus, arch, price, key_name, pem_path, sg_id,
                    bootstrap_script, requirements_file, idle_timeout,
                )
            except ClientError as e:
                if is_capacity_error(e):
                    err_msg = e.response["Error"]["Message"]
                    attempts.append((region, itype, err_msg))
                    self.fallback_log.append(f"{region}/{itype}: {err_msg}")
                    self._print(
                        f"[yellow]{itype} unavailable in {region}: "
                        f"{err_msg}[/yellow]"
                    )
                    continue
                raise

        # All options exhausted
        raise SpotCapacityError(
            f"No spot capacity available after trying {len(attempts)} options.",
            attempts=attempts,
        )

    def _do_launch_instance(
        self,
        itype: str,
        vcpus: int,
        arch: str,
        spot_price: float | None,
        key_name: str,
        pem_path: str,
        sg_id: str,
        bootstrap_script: str | None,
        requirements_file: str | None,
        idle_timeout: int,
    ) -> str:
        """Try to launch a single instance type. Returns public IP.

        Raises ClientError on capacity issues (caller handles fallback).
        """
        # Find or create AMI (must match selected architecture)
        ami_id = self.ami_mgr.find_existing(self.project_tag, arch=arch)
        if ami_id:
            self._print(f"Using existing AMI: [bold]{ami_id}[/bold] ({arch})")
        else:
            self._print(f"No existing AMI found for {arch}, building one...")
            ami_id = self.ami_mgr.create(
                key_name, pem_path, sg_id,
                bootstrap_script=bootstrap_script,
                requirements_file=requirements_file,
                project_tag=self.project_tag,
                arch=arch,
            )

        # CPU options: disable hyperthreading for x86
        if self.no_hyperthreading and arch != "arm64":
            threads_per_core = 1
            core_count = vcpus // 2
        else:
            threads_per_core = None
            core_count = None

        if self._quiet:
            self._instance_id = self.ec2.request_spot_instance(
                instance_type=itype,
                ami_id=ami_id,
                key_name=key_name,
                sg_id=sg_id,
                project_tag=self.project_tag,
                threads_per_core=threads_per_core,
                core_count=core_count,
            )
        else:
            with console.status(f"Requesting [bold]{itype}[/bold] spot instance..."):
                self._instance_id = self.ec2.request_spot_instance(
                    instance_type=itype,
                    ami_id=ami_id,
                    key_name=key_name,
                    sg_id=sg_id,
                    project_tag=self.project_tag,
                    threads_per_core=threads_per_core,
                    core_count=core_count,
                )
                console.print(f"Instance: [bold]{self._instance_id}[/bold]")

        # Save state early
        self._save_state(key_name, sg_id)

        # Wait for running + SSH
        try:
            if self._quiet:
                self._ip = self.ec2.wait_for_running(self._instance_id)
            else:
                with console.status("Waiting for instance to start..."):
                    self._ip = self.ec2.wait_for_running(self._instance_id)
                    console.print(f"Public IP: [bold]{self._ip}[/bold]")

                with console.status("Waiting for SSH..."):
                    self.ec2.wait_for_ssh(self._ip)

            if self._quiet:
                self.ec2.wait_for_ssh(self._ip)
        except Exception:
            try:
                self.teardown()
            except Exception:
                pass
            raise

        self._sync = DataSync(self._ip, pem_path)
        self._save_state(key_name, sg_id)

        if idle_timeout and idle_timeout > 0:
            self._install_idle_watchdog(idle_timeout)

        # Store launch metadata
        self.instance_type = itype
        self.vcpus = vcpus
        self.spot_price = spot_price

        self._print("[green bold]Instance ready.[/green bold]")
        return self._ip

    def _switch_region(self, region: str) -> None:
        """Switch to a different AWS region."""
        self.ec2 = EC2Manager(region=region)
        self.ami_mgr = AMIManager(self.ec2)
        self._instance_id = None
        self._ip = None
        self._sync = None
        self._pem_path = None

    def sync(self, paths: list[str], remote_base: str = "/opt/project",
             quiet: bool = False) -> None:
        """Rsync individual paths to the remote instance."""
        if not self._sync:
            raise RuntimeError("No active session. Call launch() first.")
        for path in paths:
            self._sync.rsync_to(path, remote_base, quiet=quiet)

    def sync_project(
        self,
        local_root: str = ".",
        remote_root: str = "/opt/project",
        excludes: list[str] | None = None,
        quiet: bool = False,
        n_instances: int = 1,
    ) -> None:
        """Rsync an entire project directory."""
        if not self._sync:
            raise RuntimeError("No active session. Call launch() first.")
        self._sync.rsync_project(
            local_root, remote_root, excludes=excludes,
            quiet=quiet, n_instances=n_instances,
        )

    def install_deps(self, remote_root: str = "/opt/project") -> bool:
        """Detect and install Python dependencies on the remote instance.

        Checks for requirements.txt or pyproject.toml in the remote project
        directory and installs into the existing venv.

        Returns True if dependencies were installed, False if skipped.
        """
        if not self._sync:
            raise RuntimeError("No active session. Call launch() first.")

        # Guard: skip if no venv exists (e.g. custom bootstrap that skipped it)
        venv_check = self._sync.ssh_run(
            f"test -d {remote_root}/.venv/bin", quiet=True,
        )
        if venv_check != 0:
            return False

        # Detect which deps file exists (single SSH round-trip)
        detect_cmd = (
            f"test -f {remote_root}/requirements.txt && echo requirements "
            f"|| (test -f {remote_root}/pyproject.toml && echo pyproject "
            f"|| echo none)"
        )
        try:
            result = self._sync.ssh_run(detect_cmd, capture=True)
        except subprocess.CalledProcessError:
            self._print("[yellow]Warning: could not detect dependency files[/yellow]")
            return False
        if not isinstance(result, str):
            return False
        deps_type = result.strip()

        if deps_type == "none":
            return False

        venv_pip = f"{remote_root}/.venv/bin/pip"
        venv_python = f"{remote_root}/.venv/bin/python"

        if deps_type == "requirements":
            self._print("[dim]Installing dependencies from requirements.txt...[/dim]")
            install_cmd = f"{venv_pip} install --quiet -r {remote_root}/requirements.txt"
        elif deps_type == "pyproject":
            self._print("[dim]Installing dependencies from pyproject.toml...[/dim]")
            install_cmd = (
                f"{venv_python} << 'PYEOF'\n"
                "import subprocess, sys\n"
                "try:\n"
                "    import tomllib\n"
                "except ImportError:\n"
                '    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "tomli"])\n'
                "    import tomli as tomllib\n"
                f'with open("{remote_root}/pyproject.toml", "rb") as f:\n'
                '    deps = tomllib.load(f).get("project", {}).get("dependencies", [])\n'
                "if deps:\n"
                '    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + deps)\n'
                '    print(f"Installed {len(deps)} dependencies from pyproject.toml")\n'
                "else:\n"
                '    print("No dependencies found in pyproject.toml")\n'
                "PYEOF"
            )
        else:
            return False

        exit_code = self._sync.ssh_run(install_cmd)
        if not isinstance(exit_code, int):
            exit_code = -1
        if exit_code != 0:
            self._print("[yellow]Warning: dependency installation returned non-zero exit code[/yellow]")
        return True

    def run(self, command: str, quiet: bool = False, stop_event=None,
            activate_venv: bool = True, remote_root: str = "/opt/project",
            tail_lines: int = 0) -> int:
        """Run a command on the remote instance. Returns exit code.

        Args:
            command: Shell command to execute.
            quiet: If True, suppress all output.
            stop_event: threading.Event to interrupt long-running commands.
            activate_venv: If True (default), activate the project venv before
                running the command. Safe even if no venv exists.
            remote_root: Project root on the remote instance.
            tail_lines: When > 0 and quiet=True, capture the last N lines of
                combined stdout/stderr. Available via ``self.last_output``
                after the call returns (useful for error diagnostics).
        """
        if not self._sync:
            raise RuntimeError("No active session. Call launch() first.")
        if activate_venv:
            activate = f"{remote_root}/.venv/bin/activate"
            command = (
                f"if [ -f {activate} ]; then source {activate}; fi && "
                f"({command})"
            )
        result = self._sync.ssh_run(
            command, quiet=quiet, stop_event=stop_event, tail_lines=tail_lines,
        )
        if tail_lines > 0:
            self.last_output = self._sync.last_output_tail
        if not isinstance(result, int):
            return -1
        return result

    def ssh(self) -> None:
        """Drop into an interactive SSH session (replaces this process)."""
        if not self._sync:
            raise RuntimeError("No active session. Call launch() first.")
        self._sync.ssh_interactive()

    def teardown(self) -> None:
        """Terminate the instance and clean up state.

        Logs a warning on termination failure but re-raises the exception
        so callers can handle it.  Instance metadata is always cleared
        (even on failure) to prevent retry loops on the same ID.
        """
        if self._instance_id:
            try:
                self.ec2.terminate_instance(self._instance_id)
            except Exception as e:
                console.print(
                    f"[bold yellow]Warning: failed to terminate "
                    f"{self._instance_id}: {e}[/bold yellow]"
                )
                raise
            finally:
                self._instance_id = None
                self._ip = None
                self._sync = None
        self._clear_state()

    def is_instance_running(self) -> bool | None:
        """Check if the EC2 instance is still running via the EC2 API.

        Returns:
            True if the instance state is 'running'.
            False if the instance is in any other state (terminated,
                stopped, shutting-down, etc.).
            None if the instance ID is not set or the API call fails.
        """
        if not self._instance_id:
            return None
        try:
            return self.ec2.get_instance_state(self._instance_id) == "running"
        except Exception:
            return None

    def reconnect(self, timeout: int = 120) -> None:
        """Re-establish SSH connection to a still-running instance.

        Call this when the SSH connection has dropped but the EC2 instance
        is still running (verified via :meth:`is_instance_running`).
        Creates a new :class:`~spotrun.sync.DataSync` object using the
        existing IP and PEM path.

        Does **not** re-sync project files -- the remote working directory
        is unchanged.  The caller decides whether to re-sync after
        reconnecting.

        Args:
            timeout: Seconds to wait for SSH to become available.

        Raises:
            RuntimeError: If no active instance exists.
            TimeoutError: If SSH is not reachable within *timeout*.
        """
        if not self._instance_id or not self._ip or not self._pem_path:
            raise RuntimeError(
                "Cannot reconnect: no active instance. "
                "Call launch() first or verify is_instance_running()."
            )
        self.ec2.wait_for_ssh(self._ip, timeout=timeout)
        self._sync = DataSync(self._ip, self._pem_path)
        self._print(f"[green]Reconnected to {self._ip}[/green]")

    def get_pricing_info(self) -> dict:
        """Return pricing info without launching anything."""
        prices = self.ec2.get_spot_prices(all_instance_types(include_arm=self.include_arm))
        itype, vcpus = select_instance(self.workers, prices=prices, include_arm=self.include_arm)
        spot_price = prices.get(itype)
        return {
            "instance_type": itype,
            "vcpus": vcpus,
            "spot_price": spot_price,
            "all_prices": prices,
        }

    # -- Context manager --

    def __enter__(self) -> Session:
        self.launch()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.teardown()

    # -- Internal --

    HEARTBEAT_FILE = "/tmp/spotrun-heartbeat"
    """Path to the heartbeat file checked by the idle watchdog.

    Long-running remote processes can touch this file periodically to
    signal that work is still in progress, preventing the watchdog from
    shutting down the instance even when there is no active SSH connection.

    Example (bash, run alongside your workload)::

        (while true; do touch /tmp/spotrun-heartbeat; sleep 30; done) &

    The watchdog considers the instance active when **either** an SSH
    connection exists **or** the heartbeat file was modified within the
    last *idle_timeout* seconds.
    """

    def _install_idle_watchdog(self, timeout_seconds: int) -> None:
        """Install a background watchdog that shuts down the instance after
        *timeout_seconds* of inactivity.

        Inactivity is defined as: no active SSH connections **and** no recent
        heartbeat file update.  Either signal alone keeps the instance alive.

        Remote processes can keep the instance alive by periodically touching
        ``/tmp/spotrun-heartbeat`` (see :attr:`Session.HEARTBEAT_FILE`).
        """
        self.run(
            "nohup bash -c '"
            f"HB={self.HEARTBEAT_FILE}; "
            f"while true; do sleep {timeout_seconds}; "
            # Active SSH connection? Stay alive.
            'pgrep -af "sshd:.*@" > /dev/null && continue; '
            # Recent heartbeat file? Stay alive.
            'if [ -f "$HB" ]; then '
            'age=$(( $(date +%s) - $(stat -c %Y "$HB" 2>/dev/null || echo 0) )); '
            f'[ "$age" -lt {timeout_seconds} ] && continue; '
            "fi; "
            # No SSH and no recent heartbeat — shut down.
            "sudo shutdown -h now; "
            "done"
            "' >/dev/null 2>&1 &",
            quiet=True,
            activate_venv=False,
        )

    def _show_pricing(
        self,
        instance_type: str,
        vcpus: int,
        spot_price: float | None,
        all_prices: dict[str, float],
    ) -> None:
        if self._quiet:
            return
        table = Table(title="Spot Prices", show_header=True)
        table.add_column("Instance", style="cyan")
        table.add_column("vCPUs", justify="right")
        table.add_column("$/hr", justify="right", style="green")
        from spotrun.pricing import COMPUTE_INSTANCES
        for itype, vcpu_count, arch in COMPUTE_INSTANCES:
            if itype not in all_prices:
                continue
            price = all_prices[itype]
            price_str = f"${price:.4f}"
            marker = " <--" if itype == instance_type else ""
            table.add_row(itype, str(vcpu_count), price_str + marker)
        console.print(table)
        if spot_price is not None and spot_price > 0:
            console.print(
                Panel(
                    f"[bold]{instance_type}[/bold] ({vcpus} vCPUs) @ "
                    f"[green]${spot_price:.4f}/hr[/green]",
                    title="Selected Instance",
                )
            )

    def _save_state(self, key_name: str, sg_id: str) -> None:
        if not self._save_state_enabled:
            return
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        state = {
            "instance_id": self._instance_id,
            "ip": self._ip,
            "region": self.ec2.region,
            "pem_path": self._pem_path,
            "key_name": key_name,
            "sg_id": sg_id,
        }
        content = json.dumps(state, indent=2)
        fd = os.open(
            str(STATE_FILE),
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            stat.S_IRUSR | stat.S_IWUSR,
        )
        with os.fdopen(fd, "w") as f:
            f.write(content)

    def _clear_state(self) -> None:
        if not self._save_state_enabled:
            return
        Session.clear_state_file()

    @staticmethod
    def clear_state_file() -> None:
        """Remove the state file from disk (static — usable without an instance)."""
        if STATE_FILE.exists():
            STATE_FILE.unlink()

    @staticmethod
    def load_state() -> dict | None:
        """Load saved state from disk, or None if no state file."""
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
        return None
