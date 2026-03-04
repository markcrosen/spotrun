"""EC2 infrastructure management for spotrun."""

from __future__ import annotations

import os
import socket
import stat
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from rich.console import Console


@contextmanager
def _nullcontext():
    """No-op context manager (Python 3.6 compat, avoids import)."""
    yield

console = Console()
SPOTRUN_DIR = Path.home() / ".spotrun"

# Serialize ensure_infra across threads to prevent key pair race conditions
_infra_lock = threading.Lock()

CANDIDATE_REGIONS = [
    "us-east-1",
    "us-east-2",
    "us-west-2",
    "eu-west-1",
    "eu-central-1",
    "ap-south-1",
    "ap-southeast-1",
    "ap-northeast-1",
]


_AUTH_ERROR_CODES = {
    "AuthFailure", "UnauthorizedOperation",
    "InvalidClientTokenId", "ExpiredToken",
}

CAPACITY_ERROR_CODES = {
    "InsufficientInstanceCapacity",
    "InstanceLimitExceeded",
    "MaxSpotInstanceCountExceeded",
    "SpotMaxPriceTooLow",
}


def is_capacity_error(error: ClientError) -> bool:
    """Check if a ClientError is a spot capacity issue (retryable with fallback)."""
    return error.response["Error"]["Code"] in CAPACITY_ERROR_CODES


def find_ranked_regions(
    instance_type: str,
    exclude: list[str] | None = None,
    quiet: bool = False,
) -> list[tuple[str, float]]:
    """Query spot prices across candidate regions, return all with pricing.

    Returns list of (region, price) sorted cheapest first.
    Skips regions in the exclude list.

    Raises RuntimeError if no pricing data is available in any region.
    """
    exclude_set = set(exclude or [])
    results: list[tuple[str, float]] = []

    ctx = console.status("Checking spot prices across regions...") if not quiet else _nullcontext()
    with ctx:
        for region in CANDIDATE_REGIONS:
            if region in exclude_set:
                continue
            try:
                client = boto3.client("ec2", region_name=region)
                resp = client.describe_spot_price_history(
                    InstanceTypes=[instance_type],
                    ProductDescriptions=["Linux/UNIX"],
                    MaxResults=20,
                )
                best_price = float("inf")
                for entry in resp.get("SpotPriceHistory", []):
                    price = float(entry["SpotPrice"])
                    if price < best_price:
                        best_price = price
                if best_price < float("inf"):
                    results.append((region, best_price))
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if code in _AUTH_ERROR_CODES:
                    raise
                continue

    if not results:
        raise RuntimeError(
            f"Could not find spot pricing for {instance_type} in any region."
        )

    results.sort(key=lambda x: x[1])

    if not quiet:
        best_region, best_price = results[0]
        console.print(
            f"Cheapest region: [bold]{best_region}[/bold] "
            f"([green]${best_price:.4f}/hr[/green] for {instance_type})"
        )

    return results


def find_cheapest_region(instance_type: str) -> tuple[str, float]:
    """Query spot prices across candidate regions, return (cheapest_region, price).

    Raises RuntimeError if no pricing data is available in any region.
    """
    return find_ranked_regions(instance_type)[0]


class EC2Manager:
    """Manages EC2 resources: key pairs, security groups, instances."""

    def __init__(self, region: str | None = None) -> None:
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.client = boto3.client("ec2", region_name=self.region)
        self.ec2 = boto3.resource("ec2", region_name=self.region)

    def ensure_infra(self, project_tag: str = "spotrun") -> tuple[str, str, str]:
        """Ensure key pair and security group exist.

        Thread-safe: uses a lock to prevent race conditions when multiple
        threads try to create the same key pair simultaneously.

        Returns (key_name, pem_path, sg_id).
        """
        key_name = f"{project_tag}-{self.region}"
        pem_path = str(SPOTRUN_DIR / f"{key_name}.pem")

        with _infra_lock:
            SPOTRUN_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)

            # Key pair
            need_create = False
            try:
                self.client.describe_key_pairs(KeyNames=[key_name])
                if not Path(pem_path).exists():
                    console.print(
                        f"[yellow]Key pair [bold]{key_name}[/bold] exists in AWS "
                        f"but PEM file missing locally. Recreating...[/yellow]"
                    )
                    self.client.delete_key_pair(KeyName=key_name)
                    need_create = True
                else:
                    console.print(f"[dim]Key pair [bold]{key_name}[/bold] exists[/dim]")
            except ClientError as e:
                if e.response["Error"]["Code"] not in (
                    "InvalidKeyPair.NotFound",
                ):
                    raise
                need_create = True

            if need_create:
                console.print(f"Creating key pair [bold]{key_name}[/bold]")
                resp = self.client.create_key_pair(KeyName=key_name)
                fd = os.open(pem_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR)
                with os.fdopen(fd, "w") as f:
                    f.write(resp["KeyMaterial"])

        # Security group (already idempotent, no lock needed)
        sg_name = f"{project_tag}-ssh"
        sg_id = self._ensure_security_group(sg_name, project_tag)

        return key_name, pem_path, sg_id

    def _ensure_security_group(self, sg_name: str, project_tag: str) -> str:
        try:
            resp = self.client.describe_security_groups(
                Filters=[{"Name": "group-name", "Values": [sg_name]}]
            )
            if resp["SecurityGroups"]:
                sg_id = resp["SecurityGroups"][0]["GroupId"]
                console.print(f"[dim]Security group [bold]{sg_name}[/bold] exists ({sg_id})[/dim]")
                return sg_id
        except ClientError as e:
            if e.response["Error"]["Code"] not in (
                "InvalidGroup.NotFound",
            ):
                raise

        console.print(f"Creating security group [bold]{sg_name}[/bold]")
        try:
            resp = self.client.create_security_group(
                GroupName=sg_name,
                Description=f"SSH access for {project_tag}",
            )
            sg_id = resp["GroupId"]
            self.client.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[{
                    "IpProtocol": "tcp",
                    "FromPort": 22,
                    "ToPort": 22,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH"}],
                }],
            )
            self.client.create_tags(
                Resources=[sg_id],
                Tags=[{"Key": "Project", "Value": project_tag}],
            )
            return sg_id
        except ClientError as e:
            if e.response["Error"]["Code"] == "InvalidGroup.Duplicate":
                # Race: another thread created it — re-fetch
                resp = self.client.describe_security_groups(
                    Filters=[{"Name": "group-name", "Values": [sg_name]}]
                )
                if resp["SecurityGroups"]:
                    return resp["SecurityGroups"][0]["GroupId"]
            raise

    def get_spot_prices(self, instance_types: list[str]) -> dict[str, float]:
        """Get current spot prices, returning cheapest per instance type."""
        resp = self.client.describe_spot_price_history(
            InstanceTypes=instance_types,
            ProductDescriptions=["Linux/UNIX"],
            MaxResults=len(instance_types) * 10,
        )
        prices: dict[str, float] = {}
        for entry in resp["SpotPriceHistory"]:
            itype = entry["InstanceType"]
            price = float(entry["SpotPrice"])
            if itype not in prices or price < prices[itype]:
                prices[itype] = price
        return prices

    def request_spot_instance(
        self,
        instance_type: str,
        ami_id: str,
        key_name: str,
        sg_id: str,
        user_data: str = "",
        project_tag: str = "spotrun",
        threads_per_core: int | None = None,
        core_count: int | None = None,
    ) -> str:
        """Launch a single spot instance. Returns instance_id.

        Args:
            threads_per_core: If set to 1, disables hyperthreading so each vCPU
                maps to a physical core. Ideal for CPU-bound single-threaded
                workloads (e.g. Python multiprocessing). Default None (use
                instance default, typically 2).
            core_count: Number of physical cores. Required when threads_per_core
                is set (AWS requires both CoreCount and ThreadsPerCore).
        """
        kwargs: dict = dict(
            ImageId=ami_id,
            InstanceType=instance_type,
            KeyName=key_name,
            SecurityGroupIds=[sg_id],
            MinCount=1,
            MaxCount=1,
            UserData=user_data,
            InstanceMarketOptions={
                "MarketType": "spot",
                "SpotOptions": {"SpotInstanceType": "one-time"},
            },
            TagSpecifications=[{
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Project", "Value": project_tag},
                    {"Key": "Name", "Value": f"{project_tag}-worker"},
                ],
            }],
        )
        if threads_per_core is not None and core_count is not None:
            kwargs["CpuOptions"] = {
                "CoreCount": core_count,
                "ThreadsPerCore": threads_per_core,
            }
        resp = self.client.run_instances(**kwargs)
        instance_id = resp["Instances"][0]["InstanceId"]
        return instance_id

    def wait_for_running(self, instance_id: str, timeout: int = 300) -> str:
        """Wait until the instance is running, then return its public IP."""
        waiter = self.client.get_waiter("instance_running")
        waiter.wait(
            InstanceIds=[instance_id],
            WaiterConfig={"Delay": 5, "MaxAttempts": timeout // 5},
        )
        resp = self.client.describe_instances(InstanceIds=[instance_id])
        ip = resp["Reservations"][0]["Instances"][0].get("PublicIpAddress")
        if not ip:
            raise RuntimeError(f"Instance {instance_id} has no public IP")
        return ip

    def get_instance_state(self, instance_id: str) -> str:
        """Query the current state of an EC2 instance.

        Returns one of: 'pending', 'running', 'shutting-down',
        'terminated', 'stopping', 'stopped'.

        Raises ``ClientError`` on API failure.
        """
        resp = self.client.describe_instances(InstanceIds=[instance_id])
        reservations = resp.get("Reservations", [])
        if not reservations or not reservations[0].get("Instances"):
            raise RuntimeError(f"Instance {instance_id} not found")
        return reservations[0]["Instances"][0]["State"]["Name"]

    def terminate_instance(self, instance_id: str) -> None:
        """Terminate an EC2 instance."""
        self.client.terminate_instances(InstanceIds=[instance_id])
        console.print(f"Terminated instance [bold]{instance_id}[/bold]")

    def wait_for_ssh(self, ip: str, timeout: int = 300) -> None:
        """Poll port 22 until SSH is accepting connections."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                sock = socket.create_connection((ip, 22), timeout=5)
                sock.close()
                return
            except (OSError, ConnectionRefusedError):
                time.sleep(5)
        raise TimeoutError(f"SSH not available on {ip} after {timeout}s")

    def get_ubuntu_ami(self, arch: str = "x86_64") -> str:
        """Find the latest Ubuntu 24.04 LTS AMI from Canonical.

        Args:
            arch: "x86_64" or "arm64".
        """
        name_pattern = (
            "ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-arm64-server-*"
            if arch == "arm64"
            else "ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"
        )
        resp = self.client.describe_images(
            Owners=["099720109477"],
            Filters=[
                {"Name": "name", "Values": [name_pattern]},
                {"Name": "architecture", "Values": [arch]},
                {"Name": "state", "Values": ["available"]},
            ],
        )
        images = sorted(resp["Images"], key=lambda i: i["CreationDate"], reverse=True)
        if not images:
            raise RuntimeError(f"No Ubuntu 24.04 ({arch}) AMI found in {self.region}")
        return images[0]["ImageId"]
