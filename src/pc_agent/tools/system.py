"""System inspection tools (read-only)."""

from __future__ import annotations

import getpass
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import Field

from pc_agent.config import Config
from pc_agent.core.decisions import RiskLevel, ToolResult
from pc_agent.core.permissions import is_secret_env_var
from pc_agent.tools.filesystem import format_size
from pc_agent.tools.registry import Tool, ToolArgs, ToolContext

PHYSICAL_FS = {
    "ext2", "ext3", "ext4", "btrfs", "xfs", "zfs", "f2fs", "vfat", "exfat",
    "ntfs", "ntfs3", "fuseblk", "apfs", "hfs", "jfs", "reiserfs",
}


class DiskUsageArgs(ToolArgs):
    path: str = Field(default="/", description="Any path on the filesystem to inspect")
    all_mounts: bool = Field(default=False, description="Report every physical mounted filesystem")


def _usage(path: str) -> dict:
    usage = shutil.disk_usage(path)
    return {
        "mount": path,
        "total": format_size(usage.total),
        "used": format_size(usage.used),
        "free": format_size(usage.free),
        "percent_used": round(usage.used / usage.total * 100, 1) if usage.total else 0.0,
        "free_bytes": usage.free,
    }


def _physical_mounts() -> list[str]:
    mounts_file = Path("/proc/mounts")
    if not mounts_file.exists():
        return ["/"]
    mounts, seen_devices = [], set()
    for line in mounts_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[2] not in PHYSICAL_FS:
            continue
        device, mount = parts[0], parts[1].replace("\\040", " ")
        if device in seen_devices:  # e.g. btrfs subvolumes
            continue
        seen_devices.add(device)
        mounts.append(mount)
    return mounts or ["/"]


def disk_usage(args: DiskUsageArgs, ctx: ToolContext) -> ToolResult:
    targets = _physical_mounts() if args.all_mounts else [os.path.expanduser(args.path)]
    results, errors = [], []
    for target in targets:
        try:
            results.append(_usage(target))
        except OSError as exc:
            errors.append(f"{target}: {exc.strerror or exc}")
    if not results:
        return ToolResult.fail("; ".join(errors) or "No filesystems found")
    first = results[0]
    summary = (
        f"{first['free']} free of {first['total']} on {first['mount']} ({first['percent_used']}% used)"
        if len(results) == 1
        else f"{len(results)} filesystems inspected"
    )
    return ToolResult.ok(json.dumps({"filesystems": results, "errors": errors}), summary, count=len(results))


class ProcessesArgs(ToolArgs):
    sort_by: Literal["memory", "cpu"] = "memory"
    limit: int = Field(default=15, ge=1, le=100)


def processes(args: ProcessesArgs, ctx: ToolContext) -> ToolResult:
    try:
        proc = subprocess.run(
            ["ps", "-axo", "pid=,rss=,pcpu=,pmem=,comm="],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return ToolResult.fail(f"Could not list processes: {exc}")
    rows = []
    for line in proc.stdout.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        try:
            rows.append({
                "pid": int(parts[0]),
                "rss_bytes": int(parts[1]) * 1024,
                "cpu_percent": float(parts[2]),
                "mem_percent": float(parts[3]),
                "name": os.path.basename(parts[4].strip()),
            })
        except ValueError:
            continue
    key = "rss_bytes" if args.sort_by == "memory" else "cpu_percent"
    rows.sort(key=lambda r: r[key], reverse=True)
    top = rows[: args.limit]
    for row in top:
        row["memory"] = format_size(row.pop("rss_bytes"))
    lead = f"{top[0]['name']} ({top[0]['memory']}, {top[0]['cpu_percent']}% CPU)" if top else "none"
    return ToolResult.ok(
        json.dumps({"total_processes": len(rows), "sorted_by": args.sort_by, "processes": top}),
        f"{len(rows)} processes; top by {args.sort_by}: {lead}",
        count=len(rows),
    )


class EnvironmentArgs(ToolArgs):
    pass


SAFE_ENV_VARS = ("SHELL", "LANG", "TERM", "DESKTOP_SESSION", "XDG_CURRENT_DESKTOP", "EDITOR")


def environment(args: EnvironmentArgs, ctx: ToolContext) -> ToolResult:
    try:
        user = getpass.getuser()
    except Exception:
        user = "unknown"
    info = {
        "os": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "user": user,
        "home": str(Path.home()),
        "cwd": os.getcwd(),
        "cpu_count": os.cpu_count(),
        "allowed_paths": [str(p) for p in ctx.path_policy.allowed_roots],
        "env": {k: os.environ[k] for k in SAFE_ENV_VARS if k in os.environ},
        # Names only; values of secret-looking variables are never exposed.
        "env_var_names": sorted(k for k in os.environ if not is_secret_env_var(k)),
    }
    return ToolResult.ok(
        json.dumps(info), f"{info['os']} {info['release']} ({info['machine']}), user {user}"
    )


def tools(config: Config) -> list[Tool]:
    return [
        Tool("system.disk_usage", "Show total/used/free disk space for a path or all mounted filesystems.",
             DiskUsageArgs, disk_usage, RiskLevel.READ, sandboxed=False),
        Tool("system.processes", "List the top processes by memory or CPU usage.",
             ProcessesArgs, processes, RiskLevel.READ),
        Tool("system.environment", "Basic OS, user and environment information (no secrets).",
             EnvironmentArgs, environment, RiskLevel.READ),
    ]
