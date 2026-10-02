"""Filesystem tools. Pure Python, sandboxed by PathPolicy."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from pc_agent.config import Config
from pc_agent.core.decisions import PermissionAssessment, RiskLevel, ToolResult
from pc_agent.core.permissions import PathPolicy, PathPolicyError
from pc_agent.tools.registry import Tool, ToolArgs, ToolContext

MAX_SCANNED_FILES = 200_000
TEXT_SAMPLE_BYTES = 4096


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def _mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _parse_date(value: str | None) -> float | None:
    if not value:
        return None
    return datetime.fromisoformat(value).timestamp()


def _dump(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, indent=None)


def _guarded(func):
    """Turn expected OS / policy errors into failed ToolResults."""

    def wrapper(args, ctx: ToolContext) -> ToolResult:
        try:
            return func(args, ctx)
        except PathPolicyError as exc:
            return ToolResult.fail(str(exc), reason="policy")
        except PermissionError as exc:
            return ToolResult.fail(f"Permission denied: {exc.filename or exc}")
        except FileNotFoundError as exc:
            return ToolResult.fail(f"Not found: {exc.filename or exc}")
        except (OSError, ValueError) as exc:
            return ToolResult.fail(f"{type(exc).__name__}: {exc}")

    wrapper.__name__ = func.__name__
    return wrapper


# --------------------------------------------------------------------------
# filesystem.list
# --------------------------------------------------------------------------


class ListArgs(ToolArgs):
    path: str = Field(description="Directory to list")
    include_hidden: bool = False
    max_entries: int = Field(default=200, ge=1, le=2000)


@_guarded
def list_directory(args: ListArgs, ctx: ToolContext) -> ToolResult:
    path = ctx.path_policy.resolve(args.path)
    if not path.is_dir():
        return ToolResult.fail(f"Not a directory: {path}", path=str(path))
    entries = []
    total = 0
    for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if not args.include_hidden and child.name.startswith("."):
            continue
        total += 1
        if len(entries) >= args.max_entries:
            continue
        try:
            is_dir = child.is_dir()
            entries.append(
                {
                    "name": child.name,
                    "type": "dir" if is_dir else "file",
                    "size": None if is_dir else child.stat().st_size,
                    "modified": _mtime(child),
                }
            )
        except OSError:
            entries.append({"name": child.name, "type": "unknown"})
    dirs = sum(1 for e in entries if e["type"] == "dir")
    files = len(entries) - dirs
    return ToolResult.ok(
        _dump({"path": str(path), "entries": entries, "total": total}),
        f"{files} files, {dirs} directories in {path}"
        + (f" (showing {len(entries)} of {total})" if total > len(entries) else ""),
        path=str(path),
        count=total,
    )


# --------------------------------------------------------------------------
# filesystem.search
# --------------------------------------------------------------------------


class SearchArgs(ToolArgs):
    path: str = Field(description="Directory to search")
    pattern: str = Field(default="*", description="Filename glob, e.g. '*.pdf' (case-insensitive)")
    recursive: bool = True
    min_size: int | None = Field(default=None, description="Minimum size in bytes")
    max_size: int | None = Field(default=None, description="Maximum size in bytes")
    modified_after: str | None = Field(default=None, description="ISO date, e.g. 2026-10-01")
    modified_before: str | None = Field(default=None, description="ISO date")
    sort_by: Literal["name", "size", "modified"] = "name"
    descending: bool = False
    include_hidden: bool = False
    max_results: int = Field(default=100, ge=1, le=1000)


def _walk(root: Path, recursive: bool, include_hidden: bool):
    """Yield files below root, skipping hidden and sensitive directories."""
    if not recursive:
        for child in root.iterdir():
            if child.is_file() and (include_hidden or not child.name.startswith(".")):
                yield child
        return
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        current = Path(dirpath)
        dirnames[:] = [
            d
            for d in dirnames
            if (include_hidden or not d.startswith("."))
            and not PathPolicy.is_sensitive(current / d)
        ]
        for name in filenames:
            if include_hidden or not name.startswith("."):
                yield current / name


@_guarded
def search_files(args: SearchArgs, ctx: ToolContext) -> ToolResult:
    root = ctx.path_policy.resolve(args.path)
    if not root.is_dir():
        return ToolResult.fail(f"Not a directory: {root}", path=str(root))
    after = _parse_date(args.modified_after)
    before = _parse_date(args.modified_before)
    pattern = args.pattern.lower()

    matches: list[dict] = []
    scanned = 0
    for file in _walk(root, args.recursive, args.include_hidden):
        scanned += 1
        if scanned > MAX_SCANNED_FILES:
            break
        if not fnmatch.fnmatch(file.name.lower(), pattern):
            continue
        if PathPolicy.is_sensitive(file):
            continue
        try:
            st = file.stat()
        except OSError:
            continue
        if args.min_size is not None and st.st_size < args.min_size:
            continue
        if args.max_size is not None and st.st_size > args.max_size:
            continue
        if after is not None and st.st_mtime < after:
            continue
        if before is not None and st.st_mtime > before:
            continue
        matches.append({"path": str(file), "size": st.st_size, "mtime": st.st_mtime})

    key = {"name": lambda m: m["path"].lower(), "size": lambda m: m["size"], "modified": lambda m: m["mtime"]}
    matches.sort(key=key[args.sort_by], reverse=args.descending)
    total = len(matches)
    shown = [
        {
            "path": m["path"],
            "size": m["size"],
            "size_human": format_size(m["size"]),
            "modified": datetime.fromtimestamp(m["mtime"]).isoformat(timespec="seconds"),
        }
        for m in matches[: args.max_results]
    ]
    truncated_scan = scanned > MAX_SCANNED_FILES
    summary = f"Found {total} file{'s' if total != 1 else ''} matching '{args.pattern}' in {root}"
    if total > len(shown):
        summary += f" (showing {len(shown)})"
    if truncated_scan:
        summary += " [scan limit reached]"
    return ToolResult.ok(
        _dump({"root": str(root), "total_matches": total, "files": shown, "scan_truncated": truncated_scan}),
        summary,
        count=total,
        path=str(root),
    )


# --------------------------------------------------------------------------
# filesystem.read
# --------------------------------------------------------------------------


class ReadArgs(ToolArgs):
    path: str = Field(description="File to read (text, markdown, csv, json, code, or PDF)")
    max_chars: int = Field(default=6000, ge=100, le=50_000)
    offset: int = Field(default=0, ge=0, description="Character offset to start reading from")


def _read_pdf(path: Path, limit: int) -> tuple[str, int]:
    from pypdf import PdfReader  # lazy import keeps startup fast

    reader = PdfReader(str(path))
    chunks: list[str] = []
    length = 0
    for page in reader.pages:
        text = page.extract_text() or ""
        chunks.append(text)
        length += len(text)
        if length >= limit:
            break
    return "\n".join(chunks), len(reader.pages)


def _looks_binary(path: Path) -> bool:
    with path.open("rb") as fh:
        sample = fh.read(TEXT_SAMPLE_BYTES)
    return b"\x00" in sample


@_guarded
def read_file(args: ReadArgs, ctx: ToolContext) -> ToolResult:
    path = ctx.path_policy.resolve(args.path)
    if not path.is_file():
        return ToolResult.fail(f"Not a file: {path}", path=str(path))
    limit = args.offset + args.max_chars
    metadata: dict = {"path": str(path), "size": path.stat().st_size}

    if path.suffix.lower() == ".pdf":
        try:
            text, pages = _read_pdf(path, limit + 1)
        except Exception as exc:  # pypdf raises many exception types
            return ToolResult.fail(f"Could not extract PDF text: {exc}", path=str(path))
        metadata["pages"] = pages
        if not text.strip():
            return ToolResult.fail("PDF contains no extractable text (scanned image?)", path=str(path))
    else:
        if _looks_binary(path):
            return ToolResult.fail(f"{path.name} is a binary file and cannot be read as text", path=str(path))
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            text = fh.read(limit + 1)

    content = text[args.offset : limit]
    truncated = len(text) > limit
    metadata.update(chars=len(content), truncated=truncated)
    header = f"[{path}]" + (f" (truncated; continue with offset={limit})" if truncated else "")
    return ToolResult.ok(
        f"{header}\n{content}",
        f"Read {len(content):,} chars from {path.name}" + (" (truncated)" if truncated else ""),
        **metadata,
    )


# --------------------------------------------------------------------------
# filesystem.write
# --------------------------------------------------------------------------


class WriteArgs(ToolArgs):
    path: str = Field(description="File to write")
    content: str
    overwrite: bool = Field(default=False, description="Replace the file if it already exists")
    append: bool = False


def _assess_write(args: WriteArgs, ctx: ToolContext) -> PermissionAssessment:
    path = ctx.path_policy.resolve(args.path)
    if path.exists() and not args.append:
        if not args.overwrite:
            return PermissionAssessment(
                allowed=False,
                risk=RiskLevel.WRITE,
                explanation=f"{path} already exists. Pass overwrite=true to replace it, or choose another name.",
            )
        return PermissionAssessment(
            allowed=True,
            risk=RiskLevel.DESTRUCTIVE,
            requires_confirmation=True,
            explanation=f"This will overwrite the existing file {path}.",
        )
    return PermissionAssessment(allowed=True, risk=RiskLevel.WRITE)


@_guarded
def write_file(args: WriteArgs, ctx: ToolContext) -> ToolResult:
    path = ctx.path_policy.resolve(args.path)
    existed = path.exists()
    if path.is_dir():
        return ToolResult.fail(f"{path} is a directory", path=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if args.append else "w", encoding="utf-8") as fh:
        fh.write(args.content)
    verb = "Appended to" if args.append else ("Overwrote" if existed else "Created")
    return ToolResult.ok(
        f"{verb} {path} ({len(args.content.encode())} bytes written)",
        f"{verb} {path}",
        path=str(path),
        change="modified" if existed else "created",
        bytes=len(args.content.encode()),
    )


# --------------------------------------------------------------------------
# filesystem.move / filesystem.copy
# --------------------------------------------------------------------------


class TransferArgs(ToolArgs):
    source: str
    destination: str = Field(
        description="Target path. An existing directory or a path ending in '/' means 'into this directory'."
    )


def _destination(args: TransferArgs, ctx: ToolContext, source: Path) -> Path:
    dest = ctx.path_policy.resolve(args.destination)
    if dest.is_dir() or args.destination.endswith(("/", os.sep)):
        dest = dest / source.name
    if dest.exists():
        raise ValueError(f"Destination already exists: {dest}")
    if dest == source:
        raise ValueError("Source and destination are the same")
    return dest


@_guarded
def move_path(args: TransferArgs, ctx: ToolContext) -> ToolResult:
    source = ctx.path_policy.resolve(args.source)
    if not source.exists():
        return ToolResult.fail(f"Not found: {source}")
    dest = _destination(args, ctx, source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(dest))
    return ToolResult.ok(
        f"Moved {source} -> {dest}", f"Moved {source.name} -> {dest}", source=str(source), destination=str(dest)
    )


@_guarded
def copy_path(args: TransferArgs, ctx: ToolContext) -> ToolResult:
    source = ctx.path_policy.resolve(args.source)
    if not source.exists():
        return ToolResult.fail(f"Not found: {source}")
    dest = _destination(args, ctx, source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, dest)
    else:
        shutil.copy2(source, dest)
    return ToolResult.ok(
        f"Copied {source} -> {dest}", f"Copied {source.name} -> {dest}", source=str(source), destination=str(dest)
    )


# --------------------------------------------------------------------------
# filesystem.delete
# --------------------------------------------------------------------------


class DeleteArgs(ToolArgs):
    path: str
    recursive: bool = Field(default=False, description="Required to delete a non-empty directory")


@_guarded
def delete_path(args: DeleteArgs, ctx: ToolContext) -> ToolResult:
    path = ctx.path_policy.resolve(args.path)
    if path in ctx.path_policy.allowed_roots or path == Path.home():
        return ToolResult.fail(f"Refusing to delete a root directory: {path}")
    if not path.exists():
        return ToolResult.fail(f"Not found: {path}")
    if path.is_dir():
        if args.recursive:
            shutil.rmtree(path)
        else:
            path.rmdir()  # fails if not empty
    else:
        path.unlink()
    return ToolResult.ok(f"Deleted {path}", f"Deleted {path}", path=str(path))


# --------------------------------------------------------------------------
# filesystem.duplicates
# --------------------------------------------------------------------------


class DuplicatesArgs(ToolArgs):
    path: str = Field(description="Directory to scan for duplicate files (by content hash)")
    pattern: str = Field(default="*", description="Only compare files whose name matches this glob, e.g. '*resume*'")
    recursive: bool = True
    min_size: int = Field(default=1, ge=0, description="Ignore files smaller than this many bytes")
    max_groups: int = Field(default=50, ge=1, le=500)


def _hash(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@_guarded
def find_duplicates(args: DuplicatesArgs, ctx: ToolContext) -> ToolResult:
    root = ctx.path_policy.resolve(args.path)
    if not root.is_dir():
        return ToolResult.fail(f"Not a directory: {root}")
    pattern = args.pattern.lower()
    by_size: dict[int, list[Path]] = {}
    for i, file in enumerate(_walk(root, args.recursive, include_hidden=False)):
        if i >= MAX_SCANNED_FILES:
            break
        if not fnmatch.fnmatch(file.name.lower(), pattern):
            continue
        try:
            if file.is_symlink():
                continue
            size = file.stat().st_size
        except OSError:
            continue
        if size >= args.min_size:
            by_size.setdefault(size, []).append(file)

    groups: list[dict] = []
    wasted = 0
    for size, files in sorted(by_size.items(), reverse=True):
        if len(files) < 2:
            continue
        by_hash: dict[str, list[str]] = {}
        for file in files:
            try:
                by_hash.setdefault(_hash(file), []).append(str(file))
            except OSError:
                continue
        for paths in by_hash.values():
            if len(paths) > 1:
                groups.append({"size": size, "size_human": format_size(size), "files": sorted(paths)})
                wasted += size * (len(paths) - 1)
    total = len(groups)
    return ToolResult.ok(
        _dump({"root": str(root), "duplicate_groups": groups[: args.max_groups], "total_groups": total,
               "reclaimable_bytes": wasted}),
        f"{total} duplicate group{'s' if total != 1 else ''} ({format_size(wasted)} reclaimable)",
        count=total,
        reclaimable_bytes=wasted,
    )


# --------------------------------------------------------------------------


def tools(config: Config) -> list[Tool]:
    return [
        Tool("filesystem.list", "List the contents of a directory with sizes and modification times.",
             ListArgs, list_directory, RiskLevel.READ),
        Tool("filesystem.search", "Find files by filename glob, size and date. Can sort by size to find the largest files.",
             SearchArgs, search_files, RiskLevel.READ),
        Tool("filesystem.read", "Read text from a file (plain text, markdown, csv, json, source code, PDF).",
             ReadArgs, read_file, RiskLevel.READ),
        Tool("filesystem.duplicates",
             "Find files with identical content in a directory. Use pattern to compare only specific files.",
             DuplicatesArgs, find_duplicates, RiskLevel.READ),
        Tool("filesystem.write", "Write text content to a file. Creates parent directories.",
             WriteArgs, write_file, RiskLevel.WRITE, assess=_assess_write),
        Tool("filesystem.copy", "Copy a file or directory. Never overwrites.",
             TransferArgs, copy_path, RiskLevel.WRITE),
        Tool("filesystem.move", "Move or rename a file or directory. Never overwrites. Requires user confirmation.",
             TransferArgs, move_path, RiskLevel.DESTRUCTIVE),
        Tool("filesystem.delete", "Permanently delete a file or directory. Requires user confirmation.",
             DeleteArgs, delete_path, RiskLevel.DESTRUCTIVE,
             enabled=config.allow_destructive,
             disabled_reason="destructive tools are opt-in; start with --allow-destructive"),
    ]
