from __future__ import annotations

import json
import os
from pathlib import Path

from pc_agent.core.decisions import RiskLevel
from pc_agent.tools import filesystem as fs


def make_pdf(path: Path, text: str) -> None:
    """Write a minimal single-page PDF containing `text`."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


def populate(root: Path) -> None:
    (root / "docs").mkdir()
    (root / "docs" / "kubernetes-intro.md").write_text("# Kubernetes\nPods and services.")
    (root / "docs" / "docker.md").write_text("# Docker\nContainers.")
    (root / "big.bin").write_bytes(b"\x00" * 5000)
    (root / ".hidden").write_text("secret-ish")
    (root / "notes.txt").write_text("hello world")


def test_list_directory(ctx, sandbox):
    populate(sandbox)
    result = fs.list_directory(fs.ListArgs(path=str(sandbox)), ctx)
    assert result.success
    names = [e["name"] for e in json.loads(result.output)["entries"]]
    assert names[0] == "docs"  # directories first
    assert ".hidden" not in names
    assert {"big.bin", "notes.txt"} <= set(names)


def test_list_missing_directory_fails_gracefully(ctx, sandbox):
    result = fs.list_directory(fs.ListArgs(path=str(sandbox / "nope")), ctx)
    assert not result.success
    assert "Not a directory" in result.error


def test_search_pattern_is_recursive_and_case_insensitive(ctx, sandbox):
    populate(sandbox)
    (sandbox / "docs" / "README.MD").write_text("x")
    result = fs.search_files(fs.SearchArgs(path=str(sandbox), pattern="*.md"), ctx)
    files = [Path(f["path"]).name for f in json.loads(result.output)["files"]]
    assert sorted(files) == ["README.MD", "docker.md", "kubernetes-intro.md"]
    assert result.metadata["count"] == 3


def test_search_size_filter_and_sort(ctx, sandbox):
    populate(sandbox)
    result = fs.search_files(
        fs.SearchArgs(path=str(sandbox), min_size=20, sort_by="size", descending=True), ctx
    )
    files = json.loads(result.output)["files"]
    assert Path(files[0]["path"]).name == "big.bin"
    assert all(f["size"] >= 20 for f in files)


def test_search_modified_after(ctx, sandbox):
    populate(sandbox)
    old = sandbox / "notes.txt"
    os.utime(old, (0, 0))
    result = fs.search_files(fs.SearchArgs(path=str(sandbox), modified_after="2000-01-01"), ctx)
    names = [Path(f["path"]).name for f in json.loads(result.output)["files"]]
    assert "notes.txt" not in names and "docker.md" in names


def test_read_text_with_truncation_and_offset(ctx, sandbox):
    (sandbox / "long.txt").write_text("a" * 150 + "b" * 150)
    first = fs.read_file(fs.ReadArgs(path=str(sandbox / "long.txt"), max_chars=150), ctx)
    assert first.success and first.metadata["truncated"]
    assert "offset=150" in first.output
    second = fs.read_file(fs.ReadArgs(path=str(sandbox / "long.txt"), max_chars=150, offset=150), ctx)
    assert second.output.endswith("b" * 150)
    assert not second.metadata["truncated"]


def test_read_binary_is_refused(ctx, sandbox):
    populate(sandbox)
    result = fs.read_file(fs.ReadArgs(path=str(sandbox / "big.bin")), ctx)
    assert not result.success and "binary" in result.error


def test_read_pdf(ctx, sandbox):
    make_pdf(sandbox / "k8s.pdf", "Kubernetes schedules pods")
    result = fs.read_file(fs.ReadArgs(path=str(sandbox / "k8s.pdf")), ctx)
    assert result.success, result.error
    assert "Kubernetes schedules pods" in result.output
    assert result.metadata["pages"] == 1


def test_write_creates_parents_and_tracks_change(ctx, sandbox):
    target = sandbox / "out" / "report.md"
    result = fs.write_file(fs.WriteArgs(path=str(target), content="# Report"), ctx)
    assert result.success and target.read_text() == "# Report"
    assert result.metadata["change"] == "created"


def test_write_overwrite_requires_flag_and_confirmation(ctx, sandbox):
    target = sandbox / "a.txt"
    target.write_text("old")
    refused = fs._assess_write(fs.WriteArgs(path=str(target), content="new"), ctx)
    assert not refused.allowed
    overwrite = fs._assess_write(fs.WriteArgs(path=str(target), content="new", overwrite=True), ctx)
    assert overwrite.allowed and overwrite.requires_confirmation
    assert overwrite.risk == RiskLevel.DESTRUCTIVE
    fresh = fs._assess_write(fs.WriteArgs(path=str(sandbox / "b.txt"), content="x"), ctx)
    assert fresh.allowed and not fresh.requires_confirmation


def test_move_into_new_directory(ctx, sandbox):
    populate(sandbox)
    result = fs.move_path(
        fs.TransferArgs(source=str(sandbox / "notes.txt"), destination=str(sandbox / "Text") + "/"), ctx
    )
    assert result.success, result.error
    assert (sandbox / "Text" / "notes.txt").exists()
    assert not (sandbox / "notes.txt").exists()


def test_copy_never_overwrites(ctx, sandbox):
    populate(sandbox)
    (sandbox / "copy.txt").write_text("existing")
    result = fs.copy_path(
        fs.TransferArgs(source=str(sandbox / "notes.txt"), destination=str(sandbox / "copy.txt")), ctx
    )
    assert not result.success and "already exists" in result.error
    assert (sandbox / "copy.txt").read_text() == "existing"


def test_delete_refuses_root_and_non_empty_dir(ctx, sandbox):
    populate(sandbox)
    assert not fs.delete_path(fs.DeleteArgs(path=str(sandbox)), ctx).success
    assert not fs.delete_path(fs.DeleteArgs(path=str(sandbox / "docs")), ctx).success
    assert fs.delete_path(fs.DeleteArgs(path=str(sandbox / "docs"), recursive=True), ctx).success
    assert not (sandbox / "docs").exists()


def test_duplicates(ctx, sandbox):
    (sandbox / "a.txt").write_text("same content")
    (sandbox / "sub").mkdir()
    (sandbox / "sub" / "b.txt").write_text("same content")
    (sandbox / "c.txt").write_text("different!!!")
    result = fs.find_duplicates(fs.DuplicatesArgs(path=str(sandbox)), ctx)
    groups = json.loads(result.output)["duplicate_groups"]
    assert len(groups) == 1
    assert {Path(p).name for p in groups[0]["files"]} == {"a.txt", "b.txt"}


def test_duplicates_pattern_filter(ctx, sandbox):
    for name in ("resume-a.pdf", "resume-b.pdf", "other-a.txt", "other-b.txt"):
        (sandbox / name).write_text("same" if name.startswith("resume") else "also same")
    result = fs.find_duplicates(fs.DuplicatesArgs(path=str(sandbox), pattern="*RESUME*"), ctx)
    groups = json.loads(result.output)["duplicate_groups"]
    assert [{Path(p).name for p in g["files"]} for g in groups] == [{"resume-a.pdf", "resume-b.pdf"}]


def test_sandbox_blocks_outside_paths(ctx, sandbox, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("nope")
    result = fs.read_file(fs.ReadArgs(path=str(outside)), ctx)
    assert not result.success and "outside the allowed roots" in result.error
    escape = fs.read_file(fs.ReadArgs(path=str(sandbox / ".." / "outside.txt")), ctx)
    assert not escape.success


def test_symlink_cannot_escape_sandbox(ctx, sandbox, tmp_path):
    (tmp_path / "secret.txt").write_text("top secret")
    (sandbox / "link.txt").symlink_to(tmp_path / "secret.txt")
    result = fs.read_file(fs.ReadArgs(path=str(sandbox / "link.txt")), ctx)
    assert not result.success


def test_sensitive_files_are_refused(ctx, sandbox):
    (sandbox / ".env").write_text("GROQ_API_KEY=abc")
    (sandbox / "server.pem").write_text("-----BEGIN")
    for name in (".env", "server.pem"):
        result = fs.read_file(fs.ReadArgs(path=str(sandbox / name)), ctx)
        assert not result.success and "sensitive" in result.error
    found = fs.search_files(fs.SearchArgs(path=str(sandbox), include_hidden=True), ctx)
    assert ".env" not in found.output
