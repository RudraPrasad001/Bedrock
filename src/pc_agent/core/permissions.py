"""Safety policy: path sandboxing, shell command classification, secret redaction.

Nothing in this module trusts the model. Every decision is made from the
concrete tool name and arguments.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

# Relative to the home directory. Never readable or writable by tools.
SENSITIVE_HOME_PATHS = (
    ".ssh",
    ".gnupg",
    ".aws",
    ".azure",
    ".kube",
    ".docker/config.json",
    ".netrc",
    ".pgpass",
    ".git-credentials",
    ".config/gcloud",
    ".config/gh",
    ".password-store",
    ".local/share/keyrings",
    ".mozilla",
    ".config/google-chrome",
    ".config/chromium",
)
SENSITIVE_NAMES = re.compile(
    r"^(\.env(\..*)?|id_rsa|id_ed25519|id_ecdsa|.*\.pem|.*\.key|credentials(\.json)?)$",
    re.IGNORECASE,
)
SYSTEM_SENSITIVE = (Path("/etc/shadow"), Path("/etc/sudoers"), Path("/root"))


class PathPolicyError(PermissionError):
    pass


@dataclass
class PathPolicy:
    allowed_roots: list[Path]

    def resolve(self, raw: str) -> Path:
        """Expand and resolve a user/model supplied path, enforcing the sandbox."""
        if not isinstance(raw, str) or not raw.strip():
            raise PathPolicyError("Path must be a non-empty string")
        if "\x00" in raw:
            raise PathPolicyError("Path contains a NUL byte")
        path = Path(os.path.expandvars(raw.strip())).expanduser()
        path = path.resolve()  # follows symlinks, so links can't escape the sandbox
        if self.is_sensitive(path):
            raise PathPolicyError(f"Access to sensitive path is not allowed: {path}")
        if not any(_is_relative_to(path, root) for root in self.allowed_roots):
            roots = ", ".join(str(r) for r in self.allowed_roots)
            raise PathPolicyError(f"Path {path} is outside the allowed roots ({roots})")
        return path

    @staticmethod
    def is_sensitive(path: Path) -> bool:
        home = Path.home().resolve()
        for rel in SENSITIVE_HOME_PATHS:
            if _is_relative_to(path, home / rel):
                return True
        for sys_path in SYSTEM_SENSITIVE:
            if _is_relative_to(path, sys_path):
                return True
        return any(SENSITIVE_NAMES.match(part) for part in path.parts)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------
# Shell commands
# --------------------------------------------------------------------------


class CommandClass(str, Enum):
    SAFE = "safe"  # read-only, runs without confirmation
    DANGEROUS = "dangerous"  # requires explicit confirmation
    BLOCKED = "blocked"  # never runs


SAFE_COMMANDS = {
    "pwd", "ls", "find", "du", "df", "ps", "uname", "whoami", "id", "hostname",
    "uptime", "date", "free", "stat", "file", "wc", "head", "tail", "cat",
    "grep", "tree", "which", "lsblk", "echo", "nproc", "lscpu", "sort", "uniq",
}
DANGEROUS_COMMANDS = {
    "rm": "deletes files", "rmdir": "deletes directories", "mv": "moves/renames files",
    "cp": "may overwrite files", "chmod": "changes permissions", "chown": "changes ownership",
    "chgrp": "changes group ownership", "dd": "writes raw data to devices",
    "mkfs": "formats a filesystem", "shutdown": "powers off the machine",
    "reboot": "reboots the machine", "poweroff": "powers off the machine",
    "halt": "halts the machine", "systemctl": "controls system services",
    "kill": "terminates processes", "pkill": "terminates processes",
    "killall": "terminates processes", "truncate": "truncates files",
    "shred": "irreversibly destroys files", "ln": "creates links",
    "mkdir": "creates directories", "touch": "creates/modifies files",
    "tee": "writes files", "curl": "accesses the network", "wget": "accesses the network",
    "git": "may modify repositories", "pip": "installs software",
    "npm": "installs software", "apt": "installs software", "pacman": "installs software",
    "brew": "installs software", "crontab": "modifies scheduled jobs",
}
# Commands that are never run by the agent, confirmation or not.
BLOCKED_COMMANDS = {
    "sudo": "privilege escalation is not allowed",
    "su": "privilege escalation is not allowed",
    "doas": "privilege escalation is not allowed",
    "env": "would expose environment secrets",
    "printenv": "would expose environment secrets",
    "export": "would modify the environment",
    "eval": "arbitrary evaluation is not allowed",
    "exec": "arbitrary execution is not allowed",
    "bash": "nested shells are not allowed",
    "sh": "nested shells are not allowed",
    "zsh": "nested shells are not allowed",
    "python": "use python.execute instead",
    "python3": "use python.execute instead",
}
# Flags that turn an otherwise safe command into a dangerous one.
DANGEROUS_FLAGS = {
    "find": {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprintf", "-fls"},
    "sort": {"-o", "--output"},
}
SHELL_METACHARACTERS = re.compile(r"[;&|<>`$(){}\n\r\\]")


@dataclass
class CommandAssessment:
    classification: CommandClass
    argv: list[str]
    explanation: str


def classify_command(command: str, path_policy: PathPolicy | None = None) -> CommandAssessment:
    """Classify a shell command without running it.

    Commands are executed without a shell (no pipes, redirects, globbing or
    substitution), so any shell metacharacter is rejected outright.
    """
    if not command or not command.strip():
        return CommandAssessment(CommandClass.BLOCKED, [], "Empty command")
    if SHELL_METACHARACTERS.search(command):
        return CommandAssessment(
            CommandClass.BLOCKED,
            [],
            "Shell metacharacters (; & | < > ` $ ( ) { } \\) are not allowed; "
            "commands run without a shell. Use a dedicated tool or a single command.",
        )
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return CommandAssessment(CommandClass.BLOCKED, [], f"Could not parse command: {exc}")

    program = os.path.basename(argv[0])
    if program in BLOCKED_COMMANDS:
        return CommandAssessment(CommandClass.BLOCKED, argv, f"'{program}' {BLOCKED_COMMANDS[program]}")
    if program.startswith("mkfs"):
        return CommandAssessment(CommandClass.DANGEROUS, argv, "Formats a filesystem")
    if program in DANGEROUS_COMMANDS:
        return CommandAssessment(CommandClass.DANGEROUS, argv, f"'{program}' {DANGEROUS_COMMANDS[program]}")
    if program not in SAFE_COMMANDS:
        return CommandAssessment(
            CommandClass.DANGEROUS, argv, f"'{program}' is not on the read-only allowlist"
        )

    risky = DANGEROUS_FLAGS.get(program, set()) & set(argv[1:])
    if risky:
        return CommandAssessment(
            CommandClass.DANGEROUS, argv, f"'{program}' with {', '.join(sorted(risky))} can modify files"
        )

    # Safe commands must not be used to read secrets the filesystem tools would refuse.
    for arg in argv[1:]:
        if arg.startswith("-"):
            continue
        candidate = Path(os.path.expanduser(arg))
        if PathPolicy.is_sensitive(candidate.resolve() if candidate.exists() else candidate):
            return CommandAssessment(CommandClass.BLOCKED, argv, f"'{arg}' is a sensitive path")
    return CommandAssessment(CommandClass.SAFE, argv, "Read-only command")


# --------------------------------------------------------------------------
# Secrets
# --------------------------------------------------------------------------

SECRET_ENV_PATTERN = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)", re.IGNORECASE)


def is_secret_env_var(name: str) -> bool:
    return bool(SECRET_ENV_PATTERN.search(name))


def secret_values() -> list[str]:
    """Values of secret-looking environment variables (min length avoids noise)."""
    return [v for k, v in os.environ.items() if is_secret_env_var(k) and len(v) >= 8]


def redact(text: str | None) -> str | None:
    """Remove any secret environment values from text before it reaches the LLM."""
    if not text:
        return text
    for value in secret_values():
        if value in text:
            text = text.replace(value, "[REDACTED]")
    return text


def scrubbed_environment() -> dict[str, str]:
    """A copy of os.environ with secret variables removed, for subprocesses."""
    return {k: v for k, v in os.environ.items() if not is_secret_env_var(k)}
