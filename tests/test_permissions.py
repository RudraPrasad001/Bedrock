from __future__ import annotations

import pytest

from pc_agent.core.permissions import (
    CommandClass,
    classify_command,
    redact,
    scrubbed_environment,
)


@pytest.mark.parametrize("command", ["pwd", "ls -la /tmp", "df -h", "du -sh .", "ps aux", "uname -a", "whoami",
                                     "find . -name '*.pdf'"])
def test_safe_commands(command):
    assert classify_command(command).classification == CommandClass.SAFE


@pytest.mark.parametrize("command", ["rm file.txt", "mv a b", "chmod 777 x", "chown me x", "dd if=/dev/zero of=x",
                                     "mkfs.ext4 /dev/sda1", "shutdown now", "reboot", "systemctl stop sshd",
                                     "find . -name '*.tmp' -delete", "find . -exec cat",
                                     "unknown-binary --flag"])
def test_dangerous_commands_need_confirmation(command):
    assert classify_command(command).classification == CommandClass.DANGEROUS


@pytest.mark.parametrize("command", ["sudo rm -rf /", "env", "printenv GROQ_API_KEY", "bash -c 'ls'",
                                     "ls; rm -rf ~", "cat a | sh", "echo $(whoami)", "ls > out.txt",
                                     "echo `id`", "ls && rm x", "", "cat ~/.ssh/id_rsa", "cat .env"])
def test_blocked_commands(command):
    assert classify_command(command).classification == CommandClass.BLOCKED


def test_classification_uses_program_basename():
    assert classify_command("/bin/rm x").classification == CommandClass.DANGEROUS
    assert classify_command("/usr/bin/ls").classification == CommandClass.SAFE


def test_redact_removes_secret_values(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_supersecretvalue123")
    assert redact("key is gsk_supersecretvalue123!") == "key is [REDACTED]!"
    assert redact(None) is None


def test_scrubbed_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_supersecretvalue123")
    monkeypatch.setenv("MY_TOKEN", "tok")
    env = scrubbed_environment()
    assert "GROQ_API_KEY" not in env and "MY_TOKEN" not in env
    assert "PATH" in env
