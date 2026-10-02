"""Secrets: sensitive files, key material, environment values and output redaction."""

from __future__ import annotations

import dataclasses

import pytest

from tests.benchmark.security_scenarios import AUTHORIZED, UNAUTHORIZED, case_ids, cases_in, run_case

CASES = cases_in("secret_protection")


@pytest.mark.parametrize("case", CASES, ids=case_ids(CASES))
def test_secret_is_not_exposed(case):
    record = run_case(case)
    assert not record.leaked, record.note
    assert record.prevented
    assert record.side_effects == []
    assert record.passed


def test_leak_detector_is_not_blind():
    """Negative control: redaction lives in the executor, so a direct tool call
    returns the raw secret. The benchmark must flag that as a leak."""
    case = next(c for c in UNAUTHORIZED if c.id == "output-secret-redaction")
    record = run_case(dataclasses.replace(case, layer="direct"))
    assert record.leaked and not record.passed


def test_harmless_names_containing_key_are_readable():
    case = next(c for c in AUTHORIZED if c.id == "valid-read-keyword-name" and c.layer == "executor")
    assert not run_case(case).prevented
