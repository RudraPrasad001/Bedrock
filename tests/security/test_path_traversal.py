"""Sandbox escapes: ../ traversal, absolute paths outside the roots, symlinks."""

from __future__ import annotations

import pytest

from tests.benchmark.security_scenarios import case_ids, cases_in, run_case

CASES = cases_in("path_traversal")


@pytest.mark.parametrize("case", CASES, ids=case_ids(CASES))
def test_escape_is_blocked_without_side_effects(case):
    record = run_case(case)
    assert record.prevented, record.note
    assert record.side_effects == []
    assert not record.leaked
    assert record.passed


def test_both_layers_are_exercised():
    assert {c.layer for c in CASES} == {"direct", "executor"}
