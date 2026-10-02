"""Destructive operations: confirmation, cancellation, dry-run and overwrite protection."""

from __future__ import annotations

import dataclasses

import pytest

from tests.benchmark.security_scenarios import AUTHORIZED, case_ids, cases_in, run_case

CASES = cases_in("destructive_actions")


@pytest.mark.parametrize("case", CASES, ids=case_ids(CASES))
def test_destructive_action_has_no_effect(case):
    record = run_case(case)
    assert record.prevented, record.note
    assert record.side_effects == []
    assert record.passed


def test_side_effect_detector_is_not_blind():
    """Negative control: an approved move really changes the tree, and the
    benchmark would count it as an unauthorized side effect."""
    approved = next(c for c in AUTHORIZED if c.id == "valid-approved-move")
    record = run_case(dataclasses.replace(approved, authorized=False))
    assert not record.prevented
    assert record.side_effects and not record.passed
