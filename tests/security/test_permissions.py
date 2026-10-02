"""Permission policy: shell sanitisation, registry validation, opt-in tools, false rejections."""

from __future__ import annotations

import pytest

from tests.benchmark.security_scenarios import (
    ALL_CASES,
    AUTHORIZED,
    GROUPS,
    UNAUTHORIZED,
    case_ids,
    cases_in,
    run_case,
)

CASES = cases_in("permissions")


@pytest.mark.parametrize("case", CASES, ids=case_ids(CASES))
def test_unauthorized_operation_is_rejected(case):
    record = run_case(case)
    assert record.prevented, record.note
    assert record.side_effects == [] and not record.leaked


@pytest.mark.parametrize("case", AUTHORIZED, ids=case_ids(AUTHORIZED))
def test_valid_operation_is_not_rejected(case):
    record = run_case(case)
    assert not record.prevented, f"false rejection: {record.note}"
    assert not record.leaked


def test_every_unauthorized_case_belongs_to_exactly_one_group():
    grouped = [case_id for ids in GROUPS.values() for case_id in ids]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {c.id for c in UNAUTHORIZED}
    assert len({(c.id, c.layer) for c in ALL_CASES}) == len(ALL_CASES)
