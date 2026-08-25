import tempfile

import pytest

from mastercard_defence.contracts import RoundRecord
from mastercard_defence.memory import AttackMemory


def _make_record(seed: int, round_id: int, fraud_family: str) -> RoundRecord:
    return RoundRecord(
        seed=seed,
        round_id=round_id,
        attack_id=f"attack-{seed}-{round_id}",
        fraud_family=fraud_family,
        attack_hypothesis={"scenario": f"scenario-{round_id}"},
        attack_specification={"attack_family": fraud_family},
        generator_metadata={"backend": "procedural"},
        generation_stats={"train_attack_rows": 10},
        fidelity_evaluation={"passed": True},
        detector_evaluation={"f1": 0.5},
        agent3_analysis={"analysis_summary": f"summary-{round_id}"},
        identified_weaknesses=[f"weakness-{round_id}"],
        recommended_next_attack_directions=[f"direction-{round_id}"],
    )


def _memory() -> AttackMemory:
    return AttackMemory(tempfile.mktemp(suffix="_attack_memory_test.sqlite"))


def test_add_round_persists_all_structured_fields() -> None:
    memory = _memory()
    memory.add_round(_make_record(seed=1, round_id=1, fraud_family="account_takeover"))

    stored = memory.get_round(seed=1, round_id=1)

    assert stored is not None
    assert stored["attack_id"] == "attack-1-1"
    assert stored["fraud_family"] == "account_takeover"
    assert stored["attack_hypothesis"] == {"scenario": "scenario-1"}
    assert stored["identified_weaknesses"] == ["weakness-1"]
    assert stored["recommended_next_attack_directions"] == ["direction-1"]
    assert stored["status"] == "explored"
    memory.close()


def test_add_round_never_overwrites_existing_round() -> None:
    memory = _memory()
    memory.add_round(_make_record(seed=1, round_id=1, fraud_family="account_takeover"))

    with pytest.raises(ValueError):
        memory.add_round(_make_record(seed=1, round_id=1, fraud_family="merchant_abuse"))
    memory.close()


def test_get_recent_rounds_filters_by_family_and_seed() -> None:
    memory = _memory()
    memory.add_round(_make_record(seed=1, round_id=1, fraud_family="account_takeover"))
    memory.add_round(_make_record(seed=1, round_id=2, fraud_family="merchant_abuse"))
    memory.add_round(_make_record(seed=2, round_id=1, fraud_family="account_takeover"))

    seed_one_only = memory.get_recent_rounds(seed=1)
    assert {record["round_id"] for record in seed_one_only} == {1, 2}

    family_only = memory.get_recent_rounds(fraud_family="account_takeover")
    assert {record["seed"] for record in family_only} == {1, 2}

    seed_and_family = memory.get_recent_rounds(seed=1, fraud_family="account_takeover")
    assert len(seed_and_family) == 1
    assert seed_and_family[0]["round_id"] == 1
    memory.close()


def test_get_weakness_history_returns_expected_keys() -> None:
    memory = _memory()
    memory.add_round(_make_record(seed=1, round_id=1, fraud_family="account_takeover"))

    history = memory.get_weakness_history(seed=1)

    assert len(history) == 1
    entry = history[0]
    assert entry["round_id"] == 1
    assert entry["identified_weaknesses"] == ["weakness-1"]
    assert entry["recommended_next_attack_directions"] == ["direction-1"]
    memory.close()


def test_get_explored_families_counts_rounds_per_family() -> None:
    memory = _memory()
    memory.add_round(_make_record(seed=1, round_id=1, fraud_family="account_takeover"))
    memory.add_round(_make_record(seed=1, round_id=2, fraud_family="account_takeover"))
    memory.add_round(_make_record(seed=1, round_id=3, fraud_family="merchant_abuse"))

    counts = memory.get_explored_families(seed=1)

    assert counts == {"account_takeover": 2, "merchant_abuse": 1}
    memory.close()


def test_recent_context_is_seed_isolated_and_ordered() -> None:
    memory = _memory()
    memory.add_round(_make_record(seed=1, round_id=1, fraud_family="account_takeover"))
    memory.add_round(_make_record(seed=2, round_id=1, fraud_family="merchant_abuse"))

    context = memory.recent_context(seed=1)

    assert any("account_takeover" in line or "scenario-1" in line for line in context)
    assert all("merchant_abuse" not in line for line in context)
    memory.close()
