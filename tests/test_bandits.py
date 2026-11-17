"""Multi-armed bandit policies (Course 4, section 5)."""

from __future__ import annotations

import numpy as np
import pytest

from src import bandits


def test_floor_keeps_every_arm_alive():
    p = bandits.apply_floor(np.array([1.0, 0.0]), 0.05)
    assert p.tolist() == pytest.approx([0.95, 0.05]) and p.sum() == pytest.approx(1.0)


def test_epsilon_greedy_probabilities():
    p = bandits.epsilon_greedy_probs(np.array([0.9, 0.5]), 0.1)
    assert p.tolist() == pytest.approx([0.95, 0.05])


def test_ucb1_tries_untried_arms_first_then_is_deterministic():
    assert bandits.ucb1_probs(np.array([0.9, 0.0]), np.array([10, 0])).tolist() == [0, 1]
    p = bandits.ucb1_probs(np.array([0.9, 0.5]), np.array([100, 100]))
    assert p.tolist() == [1.0, 0.0]


def test_thompson_prefers_the_better_arm():
    rng = np.random.default_rng(0)
    p = bandits.thompson_probs(np.array([90.0, 99.0]), np.array([10.0, 1.0]), rng)
    assert p[1] > 0.95
    p = bandits.thompson_probs(np.array([0.0, 0.0]), np.array([0.0, 0.0]), rng)
    assert abs(p[0] - 0.5) < 0.05  # no data: a coin flip


@pytest.mark.parametrize("policy", bandits.POLICIES)
def test_policy_probs_are_valid_distributions(policy):
    rng = np.random.default_rng(1)
    p = bandits.policy_probs(policy, np.array([8, 9]), np.array([10, 10]), rng, 0.1, 0.05)
    assert p.sum() == pytest.approx(1.0) and (p >= 0.05 - 1e-9).all()


def test_unknown_policy():
    with pytest.raises(ValueError):
        bandits.policy_probs("random", np.zeros(2), np.zeros(2), np.random.default_rng())


def test_bandits_have_lower_regret_than_ab_test():
    curves = bandits.simulate_regret([0.85, 0.97], rounds=400, seeds=5)
    final = {k: v[-1] for k, v in curves.items()}
    assert final["ab_test"] == pytest.approx(0.12 * 400 / 2, rel=1e-6)  # linear: delta*T/2
    assert final["thompson"] < final["ab_test"] / 2
    assert final["epsilon_greedy"] < final["ab_test"]
