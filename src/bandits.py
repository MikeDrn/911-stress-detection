"""Multi-armed bandit policies (Course 4, section 5). Pure NumPy, no other imports.

Arms are models (0 = champion / control, 1 = challenger / treatment). The
reward of a call is 1 when the model's alert decision matched the ground truth
reported afterwards by the dispatcher, else 0.

Every policy returns the *probability* with which each arm is chosen
(the propensity), because the course asks us to log it: it enables off-policy
evaluation later. A traffic floor keeps every arm alive (slide 5.5).

This file is also imported by the Streamlit console (offline regret
simulation), so it must stay free of project imports.
"""

from __future__ import annotations

import numpy as np

POLICIES = ("epsilon_greedy", "ucb1", "thompson")


def apply_floor(probs: np.ndarray, floor: float) -> np.ndarray:
    """Mix with the uniform distribution so each arm gets at least ``floor``."""
    k = len(probs)
    floor = min(max(floor, 0.0), 1.0 / k)
    return floor + (1.0 - k * floor) * np.asarray(probs, dtype=float)


def epsilon_greedy_probs(means: np.ndarray, epsilon: float) -> np.ndarray:
    k = len(means)
    probs = np.full(k, epsilon / k)
    best = np.flatnonzero(means == means.max())
    probs[best] += (1.0 - epsilon) / len(best)
    return probs


def ucb1_probs(means: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """UCB1 is deterministic: all probability on the arm with the highest bound."""
    k = len(means)
    probs = np.zeros(k)
    untried = np.flatnonzero(counts == 0)
    if len(untried):
        probs[untried] = 1.0 / len(untried)
        return probs
    t = counts.sum()
    bound = means + np.sqrt(2.0 * np.log(t) / counts)
    best = np.flatnonzero(bound == bound.max())
    probs[best] = 1.0 / len(best)
    return probs


def thompson_probs(
    successes: np.ndarray, failures: np.ndarray, rng: np.random.Generator, draws: int = 4000
) -> np.ndarray:
    """P(arm is best) under Beta(1 + s, 1 + f) posteriors, estimated by Monte Carlo."""
    samples = rng.beta(1 + successes[:, None], 1 + failures[:, None], size=(len(successes), draws))
    winners = samples.argmax(axis=0)
    return np.bincount(winners, minlength=len(successes)) / draws


def policy_probs(
    policy: str,
    successes: np.ndarray,
    counts: np.ndarray,
    rng: np.random.Generator,
    epsilon: float = 0.1,
    floor: float = 0.05,
) -> np.ndarray:
    successes = np.asarray(successes, dtype=float)
    counts = np.asarray(counts, dtype=float)
    means = np.divide(successes, counts, out=np.full(len(counts), 0.5), where=counts > 0)
    if policy == "epsilon_greedy":
        probs = epsilon_greedy_probs(means, epsilon)
    elif policy == "ucb1":
        probs = ucb1_probs(means, counts)
    elif policy == "thompson":
        probs = thompson_probs(successes, counts - successes, rng)
    else:
        raise ValueError(f"unknown policy '{policy}', choose one of {POLICIES}")
    return apply_floor(probs, floor)


def choose(probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(probs), p=probs / probs.sum()))


# --------------------------------------------------------------------------- #
# Offline simulation: regret of each policy vs a 50/50 A/B test (slide 5.2)
# --------------------------------------------------------------------------- #
def simulate_regret(
    true_means: list[float],
    rounds: int = 1000,
    seeds: int = 20,
    epsilon: float = 0.1,
    floor: float = 0.05,
) -> dict[str, np.ndarray]:
    """Average cumulative pseudo-regret curves, one per policy plus ``ab_test``."""
    mu = np.asarray(true_means, dtype=float)
    gaps = mu.max() - mu
    out: dict[str, np.ndarray] = {}
    for policy in (*POLICIES, "ab_test"):
        curves = np.zeros((seeds, rounds))
        for s in range(seeds):
            rng = np.random.default_rng(1000 + s)
            succ, cnt = np.zeros(len(mu)), np.zeros(len(mu))
            regret = 0.0
            for t in range(rounds):
                if policy == "ab_test":
                    arm = t % len(mu)
                elif policy == "thompson":  # one posterior draw per arm (fast, same law)
                    if rng.random() < floor * len(mu):
                        arm = int(rng.integers(len(mu)))
                    else:
                        arm = int(np.argmax(rng.beta(1 + succ, 1 + cnt - succ)))
                else:
                    arm = choose(policy_probs(policy, succ, cnt, rng, epsilon, floor), rng)
                reward = float(rng.random() < mu[arm])
                succ[arm] += reward
                cnt[arm] += 1
                regret += gaps[arm]
                curves[s, t] = regret
        out[policy] = curves.mean(axis=0)
    return out
