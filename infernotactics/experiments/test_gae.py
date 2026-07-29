"""
Hand-computed correctness check for compute_gae() (train_actor_critic_multi_v4.py),
run standalone BEFORE trusting v4's smoke test -- see that file's module
docstring's "Smoke-test verification" section.

    python -m src.train.test_gae
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.train_actor_critic_multi_v4 import compute_gae  # noqa: E402


def _approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_gae_three_step_hand_worked():
    """3 ticks, gamma=0.99, lambda=0.95, rewards=[1, 2, 3], values=[0.5, 0.6, 0.7],
    bootstrap_value=0.0 (true terminal -- fire contained on tick 3).

    Hand-worked (t indexed 0,1,2; computed backward, matching the recursion
    A_t = delta_t + gamma*lambda*A_{t+1}):
      delta_2 = r_2 + gamma*V_bootstrap - V_2 = 3 + 0.99*0.0 - 0.7 = 2.3
      A_2 = delta_2 = 2.3                      (nothing after t=2)
      delta_1 = r_1 + gamma*V_2 - V_1 = 2 + 0.99*0.7 - 0.6 = 2.093
      A_1 = delta_1 + gamma*lambda*A_2 = 2.093 + 0.99*0.95*2.3 = 2.093 + 2.16315 = 4.25615
      delta_0 = r_0 + gamma*V_1 - V_0 = 1 + 0.99*0.6 - 0.5 = 1.094
      A_0 = delta_0 + gamma*lambda*A_1 = 1.094 + 0.9405*4.25615 = 1.094 + 4.002909075 = 5.096909075
    returns = advantages + values:
      G_2 = 2.3 + 0.7 = 3.0
      G_1 = 4.25615 + 0.6 = 4.85615
      G_0 = 5.096909075 + 0.5 = 5.596909075
    """
    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 0.6, 0.7]
    gamma, lam = 0.99, 0.95

    advantages, returns = compute_gae(rewards, values, bootstrap_value=0.0, gamma=gamma, lam=lam)

    expected_adv = [5.096909075, 4.25615, 2.3]
    expected_ret = [5.596909075, 4.85615, 3.0]

    assert len(advantages) == 3 and len(returns) == 3, f"wrong length: {advantages}, {returns}"
    for i, (a, exp_a, r, exp_r) in enumerate(zip(advantages, expected_adv, returns, expected_ret)):
        assert _approx(a, exp_a, tol=1e-4), f"advantage[{i}]={a} != expected {exp_a}"
        assert _approx(r, exp_r, tol=1e-4), f"return[{i}]={r} != expected {exp_r}"
    print("  PASS: 3-step hand-worked GAE (true terminal, bootstrap_value=0.0)")


def test_gae_bootstrap_nonzero_on_timeout():
    """Same rewards/values, but bootstrap_value=1.0 (timeout truncation --
    fire NOT contained, episode cut off at MAX_TICKS with next-state value
    estimate 1.0). Only delta_2/A_2 (and everything upstream of it) changes.
      delta_2 = 3 + 0.99*1.0 - 0.7 = 3.29
      A_2 = 3.29
      delta_1 = 2 + 0.99*0.7 - 0.6 = 2.093  (unchanged -- doesn't depend on bootstrap)
      A_1 = 2.093 + 0.99*0.95*3.29 = 2.093 + 3.0952... = 5.1882...
      delta_0 = 1 + 0.99*0.6 - 0.5 = 1.094  (unchanged)
      A_0 = 1.094 + 0.99*0.95*5.1882... = 1.094 + 4.8794... = 5.9734...
    """
    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 0.6, 0.7]
    gamma, lam = 0.99, 0.95

    advantages, _ = compute_gae(rewards, values, bootstrap_value=1.0, gamma=gamma, lam=lam)
    advantages_zero_bootstrap, _ = compute_gae(rewards, values, bootstrap_value=0.0, gamma=gamma, lam=lam)

    assert _approx(advantages[2], 3.29, tol=1e-4), f"A_2={advantages[2]} != 3.29"
    # Nonzero bootstrap must change every advantage in the trajectory (it propagates backward),
    # and specifically must make them LARGER here (positive bootstrap value = "more reward still coming").
    for i in range(3):
        assert advantages[i] > advantages_zero_bootstrap[i], (
            f"advantage[{i}] with bootstrap=1.0 ({advantages[i]}) should exceed "
            f"bootstrap=0.0 ({advantages_zero_bootstrap[i]})"
        )
    print("  PASS: nonzero bootstrap value (timeout truncation) propagates backward through GAE")


def test_gae_single_step_reduces_to_td_residual():
    """With exactly 1 step, GAE has nothing to blend across -- A_0 must
    equal the plain 1-step TD residual regardless of lambda."""
    rewards = [5.0]
    values = [2.0]
    gamma = 0.99
    for lam in (0.0, 0.5, 0.95, 1.0):
        advantages, returns = compute_gae(rewards, values, bootstrap_value=0.0, gamma=gamma, lam=lam)
        expected = rewards[0] + gamma * 0.0 - values[0]
        assert _approx(advantages[0], expected), f"lambda={lam}: A_0={advantages[0]} != {expected}"
    print("  PASS: single-step GAE reduces to the plain TD residual for any lambda")


def main():
    print("Running compute_gae() unit tests...")
    test_gae_three_step_hand_worked()
    test_gae_bootstrap_nonzero_on_timeout()
    test_gae_single_step_reduces_to_td_residual()
    print("ALL GAE TESTS PASSED")


if __name__ == "__main__":
    main()
