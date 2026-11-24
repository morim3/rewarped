"""Regression for the rigid-contact adjoint on independent simulated bodies."""

import unittest
from pathlib import Path

import torch

from rewarped.envs.dflex.ant import Ant


class TestRigidContactAdjoint(unittest.TestCase):
    def test_independent_ants_have_zero_cross_gradients(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(23)
            env = Ant(
                num_envs=8, device="cpu", no_grad=False, use_graph_capture=False, randomize=True, no_env_offset=True
            )
            env.asset_dir = str(Path(__file__).resolve().parents[1] / "rewarped/assets")
            env.reset()
            for _ in range(4):
                env.initialize_trajectory()
                actions, rewards = [], []
                for _ in range(8):
                    action = torch.randn(env.num_envs, env.num_act).tanh().requires_grad_()
                    _, reward, _, _ = env.step(action)
                    actions.append(action)
                    rewards.append(reward[0])
                grad = torch.stack(torch.autograd.grad(torch.stack(rewards).sum(), actions))
                self.assertTrue(grad.isfinite().all().item())
                self.assertGreater(grad[:, 0].abs().max().item(), 0.0)
                self.assertEqual(torch.count_nonzero(grad[:, 1:]).item(), 0)
