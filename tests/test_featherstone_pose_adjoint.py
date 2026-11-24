import unittest
from unittest.mock import patch

import torch

import warp as wp
from rewarped.envs.dflex.ant import Ant
from rewarped.warp.integrator_featherstone import eval_rigid_fk


class TestFeatherstonePoseAdjoint(unittest.TestCase):
    def test_unjointed_body_retains_its_pose_and_gradient(self):
        class AntWithStaticBody(Ant):
            def create_articulation(self, builder):
                super().create_articulation(builder)
                builder.add_body(origin=wp.transform((3.0, 4.0, 5.0), wp.quat_identity()))

        env = AntWithStaticBody(num_envs=1, device="cpu", use_graph_capture=False, randomize=False)
        env.reset()
        src, dst = env.model.state(), env.model.state()
        body = env.model.body_count - 1
        self.assertNotIn(body, env.model.joint_child.numpy())
        solver_poses = []
        launch = wp.launch

        def record(kernel, *args, **kwargs):
            if kernel is eval_rigid_fk:
                solver_poses.append(kwargs["outputs"][0])
            return launch(kernel, *args, **kwargs)

        with wp.Tape() as tape, patch.object(wp, "launch", record):
            env.integrator.simulate(env.model, src, dst, env.sim_dt)
        (poses,) = solver_poses
        self.assertEqual(poses.numpy()[body, :3].tolist(), [3.0, 4.0, 5.0])
        seed = poses.numpy() * 0.0
        seed[body, 0] = 1.0
        tape.backward(grads={poses: wp.array(seed, dtype=wp.transform, device="cpu")})
        self.assertEqual(src.body_q.grad.numpy()[body, 0], 1.0)

    def test_contacting_ant_action_gradient(self):
        env = Ant(
            num_envs=3,
            device="cpu",
            seed=23,
            no_grad=False,
            use_graph_capture=False,
            randomize=False,
            no_env_offset=True,
        )
        env.reset()
        for _ in range(16):
            env.initialize_trajectory()
            env.step(torch.zeros((3, env.num_act)))

        initial = env.initialize_trajectory()
        torch.testing.assert_close(initial, initial[0:1].expand_as(initial), rtol=0, atol=0)
        epsilon = 0.004
        actions = torch.zeros((3, env.num_act))
        actions[1, 1], actions[2, 1] = epsilon, -epsilon
        actions.requires_grad_()
        _, rewards, done, _ = env.step(actions)
        self.assertFalse(done.any().item())
        grad = torch.autograd.grad(rewards[0], actions)[0]
        centered = (rewards[1].item() - rewards[2].item()) / (2 * epsilon)
        self.assertGreater(abs(centered), 0.05)
        self.assertAlmostEqual(grad[0, 1].item(), centered, delta=0.001)
        self.assertEqual(grad[1:].count_nonzero().item(), 0)
