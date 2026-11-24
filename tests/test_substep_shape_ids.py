"""Contact candidates retained by an Ant step must survive subsequent substeps."""

import unittest
from unittest.mock import patch

import torch
import warp as wp

from rewarped.envs.dflex.ant import Ant


class TestSubstepShapeIds(unittest.TestCase):
    def test_contact_arrays_survive_subsequent_substeps(self):
        env = Ant(num_envs=2, device="cpu", seed=23, no_grad=False, use_graph_capture=False)
        env.save_contact_shape_ids = True
        env.reset()
        buffers = []
        collide = wp.sim.collide

        def record(model, state):
            collide(model, state)
            buffers.append((model.rigid_contact_broad_shape0, model.rigid_contact_broad_shape1))

        with patch.object(wp.sim, "collide", record):
            env.step(torch.zeros(env.num_envs, env.num_act))
        self.assertEqual(len(buffers), env.sim_substeps)
        for slots in zip(*buffers):
            self.assertEqual(len({array.ptr for array in slots}), env.sim_substeps)
