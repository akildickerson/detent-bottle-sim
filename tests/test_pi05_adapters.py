import numpy as np
import pytest
from integration_pi05.actions import resample, ActionAdapter
from detent_bottle.core import DetentBottleSim


def test_resampling_preserves_displacement():
    chunk = np.zeros((15, 8)); chunk[:, :7] = np.arange(15)[:, None] / 15
    output = np.array(list(resample(chunk, 20)))
    np.testing.assert_allclose(output[:, :7].sum(0), chunk[:, :7].sum(0))
    with pytest.raises(ValueError): list(resample(np.zeros((10, 8)), 4))


def test_fk_does_not_write_physics():
    sim = DetentBottleSim()
    try:
        before = sim.data.qpos.copy()
        adapter = ActionAdapter(sim)
        adapter.apply(np.zeros(8))
        np.testing.assert_array_equal(sim.data.qpos, before)
        assert sim.grip_targets[0] == .045
        adapter.apply(np.r_[np.zeros(7), 1])
        assert sim.grip_targets[0] == 0
    finally: sim.close()


@pytest.mark.parametrize('ticks', [0, -1, 21, 1.5, True])
def test_resampling_rejects_invalid_horizon(ticks):
    with pytest.raises(ValueError): list(resample(np.zeros((15, 8)), ticks))


def test_fractional_tick_convention():
    chunk = np.zeros((15, 8)); chunk[:, 0] = 1
    cmds = np.array(list(resample(chunk, 4)))
    np.testing.assert_allclose(cmds[:, 0], .75)
    # Four 50ms ticks represent three normalized 1/15s increments.
    assert cmds[:, 0].sum() == pytest.approx(3)


def test_observation_closure_and_privacy():
    from integration_pi05.observations import request
    obs = {'agentview': np.zeros((224,224,3), dtype=np.uint8),
           'left_wrist': np.zeros((224,224,3), dtype=np.uint8),
           'joint_pos': np.arange(14), 'gripper_qpos': np.array([.0225,.0225,.045,.045]),
           'privileged': {'bottle_pos': [0,0,1]}}
    payload = request(obs, 'Open the bottle')
    assert payload['observation/gripper_position'][0] == pytest.approx(.5)
    assert len(payload) == 5
    np.testing.assert_array_equal(payload['observation/joint_position'], np.arange(7))
