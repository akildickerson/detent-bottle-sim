import json
import numpy as np
import pytest
from integration_pi05.evaluate import load_settings, load_episodes, run_episode


def test_manifest_expands_conditions_without_randomizing(tmp_path):
    manifest = tmp_path / 'episodes.json'
    manifest.write_text(json.dumps([{'bottle_pose': [0, .01, 5], 'seeds': [0,42], 'max_steps': 3}]))
    episodes = load_episodes(manifest)
    assert len(episodes) == 2
    assert episodes[1] == {'seed':42, 'bottle_pose':[0,.01,5], 'max_steps':3}


@pytest.mark.parametrize('override', [{'execution_ticks':21}, {'image_size':128},
    {'config':'pi05_libero'}, {'port':65536}, {'inference_timeout':0}])
def test_reject_incompatible_settings(override):
    with pytest.raises(ValueError): load_settings(overrides=override)


def test_setup_timeout_is_recorded_as_task_failure(tmp_path):
    config = load_settings()
    # Avoid GPU/rendering: mock observations only, real physics and controller.
    from unittest.mock import patch
    with patch('detent_bottle.gym_env.DetentBottleEnv._obs', return_value={}):
        final = run_episode(tmp_path / 'episode', 'hybrid', {'seed':0, 'bottle_pose':None, 'max_steps':1}, config)
    assert final['reason'] == 'time_limit'
    assert final['failure_category'] == 'setup_failure'
    assert not final['setup_complete']
    assert final['error'] is None
    trace = np.load(tmp_path / 'episode' / 'trajectory.npz')
    assert trace['eef_pos'].shape == (1,2,3)
    assert 'lid_deg' in trace and 'bottle_qpos' in trace
