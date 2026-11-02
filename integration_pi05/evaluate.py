"""Reproducible scripted/hybrid episodes: python -m integration_pi05.evaluate."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import random
import subprocess
import time

import numpy as np

from detent_bottle.gym_env import CAMERAS, DetentBottleEnv
from detent_bottle.policy import opening_sequence
from .actions import ActionAdapter, resample
from .observations import request

DEFAULTS = {
    'checkpoint': 'gs://openpi-assets/checkpoints/pi05_droid',
    'config': 'pi05_droid', 'cameras': ['agentview', 'left_wrist'],
    'image_size': 224, 'max_steps': 2400, 'execution_ticks': 4,
    'prompt': 'Fold the safety loop fully down, then press the rocker release to open the bottle. The bottle is held steady.',
    'host': 'localhost', 'port': 8000,
    'connection_timeout': 600, 'inference_timeout': 60,
}


def metadata():
    """Record client-side provenance; server provenance must be supplied separately."""
    def git(*args):
        result = subprocess.run(['git', *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    packages = {}
    for package in ('mujoco', 'numpy', 'scipy', 'gymnasium', 'Pillow', 'openpi-client'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': packages, 'simulator_revision': git('rev-parse', 'HEAD'),
            'working_tree_status': git('status', '--short'),
            'inference_rng_controlled': False,
            'inference_rng_note': 'Client seed does not set the remote policy RNG. Attach server revision, checkpoint hash and RNG settings separately.'}


def load_settings(path=None, overrides=None):
    config = dict(DEFAULTS)
    if path:
        supplied = json.loads(Path(path).read_text())
        if not isinstance(supplied, dict):
            raise ValueError('Configuration must be a JSON object')
        unknown = supplied.keys() - DEFAULTS.keys()
        if unknown:
            raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
        config.update(supplied)
    config.update({k: v for k, v in (overrides or {}).items() if v is not None})
    for name in ('image_size', 'max_steps', 'execution_ticks', 'port'):
        if not isinstance(config[name], int) or isinstance(config[name], bool) or config[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if config['execution_ticks'] > 20:
        raise ValueError('execution_ticks cannot exceed the one-second action chunk')
    if config['port'] > 65535:
        raise ValueError('port cannot exceed 65535')
    for name in ('connection_timeout', 'inference_timeout'):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if config['config'] != DEFAULTS['config'] or config['checkpoint'] != DEFAULTS['checkpoint']:
        raise ValueError('Only pi05_droid with gs://openpi-assets/checkpoints/pi05_droid is supported; configure the remote server separately')
    if config['image_size'] != 224:
        raise ValueError('pi05_droid requires image_size=224')
    cameras = config['cameras']
    if not isinstance(cameras, list) or not all(isinstance(c, str) and c in CAMERAS for c in cameras):
        raise ValueError(f'cameras must be a list of known camera names: {list(CAMERAS)}')
    if len(set(cameras)) != len(cameras):
        raise ValueError('cameras must not contain duplicates')
    if not {'agentview', 'left_wrist'}.issubset(cameras):
        raise ValueError('DROID observation adapter requires agentview and left_wrist')
    for name in ('host', 'prompt'):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f'{name} must be a nonempty string')
    return config


def load_episodes(path=None, seed=0, max_steps=2400):
    """Expand manifest conditions into concrete episodes without scene randomization."""
    conditions = json.loads(Path(path).read_text()) if path else [{'seeds': [seed]}]
    if not isinstance(conditions, list) or not conditions:
        raise ValueError('Episode manifest must be a nonempty list')
    episodes = []
    for condition in conditions:
        if not isinstance(condition, dict):
            raise ValueError('Each episode manifest condition must be a JSON object')
        unknown = condition.keys() - {'bottle_pose', 'seeds', 'max_steps'}
        if unknown:
            raise ValueError(f'Unknown episode keys: {sorted(unknown)}')
        pose = condition.get('bottle_pose')
        if pose is not None and (not isinstance(pose, list) or len(pose) != 3 or
                                 not all(isinstance(n, (int, float)) and not isinstance(n, bool) and np.isfinite(n) for n in pose)):
            raise ValueError('bottle_pose must be finite [x, y, yaw_deg]')
        limit = condition.get('max_steps', max_steps)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError('Episode max_steps must be a positive integer')
        seeds = condition.get('seeds', [seed])
        if not isinstance(seeds, list) or not seeds or any(not isinstance(s, int) or isinstance(s, bool) or not 0 <= s < 2**32 for s in seeds):
            raise ValueError('seeds must be a nonempty list of integers in [0, 2**32)')
        for episode_seed in seeds:
            episodes.append({'seed': episode_seed, 'bottle_pose': pose, 'max_steps': limit})
    return episodes


def run_episode(out, controller, episode, config):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    random.seed(episode['seed']); np.random.seed(episode['seed'])
    start = time.monotonic()
    env = None; client = None; adapter = None
    rows = []; error = None; failure_category = None; fingerprint = None
    phase = 'initialization'; setup_complete = controller == 'scripted'; first_success_tick = None
    manifest = {'controller': controller, 'episode': episode, 'configuration': config, **metadata()}
    (out / 'run.json').write_text(json.dumps(manifest, indent=2))
    try:
        env = DetentBottleEnv(image_size=config['image_size'],
                             cameras=config['cameras'] if controller == 'hybrid' else (),
                             max_steps=episode['max_steps'], bottle_pose=episode['bottle_pose'])
        # Gym's reset seed controls its RNG; simulator reset itself is deterministic.
        env.reset(seed=episode['seed'])
        s = env.sim; adapter = ActionAdapter(s)
        fingerprint = hashlib.sha256(s.data.qpos.tobytes()).hexdigest()
        np.savez_compressed(out / 'initial.npz', qpos=s.data.qpos.copy(), qvel=s.data.qvel.copy())

        def step():
            nonlocal first_success_tick
            s.step()
            if s.success and first_success_tick is None:
                first_success_tick = s.steps
            rows.append(dict(qpos=s.data.qpos.copy(), qvel=s.data.qvel.copy(),
                             targets=s.targets.copy(), rotations=s.rotations.copy(),
                             grips=s.grip_targets.copy(), eef_pos=s.data.site_xpos[s.tips].copy(),
                             eef_rotation=s.data.site_xmat[s.tips].reshape(2, 3, 3).copy(),
                             bottle_qpos=s.bottle_qpos(), wrist_ft=s.wrenches_world(),
                             step=s.steps, sim_time=s.data.time, **s.mechanism()))

        phase = 'setup' if controller == 'hybrid' else 'scripted'
        for _, arm, pos, rpy, gap, ticks in opening_sequence():
            if controller == 'hybrid' and arm == 0:
                setup_complete = True
                break
            s.target(arm, pos, rpy, gap)
            for _ in range(ticks):
                step()
                if s.reason:
                    break
            if s.reason:
                break
        if controller == 'hybrid' and not s.reason:
            phase = 'inference'
            from .client import Client
            client = Client(config['host'], config['port'], timeout=config['inference_timeout'],
                            connection_timeout=config['connection_timeout'])
            while not s.reason:
                phase = 'inference'
                obs = env._obs()
                payload = request(obs, config['prompt'])
                # Include exact RGB and robot-state request before networking, including failed calls.
                np.savez_compressed(out / f'request-{s.steps}.npz',
                                    **{k: v for k, v in payload.items() if k != 'prompt'})
                (out / f'request-{s.steps}.json').write_text(json.dumps({'step': s.steps, 'prompt': payload['prompt']}))
                chunk = client.infer(payload)
                np.save(out / f'chunk-{s.steps}.npy', chunk)
                phase = 'control'
                for cmd in resample(chunk, config['execution_ticks']):
                    adapter.apply(cmd, fraction=1)
                    step()
                    if s.reason:
                        break
        if controller == 'hybrid' and not setup_complete:
            failure_category = 'setup_failure'
        elif not s.success:
            failure_category = 'task_failure'
    except Exception as exc:
        error = repr(exc)
        failure_category = 'infrastructure_failure' if phase in ('initialization', 'inference') else ('setup_failure' if phase == 'setup' else 'control_failure')
    finally:
        np.savez_compressed(out / 'trajectory.npz', **{k: np.array([r[k] for r in rows]) for k in rows[0]} if rows else {})
        sim = env.sim if env else None
        final = {'controller': controller, **episode, 'success': bool(sim.success) if sim else False,
                 'reason': sim.reason if sim else None, 'error': error,
                 'failure_category': failure_category, 'phase': phase, 'setup_complete': setup_complete,
                 'first_success_tick': first_success_tick, 'steps': sim.steps if sim else 0,
                 'sim_s': sim.data.time if sim else 0, 'wall_s': time.monotonic() - start,
                 'initial_fingerprint': fingerprint, 'workspace_clips': adapter.clips if adapter else 0,
                 'latencies': client.latencies if client else []}
        (out / 'final.json').write_text(json.dumps(final, indent=2))
        if client:
            client.close()
        if env:
            env.close()
    return final


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller', choices=['scripted', 'hybrid'], default='scripted')
    p.add_argument('--out', required=True)
    p.add_argument('--config', help='JSON configuration; CLI values override it')
    p.add_argument('--episodes', help='JSON list of pose/seed conditions; writes one subdirectory per expanded episode')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--host'); p.add_argument('--port', type=int)
    p.add_argument('--max-steps', type=int)
    p.add_argument('--execution-ticks', type=int)
    args = p.parse_args()
    try:
        config = load_settings(args.config, {k: getattr(args, k) for k in ('host', 'port', 'max_steps', 'execution_ticks')})
        episodes = load_episodes(args.episodes, args.seed, config['max_steps'])
    except (ValueError, TypeError) as exc:
        p.error(str(exc))
    root = Path(args.out)
    if args.episodes:
        root.mkdir(parents=True, exist_ok=False)
        results = [run_episode(root / f'episode-{i:04d}-seed-{episode["seed"]}', args.controller, episode, config)
                   for i, episode in enumerate(episodes)]
        (root / 'results.json').write_text(json.dumps(results, indent=2))
    else:
        results = [run_episode(root, args.controller, episodes[0], config)]
    print(json.dumps(results, indent=2))
    if any(result['error'] for result in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
