"""Exercise network failures in real spawned processes without OpenPI/GPU."""
import time

import numpy as np
import pytest

from integration_pi05.client import Client


class FakePolicy:
    def __init__(self, **kwargs):
        pass

    def infer(self, obs):
        if obs.get('hang'):
            time.sleep(30)
        if obs.get('error'):
            raise ValueError('server rejected observation')
        if obs.get('crash'):
            import os
            os._exit(2)
        return {'actions': np.full(obs.get('shape', (15, 8)), obs.get('value', 1.))}


class HangingConnection:
    def __init__(self, **kwargs):
        time.sleep(30)


def test_success_reuses_worker_and_closes():
    client = Client(policy_factory=FakePolicy, timeout=5)
    assert client._process is None
    np.testing.assert_array_equal(client.infer({}), np.ones((15, 8)))
    pid = client._process.pid
    client.infer({'value': 2})
    assert client._process.pid == pid
    assert len(client.latencies) == 2
    client.close()
    client.close()
    assert not client._process.is_alive()
    with pytest.raises(RuntimeError, match='closed'):
        client.infer({})


@pytest.mark.parametrize('factory,obs,timeout', [
    (HangingConnection, {}, .2), (FakePolicy, {'hang': True}, .2),
])
def test_timeout_kills_worker(factory, obs, timeout):
    client = Client(policy_factory=factory, timeout=timeout,
                    connection_timeout=.5 if factory is HangingConnection else 5)
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        client.infer(obs)
    assert time.monotonic() - start < 7
    assert not client._process.is_alive()
    assert client.latencies == []


@pytest.mark.parametrize('obs,error', [
    ({'shape': (10, 8)}, ValueError), ({'value': float('nan')}, ValueError),
    ({'error': True}, RuntimeError), ({'crash': True}, RuntimeError),
])
def test_bad_response_or_worker_failure_closes(obs, error):
    client = Client(policy_factory=FakePolicy, timeout=5)
    with pytest.raises(error):
        client.infer(obs)
    assert not client._process.is_alive()


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf')])
def test_invalid_deadline(timeout):
    with pytest.raises(ValueError):
        Client(timeout=timeout)
