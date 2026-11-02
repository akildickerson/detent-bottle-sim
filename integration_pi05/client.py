"""Bounded OpenPI RPCs, isolated from potentially blocking WebSocket code."""
import multiprocessing as mp
import time

import numpy as np


def _serve(requests, replies, host, port, policy_factory):
    """Only the disposable child imports OpenPI or owns a network connection."""
    try:
        if policy_factory is None:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy
            policy_factory = WebsocketClientPolicy
        policy = policy_factory(host=host, port=port)
        metadata_getter = getattr(policy, 'get_server_metadata', None)
        metadata = metadata_getter() if callable(metadata_getter) else {}
        replies.send(('ready', metadata))
        while True:
            observation = requests.get()
            replies.send(('result', policy.infer(observation)))
    except Exception as exc:
        try:
            replies.send(('error', f'{type(exc).__name__}: {exc}'))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        replies.close()


class Client:
    """Synchronous client with a killable connection/inference worker.

    OpenPI is imported lazily in a spawned process. Connection setup and each
    inference have separate deadlines. After any RPC failure the client closes;
    construct a new instance to reconnect rather than reusing stale responses.
    ``policy_factory`` is a spawn-picklable testing seam, not a model dependency.
    """
    def __init__(self, host='localhost', port=8000, timeout=60,
                 connection_timeout=None, *, policy_factory=None):
        self.host, self.port = host, port
        self.timeout = self._positive(timeout)
        self.connection_timeout = self._positive(timeout if connection_timeout is None else connection_timeout)
        self.latencies = []
        self.server_metadata = {}
        self._factory = policy_factory
        self._process = self._requests = self._replies = None
        self._closed = False

    @staticmethod
    def _positive(value):
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Timeout must be finite and positive')
        return value

    def _receive(self, timeout):
        if not self._replies.poll(timeout):
            raise TimeoutError(f'OpenPI RPC exceeded {timeout:g} seconds')
        try:
            kind, payload = self._replies.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError('OpenPI worker exited without a response') from exc
        if kind == 'error':
            raise RuntimeError(f'OpenPI worker failed: {payload}')
        return kind, payload

    def _start(self):
        ctx = mp.get_context('spawn')
        self._requests = ctx.Queue(maxsize=1)
        self._replies, child_reply = ctx.Pipe(duplex=False)
        self._process = ctx.Process(target=_serve, args=(self._requests, child_reply,
                                   self.host, self.port, self._factory), daemon=True)
        try:
            self._process.start()
        finally:
            child_reply.close()
        kind, metadata = self._receive(self.connection_timeout)
        if kind != 'ready':
            raise RuntimeError('Invalid OpenPI worker handshake')
        if not isinstance(metadata, dict):
            raise ValueError('OpenPI server metadata must be a dictionary')
        self.server_metadata = metadata

    def infer(self, observation):
        if self._closed:
            raise RuntimeError('Client is closed; create a new client to reconnect')
        start = time.monotonic()
        try:
            if self._process is None:
                self._start()
            # Queue feeding uses a daemon thread, so a blocked child cannot block
            # the caller while it transmits the image payload.
            self._requests.put_nowait(observation)
            kind, result = self._receive(self.timeout)
            if kind != 'result':
                raise RuntimeError('Invalid OpenPI worker response')
            a = np.asarray(result['actions'])
            if a.shape != (15, 8) or not np.isfinite(a).all():
                raise ValueError('Checkpoint must return finite (15,8) actions')
        except BaseException:
            self.close()
            raise
        self.latencies.append(time.monotonic() - start)
        return a

    def close(self):
        self._closed = True
        if self._process is not None and self._process.pid is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=.5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=.5)
        if self._requests is not None:
            self._requests.cancel_join_thread()
            self._requests.close()
            self._requests = None
        if self._replies is not None:
            self._replies.close()
            self._replies = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
