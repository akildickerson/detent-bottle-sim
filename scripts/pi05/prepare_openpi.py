"""CPU-only, scheduled OpenPI setup; no model loading or inference.

Idempotent for the requested commit. Refuses repository drift and serializes
setup with a file lock. Existing files are never reset or deleted by this helper.
The upstream download utility manages its own checkpoint cache/partial files.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import venv

UV_VERSION = '0.9.5'  # Explicit bootstrap pin: https://pypi.org/project/uv/0.9.5/
UPSTREAM = 'https://github.com/Physical-Intelligence/openpi.git'
CHECKPOINT = 'gs://openpi-assets/checkpoints/pi05_droid'


def run(args, *, cwd=None, capture=False):
    print(f'[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {shlex.join(map(str, args))}', flush=True)
    return subprocess.check_output(args, cwd=cwd, text=True).strip() if capture else subprocess.run(args, cwd=cwd, check=True)


def clean_checkout(root, revision):
    actual = run(['git', 'rev-parse', 'HEAD'], cwd=root, capture=True)
    if actual != revision:
        raise RuntimeError(f'Refusing revision drift: {actual} != {revision}')
    if run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root, capture=True):
        raise RuntimeError('Refusing dirty tracked checkout or submodule')


def prepare(project, root, revision, report):
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        run(['git', 'clone', '--no-checkout', UPSTREAM, str(root)])
        run(['git', 'checkout', '--detach', revision], cwd=root)
    elif not (root / '.git').exists():
        raise RuntimeError(f'Refusing existing non-Git directory: {root}')
    clean_checkout(root, revision)
    run(['git', 'submodule', 'update', '--init', '--recursive'], cwd=root)
    clean_checkout(root, revision)
    report['submodules'] = run(['git', 'submodule', 'status', '--recursive'], cwd=root, capture=True)
    tools = project / 'tools' / f'uv-{UV_VERSION}'
    uv = tools / 'bin' / 'uv'
    if not uv.exists():
        tools.parent.mkdir(parents=True, exist_ok=True)
        venv.EnvBuilder(with_pip=True).create(tools)
        run([str(tools / 'bin' / 'python'), '-m', 'pip', 'install', '--disable-pip-version-check', f'uv=={UV_VERSION}'])
    version = run([str(uv), '--version'], capture=True)
    if version.split()[:2] != ['uv', UV_VERSION]:
        raise RuntimeError(f'Refusing unexpected uv version: {version}')
    report['uv'] = version
    target_python = root / '.venv' / 'bin' / 'python'
    if target_python.exists():
        existing = run([str(target_python), '-c', 'import sys; print("%s.%s" % sys.version_info[:2])'], capture=True)
        if existing != '3.11':
            raise RuntimeError(f'Refusing replacement of existing Python {existing} environment')
    run([str(uv), 'sync', '--frozen', '--python', '3.11'], cwd=root)
    clean_checkout(root, revision)
    report['packages'] = run([str(uv), 'pip', 'freeze', '--python', str(target_python)], cwd=root, capture=True).splitlines()
    # Only imports the download utility. JAX/torch models are not constructed.
    download_code = '''
import json, logging, pathlib, os
from openpi.shared.download import maybe_download
logging.basicConfig(level=logging.INFO)
path = maybe_download('gs://openpi-assets/checkpoints/pi05_droid')
if not (path / 'params').is_dir() or not (path / 'assets').is_dir():
    raise RuntimeError('Checkpoint missing expected params/assets directories')
files = [{'path': str(p.relative_to(path)), 'bytes': p.stat().st_size} for p in sorted(path.rglob('*')) if p.is_file()]
if not files or not any(f['bytes'] > 0 for f in files):
    raise RuntimeError('Checkpoint download contains no nonempty files')
pathlib.Path(os.environ['PI05_DOWNLOAD_REPORT']).write_text(json.dumps({'path':str(path), 'files':files, 'bytes':sum(f['bytes'] for f in files)}, indent=2))
print('CHECKPOINT DOWNLOAD COMPLETE:', path, flush=True)
'''
    os.environ['PI05_DOWNLOAD_REPORT'] = str(report['download_report'])
    run([str(target_python), '-u', '-c', download_code], cwd=root)
    report['checkpoint'] = json.loads(Path(report['download_report']).read_text())
    report['python'] = run([str(target_python), '--version'], capture=True)
    clean_checkout(root, revision)


def main():
    if sys.version_info < (3, 9):
        raise SystemExit('Setup helper needs Python >=3.9; set SETUP_PYTHON to an available module interpreter')
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run inside a scheduled Slurm job')
    revision = os.environ.get('OPENPI_REV', '')
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise SystemExit('OPENPI_REV must be a full lowercase 40-character commit SHA')
    project = Path(os.environ.get('PI05_PROJECT', '/work/users/a/k/akil/pi05'))
    root = Path(os.environ.get('OPENPI_ROOT', str(project / 'openpi')))
    if not project.is_absolute() or not root.is_absolute():
        raise SystemExit('PI05_PROJECT and OPENPI_ROOT must be absolute')
    project.mkdir(parents=True, exist_ok=True)
    for name, suffix in [('OPENPI_DATA_HOME', 'checkpoints'), ('UV_CACHE_DIR', 'uv-cache'),
                         ('UV_PYTHON_INSTALL_DIR', 'python'), ('HF_HOME', 'hf-cache'), ('PIP_CACHE_DIR', 'pip-cache')]:
        os.environ.setdefault(name, str(project / suffix))
    os.environ['GIT_LFS_SKIP_SMUDGE'] = '1'
    report_dir = project / 'setup'
    report_dir.mkdir(exist_ok=True)
    tag = f"{os.environ['SLURM_JOB_ID']}-{time.time_ns()}"
    report_path = report_dir / f'{tag}.json'
    report = {'openpi_revision': revision, 'root': str(root), 'job': os.environ['SLURM_JOB_ID'],
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'download_report': str(report_dir / f'{tag}-checkpoint.json'), 'status': 'running',
              'uv_pin': UV_VERSION, 'bootstrap_python': sys.version,
              'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'cache_environment': {k: os.environ[k] for k in ('OPENPI_DATA_HOME', 'UV_CACHE_DIR', 'UV_PYTHON_INSTALL_DIR', 'HF_HOME', 'PIP_CACHE_DIR')}}
    start = time.monotonic()
    # Separate lock file prevents simultaneous mutation of this project.
    with (project / '.setup.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another setup process owns this project; retry after it completes')
        try:
            prepare(project, root, revision, report)
            report['status'] = 'success'
        except BaseException as exc:
            report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            report['wall_s'] = time.monotonic() - start
            report_path.write_text(json.dumps(report, indent=2))
            print('SETUP REPORT:', report_path, flush=True)


if __name__ == '__main__':
    main()
