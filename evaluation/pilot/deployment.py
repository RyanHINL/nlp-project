"""Prepare or start the specific, reviewed local pilot deployment."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit

PREP = Path(__file__).resolve().parent
ROOT = PREP.parents[1]
D = json.loads((PREP / 'deployment.json').read_text())
MODEL = ROOT / D['weights']
ENV = ROOT / D['runtime_env']
OUT = ROOT / D['output_dir']
PY = ENV / 'bin/python'
UV = ENV / 'bin/uv'
WHEELS = ROOT / 'cache/vllm-metal-wheels-0.34.0'
SAVED_TOKENIZER = ROOT / 'cache/qwen2.5-7b-tokenizer'


def require(test, why):
    if not test:
        raise ValueError(why)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()


def verify_file(path, pin):
    size = path.stat().st_size
    require('bytes' not in pin or size == pin['bytes'], f'Unexpected size: {path}')
    with path.open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    if 'sha256' in pin:
        require(sha == pin['sha256'], f'SHA256 mismatch: {path}')
    else:
        git_oid = hashlib.sha1(f'blob {size}\0'.encode() + path.read_bytes()).hexdigest()
        require(git_oid == pin['git_blob_sha1'], f'Git blob mismatch: {path}')
    return {'bytes': size, 'sha256': sha}


def fetch_wheel(url, target, pin):
    if target.exists():
        verify_file(target, pin)
        return
    partial = target.with_name(target.name + '.partial')
    # Resume only a partial transfer of this exact immutable wheel URL.
    subprocess.run(['/usr/bin/curl', '--fail', '--location', '--show-error', '--retry', '3',
                    '--proto', '=https', '--proto-redir', '=https', '--continue-at', '-',
                    '--output', partial, url], check=True)
    verify_file(partial, pin)
    partial.rename(target)


def runner_config():
    # The upstream system/user prompts are read by evaluation.contract.
    model = dict(model=D['model'], revision=D['revision'], tokenizer_revision=D['revision'],
                 chat_template_sha256=D['chat_template_sha256'])
    return dict(endpoint='http://127.0.0.1:8000',
                runtime_build='<pending deployment fingerprint produced by deploy.sh freeze>',
                context_tokens=4096, context_policy='reject_overflow', timeout_seconds=120,
                temperature=0.2, top_p=1.0, seed=0,
                qa=dict(model, max_tokens=512), judge=dict(model, max_tokens=256))


def pins():
    require(sys.version_info[:2] == (3, 12) and platform.machine() == 'arm64' and sys.platform == 'darwin',
            'Expected native macOS arm64 Python 3.12')
    require(Path.cwd() == ROOT and (ROOT / 'evaluation/runner.py').is_file(), 'Run from this checkout')
    proposed = runner_config()
    for name, pin in D['model_files'].items():
        if (SAVED_TOKENIZER / name).is_file():
            verify_file(SAVED_TOKENIZER / name, pin)
    weights = [v for v in D['model_files'].values() if 'bytes' in v]
    print(f"Pinned {len(weights)} shards, {sum(v['bytes'] for v in weights):,} bytes; {len(D['wheels'])} release wheels", flush=True)
    return proposed


def install_runtime():
    stamp = ENV / '.pilot-runtime-installed'
    if stamp.is_file():
        require(stamp.read_text().strip() == digest((PREP / 'deployment.json').read_bytes()), 'Runtime spec changed')
        return
    if not ENV.exists():
        subprocess.run([sys.executable, '-m', 'venv', str(ENV)], check=True)
    require((ENV / 'pyvenv.cfg').is_file() and PY.exists(), 'Incomplete or foreign runtime directory')
    WHEELS.mkdir(exist_ok=True)
    env = dict(os.environ, UV_CACHE_DIR=str(ROOT / 'cache/uv-metal'), UV_NATIVE_TLS='1',
               PIP_CACHE_DIR=str(ROOT / 'cache/pip-metal'), PYTHONNOUSERSITE='1')
    if Path('/etc/ssl/cert.pem').is_file():
        env.setdefault('PIP_CERT', '/etc/ssl/cert.pem')
    subprocess.run([str(PY), '-m', 'pip', 'install', 'uv==' + D['uv_version']], check=True, env=env)
    # The official v0.34.0 installer follows releases/latest. Mirror its exact
    # three installation steps against digest-pinned release assets instead.
    for item in D['wheels']:
        wheel = WHEELS / unquote(Path(urlsplit(item['url']).path).name)
        fetch_wheel(item['url'], wheel, item)
        subprocess.run([str(UV), 'pip', 'install', '--python', str(PY), '--constraints',
                        str(PREP / 'runtime-constraints.txt'), str(wheel)], check=True, env=env)
    torch_pins = subprocess.check_output([str(PY), '-I', '-c',
        "from importlib.metadata import distribution; print(distribution('vllm-metal').locate_file('vllm_metal/requirements-torch.txt'))"], text=True).strip()
    require(Path(torch_pins).is_file(), 'Expected plugin-supplied torch pins')
    subprocess.run([str(UV), 'pip', 'install', '--python', str(PY), '--constraints',
                    str(PREP / 'runtime-constraints.txt'), '-r', torch_pins], check=True, env=env)
    installed_versions()
    stamp.write_text(digest((PREP / 'deployment.json').read_bytes()) + '\n')


def installed_versions():
    code = "from importlib.metadata import version; import json,sys; print(json.dumps({k:version(k) for k in sys.argv[1:]}))"
    v = json.loads(subprocess.check_output([str(PY), '-I', '-c', code, *D['runtime_versions']], text=True))
    require(v == D['runtime_versions'], f'Deployed package versions differ from the reviewed pins: {v}')
    return v


def prepare():
    require(not OUT.exists(), 'Real run already frozen; preserve its environment')
    install_runtime()
    MODEL.mkdir(exist_ok=True)
    # Keep HF transfer metadata and incomplete downloads outside the served model
    # directory, then publish verified files by rename on the same filesystem.
    staging = ROOT / 'cache/qwen2.5-download-a09a354'
    staging.mkdir(exist_ok=True)
    missing = []
    for name, pin in D['model_files'].items():
        if (MODEL / name).is_file():
            verify_file(MODEL / name, pin)
        elif (SAVED_TOKENIZER / name).is_file():
            verify_file(SAVED_TOKENIZER / name, pin)
            shutil.copyfile(SAVED_TOKENIZER / name, staging / name)
            verify_file(staging / name, pin)
            (staging / name).rename(MODEL / name)
        else:
            missing.append(name)
    if missing:
        hf_env = dict(os.environ, HF_HOME=str(ROOT / 'cache/hf-metal'),
                      HF_ENDPOINT='https://huggingface.co', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                      HF_HUB_DISABLE_TELEMETRY='1')
        subprocess.run([str(ENV / 'bin/hf'), 'download', D['model'], *missing,
                        '--revision', D['revision'], '--local-dir', str(staging), '--max-workers', '2'],
                       check=True, env=hf_env)
        for name in missing:
            verify_file(staging / name, D['model_files'][name])
            (staging / name).rename(MODEL / name)


def evaluation_revision(serving):
    if serving:
        commit = json.loads((OUT / 'deployment.frozen.json').read_text())['evaluation_commit']
    else:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    require(len(commit) == 40 and all(ch in '0123456789abcdef' for ch in commit),
            'Expected full evaluation commit SHA')
    # Handoff and review commits may follow a run. Preserve its original SHA
    # after verifying the inputs and execution code available in this checkout.
    inputs = ['configs', 'evaluation/runner.py', 'evaluation/contract.py', 'evaluation/local_model.py',
              'evaluation/development_identities.json', 'retrieval/data_tools.py', 'vendor/GroupMemBench']
    subprocess.run(['git', 'diff', '--no-ext-diff', '--quiet', commit, '--', *inputs], cwd=ROOT, check=True)
    return commit


def evidence(proposed, serving=False):
    expected = set(D['model_files'])
    require({p.name for p in MODEL.iterdir()} == expected, 'Weights directory must contain exactly the pinned files')
    manifest = {k: verify_file(MODEL / k, v) for k, v in D['model_files'].items()}
    config = json.loads((MODEL / 'config.json').read_text())
    require(config.get('torch_dtype') == 'bfloat16' and 'quantization_config' not in config, 'Precision changed')
    shards = set(json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map'].values())
    require(shards == {k for k in expected if k.endswith('.safetensors')}, 'Shards differ from model index')
    # Read the safetensors metadata, without materializing weights or loading a model.
    for name in sorted(shards):
        with (MODEL / name).open('rb') as f:
            size = int.from_bytes(f.read(8), 'little')
            header = json.loads(f.read(size))
        dtypes = {t['dtype'] for k, t in header.items() if k != '__metadata__'}
        require(dtypes == {'BF16'}, f'Unexpected on-disk tensor dtype(s) in {name}: {dtypes}')
    template = json.loads((MODEL / 'tokenizer_config.json').read_text())['chat_template'].encode('utf8')
    require(digest(template) == D['chat_template_sha256'], 'Effective template differs from pinned tokenizer')
    freeze = subprocess.check_output([str(PY), '-I', '-m', 'pip', 'freeze', '--all', '--local'])
    actual = {**D, 'status': 'frozen_for_launch', 'runtime_versions': installed_versions(),
              'platform_version': platform.platform(), 'python_version': platform.python_version(),
              'model_manifest_sha256': digest(json_bytes(manifest)), 'packages_freeze_sha256': digest(freeze),
              'config_proposal': proposed, 'evaluation_commit': evaluation_revision(serving)}
    launch = json_bytes(actual)
    final = dict(proposed, runtime_build='vllm=0.34.0+cpu, vllm-metal=0.34.0, MLX BF16; deployment_sha256=' + digest(launch))
    return {'deployment.frozen.json': launch, 'model.sha256.json': json_bytes(manifest),
            'packages.freeze.txt': freeze, 'chat_template.jinja': template, 'local.json': json_bytes(final)}


def freeze(evidence_files):
    require(not OUT.exists(), 'Refusing to overwrite an existing run')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUT.parent, prefix='.pilot-freeze-') as tmp:
        stage = Path(tmp) / 'run'
        stage.mkdir()
        for name, content in evidence_files.items():
            dest = stage / name
            dest.write_bytes(content)
            dest.chmod(0o444)
        stage.rename(OUT)
    print(f'Frozen runner config: {OUT / "local.json"}')


def serve(evidence_files):
    for name, content in evidence_files.items():
        require((OUT / name).read_bytes() == content, f'{name} differs from frozen deployment')
    executable = ENV / 'bin/vllm'
    env = {k: os.environ[k] for k in ('HOME', 'TMPDIR', 'LANG') if k in os.environ}
    env.update(PATH=f'{ENV}/bin:/usr/bin:/bin:/usr/sbin:/sbin', **D['server_environment'])
    command = [str(executable), *D['server_args']]
    # One new log per explicit launch. Stop the server with Ctrl-C in this terminal.
    with tempfile.NamedTemporaryFile(dir=OUT, prefix='server-', suffix='.log', delete=False) as log:
        print('Server log:', log.name, flush=True)
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'prepare', 'freeze', 'serve'])
    action = parser.parse_args().action
    proposed = pins()
    if action == 'check':
        return
    # Match the existing output contract by serializing writers and all server launches.
    lock_dir = ROOT / 'runs/real-pilot-prep'
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / '.deployment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if action == 'prepare':
            prepare()
        elif action == 'freeze':
            freeze(evidence(proposed))
        else:
            serve(evidence(proposed, serving=True))


if __name__ == '__main__':
    main()
