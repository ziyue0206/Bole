#!/usr/bin/env python3
"""Prepare official resources. Default: print a plan; --apply performs selected steps.

Examples:
  python3 scripts/bootstrap.py
  python3 scripts/bootstrap.py --steps python,source,assets,asset-paths,cache,config --apply
  python3 scripts/bootstrap.py --steps codex-tools,codex-base,codex-image,simulator-image --apply
  python3 scripts/bootstrap.py --steps asset-paths,cache,config --source-dir vendor/robodojo --assets-dir /data/Assets --apply

Run scripts/install_host_dependencies.py and scripts/install_apparmor.py
separately with administrator privileges when the host lacks their requirements.
This tool never starts a model, simulator, or benchmark run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from urllib.parse import urlsplit
import venv

ROOT = Path(__file__).resolve().parents[1]
STEPS = ('python', 'source', 'assets', 'asset-paths', 'codex-tools', 'codex-base',
         'codex-image', 'simulator-image', 'cache', 'config')
REQUIRED_ASSETS = ('Robots', 'Object', 'Material', 'Eval_Layout')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(handle, 'w') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def command(argv, *, cwd=None, capture=False):
    print('+ ' + shlex.join(list(map(str, argv))), flush=True)
    environment = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1', GIT_TERMINAL_PROMPT='0')
    return subprocess.run(list(map(str, argv)), cwd=cwd, env=environment, check=True,
                          text=True, stdout=subprocess.PIPE if capture else None,
                          timeout=7200).stdout


def fetch_json(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'Bole-resource-bootstrap'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, destination, digest):
    destination = Path(destination)
    if destination.is_file() and sha256(destination) == digest:
        print('Using verified download:', destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.partial')
    request = urllib.request.Request(url, headers={'User-Agent': 'Bole-resource-bootstrap'})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as stream:
        shutil.copyfileobj(response, stream, 1024 * 1024)
    if sha256(partial) != digest:
        raise RuntimeError(f'SHA256 mismatch for {destination.name}; partial file retained')
    os.replace(partial, destination)


def checkout(repository, revision, destination, *, sparse=None):
    destination = Path(destination)
    if destination.exists() and not (destination / '.git').is_dir():
        raise RuntimeError(f'Refusing to replace existing non-Git directory: {destination}')
    if not destination.exists():
        destination.mkdir(parents=True)
        command(['git', 'init', destination])
        command(['git', '-C', destination, 'remote', 'add', 'origin', repository])
    origin = command(['git', '-C', destination, 'remote', 'get-url', 'origin'], capture=True).strip()
    if origin != repository:
        raise RuntimeError(f'Existing checkout has a different origin: {destination}')
    dirty = command(['git', '-C', destination, 'status', '--porcelain', '--untracked-files=no'], capture=True)
    if dirty.strip():
        raise RuntimeError(f'Existing checkout has modified tracked files: {destination}')
    command(['git', '-C', destination, 'fetch', '--depth', '1', 'origin', revision])
    if sparse:
        command(['git', '-C', destination, 'sparse-checkout', 'init', '--cone'])
        command(['git', '-C', destination, 'sparse-checkout', 'set', sparse])
    command(['git', '-C', destination, 'checkout', '--detach', 'FETCH_HEAD'])
    actual = command(['git', '-C', destination, 'rev-parse', 'HEAD'], capture=True).strip()
    if actual != revision:
        raise RuntimeError(f'Checkout revision mismatch: {actual} != {revision}')


def generate_asset_paths(assets, destination, container_root):
    """Generate only configuration overrides; leave downloaded/shared assets intact."""
    assets, destination = Path(assets).resolve(), Path(destination).resolve()
    if destination == assets or assets in destination.parents:
        raise ValueError('Generated overrides must be outside the downloaded asset tree')
    templates = sorted((assets / 'Robots').rglob('curobo_tmp.yml'))
    if not templates:
        raise RuntimeError(f'No robot curobo_tmp.yml templates found under {assets}')
    generated = []
    expected = set()
    for template in templates:
        relative = template.relative_to(assets).with_name('curobo.yml')
        target = destination / relative
        text = template.read_text().replace('${ASSETS_PATH}', container_root).replace('$ASSETS_PATH', container_root)
        robot_root = str(PurePosixPath(container_root) / 'Assets' / relative.parent.as_posix())
        text = re.sub(r'(?m)^(\s*asset_root_path:)\s*.*$', lambda m: m[1] + ' ' + json.dumps(robot_root), text)
        if '${ASSETS_PATH}' in text or '$ASSETS_PATH' in text:
            raise RuntimeError(f'Unresolved asset path in {template}')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise RuntimeError(f'Generated output cannot be a symbolic link: {target}')
        target.write_text(text)
        expected.add(target)
        generated.append(str(relative))
    stale = set(destination.rglob('curobo.yml')) - expected
    if stale:
        raise RuntimeError(f'Stale generated configurations require review: {sorted(map(str, stale))}')
    return generated


class Bootstrap:
    def __init__(self, args):
        self.args = args
        self.manifest = json.loads(args.manifest.read_text())
        if self.manifest.get('schema_version') != 1:
            raise ValueError('Unsupported resources manifest schema')
        self.data = args.data_dir.resolve()
        self.source = args.source_dir.resolve() if args.source_dir else self.data / 'robodojo-source'
        self.assets = Path(os.path.abspath(args.assets_dir.expanduser())) if args.assets_dir else ROOT / 'assets/robodojo'
        self.generated = self.data / 'generated-assets'
        self.tools = self.data / ('codex-' + self.manifest['codex']['version'])
        self.state_path = self.data / 'bootstrap-state.json'
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {'schema_version': 1, 'completed': {}}
        if self.state.get('schema_version') != 1:
            raise ValueError('Unsupported bootstrap state schema')
        self.base = args.codex_base_image or self.manifest['codex']['base_image']
        self.simulator = args.simulator_image or self.manifest['robodojo']['image']
        self.harness = args.harness_image or self.manifest['codex']['image']
        self.steps = list(STEPS) if args.steps == 'all' else args.steps.split(',')
        if not self.steps or any(step not in STEPS for step in self.steps):
            raise ValueError('Steps must be all or a comma-separated subset of: ' + ','.join(STEPS))
        if len(set(self.steps)) != len(self.steps):
            raise ValueError('Duplicate preparation steps')
        self.steps = [step for step in STEPS if step in self.steps]
        # These policies were reviewed only for this Linux/x86_64 release pair.
        if self.manifest['codex']['version'] != self.manifest['security']['codex_version']:
            raise ValueError('Codex and reviewed security-policy versions differ; supply matching profiles')
        if self.manifest['security']['architecture'] != 'x86_64':
            raise ValueError('This provisioner currently supports the reviewed x86_64 profiles only')

    def save(self, step, value):
        self.state['completed'][step] = value
        atomic_json(self.state_path, self.state)

    def inspect_image(self, image):
        return json.loads(command(['docker', 'image', 'inspect', image], capture=True))[0]

    def disk(self, path, minimum):
        path = Path(path)
        while not path.exists():
            path = path.parent
        free = shutil.disk_usage(path).free
        if free < minimum:
            raise RuntimeError(f'Insufficient free space at {path}: {free // 2**30} GiB; need {minimum // 2**30} GiB')

    def python(self):
        target = self.args.venv.resolve()
        executable = target / 'bin/python'
        if not executable.exists():
            if target.exists() and any(target.iterdir()):
                raise RuntimeError(f'Refusing to replace incomplete environment: {target}')
            venv.EnvBuilder(with_pip=True).create(target)
        command([executable, '-m', 'pip', 'install', '-e', ROOT])
        self.save('python', {'interpreter': str(executable)})

    def source_step(self):
        item = self.manifest['robodojo']
        checkout(item['repository'], item['revision'], self.source)
        command(['git', '-C', self.source, 'submodule', 'sync', '--', *item['submodules']])
        command(['git', '-C', self.source, 'submodule', 'update', '--init', '--depth', '1', '--', *item['submodules']])
        status = command(['git', '-C', self.source, 'submodule', 'status', '--', *item['submodules']], capture=True)
        if any(line.startswith(('-', '+', 'U')) for line in status.splitlines()):
            raise RuntimeError('Submodules do not match the selected RoboDojo revision')
        self.save('source', {'path': str(self.source), 'revision': item['revision'], 'submodules': status.splitlines()})

    def assets_step(self):
        item = self.manifest['assets']
        if self.assets.is_dir() and all((self.assets / p).is_dir() for p in REQUIRED_ASSETS):
            previous = self.state['completed'].get('assets', {})
            if previous.get('revision') == item['revision'] and previous.get('path') == str(self.assets):
                print('Using previously downloaded assets:', self.assets)
                return
            raise RuntimeError('Existing assets have no matching bootstrap provenance. Use asset-paths,config to reference them without downloading or modifying them.')
        self.disk(self.data, item['minimum_free_gib'] * 2**30)
        command(['git', 'lfs', 'version'])
        repository = self.data / 'assets-repository'
        checkout(item['repository'], item['revision'], repository, sparse='Assets')
        command(['git', '-C', repository, 'lfs', 'install', '--local'])
        command(['git', '-C', repository, 'lfs', 'pull', '--include=Assets/**', '--exclude=', 'origin'])
        names = command(['git', '-C', repository, 'ls-tree', '-r', '--name-only', 'HEAD', 'Assets'], capture=True).splitlines()
        for name in names:
            path = repository / name
            if not path.is_file():
                raise RuntimeError(f'Missing downloaded asset: {name}')
            with path.open('rb') as stream:
                if stream.read(160).startswith(b'version https://git-lfs.github.com/spec/v1'):
                    raise RuntimeError(f'Unresolved Git LFS pointer: {name}')
        if not all((repository / 'Assets' / p).is_dir() for p in REQUIRED_ASSETS):
            raise RuntimeError('Required asset directories missing after download')
        self.assets.parent.mkdir(parents=True, exist_ok=True)
        if self.assets.is_symlink():
            if self.assets.resolve() != (repository / 'Assets').resolve():
                raise RuntimeError(f'Refusing to replace asset symlink: {self.assets}')
        elif self.assets.exists():
            raise RuntimeError(f'Refusing to replace existing asset destination: {self.assets}')
        else:
            self.assets.symlink_to(repository / 'Assets', target_is_directory=True)
        self.save('assets', {'path': str(self.assets), 'revision': item['revision'], 'files': len(names), 'lfs_verified_by': 'git-lfs'})

    def asset_paths(self):
        generated = generate_asset_paths(self.assets, self.generated, self.manifest['robodojo']['container_root'])
        self.save('asset-paths', {'assets': str(self.assets), 'overrides': str(self.generated), 'generated': generated})

    def codex_tools(self):
        if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'amd64'):
            raise RuntimeError('Selected native binaries and policies require Linux x86_64')
        item = self.manifest['codex']
        for name in ('cli', 'bwrap'):
            asset = item[name]
            archive = self.tools / asset['archive']
            download(asset['url'], archive, asset['sha256'])
            output = self.tools / ('codex' if name == 'cli' else 'bwrap')
            with tarfile.open(archive, 'r:gz') as tar:
                candidates = [m for m in tar.getmembers() if m.isfile() and Path(m.name).name == asset['member']]
                if len(candidates) != 1:
                    raise RuntimeError(f'Expected binary not found in verified archive: {asset["member"]}')
                temporary = output.with_name(output.name + '.unpacked')
                with tar.extractfile(candidates[0]) as src, temporary.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
            temporary.chmod(0o755)
            os.replace(temporary, output)
        shutil.copy2(ROOT / 'containers/codex/profile.sh', self.tools / 'profile.sh')
        (self.tools / '.dockerignore').write_text('*\n!codex\n!bwrap\n!profile.sh\n')
        actual = command([self.tools / 'codex', '--version'], capture=True).strip()
        if item['version'] not in actual:
            raise RuntimeError(f'Codex version mismatch: {actual}')
        self.save('codex-tools', {'version': item['version'], 'path': str(self.tools), 'archive_checksums_verified': True})

    def codex_base(self):
        previous = self.state['completed'].get('codex-base', {})
        image = previous.get('reference') if previous.get('requested') == self.base else self.base
        command(['docker', 'pull', '--platform', 'linux/amd64', image])
        inspected = self.inspect_image(image)
        digests = inspected.get('RepoDigests', [])
        reference = next((d for d in digests if d.startswith('ghcr.io/openai/codex-universal@')), None)
        if not reference:
            if '@sha256:' in image:
                reference = image
            else:
                raise RuntimeError('Cannot resolve immutable official base digest; configure a digest reference')
        self.save('codex-base', {'requested': self.base, 'reference': reference, 'image_id': inspected['Id']})

    def codex_image(self):
        base = self.state['completed'].get('codex-base', {}).get('reference')
        if not base:
            raise RuntimeError('Run codex-base first to resolve and pin the official base image')
        for name in ('codex', 'bwrap', 'profile.sh'):
            if not (self.tools / name).is_file():
                raise RuntimeError('Run codex-tools before building the thin runtime layer')
        existing = subprocess.run(['docker', 'image', 'inspect', self.harness], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if existing.returncode == 0:
            previous = self.state['completed'].get('codex-image', {})
            if previous.get('image_id') == self.inspect_image(self.harness)['Id'] and previous.get('base') == base:
                print('Using prepared Codex image:', self.harness)
                return
            raise RuntimeError('Harness image tag already exists without matching bootstrap provenance; choose another --harness-image')
        command(['docker', 'build', '--platform', 'linux/amd64', '-f', ROOT / 'containers/codex/Dockerfile',
                 '--build-arg', 'CODEX_BASE=' + base, '--build-arg', 'CODEX_VERSION=' + self.manifest['codex']['version'],
                 '-t', self.harness, self.tools])
        self.save('codex-image', {'name': self.harness, 'image_id': self.inspect_image(self.harness)['Id'], 'base': base})

    def simulator_image(self):
        item = self.manifest['robodojo']
        if not (self.source / 'Dockerfile').is_file():
            raise RuntimeError('Run source before building the official RoboDojo Dockerfile')
        actual = command(['git', '-C', self.source, 'rev-parse', 'HEAD'], capture=True).strip()
        if actual != item['revision']:
            raise RuntimeError('RoboDojo build source does not match the manifest')
        docker_root = json.loads(command(['docker', 'info', '--format', '{{json .DockerRootDir}}'], capture=True))
        existing = subprocess.run(['docker', 'image', 'inspect', self.simulator], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if existing.returncode == 0:
            previous = self.state['completed'].get('simulator-image', {})
            if previous.get('image_id') == self.inspect_image(self.simulator)['Id'] and previous.get('revision') == actual:
                print('Using prepared simulator image:', self.simulator)
                return
            raise RuntimeError('Simulator image tag already exists without matching bootstrap provenance; choose another --simulator-image')
        self.disk(docker_root, item['minimum_build_free_gib'] * 2**30)
        command(['docker', 'build', '--platform', 'linux/amd64', '-t', self.simulator, self.source])
        self.save('simulator-image', {'name': self.simulator, 'image_id': self.inspect_image(self.simulator)['Id'], 'revision': actual})

    def cache(self):
        for name in ('cache', 'extensions'):
            (self.data / 'cache' / name).mkdir(parents=True, exist_ok=True)
        self.save('cache', {'mode': 'empty_seed_directories', 'first_isaac_start_may_download_extensions': True})

    def config(self):
        target = self.args.config_output.resolve()
        expected = self.state['completed'].get('simulator-image', {})
        image_id = expected.get('image_id') if expected.get('name') == self.simulator else None
        value = {'resources': {
            'source': str(self.source), 'source_revision': self.manifest['robodojo']['revision'],
            'assets': str(self.assets), 'asset_revision': self.manifest['assets']['revision'],
            'asset_overrides': str(self.generated), 'simulator_image': self.simulator,
            'simulator_image_id': image_id, 'simulator_root': self.manifest['robodojo']['container_root'],
            'harness_image': self.harness, 'codex_python': '/usr/bin/python3',
            'codex_cli': None, 'codex_bwrap': None,
            'codex_seccomp': str(ROOT / self.manifest['security']['seccomp']),
            'codex_apparmor': self.manifest['security']['apparmor_profile'], 'gpus': self.args.gpu,
            'cache_source': str(self.data / 'cache/cache'),
            'extensions_source': str(self.data / 'cache/extensions'),
            'cache_volumes': {'cache': None, 'extensions': None}},
            'harness': {'executable': '/opt/bole-tools/codex', 'model': self.args.model,
                        'gateway': {'base_url': self.args.model_base_url,
                                    'credential_env': self.args.credential_env,
                                    'proxy': self.args.proxy}}}
        if target.exists():
            try:
                current = json.loads(target.read_text())
            except (ValueError, OSError):
                current = None
            if current == value:
                print('Configuration already matches:', target)
                return
            if not self.args.replace_config:
                raise RuntimeError(f'Refusing to replace existing config: {target}; use another --config-output or --replace-config')
            shutil.copy2(target, target.with_name(target.name + '.previous'))
        atomic_json(target, value)  # JSON is also valid YAML; no dependency needed before Python setup.
        self.save('config', {'path': str(target), 'format': 'JSON-compatible YAML', 'asset_revision_is_declared_not_full_tree_reverified': True})

    def run(self):
        report = {'mode': 'apply' if self.args.apply else 'plan', 'steps': self.steps,
                  'source': str(self.source), 'assets': str(self.assets), 'data': str(self.data),
                  'simulator_image': self.simulator, 'harness_image': self.harness,
                  'codex_base': self.base, 'codex_version': self.manifest['codex']['version'],
                  'config_output': str(self.args.config_output.resolve()),
                  'host_dependencies_step': f'sudo python3 {ROOT}/scripts/install_host_dependencies.py --apply',
                  'gpu_driver_verified': False,
                  'host': {'system': platform.system(), 'architecture': platform.machine(),
                           'python': platform.python_version(),
                           'tools': {name: bool(shutil.which(name)) for name in ('git', 'git-lfs', 'docker', 'nvidia-smi', 'apparmor_parser')}},
                  'administrator_step': shlex.join(['sudo', 'python3', str(ROOT / 'scripts/install_apparmor.py'), '--source', str(ROOT / self.manifest['security']['apparmor_source']), '--profile', self.manifest['security']['apparmor_profile'], '--apply']),
                  'first_simulator_start': 'may need internet for Isaac extensions/cloud USD assets',
                  'full_runtime_acceptance': 'deferred'}
        print(json.dumps(report, indent=2))
        if not self.args.apply:
            return
        if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'amd64'):
            raise RuntimeError('Selected images and profiles currently target Linux x86_64')
        for step in self.steps:
            print('\nPreparing:', step, flush=True)
            method = {'source': 'source_step', 'assets': 'assets_step'}.get(step, step.replace('-', '_'))
            getattr(self, method)()
        print('\nPreparation steps completed. Full runtime acceptance remains deferred.')


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--apply', action='store_true', help='perform selected steps; default only prints a plan')
    p.add_argument('--steps', default='all', help='all or comma-separated: ' + ','.join(STEPS))
    p.add_argument('--manifest', type=Path, default=ROOT / 'configs/resources.lock.json')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'runtime-deps/portable')
    p.add_argument('--source-dir', type=Path)
    p.add_argument('--assets-dir', type=Path)
    p.add_argument('--venv', type=Path, default=ROOT / '.venv')
    p.add_argument('--config-output', type=Path, default=ROOT / 'configs/portable.local.yaml')
    p.add_argument('--replace-config', action='store_true')
    p.add_argument('--simulator-image')
    p.add_argument('--harness-image')
    p.add_argument('--codex-base-image')
    p.add_argument('--gpu', type=int, action='append', default=[])
    p.add_argument('--model', default='gpt-6-astra')
    p.add_argument('--model-base-url', default='https://api.openai.com/v1')
    p.add_argument('--credential-env', default='OPENAI_API_KEY')
    p.add_argument('--proxy', help='HTTP(S) proxy for model calls; downloads/git use standard proxy variables; Docker pulls use daemon network/proxy settings')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.version_info < (3, 11):
        print('bootstrap requires Python 3.11+; use the existing .venv/bin/python or install a compatible interpreter', file=sys.stderr)
        return 2
    try:
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', args.credential_env):
            raise ValueError('--credential-env requires a variable name, not a credential value')
        for address in (args.model_base_url, args.proxy):
            if address is None:
                continue
            parsed = urlsplit(address)
            if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError('Model/proxy URLs must be HTTP(S) URLs without embedded credentials')
        Bootstrap(args).run()
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print('bootstrap: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
