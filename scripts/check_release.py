#!/usr/bin/env python3
"""Static publication checks only: no imports of Bole, containers, downloads or models."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRS = {'.venv', '__pycache__', '.pytest_cache', '.ruff_cache', 'runtime-deps',
                  'assets', 'Assets', 'runs', 'validation', 'backups', 'checkpoints'}
REQUIRED = ['README.md', 'pyproject.toml', 'uv.lock', 'THIRD_PARTY_NOTICES.md',
            'docs/getting-started.md', 'docs/configuration.md', 'docs/usage.md',
            'docs/troubleshooting.md', 'docs/development.md', 'docs/release-status.md',
            'bin/bole', 'scripts/bootstrap.py', 'scripts/install_apparmor.py',
            'scripts/install_host_dependencies.py', 'containers/codex/Dockerfile',
            'containers/codex/profile.sh', 'containers/codex/seccomp-bole-codex.json',
            'containers/codex/apparmor-bole-codex-bwrap', 'configs/resources.lock.json',
            'licenses/RoboDojo-LICENSE', 'licenses/XPolicyLab-LICENSE', 'licenses/Moby-profiles-LICENSE']
PATTERNS = {
    'private_key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'api_key_pattern': r'\bsk-[A-Za-z0-9_-]{24,}',
    'github_token_pattern': r'\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}',
    'server_path': r'/home/(?:ubuntu)/|/Users/(?:zzy)/',
    'private_gateway': r'aigw\.gino\.plus|117\.50\.181\.81',
}
REVIEWED_PRIVATE_KEY_FIXTURES = {
    # Deliberately invalid bytes literal in the PEM-rejection contract test.
    'tests/test_image_reuse.py': {'e4994c33c78f308579c25599734be1418097854c05f31b4d39d8c3a0cd2b7019'},
}


def tracked_files():
    result = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '-z'], capture_output=True, check=True)
    return [ROOT / name for name in result.stdout.decode().split('\0') if name]


def flags(path):
    tree = ast.parse(path.read_text())
    return {argument.value for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'
            for argument in node.args if isinstance(argument, ast.Constant)
            and isinstance(argument.value, str) and argument.value.startswith('--')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    errors, placeholders, reviewed_fixtures = [], [], []
    files = tracked_files()
    selected = {str(path.relative_to(ROOT)) for path in files}
    for name in REQUIRED:
        if name not in selected:
            errors.append({'path': name, 'issue': 'required_file_not_staged'})
    for path in files:
        relative = path.relative_to(ROOT)
        if (path.is_symlink() or set(relative.parts[:-1]) & FORBIDDEN_DIRS
                or any(part.startswith('._') for part in relative.parts)
                or path.name == 'resources.yaml' or '.local.' in path.name
                or path.name == '.env' or path.name.startswith('.env.')
                or path.suffix in {'.pyc', '.pyo', '.pem', '.key', '.log', '.sqlite', '.sqlite3'}):
            errors.append({'path': str(relative), 'issue': 'forbidden_publication_file'})
        if not path.is_file():
            errors.append({'path': str(relative), 'issue': 'missing_file'})
            continue
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append({'path': str(relative), 'issue': 'unexpected_large_file'})
        try:
            text = path.read_text()
        except UnicodeError:
            continue
        key_scan_text = text
        if str(relative) in REVIEWED_PRIVATE_KEY_FIXTURES:
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.Constant) and isinstance(node.value, bytes):
                    digest = hashlib.sha256(node.value).hexdigest()
                    if digest in REVIEWED_PRIVATE_KEY_FIXTURES[str(relative)]:
                        segment = ast.get_source_segment(text, node)
                        key_scan_text = key_scan_text.replace(segment, 'REVIEWED_INVALID_TEST_FIXTURE')
                        reviewed_fixtures.append({'path': str(relative), 'line': node.lineno})
        for label, pattern in PATTERNS.items():
            if re.search(pattern, key_scan_text if label == 'private_key' else text):
                errors.append({'path': str(relative), 'issue': label})
        if path.suffix == '.py':
            try:
                ast.parse(text, filename=str(relative))
            except SyntaxError as error:
                errors.append({'path': str(relative), 'issue': 'python_syntax', 'line': error.lineno})
        if path.suffix == '.md' and relative.parts[0] not in {'vendor', 'templates'}:
            body = re.sub(r'```.*?```', '', text, flags=re.S)
            for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', body):
                target = target.split(' ', 1)[0]
                if target.startswith(('http:', 'https:', 'mailto:', '#')):
                    continue
                target = unquote(target.split('#', 1)[0])
                destination = (path.parent / target).resolve()
                if destination != ROOT and ROOT not in destination.parents:
                    errors.append({'path': str(relative), 'issue': 'link_outside_repository'})
                elif not destination.exists():
                    errors.append({'path': str(relative), 'issue': 'broken_relative_link', 'target': target})
            if 'OWNER/REPOSITORY' in text:
                placeholders.append({'path': str(relative), 'kind': 'repository_address'})
    # Check explicitly documented long options against the implemented parsers.
    option_sources = {
        'scripts/bootstrap.py': ['--steps', '--apply', '--model', '--gpu', '--data-dir', '--assets-dir', '--replace-config'],
        'src/bole/cli.py': ['--resources', '--config', '--task', '--episodes', '--gpu', '--model', '--no-memory', '--harness-image'],
    }
    for source, required in option_sources.items():
        actual = flags(ROOT / source)
        # BooleanOptionalAction synthesizes --no-memory from --memory.
        actual |= {'--no-' + flag[2:] for flag in actual if flag.startswith('--')}
        for flag in required:
            if flag not in actual:
                errors.append({'path': source, 'issue': 'documented_option_missing', 'option': flag})
    for pattern in ('.venv/bin/python', 'assets/robodojo/example.usd', 'runtime-deps/state.json',
                    'configs/portable.local.yaml', 'configs/portable.local.yaml.previous',
                    'runs/example/result.json', 'validation/private.log', '.env', '._README.md'):
        result = subprocess.run(['git', '-C', str(ROOT), 'check-ignore', '-q', '--no-index', pattern])
        if result.returncode != 0:
            errors.append({'path': pattern, 'issue': 'ignore_rule_missing'})
    report = {'ok': not errors, 'scope': 'static_files_links_syntax_and_options_only',
              'files': len(files), 'bytes': sum(p.stat().st_size for p in files if p.is_file()),
              'errors': errors, 'placeholders': placeholders,
              'reviewed_invalid_test_fixtures': reviewed_fixtures,
              'runtime_acceptance': 'not_executed', 'scan_is_not_a_complete_secret_audit': True}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
