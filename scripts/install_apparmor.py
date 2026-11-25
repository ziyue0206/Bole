#!/usr/bin/env python3
"""Install the scoped Codex profile on the Linux Docker host; default is read-only.

  python3 scripts/install_apparmor.py
  sudo python3 scripts/install_apparmor.py --apply

The profile file is persistent. Host reboot acceptance is outside this task.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--apply', action='store_true')
    p.add_argument('--profile', default='bole-codex-bwrap')
    p.add_argument('--replace', action='store_true', help='back up and replace an existing different profile')
    p.add_argument('--source', type=Path, default=ROOT / 'containers/codex/apparmor-bole-codex-bwrap')
    args = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.profile) or args.profile in ('unconfined', 'docker-default'):
        p.error('Use a dedicated Bole profile name')
    content = args.source.read_text().replace('"bole-codex-bwrap"', '"' + args.profile + '"')
    destination = Path('/etc/apparmor.d') / args.profile
    enabled_file = Path('/sys/module/apparmor/parameters/enabled')
    enabled = enabled_file.exists() and enabled_file.read_text().strip() == 'Y'
    parser = shutil.which('apparmor_parser') or ('/sbin/apparmor_parser' if Path('/sbin/apparmor_parser').exists() else None)
    try:
        loaded = any(row.split(' ', 1)[0] == args.profile for row in Path('/sys/kernel/security/apparmor/profiles').read_text().splitlines())
    except OSError:
        loaded = None
    equal = destination.exists() and destination.read_text() == content
    print(json.dumps({'mode': 'apply' if args.apply else 'plan', 'host_apparmor_enabled': enabled,
                      'parser': parser, 'destination': str(destination), 'contents_match': equal,
                      'loaded': loaded, 'scope': 'named Codex containers only'}, indent=2))
    if not args.apply:
        return 0
    if os.geteuid() != 0:
        p.error('--apply requires administrator privileges on the Docker host')
    if not enabled or not parser:
        p.error('Host AppArmor and apparmor_parser are required; no policy is disabled automatically')
    if destination.exists() and not equal:
        if not args.replace:
            p.error('Existing profile differs; review it and use --replace to retain a backup and replace it')
        backup = Path('/var/backups/bole-apparmor')
        backup.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copy2(destination, backup / (destination.name + '.previous-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
    if not equal:
        handle, temporary = tempfile.mkstemp(prefix='.bole-profile-', dir=destination.parent)
        try:
            with os.fdopen(handle, 'w') as stream:
                stream.write(content)
            os.chmod(temporary, 0o644)
            # Parse without loading, then persist and load only this profile.
            subprocess.run([parser, '-Q', temporary], check=True)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    subprocess.run([parser, '-r', '-W', str(destination)], check=True)
    print('Installed and loaded dedicated profile:', args.profile)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
