#!/usr/bin/env python3
"""Prepare small Debian/Ubuntu host tools. Default: print commands only.

  python3 scripts/install_host_dependencies.py
  sudo python3 scripts/install_host_dependencies.py --apply

Provide Python 3.11+ separately for bootstrap.py. Docker Engine, NVIDIA drivers
and NVIDIA Container Toolkit are host prerequisites and are not reconfigured here.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    packages = ['git', 'git-lfs', 'curl', 'ca-certificates', 'apparmor', 'apparmor-utils']
    commands = [['apt-get', 'update'], ['apt-get', 'install', '-y', '--no-install-recommends', *packages]]
    print(json.dumps({'mode': 'apply' if args.apply else 'plan', 'commands': commands,
                      'external_prerequisites': ['Python 3.11+ with venv/pip', 'Docker Engine',
                                                 'compatible NVIDIA driver and NVIDIA Container Toolkit'],
                      'docker_reconfiguration': False, 'explicit_service_restart': False}, indent=2))
    if not args.apply:
        return 0
    if os.geteuid() != 0:
        p.error('--apply requires administrator privileges')
    if not Path('/etc/debian_version').exists() or not shutil.which('apt-get'):
        p.error('This host helper supports Debian/Ubuntu only; install the listed tools with your host package manager')
    for command in commands:
        subprocess.run(command, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
