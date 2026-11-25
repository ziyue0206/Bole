import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bole.config import DEFAULTS, load
from bole.cli import parser, overrides, main
from bole.runtime.dependencies import codex_cli_mount, asset_override_mounts, prepare_robot_overrides
from bole.runtime.snapshots import freeze

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bole_bootstrap_test', PROJECT / 'scripts/bootstrap.py')
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class PortableResourcesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_resource_selection_precedence_and_paths(self):
        env = self.root / 'env.json'
        explicit = self.root / 'explicit.json'
        env.write_text(json.dumps({'resources': {'assets': 'env-assets'}}))
        explicit.write_text(json.dumps({'resources': {'assets': 'selected-assets', 'codex_cli': 'codex', 'asset_overrides': 'generated'}}))
        with patch.dict(os.environ, {'BOLE_RESOURCES': str(env)}):
            self.assertEqual(load()['resources']['assets'], str(self.root / 'env-assets'))
            selected = load(resources_path=explicit)
            self.assertEqual(selected['resources']['assets'], str(self.root / 'selected-assets'))
            self.assertEqual(selected['resources']['codex_cli'], str(self.root / 'codex'))
            self.assertEqual(selected['resources']['asset_overrides'], str(self.root / 'generated'))

    def test_bad_environment_resource_file_fails_explicitly(self):
        with patch.dict(os.environ, {'BOLE_RESOURCES': str(self.root / 'missing')}):
            with self.assertRaises(FileNotFoundError):
                load()

    def test_missing_cli_resource_file_does_not_silently_fall_back(self):
        with patch('sys.stderr'):
            code = main(['plan', '--resources', str(self.root / 'missing')])
        self.assertEqual(code, 2)

    def test_cli_image_selection_does_not_override_unset_fields(self):
        args = parser().parse_args(['plan', '--resources', 'selected.yaml', '--simulator-image', 'custom/sim:one'])
        self.assertEqual(overrides(args), {'resources': {'simulator_image': 'custom/sim:one', 'simulator_image_id': None}})

    def test_defaults_have_no_server_assets_volumes_or_proxy(self):
        self.assertNotIn('/mnt/', DEFAULTS['resources']['assets'])
        self.assertEqual(DEFAULTS['resources']['cache_volumes'], {'cache': None, 'extensions': None})
        self.assertIsNone(DEFAULTS['harness']['gateway']['proxy'])

    def test_optional_official_cli_mount_is_readonly_and_validated(self):
        executable = self.root / 'codex'
        executable.write_bytes(b'fixture')
        executable.chmod(0o755)
        self.assertEqual(codex_cli_mount(executable, '/opt/bole-tools/codex'),
                         ['--mount', f'type=bind,src={executable},dst=/opt/bole-tools/codex,readonly'])
        with self.assertRaises(ValueError):
            codex_cli_mount(executable, '../codex')

    def test_asset_template_generation_leaves_downloaded_assets_unchanged(self):
        assets = self.root / 'Assets'
        robot = assets / 'Robots/x5'
        robot.mkdir(parents=True)
        template = robot / 'curobo_tmp.yml'
        original = 'kinematics:\n  asset_root_path: /old/server/robot\n  urdf_path: ${ASSETS_PATH}/Assets/Robots/x5/X5A.urdf\n'
        template.write_text(original)
        generated = self.root / 'generated'
        result = bootstrap.generate_asset_paths(assets, generated, '/workspace/RoboDojo')
        self.assertEqual(result, ['Robots/x5/curobo.yml'])
        self.assertEqual(template.read_text(), original)
        self.assertFalse((robot / 'curobo.yml').exists())
        content = (generated / 'Robots/x5/curobo.yml').read_text()
        self.assertIn('/workspace/RoboDojo/Assets/Robots/x5/X5A.urdf', content)
        self.assertNotIn('/old/server', content)
        self.assertEqual(bootstrap.generate_asset_paths(assets, generated, '/workspace/RoboDojo'), result)
        with self.assertRaises(ValueError):
            bootstrap.generate_asset_paths(assets, assets / 'generated', '/workspace/RoboDojo')

    def test_directory_overlay_supports_missing_upstream_curobo_file(self):
        assets = self.root / 'Assets'
        robot = assets / 'Robots/x5'
        robot.mkdir(parents=True)
        (robot / 'X5A.urdf').write_text('robot-data')
        (robot / 'curobo_tmp.yml').write_text('urdf_path: ${ASSETS_PATH}/Assets/Robots/x5/X5A.urdf\n')
        generated = self.root / 'generated'
        bootstrap.generate_asset_paths(assets, generated, '/workspace/RoboDojo')
        mounts = prepare_robot_overrides(generated, assets, self.root, '/workspace/RoboDojo')
        self.assertIn(f'type=bind,src={assets},dst=/opt/bole-assets,readonly', mounts)
        tree = self.root / 'robot-assets'
        self.assertEqual(os.readlink(tree / 'x5/X5A.urdf'), '/opt/bole-assets/Robots/x5/X5A.urdf')
        self.assertTrue((tree / 'x5/curobo.yml').is_file())
        self.assertFalse((robot / 'curobo.yml').exists())
        self.assertEqual((robot / 'X5A.urdf').read_text(), 'robot-data')

    def test_unrelated_and_linked_override_files_are_rejected(self):
        overrides_dir = self.root / 'overrides'
        overrides_dir.mkdir()
        (overrides_dir / 'outside.txt').write_text('unexpected')
        with self.assertRaises(ValueError):
            asset_override_mounts(overrides_dir, '/workspace/RoboDojo')
        (overrides_dir / 'outside.txt').unlink()
        (overrides_dir / 'link').symlink_to(self.root)
        with self.assertRaises(ValueError):
            asset_override_mounts(overrides_dir, '/workspace/RoboDojo')

    def test_planning_is_readonly_and_does_not_fetch_or_build(self):
        data = self.root / 'data'
        config = self.root / 'config.json'
        args = bootstrap.parser().parse_args(['--data-dir', str(data), '--config-output', str(config)])
        with patch.object(bootstrap, 'command', side_effect=AssertionError('must not execute external command')), patch('builtins.print'):
            bootstrap.Bootstrap(args).run()
        self.assertFalse(data.exists())
        self.assertFalse(config.exists())

    def test_config_creation_is_idempotent_and_does_not_overwrite_user_config(self):
        target = self.root / 'config.yaml'
        args = bootstrap.parser().parse_args(['--data-dir', str(self.root / 'data'), '--config-output', str(target), '--steps', 'config'])
        instance = bootstrap.Bootstrap(args)
        with patch('builtins.print'):
            instance.config()
            instance.config()
        config = load(resources_path=target)
        self.assertEqual(config['resources']['codex_python'], '/usr/bin/python3')
        self.assertEqual(config['resources']['cache_volumes'], {'cache': None, 'extensions': None})
        target.write_text('user_owned: true\n')
        with self.assertRaises(RuntimeError):
            instance.config()
        self.assertEqual(target.read_text(), 'user_owned: true\n')

    def test_download_checksum_failure_never_replaces_existing_destination(self):
        import io
        destination = self.root / 'download'
        destination.write_bytes(b'old')
        class Response(io.BytesIO):
            pass
        with patch.object(bootstrap.urllib.request, 'urlopen', return_value=Response(b'wrong bytes')):
            with self.assertRaises(RuntimeError):
                bootstrap.download('https://example.invalid/file', destination, '0' * 64)
        self.assertEqual(destination.read_bytes(), b'old')

    def test_policy_codex_version_mismatch_is_rejected_before_preparation(self):
        manifest = json.loads((PROJECT / 'configs/resources.lock.json').read_text())
        manifest['codex']['version'] = 'different'
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(manifest))
        args = bootstrap.parser().parse_args(['--manifest', str(path)])
        with self.assertRaises(ValueError):
            bootstrap.Bootstrap(args)


    def test_resource_check_accepts_a_selected_new_image_and_checks_optional_identity(self):
        from bole.benchmarks.robodojo.adapter import RoboDojoAdapter
        source = self.root / 'source'
        (source / 'XPolicyLab/client_server/ws').mkdir(parents=True)
        (source / 'XPolicyLab/client_server/ws/model_client.py').write_text('')
        assets = self.root / 'Assets'
        assets.mkdir()
        config = copy.deepcopy(DEFAULTS)
        config['resources'].update(source=str(source), assets=str(assets), gpus=[0])
        config['resources']['simulator_image'] = 'new/official-build:chosen'
        image_id = 'sha256:' + 'a' * 64
        def fake_command(args):
            if args[:3] == ['docker', 'image', 'inspect']:
                return json.dumps([{'Id': image_id}])
            return 'gpu,example'
        adapter = RoboDojoAdapter()
        with patch.object(adapter, 'plan', return_value={'episodes': [], 'coverage': {}}), \
             patch('bole.benchmarks.robodojo.adapter.command', side_effect=fake_command):
            self.assertTrue(adapter.check(config)['ok'])
            config['resources']['simulator_image_id'] = 'sha256:' + 'b' * 64
            report = adapter.check(config)
            self.assertFalse(report['ok'])
            self.assertEqual(report['issues'][0]['component'], 'simulator_image')

    def test_asset_override_configuration_is_frozen_per_run(self):
        project = self.root / 'project'
        package = project / 'src/bole'
        package.mkdir(parents=True)
        (package / '__init__.py').write_text('')
        prompt = project / 'prompt.md'
        prompt.write_text('prompt')
        generated = self.root / 'generated/Robots/x5'
        generated.mkdir(parents=True)
        (generated / 'curobo.yml').write_text('selected configuration')
        config = copy.deepcopy(DEFAULTS)
        config['benchmark']['name'] = 'fixture'
        config['harness']['instruction'] = str(prompt)
        config['resources']['asset_overrides'] = str(self.root / 'generated')
        frozen, _ = freeze(config, self.root / 'run', project_root=project)
        (generated / 'curobo.yml').write_text('later edit')
        snapshot = Path(frozen['resources']['asset_overrides']) / 'Robots/x5/curobo.yml'
        self.assertEqual(snapshot.read_text(), 'selected configuration')

    def test_official_source_checkout_uses_exact_revision_and_preserves_edits(self):
        import subprocess
        origin = self.root / 'origin'
        origin.mkdir()
        def git(*args):
            return subprocess.check_output(['git', '-C', str(origin), *args], text=True).strip()
        git('init', '-q')
        (origin / 'version.txt').write_text('selected')
        git('add', 'version.txt')
        git('-c', 'user.name=Bole Test', '-c', 'user.email=fixture@example.invalid', 'commit', '-q', '-m', 'selected')
        selected = git('rev-parse', 'HEAD')
        (origin / 'version.txt').write_text('newer')
        git('add', 'version.txt')
        git('-c', 'user.name=Bole Test', '-c', 'user.email=fixture@example.invalid', 'commit', '-q', '-m', 'newer')
        destination = self.root / 'checkout'
        with patch('builtins.print'):
            bootstrap.checkout(str(origin), selected, destination)
        self.assertEqual((destination / 'version.txt').read_text(), 'selected')
        (destination / 'version.txt').write_text('user edit')
        with self.assertRaises(RuntimeError), patch('builtins.print'):
            bootstrap.checkout(str(origin), selected, destination)
        self.assertEqual((destination / 'version.txt').read_text(), 'user edit')


    def test_downloaded_asset_symlink_keeps_its_logical_path_on_repeat(self):
        actual = self.root / 'downloaded-assets'
        for name in ('Robots', 'Object', 'Material', 'Eval_Layout'):
            (actual / name).mkdir(parents=True)
        logical = self.root / 'Assets'
        logical.symlink_to(actual, target_is_directory=True)
        data = self.root / 'data'
        data.mkdir()
        revision = json.loads((PROJECT / 'configs/resources.lock.json').read_text())['assets']['revision']
        (data / 'bootstrap-state.json').write_text(json.dumps({'schema_version': 1, 'completed': {
            'assets': {'path': str(logical), 'revision': revision}}}))
        args = bootstrap.parser().parse_args(['--data-dir', str(data), '--assets-dir', str(logical)])
        instance = bootstrap.Bootstrap(args)
        self.assertEqual(instance.assets, logical)
        with patch.object(bootstrap, 'command', side_effect=AssertionError('must reuse verified assets')), patch('builtins.print'):
            instance.assets_step()


if __name__ == '__main__':
    unittest.main()
