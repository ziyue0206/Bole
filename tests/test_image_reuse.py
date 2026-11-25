"""CPU checks for persistent bytes, fresh imports and non-destructive images."""
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from bole.cli import launch, main, overrides, parser
from unittest.mock import patch
from bole.config import DEFAULTS, load
from bole.engine.prompts import RESOURCE_DISCOVERY, render
from bole.runtime.image_assets import output_image_name, publish_image, restore_image
from bole.runtime.learning_assets import archive_tree, load_asset, snapshot_home


class ImageReuseTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_image_cli_defaults_and_explicit_selection(self):
        args = parser().parse_args(['eval', '--harness-image', 'fixture:learned', '--no-memory'])
        c = load(overrides=overrides(args), resources_path=self.root / 'absent')
        self.assertEqual(c['resources']['harness_image'], 'fixture:learned')
        self.assertFalse(c['harness']['memory']['enabled'])
        args = parser().parse_args(['learn', '--save-image', 'fixture:{task}'])
        c = load(overrides=overrides(args), mode='learn', resources_path=self.root / 'absent')
        self.assertEqual(output_image_name(c, 'run', 'task'), 'fixture:task')
        self.assertNotEqual(output_image_name(c, 'run', 'task', partial=True), 'fixture:task')

    def test_explicit_image_does_not_silently_override_or_get_overridden_by_a_manifest(self):
        with patch('builtins.print'):
            self.assertEqual(main(['plan','--harness-image','fixture:learned',
                '--task-asset','a=/nonexistent/manifest.json']), 2)

    def test_launch_pins_inspected_image_id_without_changing_user_configuration(self):
        c = copy.deepcopy(DEFAULTS)
        frozen = copy.deepcopy(c)
        pinned = 'sha256:'+'a'*64
        report = {'ok':True,'benchmark':{'resources':{'harness_image_id':pinned}}}
        with patch('bole.cli.preflight', return_value=report), \
             patch('bole.cli.new_run_root', return_value=self.root), \
             patch('bole.cli.freeze', return_value=(frozen,{'python_paths':[str(self.root)]})), \
             patch('bole.cli.subprocess.Popen') as child, patch('builtins.print'):
            child.return_value.wait.return_value = 0
            self.assertEqual(launch(c, {'episodes':[]}, object(), object()), 0)
        execution = json.loads((self.root/'execution-config.json').read_text())
        self.assertEqual(execution['resources']['harness_image'], pinned)
        self.assertEqual(c['resources']['harness_image'], DEFAULTS['resources']['harness_image'])

    def test_eval_resource_discovery_is_identical_and_not_added_to_learn(self):
        prompt = self.root / 'prompt.md'
        prompt.write_text('{{task_description}}\n{{allowed_material}}')
        c = copy.deepcopy(DEFAULTS)
        c['harness']['instruction'] = str(prompt)
        episode = {'task_id':'a'}
        info = {'instruction':'task', 'capabilities':{}}
        ordinary = render(c, episode, info, [])
        c['resources']['harness_image'] = 'fixture:learned'
        learned = render(c, episode, info, [])
        self.assertEqual(ordinary, learned)
        self.assertEqual(ordinary.count(RESOURCE_DISCOVERY), 1)
        c['mode'] = 'learn'
        self.assertNotIn(RESOURCE_DISCOVERY, render(c, episode, info, []))

    def test_arbitrary_home_files_and_wal_persist_without_raw_dialogue(self):
        home = self.root / 'home'
        (home / 'custom').mkdir(parents=True)
        (home / 'custom/method.bin').write_bytes(b'learned-byte-format')
        (home / 'AGENTS.md').write_text('Use my autonomous resources')
        (home / 'sessions').mkdir()
        (home / 'sessions/old.jsonl').write_text('old dialogue')
        (home / 'history.jsonl').write_text('old dialogue')
        (home / 'auth.json').write_text('{"key":"credential"}')
        db = sqlite3.connect(home / 'state_5.sqlite')
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE threads(id TEXT, first_user_message TEXT NOT NULL, title TEXT)')
        db.execute("INSERT INTO threads VALUES ('old-id','old dialogue','thread title')")
        db.commit()
        try:
            report = snapshot_home(home, self.root / 'snapshot')
            self.assertEqual(db.execute('SELECT first_user_message FROM threads').fetchone()[0], 'old dialogue')
            with sqlite3.connect(self.root / 'snapshot/state_5.sqlite') as saved:
                self.assertEqual(saved.execute('SELECT id, first_user_message, title FROM threads').fetchone(),
                                 ('old-id', '', 'thread title'))
        finally:
            db.close()
        self.assertEqual((self.root / 'snapshot/custom/method.bin').read_bytes(), b'learned-byte-format')
        self.assertTrue((self.root / 'snapshot/AGENTS.md').is_file())
        self.assertFalse((self.root / 'snapshot/sessions').exists())
        self.assertFalse((self.root / 'snapshot/auth.json').exists())
        self.assertEqual(report['memory_generation'], 'not_checked')

    def test_pending_legacy_asset_without_memory_is_reusable_and_isolated(self):
        asset = self.root / 'asset'; asset.mkdir()
        work = self.root / 'source'; work.mkdir()
        (work / 'autonomous.data').write_bytes(b'learned')
        archive_tree(work, asset / 'workspace.tar.gz')
        manifest = {'schema_version':1, 'export_status':'saved', 'status':'complete',
                    'finalize_status':'pending', 'components':{
                    'image':{'image_id':'sha256:'+'a'*64, 'import':'local_docker_image'},
                    'workspace':{'path':'workspace.tar.gz'}}}
        (asset / 'manifest.json').write_text(json.dumps(manifest))
        for n in (1,2):
            load_asset(asset / 'manifest.json', self.root / f'w{n}', self.root / f'h{n}')
        (self.root / 'w1/autonomous.data').write_bytes(b'mutated')
        self.assertEqual((self.root / 'w2/autonomous.data').read_bytes(), b'learned')
        self.assertEqual(list((self.root / 'h1').iterdir()), [])

    def test_public_ca_bundle_is_preserved_but_private_pem_is_rejected(self):
        work = self.root / 'work'; work.mkdir()
        (work / 'cacert.pem').write_bytes(b'-----BEGIN CERTIFICATE-----\npublic CA\n-----END CERTIFICATE-----')
        archive_tree(work, self.root / 'with-ca.tar.gz')
        (work / 'private.pem').write_bytes(b'-----BEGIN RSA PRIVATE KEY-----\nsecret\n')
        with self.assertRaisesRegex(ValueError, 'private key'):
            archive_tree(work, self.root / 'with-private.tar.gz')

    async def test_existing_output_image_is_not_overwritten(self):
        calls = []
        async def command(*args):
            calls.append(args)
            return 'existing-image'
        with self.assertRaises(FileExistsError):
            await publish_image(self.root, 'base', 'fixture:existing', command)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1:3], ('image','ls'))

    async def test_base_image_restore_does_not_start_or_inspect_memory(self):
        async def command(*args):
            raise AssertionError(args)
        self.assertIsNone(await restore_image('base', {'Config':{}}, self.root/'w', self.root/'h', command))
