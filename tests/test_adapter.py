import tempfile
import json
import base64
import unittest
from pathlib import Path
from unittest.mock import patch

from bole.benchmarks.robodojo.adapter import RoboDojoSession
from bole.results.storage import RecordStore
from bole.results.video import VideoRecorder


class NativeImageEntrypointTests(unittest.TestCase):
    def test_worker_preserves_a_image_environment_activation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"resources": {"source": str(root / "source"),
                "assets": str(root / "assets"), "simulator_image": "selected-a-image"}}
            session = RoboDojoSession({"config": config, "gpu": 0,
                "episode": {"task_id": "make_toast", "seed": 0},
                "store": RecordStore(root / "episode", {})})
            with patch.object(session, "_cache"):
                argv = session._argv(root / "private")
            self.assertNotIn("--entrypoint", argv)
            position = argv.index("selected-a-image")
            self.assertEqual(argv[position+1:position+5],
                ["python", "-u", "-m", "bole.benchmarks.robodojo.worker"])
            self.assertIn("EVAL_ENV_TYPE=sim", argv)


class InterruptedRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = RecordStore(Path(self.directory.name), {"run_id": "r", "task_id": "t", "episode_id": "e"})
        self.session = RoboDojoSession({"config": {"recording": {"video": {"enabled": True}}},
            "episode": {"task_id": "t"}, "store": self.store})
        self.store.write_json("native-evaluation/result.json", {"available": True, "success": False,
            "score": .25, "source": "native final check"})
        self.store.write_json("trajectory/native-boundary.json", {"actual_control_steps": 4,
            "first_success_control_step": 1, "termination_reason": "cancelled"})

    def test_broken_connection_recovers_final_verdict_and_finalizes_persisted_video_after_cleanup(self):
        recorder = VideoRecorder(self.store.root / "videos", identity=self.store.identity)
        png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
        for step in range(2):
            recorder.offer(step, step / 5, {camera: png for camera in recorder.cameras})
        order = []
        original_cleanup = self.session._cleanup
        def cleanup():
            order.append("cleanup")
            original_cleanup()
        def finalize(recovered, **kwargs):
            order.append("finalize")
            self.assertTrue(kwargs["partial"])
            self.assertTrue(self.session.worker_stopped)
            self.assertTrue(recovered._recovered)
            self.assertEqual(len(recovered.rows), 6)
            self.assertTrue(all((recovered.root / row["path"]).is_file() for row in recovered.rows))
            return [{"path": None, "camera": "cam_head", "status": "partial", "partial": True,
                     "playable": False, "reason": "only_one_frame"}]
        self.session.opened = True
        with patch.object(self.session, "_request", side_effect=ConnectionError("lost during encoding")):
            with patch.object(self.session, "_cleanup", side_effect=cleanup):
                with patch.object(VideoRecorder, "recover", wraps=VideoRecorder.recover) as recover:
                    with patch.object(VideoRecorder, "finalize", autospec=True, side_effect=finalize):
                        result = self.session.close("cancelled")
        self.assertEqual(order, ["cleanup", "finalize"])
        recover.assert_called_once_with(self.store.root / "videos")
        self.assertTrue(result["native_result"]["available"])
        self.assertFalse(result["native_result"]["success"])
        self.assertEqual(result["native_result"]["score"], .25)
        self.assertEqual(result["actual_control_steps"], 4)
        self.assertEqual(result["first_success_control_step"], 1)
        self.assertFalse(result["infrastructure_error"])
        self.assertIs(self.session.close("cancelled"), result)

    def test_video_recovery_failure_does_not_erase_persisted_success(self):
        self.store.write_json("native-evaluation/result.json", {"available": True, "success": True, "score": 1.})
        VideoRecorder(self.store.root / "videos", identity=self.store.identity)
        self.session.broken = True
        with patch.object(VideoRecorder, "recover", side_effect=OSError("broken camera journal")):
            result = self.session.close("infrastructure_error")
        self.assertTrue(result["native_result"]["success"])
        self.assertTrue(result["infrastructure_error"])
        self.assertEqual(result["artifacts"]["video"][0]["status"], "collection_failed")

    def test_foreign_episode_verdict_is_rejected_and_only_completed_action_prefix_recovers(self):
        path = self.store.root / "native-evaluation/result.json"
        value = json.loads(path.read_text())
        path.write_text(json.dumps({**value, "episode_id": "other"}))
        (self.store.root / "trajectory/native-boundary.json").unlink()
        self.store.append_jsonl("trajectory/control-steps.jsonl", {"control_step": 2})
        with (self.store.root / "trajectory/control-steps.jsonl").open("ab") as stream:
            stream.write(b'{"control_step":3')
        self.session.broken = True
        result = self.session.close("infrastructure_error")
        self.assertFalse(result["native_result"]["available"])
        self.assertIsNone(result["native_result"]["success"])
        self.assertEqual(result["actual_control_steps"], 2)
        self.assertEqual(result["in_flight_execution"], "unknown")


if __name__ == "__main__":
    unittest.main()
