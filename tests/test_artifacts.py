import json
from pathlib import Path
import tempfile
import unittest

from bole.results.artifacts import episode_video_records, write_artifact_index
from bole.results.storage import RecordStore


class ArtifactIndexTests(unittest.TestCase):
    def test_video_refs_resolve_from_episode_and_all_actual_evidence_is_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RecordStore(Path(directory), {"run_id": "r", "task_id": "t", "episode_id": "e"})
            for name in ("videos/cam_head.mp4", "videos/capture.jsonl", "videos/cam_head/encoded-frames.jsonl",
                         "observations/one.png", "model-events/response.json", "feedback/delivery.json",
                         "native-evaluation/result.json", "safety.json", "usage.json", "result.json"):
                store.asset(name, b"{}\n")
            (store.root / "telemetry").mkdir()
            store.asset("videos/.cam_head.partial.mp4", b"not completed")
            videos = [{"path": "cam_head.mp4", "capture_ref": "capture.jsonl",
                "encoded_frame_map": "cam_head/encoded-frames.jsonl", "camera": "cam_head",
                "playable": True, "status": "saved", "partial": False}]
            manifest = write_artifact_index(store, video=videos, video_enabled=True)
            for key in ("path", "capture_ref", "encoded_frame_map"):
                self.assertTrue((store.root / manifest["video"][0][key]).is_file())
            self.assertEqual(manifest["video"], episode_video_records(manifest["video"]))
            indexed = {row["path"]: row for row in manifest["artifacts"]}
            self.assertIn("model-events/response.json", indexed)
            self.assertIn("observations/one.png", indexed)
            self.assertIn("feedback/delivery.json", indexed)
            self.assertNotIn("videos/.cam_head.partial.mp4", indexed)
            self.assertEqual(indexed["trajectory/control-steps.jsonl"]["status"], "not_collected")
            self.assertEqual(next(row for row in manifest["collections"] if row["path"] == "telemetry")["status"], "not_collected")
            self.assertTrue(all(row["episode_id"] == "e" for row in manifest["artifacts"]))
            refreshed = write_artifact_index(store)
            self.assertEqual(refreshed["video"], manifest["video"])

    def test_damaged_tail_and_unverified_video_are_not_complete_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RecordStore(Path(directory), {})
            store.asset("model-events/raw.jsonl", b'{"ok":true}\n{"broken"')
            store.asset("videos/unverified.mp4", b"not a video")
            manifest = write_artifact_index(store)
            records = {r["path"]: r for r in manifest["artifacts"]}
            self.assertEqual(records["model-events/raw.jsonl"]["status"], "partial")
            self.assertEqual(records["model-events/raw.jsonl"]["records"], 1)
            self.assertEqual(records["videos/unverified.mp4"]["status"], "unknown")
            self.assertIsNone(records["videos/unverified.mp4"]["playable"])


if __name__ == "__main__":
    unittest.main()
