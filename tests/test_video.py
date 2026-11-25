import binascii
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
import zlib

from bole.results.video import CAMERAS, VideoRecorder


def png(rgb=(0, 0, 0)):
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", binascii.crc32(kind + body) & 0xffffffff)
    pixels = b"".join(b"\0" + bytes(rgb) * 4 for _ in range(4))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")


class VideoTests(unittest.TestCase):
    def test_simulation_clock_gap_and_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = VideoRecorder(directory)
            frame = {c: png() for c in CAMERAS}
            recorder.offer(0, 0., frame)
            self.assertFalse(recorder.wants(.01))
            self.assertEqual(recorder.offer(1, .04, lambda: self.fail("not due")), [])
            recorder.offer(15, .6, frame)
            self.assertEqual(len(recorder.rows), 12)
            self.assertEqual(sum(r["status"] == "missing" for r in recorder.rows), 6)
            self.assertEqual(len(list(Path(directory).rglob("*.png"))), 6)
            recovered = VideoRecorder.recover(directory)
            records = recovered.finalize(partial=True, ffmpeg="missing-ffmpeg")
            self.assertTrue(all(r["partial"] for r in records))
            self.assertTrue(all(r["captured_frames"] == 2 for r in records))
            self.assertTrue(all(r["missing_samples"] == 2 for r in records))
            self.assertTrue(all(not r["playable"] for r in records))

    def test_only_first_frame_is_not_reported_as_video(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = VideoRecorder(directory)
            recorder.offer(0, 0, {c: png() for c in CAMERAS})
            self.assertTrue(all(r["path"] is None for r in recorder.finalize()))

    def test_unavailable_clock_preserves_partial_capture_and_error_on_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = VideoRecorder(directory)
            recorder.offer(0, 0, {c: png() for c in CAMERAS})
            row = recorder.mark_collection_error(RuntimeError("native clock unavailable"), control_step=1)
            self.assertIsNone(row["sim_time_s"])
            self.assertFalse(recorder.wants(.2))
            recovered = VideoRecorder.recover(directory)
            self.assertIn("native clock unavailable", recovered.collection_error)
            self.assertTrue(all(r["partial"] for r in recovered.finalize(ffmpeg="missing-ffmpeg")))

    def test_long_episode_does_not_thin(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = VideoRecorder(directory, cameras=["head"])
            # More than the old 3000-frame limit; every sample must survive.
            for slot in range(3002):
                recorder.offer(slot * 5, slot / 5, {"head": png()})
            self.assertEqual(len(list(Path(directory).rglob("*.png"))), 3002)
            self.assertEqual(recorder.last_slot, 3001)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe not installed")
    def test_real_cpu_encoding_three_cameras_and_interrupted_reencoding(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = VideoRecorder(directory)
            for i in range(3):
                recorder.offer(i * 5, i / 5, {c: png((80 * i, 0, 0)) for c in CAMERAS})
            records = recorder.finalize()
            self.assertTrue(all(r["playable"] and r["encoded_frames"] == 3 and r["status"] == "saved" for r in records))
            recovered = VideoRecorder.recover(directory)
            records = recovered.finalize(partial=True)
            self.assertTrue(all(r["playable"] and r["status"] == "partial" for r in records))


if __name__ == "__main__":
    unittest.main()
