import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import torch

from lerobot.common.datasets import video_utils


class VideoUtilsTest(unittest.TestCase):
    def test_decode_video_frames_falls_back_to_opencv_without_torchvision_video_reader(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "episode_0.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (32, 24),
            )
            self.assertTrue(writer.isOpened())
            for value in (0, 120, 240):
                writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
            writer.release()

            with mock.patch.object(video_utils.torchvision, "io", types.SimpleNamespace()):
                frames = video_utils.decode_video_frames_torchvision(
                    video_path,
                    timestamps=[0.0, 0.1, 0.2],
                    tolerance_s=0.11,
                    backend="pyav",
                )

        self.assertEqual(frames.shape, torch.Size([3, 3, 24, 32]))
        self.assertEqual(frames.dtype, torch.float32)
        self.assertGreaterEqual(float(frames.min()), 0.0)
        self.assertLessEqual(float(frames.max()), 1.0)
        self.assertLess(float(frames[0].mean()), float(frames[1].mean()))
        self.assertLess(float(frames[1].mean()), float(frames[2].mean()))


if __name__ == "__main__":
    unittest.main()
