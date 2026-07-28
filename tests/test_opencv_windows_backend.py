import sys
import types
import unittest
from unittest.mock import patch


class FakeCapture:
    def __init__(self, opened=True):
        self.opened = opened

    def isOpened(self):
        return self.opened

    def release(self):
        pass

    def set(self, _prop, _value):
        return True

    def get(self, prop):
        values = {
            5: 30,
            3: 640,
            4: 480,
        }
        return values[prop]


class OpenCVWindowsBackendTest(unittest.TestCase):
    def test_windows_camera_uses_directshow_backend(self):
        fake_cv2 = types.SimpleNamespace()
        fake_cv2.CAP_DSHOW = 700
        fake_cv2.CAP_PROP_FPS = 5
        fake_cv2.CAP_PROP_FRAME_WIDTH = 3
        fake_cv2.CAP_PROP_FRAME_HEIGHT = 4
        fake_cv2.ROTATE_90_COUNTERCLOCKWISE = 0
        fake_cv2.ROTATE_90_CLOCKWISE = 1
        fake_cv2.ROTATE_180 = 2
        fake_cv2.calls = []

        def video_capture(*args):
            fake_cv2.calls.append(args)
            return FakeCapture()

        fake_cv2.VideoCapture = video_capture
        fake_cv2.setNumThreads = lambda _threads: None

        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            from lerobot.common.robot_devices.cameras import opencv

            with patch.object(opencv.platform, "system", return_value="Windows"):
                camera = opencv.OpenCVCamera(camera_index=0, fps=30, width=640, height=480)
                camera.connect()

        self.assertEqual(fake_cv2.calls, [(0, fake_cv2.CAP_DSHOW), (0, fake_cv2.CAP_DSHOW)])


if __name__ == "__main__":
    unittest.main()
