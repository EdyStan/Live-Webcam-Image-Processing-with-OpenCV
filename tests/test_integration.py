"""
Integration tests for the video pipeline.

Tests network transmission, serialization, and end-to-end crop processing.
"""

import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import zmq

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.rotated_crop import RotatedCropper


def serialize_frame(frame: np.ndarray, timestamp: float) -> bytes:
    h, w = frame.shape[:2]
    c = frame.shape[2] if len(frame.shape) == 3 else 1
    header = struct.pack('diii', timestamp, h, w, c)
    return header + frame.tobytes()


def deserialize_frame(data: bytes) -> tuple:
    header_size = struct.calcsize('diii')
    timestamp, h, w, c = struct.unpack('diii', data[:header_size])
    frame_data = data[header_size:]
    if c == 1:
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((h, w))
    else:
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((h, w, c))
    return frame, timestamp


class TestSerialization:
    """Frame serialization round-trip."""

    def test_frame_roundtrip(self):
        """Frame data and timestamp survive serialize/deserialize."""
        original = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        timestamp = time.time()
        data = serialize_frame(original, timestamp)
        recovered, recv_ts = deserialize_frame(data)
        np.testing.assert_array_equal(original, recovered)
        assert recv_ts == timestamp


class TestZmqTransmission:
    """ZeroMQ pub/sub frame transmission."""

    @pytest.fixture
    def zmq_pair(self):
        context = zmq.Context()
        pub = context.socket(zmq.PUB)
        pub.setsockopt(zmq.SNDHWM, 2)
        pub.bind("tcp://127.0.0.1:15555")
        sub = context.socket(zmq.SUB)
        sub.setsockopt(zmq.RCVHWM, 2)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        sub.connect("tcp://127.0.0.1:15555")
        time.sleep(0.1)
        yield pub, sub
        pub.close()
        sub.close()
        context.term()

    def test_frame_transmission(self, zmq_pair):
        """Frame sent over ZMQ should arrive intact."""
        pub, sub = zmq_pair
        original = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        timestamp = time.time()
        pub.send(serialize_frame(original, timestamp))
        assert sub.poll(timeout=1000), "Timeout waiting for frame"
        recovered, recv_ts = deserialize_frame(sub.recv())
        np.testing.assert_array_equal(original, recovered)
        assert recv_ts == timestamp


class TestEndToEndPipeline:
    """Full pipeline: frame -> serialize -> deserialize -> crop."""

    def test_full_pipeline(self):
        """Center crop captures a known pattern correctly."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (270, 190), (370, 290), (0, 255, 0), -1)

        data = serialize_frame(frame, time.time())
        recovered, _ = deserialize_frame(data)

        cropper = RotatedCropper({
            'alpha': 0, 'ox': 0.5, 'oy': 0.5, 'width': 0.3, 'height': 0.3
        })
        cropped = cropper.crop(recovered)

        assert cropped.shape == (int(round(480 * 0.3)), int(round(640 * 0.3)), 3)
        # Center of crop should contain the green square
        cy, cx = cropped.shape[0] // 2, cropped.shape[1] // 2
        assert np.mean(cropped[cy - 5:cy + 5, cx - 5:cx + 5, 1]) > 200


class TestPerformance:
    """Crop performance benchmarks."""

    @pytest.fixture
    def cropper(self):
        return RotatedCropper({
            'alpha': 30, 'ox': 0.5, 'oy': 0.5, 'width': 0.4, 'height': 0.4
        })

    def test_sd_crop_performance(self, cropper):
        """SD (640x480) should achieve >60 FPS."""
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        for _ in range(10):
            cropper.crop(frame)
        n = 100
        start = time.perf_counter()
        for _ in range(n):
            cropper.crop(frame)
        fps = n / (time.perf_counter() - start)
        print(f"\nSD crop: {fps:.0f} FPS")
        assert fps > 60

    def test_hd_crop_performance(self, cropper):
        """HD (1920x1080) should achieve >30 FPS."""
        frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        for _ in range(5):
            cropper.crop(frame)
        n = 50
        start = time.perf_counter()
        for _ in range(n):
            cropper.crop(frame)
        fps = n / (time.perf_counter() - start)
        print(f"\nHD crop: {fps:.0f} FPS")
        assert fps > 30


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
