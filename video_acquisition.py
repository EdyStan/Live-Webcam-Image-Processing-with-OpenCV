#!/usr/bin/env python3
"""
Video Acquisition Module

Captures video from webcam and publishes frames over ZeroMQ PUB socket.
Supports configurable camera ID, resolution, and FPS.
"""

import argparse
import signal
import struct
import sys
import time

import cv2
import numpy as np
import zmq


class VideoPublisher:
    """Captures webcam frames and publishes them over ZeroMQ."""

    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480,
                 fps: int = 30, port: int = 5555):
        """
        Initialize the video publisher.

        Args:
            camera_id: Camera device ID (default: 0)
            width: Frame width (default: 640)
            height: Frame height (default: 480)
            fps: Target frames per second (default: 30)
            port: ZeroMQ publisher port (default: 5555)
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.port = port

        self._cap = None
        self._context = None
        self._socket = None
        self._running = False

    def _init_camera(self) -> bool:
        """Initialize or reinitialize the camera."""
        if self._cap is not None:
            self._cap.release()

        self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_id}")
            return False

        # Set camera properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Read back actual values
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

        print(f"Camera initialized: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
        return True

    def _init_zmq(self):
        """Initialize ZeroMQ publisher socket."""
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 2)  # Limit send queue
        self._socket.bind(f"tcp://*:{self.port}")
        print(f"ZeroMQ publisher bound to tcp://*:{self.port}")

    def _serialize_frame(self, frame: np.ndarray, timestamp: float) -> bytes:
        """
        Serialize frame with metadata for transmission.

        Format: timestamp (8B double) + height (4B int) + width (4B int) +
                channels (4B int) + frame bytes
        """
        h, w = frame.shape[:2]
        c = frame.shape[2] if len(frame.shape) == 3 else 1

        header = struct.pack('diii', timestamp, h, w, c)
        return header + frame.tobytes()

    def start(self):
        """Start the video capture and publishing loop."""
        if not self._init_camera():
            return

        self._init_zmq()
        self._running = True

        frame_count = 0
        start_time = time.time()
        last_fps_time = start_time
        reconnect_attempts = 0
        max_reconnect_attempts = 5

        print("Starting video capture... Press Ctrl+C to stop")

        try:
            while self._running:
                ret, frame = self._cap.read()

                if not ret:
                    print("Warning: Failed to read frame")
                    reconnect_attempts += 1
                    if reconnect_attempts >= max_reconnect_attempts:
                        print("Max reconnection attempts reached, trying to reinitialize camera...")
                        if not self._init_camera():
                            print("Failed to reinitialize camera, exiting...")
                            break
                        reconnect_attempts = 0
                    time.sleep(0.1)
                    continue

                reconnect_attempts = 0
                timestamp = time.time()

                # Serialize and send
                data = self._serialize_frame(frame, timestamp)
                self._socket.send(data, zmq.NOBLOCK)

                frame_count += 1

                # Print FPS every second
                if timestamp - last_fps_time >= 1.0:
                    fps = frame_count / (timestamp - last_fps_time)
                    print(f"Publishing at {fps:.1f} FPS", end='\r')
                    frame_count = 0
                    last_fps_time = timestamp

        except KeyboardInterrupt:
            print("\nShutdown requested...")
        finally:
            self.stop()

    def stop(self):
        """Stop the publisher and release resources."""
        self._running = False

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        if self._socket is not None:
            self._socket.close()
            self._socket = None

        if self._context is not None:
            self._context.term()
            self._context = None

        print("Video publisher stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Capture webcam video and publish frames over ZeroMQ"
    )
    parser.add_argument('-c', '--camera', type=int, default=0,
                        help='Camera device ID (default: 0)')
    parser.add_argument('-W', '--width', type=int, default=640,
                        help='Frame width (default: 640)')
    parser.add_argument('-H', '--height', type=int, default=480,
                        help='Frame height (default: 480)')
    parser.add_argument('-f', '--fps', type=int, default=30,
                        help='Target FPS (default: 30)')
    parser.add_argument('-p', '--port', type=int, default=5555,
                        help='ZeroMQ publisher port (default: 5555)')

    args = parser.parse_args()

    publisher = VideoPublisher(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        port=args.port
    )

    # Handle SIGTERM gracefully
    def signal_handler(sig, frame):
        print("\nSignal received, stopping...")
        publisher.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)

    publisher.start()


if __name__ == "__main__":
    main()
