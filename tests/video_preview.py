import argparse
import json
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import zmq
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.rotated_crop import compute_rotated_corners


class VideoPreview:
    """Receives frames over ZeroMQ and displays with crop overlay."""

    def __init__(self, host: str = "localhost", port: int = 5555,
                 config_file: str = None, show_fps: bool = True):
        self.host = host
        self.port = port
        self.config_file = config_file
        self.show_fps = show_fps

        self._context = None
        self._socket = None
        self._config = {}
        self._running = False

    def _init_zmq(self):
        """Initialize ZeroMQ subscriber socket."""
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, 2)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.connect(f"tcp://{self.host}:{self.port}")
        print(f"Connected to tcp://{self.host}:{self.port}")

    def _load_config(self):
        """Load crop configuration from file."""
        if self.config_file is None:
            self._config = {
                "alpha": 0,
                "ox": 0.5,
                "oy": 0.5,
                "width": 0.5,
                "height": 0.5
            }
            print("Using default crop configuration")
            return

        config_path = Path(self.config_file)
        if not config_path.exists():
            print(f"Warning: Config file not found: {self.config_file}")
            return

        try:
            with open(config_path, 'r') as f:
                self._config = json.load(f)

            print(f"Loaded config from {self.config_file}")
            print(f"  Angle: {self._config.get('alpha', 0):.2f} deg")
            print(f"  Center: ({self._config.get('ox', 0.5):.3f}, {self._config.get('oy', 0.5):.3f})")
            print(f"  Size: {self._config.get('width', 0.5):.3f} x {self._config.get('height', 0.5):.3f}")

        except json.JSONDecodeError as e:
            print(f"Error parsing config file: {e}")
        except Exception as e:
            print(f"Error loading config: {e}")

    def _deserialize_frame(self, data: bytes) -> tuple:
        """Deserialize frame data."""
        header_size = struct.calcsize('diii')
        if len(data) < header_size:
            return None, None

        timestamp, h, w, c = struct.unpack('diii', data[:header_size])
        frame_data = data[header_size:]

        expected_size = h * w * c
        if len(frame_data) != expected_size:
            return None, None

        dtype = np.uint8
        if c == 1:
            frame = np.frombuffer(frame_data, dtype=dtype).reshape((h, w))
        else:
            frame = np.frombuffer(frame_data, dtype=dtype).reshape((h, w, c))

        return frame, timestamp

    def _draw_crop_box(self, image: np.ndarray) -> np.ndarray:
        """Draw the rotated crop bounding box on the image."""
        img_h, img_w = image.shape[:2]

        # Get crop parameters
        alpha = self._config.get('alpha', 0)
        ox = self._config.get('ox', 0.5)
        oy = self._config.get('oy', 0.5)
        crop_w = self._config.get('width', 0.5)
        crop_h = self._config.get('height', 0.5)

        # Convert normalized coords to pixels
        center_x = ox * img_w
        center_y = oy * img_h
        width = crop_w * img_w
        height = crop_h * img_h

        # Get rotated corners
        corners = compute_rotated_corners(center_x, center_y, width, height, alpha)
        corners_int = corners.astype(np.int32)

        # Draw the rotated rectangle
        cv2.polylines(image, [corners_int], isClosed=True,
                      color=(0, 255, 0), thickness=2)

        # Draw corner markers
        for i, corner in enumerate(corners_int):
            # Different colors for each corner to show orientation
            colors = [(255, 0, 0), (0, 255, 255), (0, 0, 255), (255, 0, 255)]
            cv2.circle(image, tuple(corner), 6, colors[i], -1)

        # Draw center point
        center = (int(center_x), int(center_y))
        cv2.circle(image, center, 8, (0, 255, 0), -1)
        cv2.circle(image, center, 10, (255, 255, 255), 2)

        # Draw info text
        info_lines = [
            f"Angle: {alpha:.1f} deg",
            f"Center: ({ox:.3f}, {oy:.3f})",
            f"Size: {crop_w:.3f} x {crop_h:.3f}",
            f"Pixels: {int(width)} x {int(height)}"
        ]

        y_offset = 30
        for line in info_lines:
            cv2.putText(image, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(image, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_offset += 25

        return image

    def _draw_fps(self, image: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS overlay on image."""
        text = f"FPS: {fps:.1f}"
        h = image.shape[0]
        cv2.putText(image, text, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(image, text, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return image

    def start(self):
        """Start the preview display loop."""
        self._init_zmq()
        self._load_config()
        self._running = True

        frame_count = 0
        last_fps_time = time.time()
        display_fps = 0.0

        window_name = "Video Preview (with crop box)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("\nStarting video preview...")
        print("Controls:")
        print("  q - Quit")
        print("  r - Reload config file")
        print("\nCorner colors: Blue=TL, Cyan=TR, Red=BR, Magenta=BL")

        try:
            while self._running:
                if self._socket.poll(timeout=1000):
                    data = self._socket.recv()
                    frame, timestamp = self._deserialize_frame(data)

                    if frame is None:
                        continue

                    # Make a copy to draw on
                    display = frame.copy()

                    # Draw crop bounding box
                    display = self._draw_crop_box(display)

                    # Calculate and display FPS
                    frame_count += 1
                    now = time.time()
                    if now - last_fps_time >= 1.0:
                        display_fps = frame_count / (now - last_fps_time)
                        frame_count = 0
                        last_fps_time = now

                    if self.show_fps:
                        display = self._draw_fps(display, display_fps)

                    cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuit requested...")
                    break
                elif key == ord('r'):
                    self._load_config()
                    print("Config reloaded")

        except KeyboardInterrupt:
            print("\nShutdown requested...")
        finally:
            self.stop()

    def stop(self):
        """Stop the preview and release resources."""
        self._running = False
        cv2.destroyAllWindows()

        if self._socket is not None:
            self._socket.close()
            self._socket = None

        if self._context is not None:
            self._context.term()
            self._context = None

        print("Video preview stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Preview video stream with crop bounding box overlay"
    )
    parser.add_argument('-H', '--host', type=str, default='localhost',
                        help='ZeroMQ publisher host (default: localhost)')
    parser.add_argument('-p', '--port', type=int, default=5555,
                        help='ZeroMQ publisher port (default: 5555)')
    parser.add_argument('-c', '--config', type=str, default='rcrop_parameters.json',
                        help='JSON crop configuration file (default: rcrop_parameters.json)')
    parser.add_argument('--no-fps', action='store_true',
                        help='Disable FPS overlay')

    args = parser.parse_args()

    preview = VideoPreview(
        host=args.host,
        port=args.port,
        config_file=args.config,
        show_fps=not args.no_fps
    )

    preview.start()


if __name__ == "__main__":
    main()
