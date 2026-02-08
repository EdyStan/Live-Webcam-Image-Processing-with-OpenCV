import argparse
import json
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import zmq

from utils.rotated_crop import RotatedCropper


class VideoSubscriber:
    """Receives frames over ZeroMQ and applies rotated crop."""

    def __init__(self, host="localhost", port=5555,
                 config_file=None, show_fps=True):
        self.host = host
        self.port = port
        self.config_file = config_file
        self.show_fps = show_fps

        self._context = None
        self._socket = None
        self._cropper = RotatedCropper()
        self._running = False

    def _init_zmq(self):
        """Initialize ZeroMQ subscriber socket."""
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, 2)  # Limit receive queue
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all
        self._socket.connect(f"tcp://{self.host}:{self.port}")
        print(f"Connected to tcp://{self.host}:{self.port}")

    def _load_config(self):
        """Load crop configuration from file."""
        if self.config_file is None:
            # Use default config
            default_config = {
                "alpha": 0,
                "ox": 0.5,
                "oy": 0.5,
                "width": 0.5,
                "height": 0.5
            }
            self._cropper.load_config(default_config)
            print("Using default crop configuration")
            return

        config_path = Path(self.config_file)
        if not config_path.exists():
            print(f"Warning: Config file not found: {self.config_file}")
            print("Using default configuration")
            return

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Validate required keys
            required_keys = ['alpha', 'ox', 'oy', 'width', 'height']
            missing = [k for k in required_keys if k not in config]
            if missing:
                print(f"Warning: Config missing keys: {missing}")
                return

            # Validate ranges
            for key in ['ox', 'oy', 'width', 'height']:
                val = config[key]
                if not (0 <= val <= 1):
                    print(f"Warning: {key}={val} should be in range [0, 1]")

            self._cropper.load_config(config)
            print(f"Loaded config from {self.config_file}")
            print(f"  Angle: {config['alpha']:.2f} deg")
            print(f"  Center: ({config['ox']:.3f}, {config['oy']:.3f})")
            print(f"  Size: {config['width']:.3f} x {config['height']:.3f}")

        except json.JSONDecodeError as e:
            print(f"Error parsing config file: {e}")
        except Exception as e:
            print(f"Error loading config: {e}")

    def _deserialize_frame(self, data):
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

    def _draw_fps(self, image, fps):
        """Draw FPS overlay on image."""
        text = f"FPS: {fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        # Get text size for background rectangle
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Draw background rectangle
        cv2.rectangle(image, (5, 5), (15 + text_w, 15 + text_h + baseline),
                      (0, 0, 0), -1)

        # Draw text
        cv2.putText(image, text, (10, 10 + text_h), font, font_scale,
                    (0, 255, 0), thickness)

        return image

    def start(self):
        """Start the subscriber and display loop."""
        self._init_zmq()
        self._load_config()
        self._running = True

        frame_count = 0
        last_fps_time = time.time()
        display_fps = 0.0

        window_name = "Rotated Crop"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("Starting video display... Press 'q' to quit")

        try:
            while self._running:
                # Try to receive frame with timeout
                if self._socket.poll(timeout=1000):
                    data = self._socket.recv()
                    frame, timestamp = self._deserialize_frame(data)

                    if frame is None:
                        continue

                    # Apply crop
                    cropped = self._cropper.crop(frame)

                    if cropped is None:
                        continue

                    # Calculate and display FPS
                    frame_count += 1
                    now = time.time()
                    if now - last_fps_time >= 1.0:
                        display_fps = frame_count / (now - last_fps_time)
                        frame_count = 0
                        last_fps_time = now

                    if self.show_fps:
                        cropped = self._draw_fps(cropped.copy(), display_fps)

                    # Display
                    cv2.imshow(window_name, cropped)

                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuit requested...")
                    break
                elif key == ord('r'):
                    # Reload config on 'r' key
                    self._load_config()
                    print("Config reloaded")

        except KeyboardInterrupt:
            print("\nShutdown requested...")
        finally:
            self.stop()

    def stop(self):
        """Stop the subscriber and release resources."""
        self._running = False

        cv2.destroyAllWindows()

        if self._socket is not None:
            self._socket.close()
            self._socket = None

        if self._context is not None:
            self._context.term()
            self._context = None

        print("Video subscriber stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Subscribe to video stream, apply rotated crop, and display"
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

    subscriber = VideoSubscriber(
        host=args.host,
        port=args.port,
        config_file=args.config,
        show_fps=not args.no_fps
    )

    subscriber.start()


if __name__ == "__main__":
    main()
