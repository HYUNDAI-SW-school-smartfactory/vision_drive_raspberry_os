import time
import os
import re
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder, JpegEncoder
from picamera2.outputs import FileOutput
from libcamera import Transform
from threading import Condition
import io

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        """Initialize the StreamingOutput class."""
        self.frame = None
        self.condition = Condition()  # Initialize the condition variable for thread synchronization

    def write(self, buf: bytes) -> int:
        """Write a buffer to the frame and notify all waiting threads."""
        with self.condition:
            self.frame = buf             # Update the frame buffer with new data
            self.condition.notify_all()  # Notify all waiting threads that new data is available
        return len(buf)

class Camera:
    def __init__(self, preview_size: tuple = (640, 480), hflip: bool = False, vflip: bool = False, stream_size: tuple = (400, 300)):
        """Initialize the Camera class."""
        self.camera = Picamera2()  # Initialize the Picamera2 object
        self.transform = Transform(hflip=1 if hflip else 0, vflip=1 if vflip else 0)  # Set the transformation for flipping the image
        preview_config = self.camera.create_preview_configuration(main={"size": preview_size}, transform=self.transform)  # Create the preview configuration
        self.camera.configure(preview_config)  # Configure the camera with the preview settings
        
        # Configure video stream
        self.stream_size = stream_size  # Set the size of the video stream
        self.stream_config = self.camera.create_video_configuration(main={"size": stream_size}, transform=self.transform)  # Create the video configuration
        self.streaming_output = StreamingOutput()  # Initialize the streaming output object
        self.streaming = False  # Initialize the streaming flag

    def start_image(self, use_preview: bool = True) -> None:
        """Start the camera preview and capture."""
        if use_preview:
            self.camera.start_preview(Preview.QTGL)  # Start the camera preview using the QTGL backend
        self.camera.start()                          # Start the camera

    def save_image(self, filename: str) -> dict:
        """Capture and save an image to the specified file."""
        try:
            metadata = self.camera.capture_file(filename)  # Capture an image and save it to the specified file
            return metadata                              # Return the metadata of the captured image
        except Exception as e:
            print(f"Error capturing image: {e}")         # Print error message if capturing fails
            return None                                  # Return None if capturing fails

    def start_stream(self, filename: str = None) -> None:
        """Start the video stream or recording."""
        if not self.streaming:
            if self.camera.started:
                self.camera.stop()                         # Stop the camera if it is currently running
            
            self.camera.configure(self.stream_config)      # Configure the camera with the video stream settings
            if filename:
                encoder = H264Encoder()                    # Use H264 encoder for video recording
                output = FileOutput(filename)              # Set the output file for the recorded video
            else:
                encoder = JpegEncoder()                    # Use Jpeg encoder for streaming
                output = FileOutput(self.streaming_output) # Set the streaming output object
            self.camera.start_recording(encoder, output)   # Start recording or streaming
            self.streaming = True                          # Set the streaming flag to True

    def stop_stream(self) -> None:
        """Stop the video stream or recording."""
        if self.streaming:
            try:
                self.camera.stop_recording()               # Stop the recording or streaming
                self.streaming = False                     # Set the streaming flag to False
            except Exception as e:
                print(f"Error stopping stream: {e}")       # Print error message if stopping fails

    def get_frame(self) -> bytes:
        """Get the current frame from the streaming output."""
        with self.streaming_output.condition:
            self.streaming_output.condition.wait()         # Wait for a new frame to be available
            return self.streaming_output.frame             # Return the current frame

    def save_video(self, filename: str, duration: int = 10) -> None:
        """Save a video for the specified duration."""
        self.start_stream(filename)                        # Start the video recording
        time.sleep(duration)                               # Record for the specified duration
        self.stop_stream()                                 # Stop the video recording

    def close(self) -> None:
        """Close the camera."""
        if self.streaming:
            self.stop_stream()                             # Stop the streaming if it is active
        self.camera.close()                                # Close the camera

if __name__ == '__main__':
    print('Program is starting ... ')
    camera = Camera()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, "image")
    os.makedirs(image_dir, exist_ok=True)

    def normalize_name(name: str) -> str:
        cleaned = re.sub(r'[^A-Za-z0-9_-]', '_', name.strip())
        return cleaned if cleaned else "image"

    def build_output_path(base_name: str) -> str:
        first_path = os.path.join(image_dir, f"{base_name}.jpg")
        if not os.path.exists(first_path):
            return first_path
        index = 1
        while True:
            candidate = os.path.join(image_dir, f"{base_name}_{index}.jpg")
            if not os.path.exists(candidate):
                return candidate
            index += 1

    try:
        print("View image...")
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if not has_display:
            print("No GUI display detected. Running in headless mode (no preview window).")
        camera.start_image(use_preview=has_display)
        print("Command: c <name> (capture), q (quit)")

        while True:
            user_input = input(">> ").strip()
            if not user_input:
                continue
            if user_input.lower() == "q":
                print("Exit requested.")
                break
            if user_input.lower().startswith("c"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 2 and parts[1].strip():
                    raw_name = parts[1].strip()
                else:
                    raw_name = input("Image name: ").strip()
                base_name = normalize_name(raw_name)
                output_path = build_output_path(base_name)
                metadata = camera.save_image(filename=output_path)
                if metadata is not None:
                    print(f"Saved: {output_path}")
                continue
            print("Unknown command. Use: c <name> or q")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("Close camera...")
        camera.close()
