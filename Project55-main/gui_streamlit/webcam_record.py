import cv2
import time
import datetime
import os
import pyaudio
import wave
import threading
import ffmpeg

class WebcamRecorder:
    def __init__(self, output_directory="recordings", video_format=".mp4", resolution=(640, 480), fps=30.0, audio_format=".wav"):
        """
        Initialize the webcam recorder
        
        Args:
            output_directory (str): Directory to save recordings
            video_format (str): Video file extension
            resolution (tuple): Video resolution as (width, height)
            fps (float): Frames per second
            audio_format (str): Audio file extension
        """
        self.output_directory = output_directory
        self.video_format = video_format
        self.audio_format = audio_format
        self.resolution = resolution
        self.fps = fps
        self.cap = None
        self.out = None
        self.is_recording = False
        self.window_name = "Recording Preview"
        self.audio_thread = None
        self.audio_frames = []
        self.video_filepath = None
        self.audio_filepath = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
    
    def start_recording(self, duration=None, show_preview=True):
        """
        Start recording from webcam
        
        Args:
            duration (float, optional): Recording duration in seconds. If None, records until stop_recording is called.
            show_preview (bool): Whether to show preview window during recording
        
        Returns:
            bool: True if recording started successfully, False otherwise
        """
        if self.is_recording:
            print("Already recording!")
            return False
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return False
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Create filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_filename = f"webcam_{timestamp}{self.video_format}"
        self.audio_filename = f"audio_{timestamp}{self.audio_format}"
        self.video_filepath = os.path.join(self.output_directory, self.video_filename)
        self.audio_filepath = os.path.join(self.output_directory, self.audio_filename)
        
        # Initialize video writer
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(self.video_filepath, -1, self.fps, self.resolution)
        
        self.is_recording = True
        print(f"Recording started. Saving to {self.video_filepath} and {self.audio_filepath}")
        
        # Start audio recording in a separate thread
        self.audio_thread = threading.Thread(target=self._record_audio)
        self.audio_thread.start()
        
        # Create preview window if requested
        if show_preview:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
            
        if duration is not None:
            # Record for specified duration with preview
            start_time = time.time()
            while time.time() - start_time < duration:
                self._capture_frame(show_preview)
                # Add a small delay to control frame rate
                time.sleep(1/self.fps)
                # Check for window close or any key press to stop recording
                if cv2.waitKey(1) != -1:
                    break
            self.stop_recording()
        else:
            # For non-duration recording, we'll let the caller handle the loop
            if show_preview:
                print("Preview window opened. Press 'q' to stop recording.")
            
        return True
    
    def _capture_frame(self, show_preview=True):
        """
        Capture a single frame and write to video file
        
        Args:
            show_preview (bool): Whether to show the frame in a preview window
            
        Returns:
            bool: True if frame was captured successfully, False otherwise
        """
        if not self.is_recording:
            return False
        
        ret, frame = self.cap.read()
        if ret:
            # Write frame to video file
            self.out.write(frame)
            
            # Show preview if requested
            if show_preview:
                # Add recording indicator
                cv2.putText(
                    frame, 
                    "REC", 
                    (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 0, 255), 
                    2
                )
                
                # Add recording time indicator
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                cv2.putText(
                    frame, 
                    timestamp, 
                    (frame.shape[1] - 120, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    1
                )
                
                cv2.imshow(self.window_name, frame)
            
            return True
        return False
    
    def _record_audio(self):
        """Record audio from the microphone"""
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=44100,
                        input=True,
                        frames_per_buffer=1024)
        
        self.audio_frames = []
        
        while self.is_recording:
            data = stream.read(1024)
            self.audio_frames.append(data)
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save the recorded audio to a file
        wf = wave.open(self.audio_filepath, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(self.audio_frames))
        wf.close()
    
    def stop_recording(self):
        """Stop recording and release resources"""
        if not self.is_recording:
            print("Not currently recording!")
            return False
        
        self.is_recording = False
        
        # Wait for the audio thread to finish
        if self.audio_thread is not None:
            self.audio_thread.join()
        
        # Release resources
        if self.out:
            self.out.release()
        
        if self.cap:
            self.cap.release()
        
        # Close any open windows
        cv2.destroyAllWindows()
        
        print("Recording stopped.")
        
        # Combine audio and video using ffmpeg
        # self._combine_audio_video()
        
        return True
    
    # def _combine_audio_video(self):
    #     """Combine recorded audio and video into a single file"""
    #     output_filepath = os.path.join(self.output_directory, f"output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    #     input_video = ffmpeg.input(self.video_filepath)
    #     input_audio = ffmpeg.input(self.audio_filepath)
    #     ffmpeg.output(input_video, input_audio, output_filepath, vcodec='copy', acodec='aac', strict='experimental').run()
    #     print(f"Combined audio and video saved to {output_filepath}")
    #     self.combined_filepath = output_filepath
    
    def get_most_recent_filepaths(self):
        """
        Get the file paths of the most recent video and audio files
        
        Returns:
            tuple: (video_filepath, audio_filepath, combined_filepath)
        """
        return self.video_filepath, self.audio_filepath, getattr(self, 'combined_filepath', None)
    
    def record_with_preview(self, duration=None):
        """
        Record with preview window until duration is reached or 'q' is pressed
        
        Args:
            duration (float, optional): Recording duration in seconds
        """
        if not self.start_recording(show_preview=True):
            return
        
        start_time = time.time()
        try:
            while self.is_recording:
                # Check if duration is set and reached
                if duration and (time.time() - start_time >= duration):
                    break
                
                # Capture frame and show preview
                if not self._capture_frame(show_preview=True):
                    break
                
                # Break loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Small delay to control frame rate
                time.sleep(1/self.fps)
                
        finally:
            self.stop_recording()

# Example usage
if __name__ == "__main__":
    # Create recorder instance
    recorder = WebcamRecorder()
    
    # Example 1: Record for 10 seconds with preview
    # recorder.start_recording(duration=10, show_preview=True)
    
    # Example 2: Record until 'q' is pressed
    #print("\nRecording until 'q' is pressed...")
    #recorder.record_with_preview()
    
    # Get the most recent file paths
    #video_filepath, audio_filepath, combined_filepath = recorder.get_most_recent_filepaths()
    #print(f"Video file saved to: {video_filepath}")
    #print(f"Audio file saved to: {audio_filepath}")
    #print(f"Combined file saved to: {combined_filepath}")