import data.head_pose as head_pose 
import threading as th 
import time
from data.sound import AudioCheatDetector  # Replace with the actual module name

# Set the exit flag to false initially
head_pose.EXIT_FLAG = False

def main():
    # Create instances for both head pose and audio cheat detection
    audio_detector = AudioCheatDetector()
    
    # Start the audio detection thread
    audio_thread = audio_detector.start()
    
    # Create and start the head pose estimation thread
    head_pose_thread = th.Thread(target=head_pose.pose)
    head_pose_thread.start()

    try:
        # Keep the main program running
        while True:
            if audio_detector.audio_cheat:
                print("Suspicious audio detected!")
            else:
                print("Audio detection is running, no suspicious activity.")  # Confirm audio detection is running
            time.sleep(1)  # Adjust as needed for responsiveness
    except KeyboardInterrupt:
        # Stop both threads on keyboard interrupt
        audio_detector.stop()  # Stop audio detection
        head_pose_thread.join()  # Wait for head pose thread to finish
        audio_thread.join()  # Wait for audio thread to finish
        print("Program terminated by user.")

if __name__ == "__main__":
    main()
