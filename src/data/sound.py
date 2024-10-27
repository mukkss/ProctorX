import sounddevice as sd
import numpy as np
import threading as th
import csv
import time

class AudioCheatDetector:
    def __init__(self, csv_file='audio_cheat_data.csv'):
        self.sound_amplitude = 0
        self.audio_cheat = 0

        self.callbacks_per_second = 38
        self.sus_finding_frequency = 2
        self.sound_amplitude_threshold = 20

        self.frames_count = int(self.callbacks_per_second / self.sus_finding_frequency)
        self.amplitude_list = [0] * self.frames_count
        self.sus_count = 0
        self.count = 0
        self.running = True

        self.csv_file = csv_file
        self.init_csv()  # Initialize CSV file and header

    def init_csv(self):
        """Initialize the CSV file with a header."""
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Sound Amplitude', 'Audio Cheat'])

    def log_to_csv(self, timestamp, sound_amplitude, audio_cheat):
        """Log the data to the CSV file."""
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, sound_amplitude, audio_cheat])

    def print_sound(self, indata, outdata, frames, time, status):
        """Callback function to process audio data."""
        vnorm = int(np.linalg.norm(indata) * 10)
        self.amplitude_list.append(vnorm)
        self.count += 1
        self.amplitude_list.pop(0)

        if self.count >= self.frames_count:
            avg_amp = sum(self.amplitude_list) / self.frames_count
            self.sound_amplitude = avg_amp

            if avg_amp > self.sound_amplitude_threshold:
                self.sus_count += 1
                if self.sus_count >= 2:
                    self.audio_cheat = 1
                    print("Suspicious audio detected!")  # Log when suspicious audio is detected
                    self.sus_count = 0
            else:
                self.sus_count = 0
                self.audio_cheat = 0

            # Log data to CSV
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            self.log_to_csv(timestamp, self.sound_amplitude, self.audio_cheat)

            self.count = 0

    def sound(self):
        """Start the sound stream."""
        with sd.Stream(callback=self.print_sound):
            print("Sound stream started.")
            while self.running:
                sd.sleep(100)  # Sleep briefly to allow callback processing

    def start(self):
        """Start the sound detection in a separate thread."""
        sound_thread = th.Thread(target=self.sound)
        sound_thread.start()
        return sound_thread

    def stop(self):
        """Stop the sound detection."""
        self.running = False
        print("Sound stream stopped.")
