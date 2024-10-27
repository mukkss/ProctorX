import time
import csv

# Paths to CSV files
AUDIO_CSV_FILE = 'D:\Projects\ProctorX\src\audio_cheat_data.csv'
HEAD_POSE_CSV_FILE = 'D:\Projects\ProctorX\src\head_pose_data.csv'  # Placeholder file name; update as needed

CHEAT_THRESH = 0.6

def avg(current, previous):
    if previous > 1:
        return 0.65
    if current == 0:
        if previous < 0.01:
            return 0.01
        return previous / 1.01
    if previous == 0:
        return current
    return 1 * previous + 0.1 * current

def read_latest_from_csv(csv_file, columns):
    """Read the latest values from the specified CSV file."""
    try:
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            # Read the last row (latest entry)
            last_row = None
            for row in reader:
                last_row = row
            if last_row:
                return {col: float(last_row[col]) for col in columns}
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Error reading {csv_file}: {e}")
        return {col: 0 for col in columns}

def process():
    # Read the latest values from CSV files
    audio_data = read_latest_from_csv(AUDIO_CSV_FILE, ['Sound Amplitude', 'Audio Cheat'])
    head_pose_data = read_latest_from_csv(HEAD_POSE_CSV_FILE, ['X_AXIS_CHEAT', 'Y_AXIS_CHEAT'])

    audio_cheat = audio_data['Audio Cheat']
    x_cheat = head_pose_data['X_AXIS_CHEAT']
    y_cheat = head_pose_data['Y_AXIS_CHEAT']

    if x_cheat == 0 and y_cheat == 0:
        return avg(audio_cheat * 0.2, 0)  # Calculate based on audio
    else:
        return avg(0.4, 0)  # Placeholder logic if head pose indicates possible cheating

def run_detection(data_queue):
    while True:
        PERCENTAGE_CHEAT = process()  # Calculate cheat percentage
        GLOBAL_CHEAT = 1 if PERCENTAGE_CHEAT > CHEAT_THRESH else 0

        # Send the percentage to the main thread for plotting
        data_queue.put(PERCENTAGE_CHEAT)
        
        # For demonstration purposes
        print("Cheat percent:", PERCENTAGE_CHEAT, GLOBAL_CHEAT)

        time.sleep(0.05)  # Adjust sleep for refresh rate
