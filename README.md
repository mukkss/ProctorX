# ProctorX (Automated Exam Proctoring using Python)

## Configuration
- **Anaconda Virtual Environment**
- **Python Version**: 3.11.7
- **Install External Libraries**:
   - Install all required libraries in the current directory by running:
   ```bash
   pip install -r requirements.txt
   ``

## Running the Program
1. **Start Audio and Head Pose Detection**:
   - Run the `main.py` file to initiate both audio detection and head pose estimation.
  
2. **Store Data**:
   - Data related to audio detection, head pose, and potential cheating will be collected and saved in a CSV file for later analysis.

3. **Functionality**:
   - The system will monitor audio levels to detect suspicious activity and analyze head pose angles to identify potential cheating.
   - Future versions will include additional detection capabilities for tab switching and copy-paste actions to enhance proctoring accuracy.

4. **Viewing Results**:
   - You can modify the code to visualize or further analyze the data stored in the CSV file based on your needs.

## Additional Features
- Future versions will include detection for:
  - Tab switching
  - Copy-paste actions


**Note**: Feel free to use a database of your convenience and tweak the code accordingly. It’s suggested to use SQLite3 for simplicity.

If you find this project useful, please consider giving it a star and forking it! 😊

