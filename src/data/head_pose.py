import cv2
import mediapipe as mp
import numpy as np
import threading as th
import time
import csv  # Import CSV module for file handling

# Placeholders and global variables
x = 0  # X axis head pose
y = 0  # Y axis head pose
X_AXIS_CHEAT = 0
Y_AXIS_CHEAT = 0
EXIT_FLAG = False  # Global flag for clean exit

# To track the last state to print only when state changes
last_state = None

def pose(csv_file_path='head_pose_data.csv'):
    global x, y, X_AXIS_CHEAT, Y_AXIS_CHEAT, EXIT_FLAG, last_state
    # Initialize MediaPipe Pose
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Camera not accessible.")
            return
        mp_drawing = mp.solutions.drawing_utils
        fps = 30  # Target frames per second

        # Open the CSV file for writing
        with open(csv_file_path, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'x_angle', 'y_angle', 'current_state', 'x_axis_cheat', 'y_axis_cheat']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            while cap.isOpened() and not EXIT_FLAG:
                start_time = time.time()

                success, image = cap.read()
                if not success:
                    print("Error: Failed to capture image.")
                    break

                # Flip the image horizontally for a selfie-view display
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False  # To improve performance
                results = face_mesh.process(image)
                image.flags.writeable = True  # Re-enable write
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                img_h, img_w, img_c = image.shape
                face_3d = []
                face_2d = []
                face_ids = [33, 263, 1, 61, 291, 199]

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None)
                        for idx, lm in enumerate(face_landmarks.landmark):
                            if idx in face_ids:
                                if idx == 1:
                                    nose_2d = (lm.x * img_w, lm.y * img_h)
                                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                                x, y = int(lm.x * img_w), int(lm.y * img_h)
                                face_2d.append([x, y])
                                face_3d.append([x, y, lm.z])

                        face_2d = np.array(face_2d, dtype=np.float64)
                        face_3d = np.array(face_3d, dtype=np.float64)

                        focal_length = 1 * img_w
                        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                               [0, focal_length, img_w / 2],
                                               [0, 0, 1]])
                        dist_matrix = np.zeros((4, 1), dtype=np.float64)

                        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                        rmat, jac = cv2.Rodrigues(rot_vec)
                        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                        x = angles[0] * 360
                        y = angles[1] * 360

                        current_state = "Forward"
                        if y < -10:  # Looking left
                            current_state = "Looking Left"
                            X_AXIS_CHEAT = 1
                        elif y > 10:  # Looking right
                            current_state = "Looking Right"
                            X_AXIS_CHEAT = 1
                        elif x < -10:  # Looking down
                            current_state = "Looking Down"
                            Y_AXIS_CHEAT = 1
                        else:
                            X_AXIS_CHEAT = 0
                            Y_AXIS_CHEAT = 0

                        # Only print when state changes
                        if current_state != last_state:
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                            print(f"[{timestamp}] State changed: {current_state}")
                            last_state = current_state  # Update the last state

                        # Write data to the CSV file
                        writer.writerow({
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                            'x_angle': x,
                            'y_angle': y,
                            'current_state': current_state,
                            'x_axis_cheat': X_AXIS_CHEAT,
                            'y_axis_cheat': Y_AXIS_CHEAT
                        })

                        # Projection to show head direction visually
                        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                        p1 = (int(nose_2d[0]), int(nose_2d[1]))
                        p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                        cv2.line(image, p1, p2, (255, 0, 0), 2)

                        cv2.putText(image, current_state, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow('Head Pose Estimation', image)

                if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
                    EXIT_FLAG = True

                # FPS control: limit the frame rate to the target fps
                end_time = time.time()
                time_elapsed = end_time - start_time
                if time_elapsed < 1 / fps:
                    time.sleep(1 / fps - time_elapsed)

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")

if __name__ == "__main__":
    t1 = th.Thread(target=pose)
    t1.start()

    try:
        t1.join()  # Wait for the thread to finish
    except KeyboardInterrupt:
        EXIT_FLAG = True
        t1.join()
        print("Program terminated by user.")
