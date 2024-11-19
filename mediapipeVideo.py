import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import numpy as np
import mediapipe as mp
import cv2 as cv
import os
import warnings
import tensorflow as tf




# Work on getting better data and change it so we get all hte points think about it as getting a daque of hands instead of finger tips

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
def main_video():
    # Create a hand landmarker instance with the video mode:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO)
    use_brect = True
    
    
    
    # Read labels
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]
        
    
    # Mode to history screen shot
    mode = 1
    
    # Directory where .mov files are stored
    directory = 'videos/FROM_LOCAL_PATH'
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Build the path to the current item
        dir_path = os.path.join(directory, filename)
        
        # Check if it's a directory
        if not os.path.isdir(dir_path):
            continue  # Skip if it's not a directory
        
        number =  int(filename[-1])
        for file in os.listdir(os.path.join(directory, filename)):
            if file.endswith('.mp4'):
                file_path = os.path.join(directory, filename, file)
                print(f"File: {file_path}")
                # Coordinate history
                history_length = 16
                point_history = deque(maxlen=history_length)
                finger_gesture_history = deque(maxlen=history_length)

                with HandLandmarker.create_from_options(options) as landmarker:
                    # Use OpenCV’s VideoCapture to load the input video
                    cap = cv.VideoCapture(file_path)
                    fps = cap.get(cv.CAP_PROP_FPS)
                    frame_count = 0  # Initialize a counter to keep track of frames
                    if not cap.isOpened():
                        print(f"Error opening video file: {file}")
                        continue
                    
                    # Loop through each frame in the video
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break  # Exit the loop if no frame is captured
                        
                        # Get the current timestamp in milliseconds
                        frame_timestamp_ms = int(cap.get(cv.CAP_PROP_POS_MSEC))

                        # Convert the frame from BGR (OpenCV format) to RGB for MediaPipe
                        # Create a copy of the frame to draw landmarks on
                        frame = cv.flip(frame, 1)
                        debug_image = copy.deepcopy(frame)
                        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                        # Perform hand landmarks detection on the single image
                        # The hand landmarker must be created with the video mode.
                        hand_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                        # Process each detected hand
                        
                        
                            
                        
                        # To print to the screen
                        # iterating through the list of landmarks
                        # Going through the list of List[List[landmark_modle.NormalizedLandmark]]
                        # count should reach 3
                        if len(hand_landmarker_result.hand_landmarks) > 0:
                            # 21 hand landmarks 
                            # x and y coordinates are normalized to [0.0, 1.0] by the image width and height, respectively
                            for landmarks in hand_landmarker_result.hand_landmarks:
                                
                                landmark_list = calc_landmark_list(debug_image, landmarks)
                                # Landmark drawing
                                # 0. WRIST
                                # 8. INDEX_FINGER_TIP
                                # 12. MIDDLE_FINGER_TIP
                                # 16. RING_FINGER_TIP
                                # getting all landmarks from land_list
                                for landmark in landmark_list:
                                    point_history.append(landmark)
                                
                                

                                # Conversion to relative coordinates / normalized coordinates
                                pre_processed_landmark_list = pre_process_landmark(
                                    landmark_list)
                                pre_processed_point_history_list = pre_process_point_history(
                                                debug_image, landmark_list)
                                # Drawing part
                                frame = draw_landmarks(debug_image, landmark_list)
                                # Write to the dataset file
                                logging_csv(number, mode, pre_processed_point_history_list, pre_processed_landmark_list)
                            # Display the frame
                            cv.imshow('Hand Landmarks' + file_path, debug_image)
                            # Optional: display the frame
                            # Process to end
                            key = cv.waitKey(10)
                            if key == 27:  # ESC
                                break
                        else:
                            point_history.append([0, 0])
                        # Increment the frame count
                        frame_count += 1
                    
                    # Release the video capture and close display windows
                    cap.release()
                    cv.destroyAllWindows()
                
        
        
def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 107:  # k
        mode = 1
    return number, mode
        
def calc_bounding_rect(image, landmark):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)
    landmark_x = min(int(landmark.x * image_width), image_width - 1)
    landmark_y = min(int(landmark.y * image_height), image_height - 1)

    landmark_point = [np.array((landmark_x, landmark_y))]

    landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]  
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    # Keypoint

    for _, landmark in enumerate(landmarks):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history

def logging_csv(number, mode, point_history_list, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/landmark_points_classifier/landmark_points.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return 

def draw_landmarks(image, landmarks, color=(102, 25, 179)):
    # Thumb
    if len(landmarks) > 0:
        # Thumb
        cv.line(image, tuple(landmarks[2]), tuple(landmarks[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmarks[2]), tuple(landmarks[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmarks[3]), tuple(landmarks[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmarks[3]), tuple(landmarks[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmarks[5]), tuple(landmarks[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmarks[5]), tuple(landmarks[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmarks[6]), tuple(landmarks[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmarks[6]), tuple(landmarks[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmarks[7]), tuple(landmarks[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmarks[7]), tuple(landmarks[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmarks[9]), tuple(landmarks[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmarks[9]), tuple(landmarks[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmarks[10]), tuple(landmarks[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmarks[10]), tuple(landmarks[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmarks[11]), tuple(landmarks[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmarks[11]), tuple(landmarks[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmarks[13]), tuple(landmarks[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmarks[13]), tuple(landmarks[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmarks[14]), tuple(landmarks[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmarks[14]), tuple(landmarks[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmarks[15]), tuple(landmarks[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmarks[15]), tuple(landmarks[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmarks[17]), tuple(landmarks[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmarks[17]), tuple(landmarks[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmarks[18]), tuple(landmarks[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmarks[18]), tuple(landmarks[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmarks[19]), tuple(landmarks[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmarks[19]), tuple(landmarks[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmarks[0]), tuple(landmarks[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmarks[0]), tuple(landmarks[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmarks[1]), tuple(landmarks[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmarks[1]), tuple(landmarks[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmarks[2]), tuple(landmarks[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmarks[2]), tuple(landmarks[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmarks[5]), tuple(landmarks[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmarks[5]), tuple(landmarks[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmarks[9]), tuple(landmarks[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmarks[9]), tuple(landmarks[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmarks[13]), tuple(landmarks[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmarks[13]), tuple(landmarks[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmarks[17]), tuple(landmarks[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmarks[17]), tuple(landmarks[0]),
                (255, 255, 255), 2)
        
    # Key Points
    for index, landmark in enumerate(landmarks):
        if index == 0:  # wrist1
                cv.circle(image, (landmark[0], landmark[1]), 5, color,
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # wrist2
            cv.circle(image, (landmark[0], landmark[1]), 5, color,
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # Thumb: base
            cv.circle(image, (landmark[0], landmark[1]), 5, color,
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # Thumb: 1st joint
            cv.circle(image, (landmark[0], landmark[1]), 5, color,
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # Thumb: fingertip
            cv.circle(image, (landmark[0], landmark[1]), 8, color,
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # Index finger: base
            cv.circle(image, (landmark[0], landmark[1]), 5, color,
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # Index finger: 2nd joint
            cv.circle(image, (landmark[0], landmark[1]), 5, color,
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # Index finger: 1st joint
            cv.circle(image, (landmark[0], landmark[1]), 5, color,
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # Index finger: fingertip
            cv.circle(image, (landmark[0], landmark[1]), 10, color,
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # Middle finger: base
            cv.circle(image, (landmark[0], landmark[1]), 5, color,
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # Middle finger: 2nd joint
            cv.circle(image, (landmark[0], landmark[1]), 5, color,
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # Middle finger: 1st joint
            cv.circle(image, (landmark[0], landmark[1]), 5, color,
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # Middle finger: fingertip
            cv.circle(image, (landmark[0], landmark[1]), 8, color,
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # Ring finger: base
            cv.circle(image, (landmark[0], landmark[1]), 5, color,
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # Ring finger: 2nd joint
            cv.circle(image, (landmark[0], landmark[1]), 5, color,
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # Ring finger: 1st joint
            cv.circle(image, (landmark[0], landmark[1]), 5, color,
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # Ring finger: fingertip
            cv.circle(image, (landmark[0], landmark[1]), 8, color,
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 17:  # Pinky: base
            cv.circle(image, (landmark[0], landmark[1]), 5, color,
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # Pinky: 2nd joint
            cv.circle(image, (landmark[0], landmark[1]), 5, color,
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # Pinky: 1st joint
            cv.circle(image, (landmark[0], landmark[1]), 5, color,
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # Pinky: fingertip
            cv.circle(image, (landmark[0], landmark[1]), 8, color,
                        -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image

def draw_info(image, mode, number):
    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image

def draw_info_text(image, brect, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    # if finger_gesture_text != "":
    #     cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
    #                cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    #     cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
    #                cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
    #                cv.LINE_AA)

    return image

if __name__ == '__main__':
    main_video()
        
        
