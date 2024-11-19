import mediapipe as mp
import cv2 as cv
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from model import PointHistoryClassifier
from model import LandmarkPointsClassifier
import copy
import itertools
import csv
from collections import Counter
from collections import deque
import pyautogui as pag

# Change the scope of the for loop please

# Initialize necessary MediaPipe classes and options
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Variable to store the latest landmarks
latest_landmarks = None
latest_handedness = None

# Define the callback to handle results
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_landmarks
    global latest_handedness
    # hand_landmarks: List[List[landmark_module.NormalizedLandmark]]
    latest_landmarks = result.hand_landmarks  # Store the latest hand landmarks
    # handedness: List[List[category_module.Category]]
    latest_handedness = result.handedness  # Store the latest handedness


def main():
    # Create hand landmarker options
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result
    )
    
    # Open label
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]
    with open(
            'model/landmark_points_classifier/landmark_points_labels.csv',
            encoding='utf-8-sig') as f:
        landmark_points_classifier_labels = csv.reader(f)
        landmark_points_classifier_labels = [
            row[0] for row in landmark_points_classifier_labels
        ]

    # Read the model
    
    # TODO WORK WITH MODEL
    # point_history_classifier = PointHistoryClassifier()
    landmark_points_classifier = LandmarkPointsClassifier(num_classes=len(landmark_points_classifier_labels), length=42)
    
    # point_history_classifier.run()
    cap = cv.VideoCapture(0)
    
    # Coordinate history
    history_length = 21 * 16
    point_history = deque(maxlen=history_length)
    
    
    # Finger gesture history
    landmark_history = 21
    finger_gesture_history = deque(maxlen=landmark_history)
    STOP = False
    with HandLandmarker.create_from_options(options) as landmarker:
        # Use OpenCV to capture from the webcam
        frame_timestamp_ms = 0
        
        if not cap.isOpened():
            print("Cannot open camera")
                    
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break


            # Create a copy of the frame to draw landmarks on
            frame = cv.flip(frame, 1)
            debug_image = copy.deepcopy(frame)
            
            # Convert the frame to RGB for MediaPipe and to MediaPipe Image format
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Perform asynchronous hand landmark detection
            landmarker.detect_async(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += 1
            

            
            # Draw the latest landmarks on the frame
            if latest_landmarks:
                for landmarks in latest_landmarks:
                    
                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, landmarks)
                    landmark_list = calc_landmark_list(debug_image, landmarks)
                    # Landmark drawing
                    # 0. WRIST
                    # 8. INDEX_FINGER_TIP
                    # 12. MIDDLE_FINGER_TIP
                    # 16. RING_FINGER_TIP
                    # Getting all 21 landmarks
                    for landmark in landmark_list:
                        point_history.append(landmark)
                    hand_sign_id = 0
                    pre_processed_point_history_list = pre_process_point_history(
                debug_image, point_history)
                    
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    
                    hand_sign_id = landmark_points_classifier(pre_processed_landmark_list)
                    if (hand_sign_id == 2):
                        cords = landmark_list[8]
                        moving_mouse(cords[0], cords[1])
                    if (hand_sign_id == 3):
                        click_mouse()
                    if (hand_sign_id == 4):
                        STOP = True
                    
                    # # Finger gesture classification
                    # finger_gesture_id = 0
                    # point_history_len = len(pre_processed_point_history_list)
                    # print(point_history_len)
                    # # 1286 hard coded for now but  1286 is the input for the nn
                    # if point_history_len == (point_history_len * 2):
                    #     # TODO WORK WITH MODEL
                    #     finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
                    
                    # Calculates the gesture IDs in the latest detection
                

                    # Drawing part
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    
                    # Calculates the gesture IDs in the latest detection
                    # finger_gesture_history.append(finger_gesture_id)
                    # most_common_fg_id = Counter(finger_gesture_history).most_common()
                    
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        latest_handedness,
                        landmark_points_classifier_labels[hand_sign_id],
                        point_history_classifier_labels[0],
                    )
            else:
                point_history.append([0, 0])

            # Show the frame with landmarks drawn
            cv.imshow("Hand Landmarks", debug_image)
        
            # Exit loop when 'q' is pressed
            if cv.waitKey(1) & 0xFF == ord('q') or STOP:
                break
            
            
            

    cap.release()
    cv.destroyAllWindows()

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks):
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


def draw_landmarks(image, landmarks, color=(102, 25, 179)) -> cv:
        # Thumb
    # Key Points
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
            cv.circle(image, (landmark[0], landmark[1]), 10, (84, 157, 138),
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
            cv.circle(image, (landmark[0], landmark[1]), 8, (84, 157, 138),
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
            cv.circle(image, (landmark[0], landmark[1]), 8, (84, 157, 138),
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

def draw_info_text(frame, brect, handedness,
                  hand_sign_text,
                    finger_gesture_text
                ):
    
    fixed_x = 10
    fixed_y = 60
    scale = 2.0
    
    
    if handedness:
        info_text = latest_handedness[0][0].category_name
        if info_text == "Left":
            info_text = "Right Hand"
        elif info_text == "Right":
            info_text = "Left Hand"
        cv.putText(frame, info_text, (fixed_x, fixed_y + 55),
                cv.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(frame, info_text, (fixed_x, fixed_y + 55),
                cv.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), 2, cv.LINE_AA)
        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text
            cv.putText(frame, info_text, (fixed_x, fixed_y + 110),
                    cv.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), 4, cv.LINE_AA)
            
    # if finger_gesture_text != "":
    #     cv.putText(frame, "Finger Gesture:" + finger_gesture_text, (fixed_x, fixed_y),
    #                cv.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), 4, cv.LINE_AA)
    #     cv.putText(frame, "Finger Gesture:" + finger_gesture_text, (fixed_x, fixed_y),
    #                cv.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), 2,
    #                cv.LINE_AA)

    return frame

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


def moving_mouse(x, y):
    pag.moveTo(x, y)
    pag.PAUSE = 0.001
def click_mouse():
    cords = pag.position()
    pag.click(cords.x,cords.y)
    
    
if __name__ == '__main__':
    main()
