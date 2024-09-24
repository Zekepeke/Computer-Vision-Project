import mediapipe as mp
import cv2 as cv
import pyautogui as pag

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


cap = cv.VideoCapture(0)

screenWidth, screenHeight = pag.size()
id = 0

with mp_hands.Hands(max_num_hands=1,model_complexity=0,
                    min_detection_confidence=0.8, 
                    min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Frame not register
        if not ret:
            break
        
        # Frame Height and Width
        frameHeight, frameWidth, _ = frame.shape
        
        # Convert BGR to RGB for Mediapipe
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        # print(results.multi_hand_world_landmarks)

        # Set flag to false
        image.flags.writeable = False
        
        # Convert the image back to BGR for rendering
        image = cv.cvtColor(image,cv.COLOR_RGB2BGR)
        
                
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=4, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(217, 228, 212), thickness=2, circle_radius=3),
                                        )
                landmarks = hand.landmark
                print('Landmarks',landmarks)
                for id, landmark in enumerate(landmarks):
                    if id == 8:
                        print('Index',landmark)
                        x = int(landmark.x*frameWidth)
                        y = int(landmark.y*frameHeight)
                        cv.circle(img=frame, center=(x,y), radius=30, color=(255,0,0))
                        mousePositionX = screenWidth/frameWidth*x
                        mousePositionY = screenHeight/frameHeight*y
                        pag.moveTo(mousePositionX, mousePositionY)
                                    
                            
        # Exist out
        if cv.waitKey(20) & 0xFF == ord('q'):
            break
        # Flip on horizontal
        cv.imshow('Camera', image)

cap.release()
cv.destroyAllWindows()
