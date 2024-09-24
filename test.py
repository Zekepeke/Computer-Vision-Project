# import cv2
# import mediapipe as mp
# import pyautogui




# mp_hands = mp.solutions.hands.Hands(
#         max_num_hands=1,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.8)
# drawingTools = mp.solutions.drawing_utils


# video = cv2.VideoCapture(0)

# screenWidth, screenHeight = pyautogui.size()


# while True:
#    _, frame = video.read()
#    frameHeight, frameWidth, _ = frame.shape
#    frame = cv2.flip(frame, 1)
#    rgbConvertedFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#    output = mp_hands.process(rgbConvertedFrame)
#    hands = output.multi_hand_landmarks
#    if cv2.waitKey(20) & 0xFF==ord('d'): 
#         break


#    if hands:
#        for hand in hands:
#            drawingTools.draw_landmarks(frame, hand,)
#            landmarks = hand.landmark
#            for id, landmark in enumerate(landmarks):
#                if id == 8:
#                    x = int(landmark.x*frameWidth)
#                    y = int(landmark.y*frameHeight)
#                    cv2.circle(img=frame, center=(x,y), radius=30, color='#136207')
#                    mousePositionX = screenWidth/frameWidth*x
#                    mousePositionY = screenHeight/frameHeight*y
#                    pyautogui.moveTo(mousePositionX, mousePositionY)




#    cv2.imshow('Mouse', frame)
#cv2.waitKey(1)




# PROBLEM IS THAT HANDS WAS NEVER CLOSED SO THIS PROGRAM USES UP MEMORY