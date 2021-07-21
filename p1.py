import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
x1=0
x2=0
y1=0
y2=0
distance=0

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        for id , fix in enumerate(hand_landmarks.landmark):
          #here we want to find the dimensions of landmark 4 and 20
          h,w,c=image.shape
          cx ,cy=int(fix.x*w),int(fix.y*h)
          if id==4:
            x1=cx
            y1=cy
            print(id, x1, y1)
          if id==20:
            x2=cx
            y2=cy
            print(id,x2,y2)

        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    distance = round(np.sqrt(((x2 - x1) ** 2) - ((y2 - y1) ** 2)), 2)
    image=cv2.line(image, (x1, y1), (x2, y2), (214, 200, 214), 2)
    image = cv2.putText(image, 'distance='+str(distance), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (214, 200, 214), 2, cv2.LINE_AA)

    cv2.imshow('image', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()