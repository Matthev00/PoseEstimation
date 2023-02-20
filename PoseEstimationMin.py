import cv2
import mediapipe as mp

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
cap = cv2.VideoCapture('1.mp4')

while True:
    success, img = cap.read()

    scale_percent = 50  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    imgr = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    imgRGB = cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)

    result = pose.process(imgRGB)
    if result.pose_world_landmarks:
        mpDraw.draw_landmarks(imgr,
                              result.pose_landmarks,
                              mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(result.pose_landmarks.landmark):
            h, w, c = imgr.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(imgr, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

    cv2.imshow('Image', imgr)
    cv2.waitKey(1)
