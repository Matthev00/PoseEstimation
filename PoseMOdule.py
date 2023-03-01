import cv2
import mediapipe as mp
import math


class PoseDetector():
    def __init__(
        self,
        mode=False,
        upBody=False,
        smooth=True,
        detectionCon=0.5,
        trackCon=0.5
    ):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            self.mode,
            self.upBody,
            self.smooth,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):

        img = resize(img)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.pose.process(imgRGB)

        if self.result.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.result.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmlist = []
        if self.result.pose_landmarks:
            for id, lm in enumerate(self.result.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return self.lmlist

    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]
        x3, y3 = self.lmlist[p3][1:]

        # Angle
        angle = math.degrees(math.atan2(y1-y2, x1-x2)-math.atan2(y3-y2, x3-x2))
        if angle < 0:
            angle += 360
        angle = angle % 180

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (0, 0, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), ((x2-50), (y2+50)),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle


def resize(img, scale_percent=20):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    imgr = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return imgr


def main():
    cap = cv2.VideoCapture('1.mp4')
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmlist = detector.findPosition(img, False)
        if len(lmlist) != 0:
            print(lmlist)
        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
