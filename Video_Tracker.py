import cv2
import numpy as np


def gaussianBlur(path, contourAreaSize):
    cap = cv2.VideoCapture(path)
    ret, frame1 = cap.read()
    assert ret, "failed reading image 1"
    ret, frame2 = cap.read()
    assert ret, "failed reading image 2"
    height, width, channels = frame1.shape

    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, np.ones((5, 5), np.uint8), iterations=5)
        _, contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < contourAreaSize:
                continue
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame1, 'Press Q to quit', (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("GaussianBlur", frame1)
        frame1 = frame2
        ret, frame2 = cap.read()

        if cv2.waitKey(40) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()