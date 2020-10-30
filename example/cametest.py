import cv2

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

while True:
    _, frame = cap.read()
    _, frame2 = cap2.read()
    cv2.imshow("image1", frame)
    cv2.imshow("image2", frame2)
    cv2.waitKey(3)

