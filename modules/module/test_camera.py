import cv2

url = "http://10.30.79.120:8080/video"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Camera connection failed!")
    exit()

print("IP Camera connected successfully!")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not received")
        break

    cv2.imshow("K-Fitness IP Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()