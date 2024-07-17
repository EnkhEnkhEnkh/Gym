import cv2
import time
import datetime

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

if face_cascade.empty():
    raise IOError("Error loading face cascade classifier")
if body_cascade.empty():
    raise IOError("Error loading body cascade classifier")

# cap = cv2.VideoCapture(0)
ip_camera_url = 'rtsp://admin:123456@192.168.1.2:554/stream1'

cap = cv2.VideoCapture(ip_camera_url)

detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

open("logfile.txt", "w").close()
file = open("logfile.txt", "a")

frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    num_faces = len(faces)
    num_bodies = len(bodies)

    if num_faces > 1:
        detection = True
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        file.write(f"{current_time}"+str(4))
    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                file.write(f"{current_time}"+str(4))
        else:
            timer_started = True
            detection_stopped_time = time.time()

    # if detection:
    #     out.write(frame)

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
    for (x, y, width, height) in bodies:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

file.close()
out.release()
cap.release()
cv2.destroyAllWindows()