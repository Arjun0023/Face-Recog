import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# first we will capture video using cv2.videoCapture

video_capture = cv2.VideoCapture(0)  # Depends on which webcam are we using

# Now we will load known faces

arjuns_image = face_recognition.load_image_file("faces/arjuns.jpg")
arjuns_encoding = face_recognition.face_encodings(arjuns_image)[0]  # 0 because we want to recog 1st face

rohan_image = face_recognition.load_image_file("faces/rohan.jpg")
rohan_encoding = face_recognition.face_encodings(rohan_image)[0]

# now we will store the names
known_face_encoding = [arjuns_encoding, rohan_encoding]
known_face_names = ["Arjun", "rohan"]

# list of expected student
students = known_face_names.copy()

face_locations = []
face_encoding = []

# we will get the current date and time to regester attendence 

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# now we will make a csv writer

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

# now we will create an infinite while loop
while True:
    _ , frame = video_capture.read()
    small_frame = cv2.resize(frame, (0,0), fx=0.50, fy=0.50)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

# recognize faces
face_locations = face_recognition.face_locations(rgb_small_frame)
face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

for face_encoding in face_encodings:
    matches = face_recognition.compare_faces(known_face_encoding,face_encoding)  # it matches known face and face encoding
    face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)  # It rells similarity of known face and face encoding
    best_match_index = np.argmin(face_distance)

    # this loop show if the face has matched or not
    if (matches[best_match_index]):
        name = known_face_names[best_match_index]  # if the face is matched , we will get the name

        # adding text for attendence
    if name in known_face_names:
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 100)
        fontScale = 1.5
        fontColor - (255, 0, 0)
        thickness = 3
        lineType = 2
        cv2.putText(frame, name + "Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

    if name in students:
        students.remove(name)
        current_time = now.strftime("%H-%M-%S")
        lnwriter.writerow([name, current_time])
    cv2.imshow("Attendence", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break 

video_capture.release()
cv2.destroyAllWindows()
f.close()
