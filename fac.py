import cv2
import face_recognition

img = cv2.imread("C:\\Users\\91981\\Desktop\\innovation week\\data\\data\\th.jpeg")

rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

img2 = cv2.imread("C:\\Users\\91981\\Desktop\\th (1).jpeg")


rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

result = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result: ", result)

import cv2
import face_recognition
import os
from matplotlib import pyplot as plt
from IPython.display import display, clear_output
import matplotlib.animation as animation

known_face_encodings = []
known_face_names = []

data_folder = r"C:\Users\91981\Desktop\innovation week\data"

for filename in os.listdir(data_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        name = os.path.splitext(filename)[0]  # Extract name from filename without extension
        image_path = os.path.join(data_folder, filename)
        face_encoding = face_recognition.face_encodings(face_recognition.load_image_file(image_path))[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)

video_capture = cv2.VideoCapture(0)        
fig, ax = plt.subplots()
img_plot = ax.imshow(cv2.cvtColor(video_capture.read()[1], cv2.COLOR_BGR2RGB))
plt.axis('off')

def update_frame(frame):
    ret, frame = video_capture.read()

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # Compare the current face encoding with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, use the name of the known face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Update the plot with the current frame
    img_plot.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

ani = animation.FuncAnimation(fig, update_frame, frames=range(100), interval=50, repeat=False)

plt.show()