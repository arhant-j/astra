import cv2
import face_recognition
import os

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load known faces from the 'data' folder
known_faces = []
known_face_names = []
data_folder_path = r"C:\Users\harsh\Desktop\innovation week\data\dataaa"
for filename in os.listdir(r"C:\Users\harsh\Desktop\innovation week\data\dataaa"):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        
        known_image = face_recognition.load_image_file(os.path.join(data_folder_path, filename))
        known_face_encodings = face_recognition.face_encodings(known_image)

        if len(known_face_encodings) > 0:
            known_face_encoding = known_face_encodings[0]
            known_faces.append(known_face_encoding)
            known_face_names.append(os.path.splitext(filename)[0])
           
        else:
            print(f"No face found in {filename}")
# Load an image or use a video stream
# If using an image:
# img = cv2.imread("path/to/image.jpg")
# If using a video stream:
print("Starting...")
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Convert the frame to RGB for face recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Display the name of the image file if known face is detected, otherwise display "unknown"
    for (x, y, w, h), face_encoding in zip(faces, face_encodings):
        # Check if the detected face matches any known faces
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        print(matches)
        name = "unknown"
                  

        if True in matches:
            # Use the index of the first match to get the corresponding name
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection and Recognition', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
