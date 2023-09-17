import cv2
import dlib
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load the face recognition model (you need to have dlib's pre-trained model)
face_recognizer = dlib.face_recognition_model_v1(
    "shape_predictor_68_face_landmarks.dat"
)

# Load the face recognition dataset (your friends' faces with their names)
# Replace this with your actual dataset
known_faces = {
    "Friend1": dlib.load_rgb_image("about1.jpg"),
    "Friend2": dlib.load_rgb_image("anup.jpg"),
    # Add more friends with their images and names
}

# Initialize the face detector
detector = dlib.get_frontal_face_detector()

# Open the video capture from the webcam (camera index 0)
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to RGB format (dlib uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    faces = detector(rgb_frame)

    for face in faces:
        # Recognize the face
        face_descriptor = face_recognizer.compute_face_descriptor(rgb_frame, face)

        # Compare the face descriptor with known faces
        for name, known_descriptor in known_faces.items():
            # You can use any distance metric here (e.g., Euclidean distance)
            distance = dlib.distance(face_descriptor, known_descriptor)

            if distance < 0.6:  # Adjust the threshold as needed
                # Recognized a known person, speak their name
                print(f"Recognized: {name}")
                engine.say(f"Hello, {name}!")
                engine.runAndWait()

    # Display the frame with detected faces
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
