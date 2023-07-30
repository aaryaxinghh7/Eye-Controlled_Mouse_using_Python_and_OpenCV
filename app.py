import cv2
import mediapipe as mp
import pyautogui

# Initialize the webcam and FaceMesh model
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get the screen size for mouse movement
screen_w, screen_h = pyautogui.size()

while True:
    # Read a frame from the webcam
    _, frame = cam.read()
    
    # Flip the frame horizontally to avoid mirroring effect
    frame = cv2.flip(frame, 1)
    
    # Convert the frame from BGR to RGB (required by Mediapipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame using FaceMesh to get facial landmark points
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        # Extract landmark points for the left eye
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            # Calculate the pixel coordinates of the landmark points
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            # Draw a green circle around the eye landmark points
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            if id == 1:
                # Calculate the screen coordinates for mouse movement
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                # Move the mouse cursor to the corresponding location
                pyautogui.moveTo(screen_x, screen_y)

        # Extract landmark points for the left eyebrow
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            # Calculate the pixel coordinates of the landmark points
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            # Draw a cyan circle around the eyebrow landmark points
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

        # Check if the user's eye is closed based on eyebrow movement
        if (left[0].y - left[1].y) < 0.004:
            # Perform a mouse click action if the eye is closed
            pyautogui.click()
            pyautogui.sleep(1)

    # Show the processed frame with landmark points
    cv2.imshow('Eye Controlled Mouse', frame)

    # Check for the 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
