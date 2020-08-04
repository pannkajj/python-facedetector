import cv2

# Import the image
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# CAPTURE video from webcam
webcam = cv2.VideoCapture(0)

while True:
  # Make frames from webcame video
  successful_frame_read, frame = webcam.read()

  # Convert Image to greyscale
  greyscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Detect Faces
  face_coordinates = trained_face_data.detectMultiScale(greyscaled_img)

  # Draw rectangles around the faces
  for (x, y, w, h) in face_coordinates:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)

  # Show the image
  cv2.imshow('Pankaj Choudhary Face Detector', frame)
  key = cv2.waitKey(1)

  # Stop if Q key is pressed
  if key==81 or key==113:
    break

print("Code Completed")
