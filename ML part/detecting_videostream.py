from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("ML part/best.pt")

# Set the dimensions for captured frames
frame_width = 640
frame_height = 480

# known freely available urls with traffic stream:
# http://204.106.237.68:88/mjpg/1/video.mjpg

# URL of the video stream - preferably use motion jpg urls
stream_url = "http://kamera.mikulov.cz:8888/mjpg/video.mjpg"

# start capturing
cap = cv2.VideoCapture(stream_url)

while cap.isOpened():
    # boolean success flag and the video frame
    # if the video has ended, the success flag is False
    is_frame, frame = cap.read()
    if not is_frame:
        break

    resized_frame = cv2.resize(frame, (frame_width, frame_height))

    # Save the resized frame as an image in a temporary directory
    temp_image_path = "ML part/temp/temp.jpg"
    cv2.imwrite(temp_image_path, resized_frame)

    # Perform object detection on the image and show the results
    model.predict(source=temp_image_path, show=True)

    # Check for the 'q' key press to quit
    if cv2.waitKey(1) == 0xff & ord('q'):
        break

# Release the video capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
