from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("ML part/best.pt")

detection_result = []


# detection
class Detection:
    def prediction(path):
        save_path = "ML part/results/"

        # detection
        results = model.predict(source=path, show=True)

        if len(results):
            result = results[0]

            if len(result.boxes):
                box = result.boxes[0]

                # extracting data to appropriate variables
                for box in result.boxes:
                    class_id = box.cls[0].item()
                    cords = box.xyxy[0].tolist()
                    cords = [round(x) for x in cords]
                    conf = round(box.conf[0].item(), 2)

                    print("Object type:", class_id)
                    print("Coordinates:", cords)
                    print("Confidence Score:", conf)
                    print("---")

                if conf:
                    print(f"If conf: {conf}")
                    detection_result.append(class_id)
                    detection_result.append(conf)
                    return detection_result

        # if no accident is detected, return a zero list
        return [0, 0]

    def staticDetection():
        # path variables
        image_path = "ML part/inputs/images/image5.jpg"
        video_path = "ML part/inputs/videos/video3.mp4"

        detection_result = Detection.prediction(image_path)
        return detection_result

    def videoStreamDetection():
        # Set the dimensions for captured frames
        frame_width = 1280
        frame_height = 720

        # known freely available urls with traffic stream:
        # http://204.106.237.68:88/mjpg/1/video.mjpg
        # URL of the video stream
        stream_url = "http://195.196.36.242/mjpg/video.mjpg"

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
            detection_result = Detection.prediction(temp_image_path)

            # Check for the 'q' key press to quit
            if cv2.waitKey(1) == 0xff & ord('q'):
                exit

            return detection_result

        # Release the video capture and close any OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
