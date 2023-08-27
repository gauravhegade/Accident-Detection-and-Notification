from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("ML part/best.pt")


# detection
class Detection:
    # # constructor
    # def __init__(self):
    #     pass

    def prediction(path):
        save_path = "ML part/results/"

        # detection
        results = model.predict(source=path, show=True, save = True, project=save_path)

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
                    print("Probability:", conf)
                    print("---")

                return class_id

        return -1

    def staticDetection():
        # path variables
        image_path = "ML part/sample_input/images/image1.jpg"
        video_path = "ML part/sample_input/videos/video3.mp4"

        class_id = Detection.prediction(image_path)
        return class_id

    def videoStreamDetection():
        # Set the dimensions for captured frames
        frame_width = 640
        frame_height = 480

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

            # Save the resized frame as an image in temporary directory
            temp_image_path = "ML part/temp/temp.jpg"
            cv2.imwrite(temp_image_path, resized_frame)

            # Perform object detection on the image and show the results
            class_id = Detection.prediction(temp_image_path)

            # Check for the 'q' key press to quit
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q") or key == 27:
                break

            return class_id

        # Release the video capture and close any OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
