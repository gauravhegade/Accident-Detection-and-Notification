import cv2
from ultralytics import YOLO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from twilio.rest import Client

# Load a model
model = YOLO("ML part/best.pt")


def send_whatsapp_message(to_whatsapp_number, message):
    # Your Twilio Account SID and Auth Token
    account_sid = ""
    auth_token = ""

    # Create a Twilio client
    client = Client(account_sid, auth_token)

    # The number you purchased from Twilio or verified on Twilio
    from_whatsapp_number = "+14155238886"

    try:
        # Send the WhatsApp message
        message = client.messages.create(
            body=message,
            from_="whatsapp:" + from_whatsapp_number,
            to="whatsapp:" + to_whatsapp_number,
        )
        print("WhatsApp message sent successfully:", message.sid)
        # true if whatsapp message sent
        return True

    except Exception as e:
        print("Error sending WhatsApp message:", str(e))
        # false if any error
        return False


def send_email_with_frame(frame_path):
    msg = MIMEMultipart()
    msg["From"] = "accidentdetectionmces@gmail.com"
    msg["To"] = "kir4nchavan@gmail.com"
    msg["Subject"] = "Suspicious Activity Detected"

    # Attach the image
    with open(frame_path, "rb") as img_file:
        image = MIMEImage(img_file.read())
        msg.attach(image)

    # Send email using SMTP
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_username = "accidentdetectionmces@gmail.com"
    smtp_password = "cwouqflkjtrvqynu"
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(smtp_username, smtp_password)
    server.sendmail(
        "accidentdetectionmces@gmail.com", "kir4nchavan@gmail.com", msg.as_string()
    )
    server.quit()

    return True  # Return success flag


# detection
class Detection:
    detection_result = []

    def prediction(path):
        # detection
        results = model.predict(source=path, show=False)

        for result in results:
            for box in result.boxes:
                # extracting data to appropriate variables if an accident is detected means a box is created around that accident area coords
                class_id = box.cls[0].item()
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]
                conf = round(box.conf[0].item(), 2)

                if conf >= 0.5:  # Adjust the confidence threshold as needed
                    print("Object type:", class_id)
                    print("Coordinates:", cords)
                    print("Confidence Score:", conf)
                    print("---")

                    # Send email notification
                    if send_email_with_frame(path):
                        print("Email sent!")

                    location = "location"
                    message = f"Accident detected at location {location}. Severity level: {conf*100}%. Check email for image."
                    send_whatsapp_message("+918431342228", message)

                    # reinitialize the result list to empty to get updated values for the next detection
                    Detection.detection_result = []
                    Detection.detection_result.append(class_id)
                    Detection.detection_result.append(conf)
                    return Detection.detection_result

        # return dummy list if nothing detected
        return [0, 0]

    def staticDetection():
        # path variables
        image_path = "ML part/sample_input/images/image9.jpg"
        # video_path = "ML part/sample_input/videos/video3.mp4"

        Detection.detection_result = Detection.prediction(image_path)
        return Detection.detection_result

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
            temp_image_path = "ML part/temp.jpg"
            cv2.imwrite(temp_image_path, resized_frame)

            # Perform object detection on the image and show the results
            Detection.detection_result = Detection.prediction(temp_image_path)

            # Check for the 'q' key press to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            return Detection.detection_result

        # Release the video capture and close any OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
