import cv2
from ultralytics import YOLO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from twilio.rest import Client
import env_vars as ev

# Load the trained YOLO model
model = YOLO("ML part/best.pt")


def send_whatsapp_message(to_whatsapp_number, message):
    # Function to send WhatsApp message using Twilio API
    account_sid = ev.account_sid
    auth_token = ev.auth_token
    from_whatsapp_number = ev.from_whatsapp_number

    client = Client(account_sid, auth_token)

    try:
        # Send WhatsApp message
        message = client.messages.create(
            body=message,
            from_="whatsapp:" + from_whatsapp_number,
            to="whatsapp:" + to_whatsapp_number,
        )
        print("WhatsApp message sent successfully:", message.sid)
        return True

    except Exception as e:
        print("Error sending WhatsApp message:", str(e))
        return False


def send_email_with_frame(frame_path, class_name, rounded_conf, location):
    try:
        # Function to send email with an attached image
        msg = MIMEMultipart()
        msg["From"] = ev.from_email
        msg["To"] = ev.to_email
        msg["Subject"] = "Accident Detected"

        # Create HTML content with formatting
        body = f"""
        <html>
            <body>
                <p>Accident detected of class <b>{class_name}</b> with severity level <b>{rounded_conf}%</b> at location <b>{location}</b>. Check the attached image.</p>
            </body>
        </html>
        """

        msg.attach(MIMEText(body, 'html'))

        with open(frame_path, "rb") as img_file:
            image = MIMEImage(img_file.read())
            msg.attach(image)

        # Send email using SMTP
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        smtp_username = ev.smtp_username
        smtp_password = ev.smtp_password
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(ev.from_email, ev.to_email, msg.as_string())
        server.quit()
        
        return True  # Return success flag

    except Exception as e:
        print("Error sending email:", str(e))
        return False


class Detection:
    detection_result = []

    @staticmethod
    def prediction(path):
        # Function to perform object detection using YOLO model
        results = model.predict(source=path, show=False)

        for result in results:
            for box in result.boxes:
                class_id = box.cls[0].item()
                cords = [round(x) for x in box.xyxy[0].tolist()]
                conf = round(box.conf[0].item(), 2)

                if conf >= 0.5:  # Adjust the confidence threshold as needed
                    print("Object type:", class_id)
                    print("Coordinates:", cords)
                    print("Confidence Score:", conf)
                    print("---")

                    location = "*LOCATION*"
                    class_name = "*MODERATE*" if class_id == 0 else "*SEVERE*"
                    rounded_conf = str(round(conf * 100, 2))

                    # Send email notification
                    if send_email_with_frame(path, class_name, rounded_conf, location):
                        print("Email sent!")

                    message = f"Accident detected of class { class_name } at location { location }. Severity level: { rounded_conf }%. Check email for image."
                    send_whatsapp_message(ev.to_whatsapp_number, message)

                    # Reinitialize the result list to empty to get updated values for the next detection
                    Detection.detection_result = []
                    Detection.detection_result.append(class_id)
                    Detection.detection_result.append(conf)
                    return Detection.detection_result

        # Return dummy list if nothing detected
        return [0, 0]

    @staticmethod
    def static_detection():
        # Function to perform static image detection
        image_path = "ML part/inputs/images/image5.jpg"
        Detection.detection_result = Detection.prediction(image_path)
        return Detection.detection_result

    @staticmethod
    def video_stream_detection():
        # Function to perform video stream detection
        frame_width, frame_height = 1280, 720
        stream_url = "http://195.196.36.242/mjpg/video.mjpg"

        cap = cv2.VideoCapture(stream_url)

        while cap.isOpened():
            # Boolean success flag and the video frame
            is_frame, frame = cap.read()
            if not is_frame:
                break

            resized_frame = cv2.resize(frame, (frame_width, frame_height))
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
