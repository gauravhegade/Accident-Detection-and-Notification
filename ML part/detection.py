import logging
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional, Union

import cv2
import env_vars as ev
from twilio.rest import Client
from ultralytics import YOLO


class NotificationService:
    def __init__(self):
        """
        Initialize notification service with logging and Twilio client
        """

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        self.twilio_client = Client(ev.account_sid, ev.auth_token)

    def send_whatsapp_message(self, message: str) -> bool:
        """
        Send a WhatsApp message using Twilio

        Args:
            message (str): Message to send

        Returns:
            bool: Success status of message sending
        """

        try:
            recipient = ev.to_whatsapp_number

            message = self.twilio_client.messages.create(
                body=message,
                from_="whatsapp:" + ev.from_whatsapp_number,
                to="whatsapp:" + recipient,
            )
            self.logger.info(f"WhatsApp message sent successfully: {message.sid}")
            return True

        except Exception as e:
            self.logger.error(f"Error sending WhatsApp message: {str(e)}")
            return False

    def send_email_with_frame(
        self, frame_path: str, class_name: str, confidence: float, location: str
    ) -> bool:
        """
        Send email with detection frame and details

        Args:
            frame_path (str): Path to the image frame
            class_name (str): Detected class name
            confidence (float): Detection confidence
            location (str): Detection location

        Returns:
            bool: Success status of email sending
        """
        try:
            msg = MIMEMultipart()
            msg["From"] = ev.from_email
            msg["To"] = ev.to_email
            msg["Subject"] = "Accident Detected"

            body = f"""
            <html>
                <body>
                    <p>Accident detected of class <b>{class_name}</b> 
                    with severity level <b>{confidence}%</b> 
                    at location <b>{location}</b>. 
                    Check the attached image.</p>
                </body>
            </html>
            """

            msg.attach(MIMEText(body, "html"))

            with open(frame_path, "rb") as img_file:
                image = MIMEImage(img_file.read())
                msg.attach(image)

            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(ev.smtp_username, ev.smtp_password)
                server.sendmail(ev.from_email, ev.to_email, msg.as_string())

            self.logger.info("Email sent successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error sending email: {str(e)}")
            return False


class Detection:
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        ip_address: Optional[str] = None,
        path: Optional[str] = None,
        should_send_notifications: bool = False,
        model_path: str = "ML part/best.pt",
    ):
        """
        Initialize detection configuration

        Args:
            confidence_threshold (float): Minimum confidence for detection
            ip_address (Optional[str]): IP address for video stream
            path (Optional[str]): Path to image or video file
            should_send_notifications (bool): Flag to enable/disable notifications
            model_path (str): Path to YOLO model
        """
        self.confidence_threshold = confidence_threshold
        self.ip_address = ip_address
        self.path = path
        self.should_send_notifications = should_send_notifications

        self.model = YOLO(model_path)
        self.detection_result: List[Union[int, float]] = []

        self.notification_service = (
            NotificationService() if should_send_notifications else None
        )

    def process_prediction(
        self,
        results,
        image,
    ) -> List[Union[int, float]]:
        """
        Process YOLO detection results

        Args:
            results: YOLO detection results
            frame_path (Optional[str]): Path to the frame image

        Returns:
            List[Union[int, float]]: Detection results
        """
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                coordinates = box.xyxy[0].astype(int)  # Get corner points as int
                confidence = round(box.conf[0].item(), 2)
                image = cv2.rectangle(
                    image, coordinates[:2], coordinates[2:], (0, 255, 0), 2
                )
                cv2.imshow("Image", image)

                if confidence >= self.confidence_threshold:
                    print(f"Accident type: {class_id, class_name}")
                    print(f"Box: {coordinates}")
                    print(f"Confidence Score: {confidence}")

                    location = "*LOCATION*"  # add functionality to get the location of the accident from the IP camera
                    rounded_conf = round(confidence * 100, 2)

                    if self.notification_service:  # feature flag to send notifications
                        message = (
                            f"Accident detected of class {class_name} "
                            f"at location {location}. "
                            f"Severity level: {rounded_conf}%. "
                            "Check email for image."
                        )
                        self.notification_service.send_whatsapp_message(message)

                        if image:
                            self.notification_service.send_email_with_frame(
                                image, class_name, rounded_conf, location
                            )

                    self.detection_result = [class_id, confidence]
                    return self.detection_result

        return [0, 0]

    def static_detection(self) -> List[Union[int, float]]:
        """
        Perform detection on a static image

        Returns:
            List[Union[int, float]]: Detection results
        """
        if not self.path:
            raise ValueError("No image path provided for static detection")

        # read the image and run predictions
        img = cv2.imread(self.path)
        results = self.model.predict(source=img, show=False)
        return self.process_prediction(results, img)

    def video_stream_detection(
        self, frame_width: int = 1280, frame_height: int = 720
    ) -> Optional[List[Union[int, float]]]:
        """
        Perform detection on a video stream

        Args:
            frame_width (int): Width of the frame
            frame_height (int): Height of the frame

        Returns:
            Optional[List[Union[int, float]]]: Detection results or None
        """
        if not self.ip_address:
            raise ValueError("No IP address provided for video stream detection")

        cap = cv2.VideoCapture(self.ip_address)
        try:
            detection_results = []
            while cap.isOpened():
                is_frame, frame = cap.read()
                if not is_frame:
                    break

                resized_frame = cv2.resize(frame, (frame_width, frame_height))

                # temp_image_path = "ML part/temp.jpg"
                # cv2.imwrite(temp_image_path, resized_frame)

                results = self.model.predict(
                    source=resized_frame, show=False, save_frames=True
                )
                detection_result = self.process_prediction(results)
                if detection_result != [0, 0]:
                    detection_results.append(detection_result)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except Exception as e:
            print(f"Error in video stream detection: {e}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            return detection_results


def main():
    static_detector = Detection(
        confidence_threshold=0.5,
        path="ML part/inputs/images/image5.jpg",
        should_send_notifications=False,
    )
    static_result = static_detector.static_detection()
    print("Static Detection Result:", static_result)

    # # Video stream detection
    # stream_detector = Detection(
    #     confidence_threshold=0.5,
    #     ip_address="http://195.196.36.242/mjpg/video.mjpg",
    #     should_send_notifications=False,
    # )
    # stream_result = stream_detector.video_stream_detection()
    # print("Stream Detection Result:", stream_result)


if __name__ == "__main__":
    main()
