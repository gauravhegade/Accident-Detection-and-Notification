from dotenv import load_dotenv
import os

load_dotenv()
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
from_whatsapp_number = os.getenv("FROM_WHATSAPP_NUMBER")
to_whatsapp_number = os.getenv("TO_WHATSAPP_NUMBER")
smtp_username = os.getenv("SMTP_USERNAME")
smtp_password = os.getenv("SMTP_PASSWORD")
from_email = os.getenv("FROM_EMAIL")
to_email = os.getenv("TO_EMAIL")
