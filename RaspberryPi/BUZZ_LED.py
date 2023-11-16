# Libraries
import RPi.GPIO as GPIO
from time import sleep
import sys

# importing folder
folder_path = "ML part"
sys.path.append(folder_path)
from integrated import Detection as d

# Disable warnings (optional)
GPIO.setwarnings(False)

# Select GPIO mode
GPIO.setmode(GPIO.BCM)

# # These are GPIO pin numbers, not the hardware pin numbers
# Set buzzer - GPIO.23 as output
buzzer = 23
GPIO.setup(buzzer, GPIO.OUT)

# set red, green and blue pins.
redPin = 12
greenPin = 19
bluePin = 13

# set pins as outputs
GPIO.setup(redPin, GPIO.OUT)
GPIO.setup(greenPin, GPIO.OUT)
GPIO.setup(bluePin, GPIO.OUT)


# Define LED functions, LED is common anode (assuming)
def turnOff():
    print("Turned off all LED pins!")
    # GPIO.output(buzzer, GPIO.HIGH)
    GPIO.output(redPin, GPIO.HIGH)
    GPIO.output(greenPin, GPIO.HIGH)
    GPIO.output(bluePin, GPIO.HIGH)


def red():
    print("RED LED!")
    GPIO.output(redPin, GPIO.LOW)
    GPIO.output(greenPin, GPIO.HIGH)
    GPIO.output(bluePin, GPIO.HIGH)


def green():
    print("GREEN LED!")
    GPIO.output(redPin, GPIO.HIGH)
    GPIO.output(greenPin, GPIO.LOW)
    GPIO.output(bluePin, GPIO.HIGH)


def yellow():
    print("YELLOW LED!")
    GPIO.output(redPin, GPIO.LOW)
    GPIO.output(greenPin, GPIO.LOW)
    GPIO.output(bluePin, GPIO.HIGH)


def accident_detected():
    # If accident is severe, fast buzzing and red flashing lights
    if class_id == 1:
        while True:
            GPIO.output(buzzer, GPIO.HIGH)
            print("Beep")
            sleep(0.5)

            GPIO.output(buzzer, GPIO.LOW)
            print("No Beep")
            sleep(0.5)

            turnOff()
            sleep(0.5)
            red()
            sleep(0.5)

    # if accident is moderate, relatively slower buzzing and yellow flashing lights
    elif class_id == 0:
        while True:
            GPIO.output(buzzer, GPIO.HIGH)
            print("Beep")
            sleep(2)  # Delay in seconds

            GPIO.output(buzzer, GPIO.LOW)
            print("No Beep")
            sleep(2)

            turnOff()
            sleep(2)
            yellow()
            sleep(2)


while True:
    # Run forever loop with a blinking green LED to indicate the system is running
    turnOff()
    sleep(1)  # 1second
    green()
    sleep(1)

    # mapping class ID and confidence score to returned list of values
    [class_id, confidence_score] = d.staticDetection()

    print(f"Class ID: {class_id}")
    print(f"Confidence Score: {confidence_score}")

    if confidence_score >= 0.7:
        accident_detected()
