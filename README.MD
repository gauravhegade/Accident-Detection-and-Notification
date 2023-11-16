## Accident Detection and Notification using YOLOv8

Accidents on the road can have serious consequences, and quick response times are crucial for minimizing harm. This project aims to enhance road safety by automatically detecting accidents through object detection in real-time video streams (also in static images and videos) and sending a notification to concerned authorities. This repository contains the implementation of an accident detection system utilizing the YOLOv8 object detection model.

### Step 1: Setup

- Create a virtual environment and activate it: `python -m venv venv && source venv/bin/activate`
- Install Prerequisites: `pip install -r requirements.txt`
- Clone this repository: `git clone https://github.com/gauravhegade/MCES-Accident-Detection.git Accident-Detection`
- `cd` into the directory: `cd Accident-Detection`, then proceed to Step 2

### Step 2: Detection

#### Step 2.1: Static Image/Video Detection

- To perform static image/video detection, use the file named `detecting_static.py` available in `ML Part` folder
- Run the script: `python detecting_static.py`
- The script will load the model, perform object detection on the specified image (sample input images available in `inputs` folder), and save the results in the `results` directory.

#### Step 2.2: Videostream Detection

- To perform videostream detection, use the file named `detecting_videostream.py` available in `ML Part` folder
- Run the script: `python detecting_videostream.py`
- The script will load the model, perform object detection on the input videostream (specified by the `stream_url`) by converting it to individual frames.
