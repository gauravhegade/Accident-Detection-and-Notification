from detection import Detection as d

# uncomment whichever one you want to test, results will be displayed on the terminal

# Static detection
[class_id, confidence_score] = d.static_detection()
print("Test results for static detection:")
print("Class id", class_id)
print("Confidence Score", confidence_score * 100)

# # Video stream detection
# while True:
#     [class_id, confidence_score] = d.video_stream_detection()
#     print("Test results for video stream detection:")
#     print("Class id", class_id)
#     print("Confidence Score", confidence_score * 100)
