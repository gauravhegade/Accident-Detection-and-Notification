from detection import Detection as d

# # Static detection
# expected_class_id = d.staticDetection()
# print(expected_class_id)
# print(expected_class_id == 1)

# # Video stream detection
while True:
    expected_class_id = d.videoStreamDetection()
    print(expected_class_id)
    print(expected_class_id == 1)
