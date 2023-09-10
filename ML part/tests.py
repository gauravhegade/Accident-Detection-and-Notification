from integrated import Detection as d

result_list = []

# Static detection
# result_list = d.staticDetection()
# print(class_id)
# print(confidence_score)

# Video stream detection
while True:
    result_list = d.videoStreamDetection()
    print(result_list)
    # print(class_id)
    # print(confidence_score)
