from ultralytics import YOLO

data_path = "./data.yaml"

# training the model
model = YOLO("yolov8n.pt")
model.to("cuda")
model.train(data=data_path, epochs=50, amp=False)

# automatically evaluate the data you trained.
model.val()
