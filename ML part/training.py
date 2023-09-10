# from ultralytics import YOLO

# data_path = "./data.yaml"

# # training the model
# model = YOLO("yolov8n.pt")
# model.to("cuda")
# model.train(data=data_path, epochs=100, amp=False)

# # automatically evaluate the data you trained.
# model.val()


# To check for CUDA support
import torch
print(f"Version: {torch.__version__}")
print(f"Cuda available? {torch.cuda.is_available()}")
