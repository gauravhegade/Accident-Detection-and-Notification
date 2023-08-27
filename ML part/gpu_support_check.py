# to check if GPU supports training
import torch

print("Torch version:", torch.__version__)

print("Is CUDA enabled?", torch.cuda.is_available())
