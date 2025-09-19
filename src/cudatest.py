import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():    print("Device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0))
print("CUDA version in PyTorch:", torch.version.cuda)
