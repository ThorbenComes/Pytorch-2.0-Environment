import torch

"""Tests gpu availability and gives information about the gpu types"""

print("GPU Test started")
print("Cuda available? " + torch.cuda.is_available().__str__())
num_of_gpus = torch.cuda.device_count()
print("Number of gpus available: " + num_of_gpus.__str__())
for i in range(num_of_gpus):
    print("GPU " + i.__str__() + " " + torch.cuda.get_device_properties(i).name)
status = "finished successfully" if num_of_gpus != 0 else "failed"
print("GPU Test " + status)
print("Number of GPU's found: " + num_of_gpus.__str__())
