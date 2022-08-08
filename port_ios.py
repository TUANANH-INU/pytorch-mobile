import torch
import torchvision

model = torchvision.models.resnet18()
# only one input
example = torch.rand(1, 3, 224, 224)

# multiple inputs
example_1 = torch.rand(1, 3, 224, 224)
example_2 = torch.rand(1, 10, 2, 2)
example_3 = torch.rand(1, 10, 2)
example = (example_1, example_2, example_3)

# convert
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("ios.pt")
