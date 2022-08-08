import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

model = torchvision.models.mobilenet_v2(pretrained=True)
model.eval()

# only one input
example = torch.rand(1, 3, 224, 224)

# multiple inputs
example_1 = torch.rand(1, 3, 224, 224)
example_2 = torch.rand(1, 10, 2, 2)
example_3 = torch.rand(1, 10, 2)
example = (example_1, example_2, example_3)

# convert
traced_script_module = torch.jit.trace(model, example)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module_optimized._save_for_lite_interpreter("android.ptl")


