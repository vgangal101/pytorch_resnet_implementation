import torch 
from model_arch2 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152 



model = ResNet152()

input_t = torch.ones(1,3,224,224)

out = model(input_t)

