import torch 
import model_arch2
import torchvision_resnet


#model = model_arch2.ResNet50()



model = torchvision_resnet.ResNet50(1000)

#input_t = torch.ones(1,3,224,224)

#out = model(input_t)

print(model)


