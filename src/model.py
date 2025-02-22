import torchvision.models as models
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor

class VGG_model(nn.Module):
    '''Class for the model. It is based on VGG19, but with Average Pool instead of Max Pool layers.'''
    def __init__(self):
        #call parent class constructor
        super().__init__()

        #use model vgg19, avoiding the linear layers
        vgg19 = models.vgg19(weights = 'DEFAULT').features

        #list for the layers of the new model
        new_layers = []
        #loop for every layer in the model
        for layer in vgg19:
            #if there is a MaxPool layer, instead use an AvgPool layer, the rest of layers remain the same and append to the list
            if isinstance(layer, nn.MaxPool2d):
                new_layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding= 0))
            else:
                new_layers.append(layer)
        #Sequential to create the new model
        self.model_vgg19 = nn.Sequential(*new_layers)

        #avoid computing the gradient
        for param in self.model_vgg19.parameters():
            param.requires_grad = False

        #rename layers for the cost function
        self.model = create_feature_extractor(self.model_vgg19, {
            '1': 'conv1_1',
            '6': 'conv2_1',
            '11': 'conv3_1',
            '20': 'conv4_1',
            '29': 'conv5_1',
        })

    #return model
    def forward(self, x):
        return self.model(x)     