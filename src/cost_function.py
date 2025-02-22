import torch
import torch.nn.functional as F
import torch.nn as nn

class StyleCost(nn.Module):
    '''Class that creates the loss function that combines style and content features'''
    #a and b are the weight parameters needed for the loss function
    def __init__(self,a,b):
        super().__init__()
        self.alpha = a
        self.beta = b

    #content contribution
    def content_contribution(self, content_features, target):
        #returns mean squared error from content features and target image
        return F.mse_loss(content_features, target)
        
    
    #gram matrix
    def gram_matrix(self, x):
        b, c, h, w = x.size()  #batch, channels, height, width
        features = x.view(b, c, h * w)  #reshape to (batch, channels, height * width)
        
        #compute the gram matrix
        gram = torch.bmm(features, features.transpose(1, 2))  #batch matrix-matrix product
        
        #normalize the gram matrix
        gram /= (c * h * w)

        #return normalized gram matrix
        return gram
    
    #style contribution
    def style_contribution(self, style_features, target):
        gram_style = self.gram_matrix(style_features) #gram matrix for the style features
        gram_target = self.gram_matrix(target) #gram matrix for the target image

        #return mean squared error from both gram matrixes
        return F.mse_loss(gram_style, gram_target)
    
    #call function
    def forward(self, content_features, style_features, target):
        content_loss = self.content_contribution(content_features['conv4_1'], target['conv4_1']) #21 is the only layer that contributes to the content loss
        
        #initialize style loss
        style_loss = 0
        
        style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'] #list of layers that contribute to the style loss

        #loop through style layers
        for layer in style_layers:
            #apply style contribution function and divide between the total amount of layers to assign equal weight
            style_loss += self.style_contribution(style_features[layer], target[layer])/5

        #return the complete cost function that includes content loss and style loss
        return self.alpha * content_loss + self.beta * style_loss