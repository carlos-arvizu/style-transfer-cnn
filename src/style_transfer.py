from model import VGG_model
from image_utils import transform_image
from cost_function import StyleCost
import torch
import torchvision.transforms as transforms
import os
from PIL import Image

def style_transfer(content, style, target, lr, steps, alpha, beta, name = 'output.jpg'):
    '''
    content - image we want with new style
    style - image we want to it's style to be transferred
    target - white noise image (can be the content image). It is the output
    model - model to train (we use VGG19 with some modifications)
    lr - learning rate
    steps - iterations to transfer stlye
    name - name for the output file
    '''
    #initialize model
    model = VGG_model()

    #set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} as device.')

    #use gpu
    model = model.to(device = device).requires_grad_(False)

    #save size of original image
    with Image.open(target) as target_image:
        target_size = target_image.size

    #apply transformations for images
    content_image = transform_image(content).unsqueeze(0).to(device)
    style_image = transform_image(style).unsqueeze(0).to(device)
    target_image = transform_image(target).unsqueeze(0).to(device).clone().requires_grad_(True)

    model.eval()

    #obtain features from content and style images
    content_features = model(content_image)
    style_features = model(style_image)

    #define optimiser
    optimiser = torch.optim.Adam([target_image], lr = lr)

    #loss function
    transfer=StyleCost(alpha, beta)


    for step in range(steps):

        #model.train()
        #model.requires_grad_(True)

        #obtain target_features
        target_features = model(target_image)

        loss_function = transfer(content_features, style_features, target_features)

        optimiser.zero_grad()
        loss_function.backward(retain_graph=True)
        optimiser.step() 

    #denormalise
    denormalise = transforms.Normalize([-2.12, -2.04, -1.80], [4.37, 4.46, 4.44])
    denormalised_tensor = denormalise(target_image.detach().squeeze(0))

    #clip values to range [0, 1] to make it a valid image
    denormalised_tensor = torch.clamp(denormalised_tensor, 0, 1)
    #denormalised_tensor = torch.clamp(target_image.detach().squeeze(0), 0 ,1)

    #convert the tensor to a PIL image
    to_pil = transforms.ToPILImage()
    image = to_pil(denormalised_tensor)

    #resize to original dimensions
    image = image.resize(target_size)

    #save the final image to the 'output' folder
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)  #ensure the folder exists
    final_image_path = os.path.join(output_folder, name)
    image.save(final_image_path)

    print(f"Final stylized image saved at {final_image_path}")