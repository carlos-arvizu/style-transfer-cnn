from PIL import Image
import torchvision.transforms as transforms

def transform_image(image):
    #transformations needed
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  #change the size of the image
        transforms.ToTensor(),  #convert to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #normalize image
    ])
    #open image
    open_image = Image.open(image)
    #transform image
    out_image = transform(open_image)
    return out_image