
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import torchvision.models as models
import yaml
with open("class.yaml", "r") as file:
    CLASS_MAP = yaml.safe_load(file)

# Define a function to perform inference on an input image
def infer(image_path, model, device):
    # Load the image
    img = Image.open(image_path)
    
    # Apply the transformations
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to the device
    
    # Forward pass to get predictions
    with torch.no_grad():  # No need to track gradients during inference
        outputs = model(img_tensor)
        
    # Get the predicted class
    _, predicted_class = torch.max(outputs, 1)
    
    # Convert the predicted class index back to class label if necessary
    class_idx = predicted_class.item()
    final_pred = CLASS_MAP[int(class_idx)]
    print(f"Predicted class : {final_pred}")
    # If you have class names, you can map it to the corresponding label like this:
    # class_names = train_dataset.classes  # The class names from your ImageFolder dataset
    # predicted_label = class_names[class_idx]
    # print(f"Predicted class label: {predicted_label}")

    return final_pred  # Return the predicted class index

if __name__ == "__main__":
    import argparse
    # 1. Create the parser
    parser = argparse.ArgumentParser(
        description="Example script that takes one variable from the terminal"
    )

    # 2. Add a positional argument called "name"
    parser.add_argument(
        "--image_path",            # for an optional arg you’d use "--name" instead
        help="The name you want to pass in",
        type=str
    )

    # 3. Parse the command-line arguments
    args = parser.parse_args()
    image_path = args.image_path

    # Load the pre-trained ResNet-18 model
    model = models.resnet18(pretrained=True)
    # Load the state_dict
    num_classes = 4
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(torch.load('checkpoints/model.pth'))

    # Set the model to evaluation mode (important for inference)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print (device)
    model.to(device)

    # Load the trained model (assuming it's saved and restored, otherwise use the code you already have)
    model.eval()  # Set the model to evaluation mode

    # Define the same transformations used during training
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # Example usage
    final_pred = infer(image_path, model, device)
