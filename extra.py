import torch
import onnx

# Define the PyTorch model class
class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=5)  # 3 channels for RGB images
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(20 * 53 * 53, 50)
        self.fc2 = torch.nn.Linear(50, 11)  # Changed to 11 classes

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 20 * 53 * 53)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the model state
model_state = torch.load('cnn_model.pth', map_location=torch.device('cpu'))

# Create a PyTorch model instance and load the state
model = CNNModel()
model.load_state_dict(model_state)

# Convert the PyTorch model to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, 'model.onnx', opset_version=11)

