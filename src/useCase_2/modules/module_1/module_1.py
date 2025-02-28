#####################################
# First Container 
# module_1.py
# TCP Socket SERVER Deployment For generate inference with a GAN Train on MNIST with pytorch

#####################################
# Libraries
## Basics
import random
import numpy as np 
import types
import os
from PIL import Image
import logging
## Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
## Watermark
import sys
sys.path.append('./UCHI.py')
from UCHI import Uchi_tools
#####################################
from otter_net_utils import OtterUtils
tcp_tools = OtterUtils()
print("OTTER Utils class imported successfully!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####################################
# Generator Network for GAN
class Generator(nn.Module):
    """
    Defines the Generator model for the GAN network.
    The generator takes a latent noise vector as input and generates synthetic data 
    through a series of fully connected layers with Leaky ReLU activation.

    Args:
        g_input_dim (int): Dimensionality of the input noise vector.
        g_output_dim (int): Dimensionality of the generated output.
    """

    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)
    
    def forward(self, x): 
        """
        Forward pass through the generator network.
        Args:
            x (torch.Tensor): Input noise vector.
        Returns:
            torch.Tensor: Generated output data.
        """
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


# Inference class for GAN model
class InferenceGAN:
    """
    Handles the loading, watermark extraction, and inference of a pre-trained GAN model.

    Attributes:
        PATH_TO_MODEL (str): Path to the trained generator model.
        PATH_TO_WATERMARK_DICT (str): Path to the stored watermark dictionary.
        PATH_TO_GENERATE_SAMPLE (str): Path where generated samples will be saved.
        BS (int): Batch size for inference.
        Z_DIM (int): Latent space dimensionality.
        DATA_DIM (int): Dimension of the generated data.
        T (float): Threshold parameter (if used in watermark detection).
        WEIGHT_NAME (str): Name of the stored model weights.
        G (Generator): Generator model instance.
        CHANEL (int): Number of image channels (e.g., 1 for grayscale, 3 for RGB).
        W (int): Width of generated images.
        H (int): Height of generated images.
    """

    PATH_TO_MODEL = None
    PATH_TO_WATERMARK_DICT = None
    PATH_TO_GENERATE_SAMPLE = None
    BS = None
    Z_DIM = None
    DATA_DIM = None
    T = None
    WEIGHT_NAME = None
    G = None
    CHANEL = None
    W = None
    H = None

    def build_generator(self):
        """
        Initializes and loads the pre-trained generator model.
        The generator is built using the stored latent space and data dimension sizes. 
        The pre-trained weights are loaded and the model is set to evaluation mode.
        Returns:
            None
        """
        self.G = Generator(self.Z_DIM, self.DATA_DIM).to(device)
        self.G.load_state_dict(torch.load(self.PATH_TO_MODEL))
        self.G.eval()

    def extraction(self):
        """
        Extracts a watermark from the weight using Uchida's watermarking method.
        Loads the watermark dictionary and applies a watermark detection tool 
        Returns:
            None
        """
        tools = Uchi_tools()
        logging.info(f"Uchida class imported")
        
        watermarking_dict = np.load(self.PATH_TO_WATERMARK_DICT, allow_pickle=True).item()
        extraction_score = tools.detection(self.G, watermarking_dict)[1]

        if extraction_score == 0.0:
            logging.info(f"Watermark extracted with score: {extraction_score}")
        else:
            logging.info(f"Watermark undetected with score: {extraction_score}")

    def inference(self):
        """
        Generates synthetic data using the trained generator.
        A batch of random latent vectors is passed through the generator to 
        create synthetic images. The generated images are then saved to the specified path.
        Returns:
            torch.Tensor: A batch of generated images.
        """
        with torch.no_grad():
            test_z = Variable(torch.randn(self.BS, self.Z_DIM).to(device))
            generated = self.G(test_z)
            generated_images = generated.view(self.BS, self.CHANEL, self.H, self.W)
            save_image(generated_images, self.PATH_TO_GENERATE_SAMPLE)
        return generated_images

    def run(self):
        """
        Executes the complete inference pipeline.
        This includes:
        1. Loading the pre-trained generator model.
        2. Extracting the watermark from generated images.
        3. Generating and saving synthetic images.
        Returns:
            torch.Tensor: The generated batch of images.
        """
        self.build_generator()
        self.extraction()
        inference = self.inference()
        return inference
#####################################

if __name__ == "__main__":
    # Log Initialization
    log_file_path = "/app/logs/container.log"
    tcp_tools.build_log_file(log_file_path)

    ##################################### NETWORK PART #####################################
    # Network Parameters
    HOST = '0.0.0.0'  # Listen on all network interfaces (Server configuration)
    PORT = 5000       # Port for listening
    conn = tcp_tools.init_server_TCP_connection(HOST, PORT)

    ##########################################
    # Placeholder for future AI prediction 
    ## Seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Build the model
    GAN_MODEL = InferenceGAN()
    GAN_MODEL.PATH_TO_MODEL = "/app/inputs/Gmodel_GANW_500_weights.pth"
    GAN_MODEL.PATH_TO_WATERMARK_DICT = "/app/inputs/watermark_dict_GANW.npy"
    GAN_MODEL.PATH_TO_GENERATE_SAMPLE ='./sample_final.png'
    GAN_MODEL.BS = 1
    GAN_MODEL.Z_DIM = 100
    GAN_MODEL.CHANEL = 1
    GAN_MODEL.W = 28
    GAN_MODEL.H = 28
    GAN_MODEL.DATA_DIM = GAN_MODEL.CHANEL*GAN_MODEL.W*GAN_MODEL.H
    GAN_MODEL.T = 64
    GAN_MODEL.WEIGHT_NAME = 'fc3'

    prediction = GAN_MODEL.run()

    ##########################################

    ##################################### NETWORK PART #####################################
    tcp_tools.send_variable_to_container_TCP(conn, prediction.cpu().numpy())











