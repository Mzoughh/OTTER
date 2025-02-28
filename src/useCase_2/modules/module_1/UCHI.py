# Implementation by Carl de Sousa Trias and modified for GAN adaptation by Mateo Zoughebi
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Uchi_tools():
    """
    A utility class for performing watermark insertion, extraction, and loss calculation 
    for GAN, using Uchida's watermarking method.
    """

    def __init__(self) -> None:
        """
        Initializes the Uchi_tools class. No specific parameters are required.
        """
        super(Uchi_tools, self).__init__()

    def insertion(self, net, trainloader, optimizer, criterion, watermarking_dict):
        """
        Inserts a watermark into the model during training, and calculates the loss (global, task, and watermark losses).
        
        Args:
            net (nn.Module): The neural network model to insert the watermark into.
            trainloader (DataLoader): The training data loader.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            criterion (nn.Module): The loss function used for training.
            watermarking_dict (dict): A dictionary containing watermark elements (e.g., lambda, watermark, etc.).

        Returns:
            tuple: A tuple containing the total loss, task loss, and watermark loss for the current batch.
        """
        running_loss = 0
        running_loss_nn = 0
        running_loss_watermark = 0
        for i, data in enumerate(trainloader, 0):
            # split data into the image and its label
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            if inputs.size()[1] == 1:
                inputs.squeeze_(1)
                inputs = torch.stack([inputs, inputs, inputs], 1)
            # initialise the optimiser
            optimizer.zero_grad()

            # forward
            outputs = net(inputs)
            # backward
            loss_nn = criterion(outputs, labels)
            # watermark
            loss_watermark = self.loss(net, watermarking_dict['weight_name'], watermarking_dict['X'], watermarking_dict['watermark'])

            loss = loss_nn + watermarking_dict['lambd'] * loss_watermark  # Uchida

            loss.backward()
            # update the optimizer
            optimizer.step()

            # loss
            running_loss += loss.item()
            running_loss_nn += loss_nn.item()
            running_loss_watermark += loss_watermark.item()
        return running_loss, running_loss_nn, running_loss_watermark

    def detection(self, net, watermarking_dict):
        """
        Detects the watermark from the model's weights and compares it to the original watermark.

        Args:
            net (nn.Module): The neural network model from which to extract the watermark.
            watermarking_dict (dict): A dictionary containing the watermark and related information.

        Returns:
            tuple: A tuple containing the extracted watermark and the hamming distance between the extracted and original watermark (in percentage).
        """
        watermark = watermarking_dict['watermark'].to(device)
        X = watermarking_dict['X'].to(device)
        weight_name = watermarking_dict["weight_name"]
        extraction = self.extraction(net, weight_name, X)
        extraction_r = torch.round(extraction) # <.5 = 0 and >.5 = 1
        res = self.hamming(watermark, extraction_r)/len(watermark)
        return extraction, float(res)*100

    def init(self, net, watermarking_dict, save=None):
        """
        Initializes the watermarking procedure by generating a secret key matrix (X), 
        and optionally saves the watermarking dictionary to a file.

        Args:
            net (nn.Module): The neural network model to embed the watermark into.
            watermarking_dict (dict): A dictionary containing watermarking parameters.
            save (str, optional): Path to save the watermarking dictionary. Defaults to None.

        Returns:
            dict: The updated watermarking dictionary with the new secret key matrix (X).
        """
        M = self.size_of_M(net, watermarking_dict['weight_name'])
        T = len(watermarking_dict['watermark'])
        X = torch.randn((T, M), device=device)
        watermarking_dict['X'] = X
        if save != None:
            np.save(save, watermarking_dict)
        return watermarking_dict

    def projection(self, X, w):
        """
        Projects the secret key matrix (X) onto the flattened weight vector (w) using a sigmoid activation function.

        Args:
            X (torch.Tensor): The secret key matrix.
            w (torch.Tensor): The flattened weight vector.

        Returns:
            torch.Tensor: The result of the projection after applying the sigmoid function.
        """
        sigmoid_func = nn.Sigmoid()
        res = torch.matmul(X, w)
        sigmoid = sigmoid_func(res)
        return sigmoid

    def flattened_weight(self, net, weights_name):
        """
        Flattens the weights of a specific layer in the network into a 1D vector.

        Args:
            net (nn.Module): The neural network model.
            weights_name (str): The name of the layer whose weights to flatten.

        Returns:
            torch.Tensor: The flattened weight vector.
        """
        for name, parameters in net.named_parameters():
            if weights_name in name:
                f_weights = torch.mean(parameters, dim=0)
                f_weights = f_weights.view(-1, )
                return f_weights

    def extraction(self, net, weights_name, X):
        """
        Extracts a watermark from the model's weights using the secret key matrix (X).

        Args:
            net (nn.Module): The neural network model from which to extract the watermark.
            weights_name (str): The name of the layer to extract the watermark from.
            X (torch.Tensor): The secret key matrix.

        Returns:
            torch.Tensor: The extracted binary watermark vector.
        """
        W = self.flattened_weight(net, weights_name)
        return self.projection(X, W)

    def hamming(self, s1, s2):
        """
        Computes the Hamming distance between two binary sequences.

        Args:
            s1 (torch.Tensor): The first binary sequence.
            s2 (torch.Tensor): The second binary sequence.

        Returns:
            int: The Hamming distance between the two sequences.
        """
        assert len(s1) == len(s2)
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    def loss(self, net, weights_name, X, watermark):
        """
        Computes Uchida's loss function for watermark embedding.

        Args:
            net (nn.Module): The neural network model.
            weights_name (str): The name of the layer to apply the watermark.
            X (torch.Tensor): The secret key matrix.
            watermark (torch.Tensor): The binary watermark vector.

        Returns:
            torch.Tensor: The computed Uchida loss.
        """
        loss = 0
        W = self.flattened_weight(net, weights_name)
        yj = self.projection(X, W)
        for i in range(len(watermark)):
            loss += watermark[i] * torch.log2(yj[i]) + (1 - watermark[i]) * torch.log2(1 - yj[i])
        return -loss/len(watermark)

    def size_of_M(self, net, weight_name):
        """
        Determines the size of the flattened weight vector for a specific layer in the network.

        Args:
            net (nn.Module): The neural network model.
            weight_name (str): The name of the layer whose weights to analyze.

        Returns:
            int: The size of the flattened weight vector.
        
        Raises:
            ValueError: If the layer's weight shape is unsupported or if the layer is not found.
        """
        for name, parameters in net.named_parameters():
            if weight_name in name:
                if len(parameters.size()) == 2:  
                    return parameters.size()[1]
                elif len(parameters.size()) == 4:  
                    return parameters.size()[1] * parameters.size()[2] * parameters.size()[3]
                else:
                    raise ValueError(f"Unsupported parameter shape for {name}: {parameters.size()}")
        raise ValueError(f"Weight name {weight_name} not found in the network.")


    # you can copy-paste this section into main to test Uchida's method
    '''
    tools=Uchi_tools()
    weight_name = 'features.19.weight'
    T = 64
    watermark = torch.tensor(np.random.choice([0, 1], size=(T), p=[1. / 3, 2. / 3]), device=device)
    watermarking_dict={'lambd':0.1, 'weight_name':weight_name,'watermark':watermark}
    '''