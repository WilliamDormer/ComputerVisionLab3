#I'm aware that there are many different architecetural improvements that can be made to the model following those in the vision recognition challenge.

# 	- Extra convolutional layers for classification?
# 	- Train the VGG encoder (don't just keep it static)
# 	- 1D convolutions? Maybe on the encoder outputs for some further processing?
# 	- Sparse network like in inception, where they have the concatenation of multiple convolutional layers
# 	- Intermediate loss signal (e.g. Use the coarse label as an extra loss function implemented at earlier in the network) Auxilliary classifiers
# 	- Spatial factorization by assymmetric convolutions
# 	- Factorizing into smaller convolutions
# 	- Skip connections like in resnet
# 	- 1x1 convolutions (SE net)
# 	- Split transform merge from resnext
# 	- Trimps souchen is identifying the hardest categories and training the model explicitly to be better with those. Multiscale Fusion
# 	- Squeeze and Excitation Block
# 	- Dropout, more a training hyperparameter than actually an architectural mod.
# 	- Regularization in the loss function (L1 or L2, to prevent high weights and prevent overfitting)
# 	- Batch normalization (to help stabilize training by normalizing activations in each layer) need to look up if this actually helps
# 	- Learning Rate Scheduling
# 	- Momentum and Nesterov Accelerated Gradient
# 	- Normalization of data (I think I already do this)
# 	- Data Cleaning (getting rid of noisy or irrelevant training data points)
# 	- Data Augmentation:
# 		○ Give more data, which means we fit more per epoch.
# 		○ Rotation
# 		○ Scaling
# 		○ Flipping
# 	      Cropping

#I however wanted to try to implement a vision transformer, because I believe that is the state of the art.
# https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c

import torch
import torch.nn as nn
import numpy as np

from ..model_definitions.vanilla_classifier import VanillaClassifier
from ..model_definitions.default_encoder import encoder as default_encoder

#first attempt will use skip connections in an enhanced VGG encoder
class UpgradedModel(nn.Module):
    def __init__(self, image_dimensions, initial_encoder_path, hidden_size_1 = 4000, hidden_size_2 = 2000, hidden_size_3 = 700, target_size=100):
        super(UpgradedModel, self).__init__()

        #the size of the images being inputted into the network.
        self.image_dimensions = image_dimensions

        self.target_size = target_size

        #define the encoder layers:
        self.refpad = nn.ReflectionPad2d((1,1,1,1))
        self.maxpool = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 3, (1, 1))
        self.conv2 = nn.Conv2d(3, 64, (3, 3))
        self.conv3 = nn.Conv2d(64, 64, (3, 3))
        self.conv4 = nn.Conv2d(64, 128, (3, 3))
        self.conv5 = nn.Conv2d(128, 128, (3, 3))
        self.conv6 = nn.Conv2d(128, 256, (3, 3))
        self.conv7 = nn.Conv2d(256, 256, (3, 3))
        self.conv8 = nn.Conv2d(256, 256, (3, 3))
        self.conv9 = nn.Conv2d(256, 256, (3, 3))
        self.conv10 = nn.Conv2d(256, 512, (3, 3))

        self.initial_encoder_path = initial_encoder_path

        encoder_state_dict = torch.load(initial_encoder_path)
        # print(encoder_state_dict.keys())

        # enc = default_encoder
        # encoder_state_dict = enc.load_state_dict(torch.load(initial_encoder_path))
        # print("encoder state dict")
        # print(type(encoder_state_dict))
        # print(encoder_state_dict.keys())
        #keys from the original decoder # odict_keys(['0.weight', '0.bias', '2.weight', '2.bias', '5.weight', '5.bias', '9.weight', '9.bias', '12.weight', '12.bias', '16.weight', '16.bias', '19.weight', '19.bias', '22.weight', '22.bias', '25.weight', '25.bias', '29.weight', '29.bias'])

        #apply the original decoder weights to the correct weights in the new encoder with residuals.
        self.conv1.weight = nn.Parameter(encoder_state_dict["0.weight"])
        self.conv1.bias = nn.Parameter(encoder_state_dict["0.bias"])
        self.conv2.weight = nn.Parameter(encoder_state_dict["2.weight"])
        self.conv2.bias = nn.Parameter(encoder_state_dict["2.bias"])
        self.conv3.weight = nn.Parameter(encoder_state_dict["5.weight"])
        self.conv3.bias = nn.Parameter(encoder_state_dict["5.bias"])
        self.conv4.weight = nn.Parameter(encoder_state_dict["9.weight"])
        self.conv4.bias = nn.Parameter(encoder_state_dict["9.bias"])
        self.conv5.weight = nn.Parameter(encoder_state_dict["12.weight"])
        self.conv5.bias = nn.Parameter(encoder_state_dict["12.bias"])
        self.conv6.weight = nn.Parameter(encoder_state_dict["16.weight"])
        self.conv6.bias = nn.Parameter(encoder_state_dict["16.bias"])
        self.conv7.weight = nn.Parameter(encoder_state_dict["19.weight"])
        self.conv7.bias = nn.Parameter(encoder_state_dict["19.bias"])
        self.conv8.weight = nn.Parameter(encoder_state_dict["22.weight"])
        self.conv8.bias = nn.Parameter(encoder_state_dict["22.bias"])
        self.conv9.weight = nn.Parameter(encoder_state_dict["25.weight"])
        self.conv9.bias = nn.Parameter(encoder_state_dict["25.bias"])
        self.conv10.weight = nn.Parameter(encoder_state_dict["29.weight"])
        self.conv10.bias = nn.Parameter(encoder_state_dict["29.bias"])

        #define the bottleneck blocks that will be used to modify the skip connection dimensions to match
        encoder_block_1_dimension, encoder_block_2_dimension = self.get_encoder_dimensions()
        self.bottleneck_1 = self.construct_bottlneck_block(input_dimension=self.image_dimensions, output_dimension=encoder_block_1_dimension)
        self.bottleneck_2 = self.construct_bottlneck_block(input_dimension=encoder_block_1_dimension, output_dimension=encoder_block_2_dimension)

        self.classifier = VanillaClassifier(encoder_block_2_dimension, hidden_size_1, hidden_size_2, hidden_size_3, target_size).sequential



    #takes in the input and output dimensions you want, and gives back an nn.Sequential that will do the translation.
    def construct_bottlneck_block(self, input_dimension, output_dimension):

        #define the empty sequence:
        bottleneck_modules = []

        #find the difference in each dimension
        difference = [a - b for a, b in zip(output_dimension, input_dimension)]
        # print(difference)

        #if there is a need to reduce channels, then we do that first
        if difference[0] < 0:
            raise Exception("I haven't programmed this yet")

        #then we do the spatial dimension reduction
        #assume that it's square
        if difference[1] > 0 or difference[2] > 0:
            raise Exception("only programmed to handle shrinking spatial dimension")
        elif (difference[1] != difference[2]):
            raise Exception("only expecting square spatial dimensions.")
        else:
            num_3x3_convs = int(abs(difference[1]) / 2) #because each 3x3 conv takes off 1 pixel on each side
            #add that number of 3x3 convs to the sequence
            for i in range(num_3x3_convs):
                #input and out channels should stay the same.
                bottleneck_modules.append(nn.Conv2d(in_channels=input_dimension[0], out_channels=input_dimension[0], kernel_size=(3,3)))

        #if there is a need to increase channels, then we do that last.
        bottleneck_modules.append(nn.Conv2d(in_channels=input_dimension[0],out_channels=output_dimension[0],kernel_size=(1,1)))

        sequential = nn.Sequential(*bottleneck_modules)
        return sequential

    def forward_encoder_block_1(self,x):
        out = self.conv1(x)
        out = self.refpad(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.refpad(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.refpad(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.refpad(out)
        out = self.conv5(out)
        return out

    def forward_encoder_block_2(self,x):
        out = self.relu(x)
        out = self.maxpool(out)
        out = self.refpad(out)
        out = self.conv6(out)
        out = self.relu(out)
        out = self.refpad(out)
        out = self.conv7(out)
        out = self.relu(out)
        out = self.refpad(out)
        out = self.conv8(out)
        out = self.relu(out)
        out = self.refpad(out)
        out = self.conv9(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.refpad(out)
        out = self.conv10(out)
        return out
    def forward_encoder(self, x):

        #need to apply a dimension matching system to make the residual the same size as the output of the conv layer
        # I will do this with bottleneck blocks, which are a series of simple cnn layers with no activations
        # that use the 1x1 conv, and 3x3 conv to change channel dimension, and change spatial dimension respectively.

        # identity = x

        identity = self.bottleneck_1(x)
        out = self.forward_encoder_block_1(x)

        # print("identity shape: ", identity.size())
        # print("out shape: ", out.size())

        # apply the identity
        out += identity
        identity2 = self.bottleneck_2(out)
        out = self.forward_encoder_block_2(out)

        # apply the second identity
        out += identity2

        out = self.relu(out)
        return out
    def forward(self, x):
        out = self.forward_encoder(x)
        # print("shape after the forward pass of the encoder: ", out.size())
        #end of the encoder, now do the same linear layers as before for classification.
        out = self.classifier.forward(out)
        return out

    #where x is a single input image vector.
    #returns the dimensions of the output of the convolutional part of the network.
    #this is necessary for configuring the network.
    def get_encoder_dimensions(self):
        # run through the encoder part of the model with a random input to get the output shape
        # generate a random input vector of size self.image_dimensions
        sample = torch.rand(size=self.image_dimensions)
        sample = self.forward_encoder_block_1(sample)
        dimension1 = sample.size()
        sample = self.forward_encoder_block_2(sample)
        dimension2 = sample.size()
        return dimension1, dimension2



#more complicated upgraded model (maybe I'll use it later for the challenge)

#cifar images are 3 x 32 x 32 , where 3 is the colour dimension
#want to reshape our input of (N, 3, 32, 32) batch size, colour, height, width
# to (N, PxP, HxC/P x WxC/P)
#
# #inefficient version of patchify, need to look for a faster way to do it:
# def patchify(images, n_patches):
#     n, c, h, w = images.shape
#     assert h == w, "Patchify method is implemented for square images only"
#
#     patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
#     patch_size = h // n_patches
#
#     for idx, image in enumerate(images):
#         for i in range(n_patches):
#             for j in range(n_patches):
#                 patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
#                 patches[idx, i * n_patches + j] = patch.flatten()
#
#     return patches
#
# def get_positional_embeddings(sequence_length, d):
#     result = torch.ones(sequence_length, d)
#     for i in range(sequence_length):
#         for j in range(d):
#             result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
#     return result
#
# class MyViT(nn.Module):
#     def __init__(self, chw=(3, 32, 32), n_patches=8, hidden_d=12):
#         super(MyViT, self).__init__()
#         self.chw = chw # (C, W, H)
#         self.n_patches = n_patches
#         self.hidden_d = hidden_d # The hidden dimension that it will be reduced to before embedding
#
#         assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
#         assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
#         self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)
#
#         #Linear mapping, reduces last dimension?
#         self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
#         self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
#
#         #learnable classification token
#         self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
#
#         # Positional embedding
#         self.pos_embed = nn.Parameter(torch.tensor(get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_d)))
#         self.pos_embed.requires_grad = False
#
#     def forward(self, images):
#         #split the image up into patches
#         patches = patchify(images, self.n_patches)
#         tokens = self.linear_mapper(patches)
#
#         #adding classification token to the tokens list, len(tokens) returns the length of the first dimension (aka the batch size or num samples)
#         tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])
#
#         #adding positional embedding
#         #assume this
#         n = len(tokens) #aka the length of the first dimension.
#         pos_embed = self.pos_embed.repeat(n, 1, 1)
#         out = tokens + pos_embed
#         return out
#
#
#
