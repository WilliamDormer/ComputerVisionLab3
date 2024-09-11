import torch
import argparse
from datetime import datetime
from torch.utils import data
import tqdm
from pathlib import Path
import os
import sys
import torchvision
import torchvision.transforms as transforms

from ..model_definitions.default_encoder import encoder
from ..model_definitions.vanilla_model import Vanilla_Model
from ..model_definitions.vanilla_classifier import VanillaClassifier

import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    # checks to see if file is valid.
    def dir_path(string):
        if os.path.isdir(string) or os.path.isfile(string):
            return string
        else:
            if os.path.isdir(string) == False:
                raise NotADirectoryError(string)
            else:
                raise FileNotFoundError(string)

    #define the command line input structure:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=5, help="how many epochs to train the classifier for")
    parser.add_argument("-b","--batch_size", type=int, default=32, help="defines the batch size (number of images per batch)")
    parser.add_argument("-l","--encoder_path", default="model_parameters/encoder/default_encoder.pth", type=dir_path, help="path to the saved weights of the default encoder")
    parser.add_argument("-n", "--experiment_name", default=None, type=str)
    parser.add_argument("-d", "--model_save_directory", default="model_parameters/classifier/")
    parser.add_argument("-f", "--figure_directory", default="figures/")
    parser.add_argument('-lr', '--learning_rate', default=0.00001, type=float)
    args = parser.parse_args()

    #argument checking
    if args.experiment_name == None:
        #Then we give it a sensible default value
        args.experiment_name = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    else:
        args.experiment_name = args.experiment_name + "_" + str(datetime.now().strftime('%Y%m%d_%H%M%S'))

    print("starting training script for vanilla model with the following parameters: ")
    print("epochs: ", args.epochs)
    print("batch size: ", args.batch_size)



    #obtain CIFAR datasets
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



    #cifar 10 version (for initial testing)
    # target_size = 10
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=True, transform=transform)
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                        download=True, transform=transform)

    #cifar 100 version
    target_size = 100
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    #we get the fine labels by default.
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    #makes sure the split is the same every time.
    testval_generator = torch.Generator().manual_seed(42)
    #split into validation and test sets.
    testval_datasets = torch.utils.data.random_split(testset,[0.5, 0.5], generator=testval_generator)

    test_dataset = testval_datasets[0]
    val_dataset = testval_datasets[1]

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    classes = trainset.classes

    #1652
    print("length of val_loader: ", len(valloader))
    print("length of train loader: ", len(trainloader))

    #get the structure of the encoder and classifier
    encoder = encoder

    #feed a sample image through the encoder to determine shape
    # sample_image = None
    # sample_label = None
    # for images, labels in trainloader:
    #     sample_image = images[0]
    #     sample_label = labels[0]
    #     break
    print("getting first item in train set: ")
    sample = torch.utils.data.Subset(trainset, [0])
    sample_loader = torch.utils.data.DataLoader(sample, batch_size=1, num_workers=0, shuffle=False)
    sample_image = None
    sample_label = None
    for images, labels in sample_loader:
        sample_image = images[0]
        sample_label = labels[0]

    encoded_sample = encoder(sample_image)
    print("size of encoded sample", encoded_sample.size())
    encoded_size = encoded_sample.size()

    #initialize the classifier module.
    hidden_size_1 = 4000
    hidden_size_2 = 2000
    hidden_size_3 = 700
    classifier = VanillaClassifier(encoded_size, hidden_size_1, hidden_size_2, hidden_size_3, target_size)
    print(classifier.get_info())
    #pull out the sequential component
    classifier = classifier.sequential

    #load in the initial weights for the default encoder
    encoder.load_state_dict(torch.load(args.encoder_path))

    network = Vanilla_Model(encoder, classifier)

    # print("checking layers")
    # for layer in network.classifier:
    #     if hasattr(layer, "weight"):
    #         print(f"Layer Type: {layer.__class__.__name__}")
    #         print(f"Weight Initialization: {layer.weight.data}")

    optimizer = torch.optim.Adam(network.classifier.parameters(), lr=args.learning_rate)

    #set it to run on GPU.
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    network.to(device)

    #set the network into train mode.
    network.train()

    #display some of the images:
    # functions to show an image
    # def imshow(img):
    #     img = img / 2 + 0.5     # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # # get some random training images
    # dataiter = iter(valloader)
    # images, labels = next(dataiter)
    #
    # # print labels
    # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(args.batch_size)))
    # # show images
    # imshow(torchvision.utils.make_grid(images))

    def train_one_epoch():
        running_loss = 0.
        network.train()
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(trainloader):
            # Every data instance is an input + label pair
            inputs, labels = data

            #send them to device if possible
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # compute loss with forward function
            loss = network(inputs, labels)
            # back prop
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            avg_batch_loss = loss / len(inputs)
            if i % 100 == 0:
                print(f"batch {i} loss {avg_batch_loss}")

            running_loss += avg_batch_loss

        #calculate the average loss for the epoch.
        avg_loss = running_loss / len(trainloader)


        return avg_loss

    loss_record = torch.empty((args.epochs,1), requires_grad=False)
    v_loss_record = torch.empty((args.epochs,1), requires_grad=False)

    best_vloss = 1_000_000.
    best_model_path = None

    for epoch in range(args.epochs):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        network.train(True)
        avg_loss = train_one_epoch()

        # run validation
        valid_loss = 0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        network.eval()

        network.eval()
        with torch.no_grad():
            for i, v_data in enumerate(valloader):
                v_inputs, v_labels = v_data
                if torch.cuda.is_available():
                    v_inputs, v_labels = v_inputs.cuda(), v_labels.cuda()

                predicted = network(v_inputs)
                v_loss = network.loss_function(predicted, v_labels)
                # v_loss = v_loss.item() * v_inputs.size(0)
                avg_batch_v_loss = v_loss.item() / len(v_inputs)
                valid_loss += avg_batch_v_loss

        #divide the sum of the average valid loss over all batches, and divide by the number of batches.
        avg_vloss = valid_loss / len(valloader)

        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        loss_record[epoch] = avg_loss
        v_loss_record[epoch] = avg_vloss

        # Track the best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = args.model_save_directory + args.experiment_name + "_" + str(epoch+1)
            best_model_path = model_path
            torch.save(network.state_dict(), model_path)

    #run the test dataset to measure accuracy.
    #load the best model version:
    network.load_state_dict(torch.load(best_model_path))
    network.eval()
    total_acc = 0
    with torch.no_grad():
        for i, t_data in enumerate(testloader):
            t_inputs, t_labels = t_data
            if torch.cuda.is_available():
                t_inputs, t_labels = t_inputs.cuda(), t_labels.cuda()
            predicted = network(t_inputs)
            # print(predicted.size())
            _, predicted = torch.max(predicted.data, 1)
            # print(predicted.size())
            # print(t_labels.size())
            acc = (predicted == t_labels).sum().item()
            total_acc += acc / len(t_inputs)
    total_acc = total_acc / len(testloader)
    print("total accuracy on test set: ", total_acc)

    loss_record = loss_record.cpu().detach().numpy()
    v_loss_record = v_loss_record.cpu().detach().numpy()

    # Generate and save the loss plots
    plt.figure(figsize = (10,6))
    epochs = list(range(1, len(loss_record)+1))
    plt.plot(epochs, loss_record, label="Training Loss", marker='o')
    plt.plot(epochs, v_loss_record, label="Validation Loss", marker='o')

    plt.title('Loss Over Epochs')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.savefig(args.figure_directory + args.experiment_name + ".jpg")

    plt.show()