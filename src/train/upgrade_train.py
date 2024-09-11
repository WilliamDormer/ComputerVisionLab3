import numpy as np
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datetime import datetime
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision
import os
import argparse
from ..model_definitions.upgrade_model import UpgradedModel
from ..model_definitions.default_encoder import encoder as default_encoder

import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

if __name__ == '__main__':
    # transform = ToTensor() #may need to change this

    def dir_path(string):
        if os.path.isdir(string) or os.path.isfile(string):
            return string
        else:
            if os.path.isdir(string) == False:
                raise NotADirectoryError(string)
            else:
                raise FileNotFoundError(string)

    # define the command line input structure:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=5, help="how many epochs to train the classifier for")
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help="defines the batch size (number of images per batch)")
    parser.add_argument("-l", "--encoder_path", default="model_parameters/encoder/default_encoder.pth",
                        type=dir_path, help="path to the saved weights of the default encoder")
    parser.add_argument("-n", "--experiment_name", default=None, type=str)
    parser.add_argument("-d", "--model_save_directory", default="model_parameters/classifier/")
    parser.add_argument("-f", "--figure_directory", default="figures/")
    parser.add_argument('-lr', '--learning_rate', default=0.00001, type=float)
    args = parser.parse_args()

    # argument checking
    if args.experiment_name == None:
        # Then we give it a sensible default value
        args.experiment_name = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    else:
        args.experiment_name = args.experiment_name + "_" + str(datetime.now().strftime('%Y%m%d_%H%M%S'))

    #attempt to load the state dict of the encoder to see what the keys are
    # default_encoder = default_encoder
    # default_encoder.load_state_dict(torch.load(args.encoder_path))
    # print(default_encoder.state_dict().keys())




    print("starting training script for vanilla model with the following parameters: ")
    print("epochs: ", args.epochs)
    print("batch size: ", args.batch_size)

    # obtain CIFAR datasets
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    target_size = 100

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    # we get the fine labels by default.
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # makes sure the split is the same every time.
    testval_generator = torch.Generator().manual_seed(42)
    # split into validation and test sets.
    testval_datasets = torch.utils.data.random_split(testset, [0.5, 0.5], generator=testval_generator)

    test_dataset = testval_datasets[0]
    val_dataset = testval_datasets[1]

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=2)

    classes = trainset.classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: ", device)

    sample = torch.utils.data.Subset(trainset, [0])
    sample_loader = torch.utils.data.DataLoader(sample, batch_size=1, num_workers=0, shuffle=False)
    sample_image = None
    sample_label = None
    for images, labels in sample_loader:
        sample_image = images[0]
        sample_label = labels[0]

    image_dimensions = sample_image.size()

    model = UpgradedModel(image_dimensions=image_dimensions, initial_encoder_path=args.encoder_path)
    # print(model)
    # print("getting dimensions")
    # print(model.get_encoder_dimensions())
    # print("attemping a bottlneck block")
    # example_bottleneck = model.construct_bottlneck_block(input_dimension=torch.Size([128, 16, 16]), output_dimension=torch.Size([512, 4, 4]))
    # print(example_bottleneck)
    # print("attemping forward")
    # print(model.forward(sample_image).size())

    # for images, labels in valloader:
    #     result = model.forward(images)
    #     print(result.size())
    #     print(result)
    #     break



    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    model.to(device)
    model.train()

    def train_one_epoch():
        running_loss = 0.
        model.train()
        for i, data in enumerate(trainloader):
            inputs, labels = data

            #send them to device if possible
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            #zero your gradient for the batch
            optimizer.zero_grad()

            #compute the output of the network
            result = model.forward(inputs)
            #calculate the loss
            loss = loss_function(result, labels)
            #back prop
            loss.backward()

            #adjust learning rate
            optimizer.step()

            #gather data and report
            avg_batch_loss = loss / len(inputs)

            if i % 100 == 0:
                print(f"batch {i} loss {avg_batch_loss}")

            running_loss += avg_batch_loss

        # calculate the average loss for the epoch.
        avg_loss = running_loss / len(trainloader)
        return avg_loss

    loss_record = torch.empty((args.epochs, 1), requires_grad=False)
    v_loss_record = torch.empty((args.epochs, 1), requires_grad=False)

    best_vloss = 1_000_000.
    best_model_path = None

    for epoch in range(args.epochs):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch()

        # run validation
        valid_loss = 0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        model.eval()
        with torch.no_grad():
            for i, v_data in enumerate(valloader):
                v_inputs, v_labels = v_data
                if torch.cuda.is_available():
                    v_inputs, v_labels = v_inputs.cuda(), v_labels.cuda()

                predicted = model(v_inputs)
                v_loss = loss_function(predicted, v_labels)
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
            torch.save(model.state_dict(), model_path)

    #run the test dataset to measure accuracy.
    #load the best model version:
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    total_acc = 0
    with torch.no_grad():
        for i, t_data in enumerate(testloader):
            t_inputs, t_labels = t_data
            if torch.cuda.is_available():
                t_inputs, t_labels = t_inputs.cuda(), t_labels.cuda()
            predicted = model(t_inputs)
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

    #
    # # Defining model and training options
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    # model = MyViT((1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)
    # N_EPOCHS = 5
    # LR = 0.005
    #
    # # Training loop
    # optimizer = Adam(model.parameters(), lr=LR)
    # criterion = CrossEntropyLoss()
    # for epoch in trange(N_EPOCHS, desc="Training"):
    #     train_loss = 0.0
    #     for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
    #         x, y = batch
    #         x, y = x.to(device), y.to(device)
    #         y_hat = model(x)
    #         loss = criterion(y_hat, y)
    #
    #         train_loss += loss.detach().cpu().item() / len(train_loader)
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #     print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")
    #
    # # Test loop
    # with torch.no_grad():
    #     correct, total = 0, 0
    #     test_loss = 0.0
    #     for batch in tqdm(test_loader, desc="Testing"):
    #         x, y = batch
    #         x, y = x.to(device), y.to(device)
    #         y_hat = model(x)
    #         loss = criterion(y_hat, y)
    #         test_loss += loss.detach().cpu().item() / len(test_loader)
    #
    #         correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
    #         total += len(x)
    #     print(f"Test loss: {test_loss:.2f}")
    #     print(f"Test accuracy: {correct / total * 100:.2f}%")
