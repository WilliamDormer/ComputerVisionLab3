import torchvision.transforms as transforms
import torchvision
import torch
from torch.utils import data
from torchsummary import summary

from ..model_definitions.vanilla_model import Vanilla_Model
from ..model_definitions.vanilla_classifier import VanillaClassifier
from ..model_definitions.default_encoder import encoder
from ..model_definitions.upgrade_model import UpgradedModel

from ..test.test_helper import *

#randomly selects 6 images from the test dataset
#calculates the outputs for them
#then displays the labels along with the images in a grid

#it should also calculate the top 1 and top 5 error rates on the test set

if __name__ == '__main__':

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    target_size = 100
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    # print("getting first item in train set: ")
    sample = torch.utils.data.Subset(testset, [0])
    sample_loader = torch.utils.data.DataLoader(sample, batch_size=1, num_workers=0, shuffle=False)
    sample_image = None
    sample_label = None
    for images, labels in sample_loader:
        # if torch.cuda.is_available():
        #     images, labels = images.cuda(), labels.cuda()
        sample_image = images[0]
        sample_label = labels[0]

    # encoder = encoder.to("cuda")
    sample_image_cpu = sample_image.cpu()
    encoder_path = 'model_parameters/encoder/default_encoder.pth'
    network = UpgradedModel(image_dimensions=sample_image_cpu.size(), initial_encoder_path=encoder_path)
    network.load_state_dict(torch.load("model_parameters/classifier/UpgradeModelV1.1_20231105_104925_20"))
    network.eval()
    #run the network on a small sample of the training data
    sample = torch.utils.data.Subset(testset, [0,1,2,3,4,5])
    sample_loader = torch.utils.data.DataLoader(sample, batch_size=6, num_workers=0, shuffle=False)
    sample_images = None
    sample_labels = None
    sample_outputs = None
    with torch.no_grad():
        for images, labels in sample_loader:
            # if torch.cuda.is_available():
            #     images, labels = images.cuda(), labels.cuda()
            sample_images = images
            sample_labels = labels
            sample_outputs = network.forward(images)
        # print("size of sample outputs: ", sample_outputs.size())
        # print(sample_outputs)

    # #get the top 1 and top 5 prediction
    # top1_indices = sample_outputs.topk(1, dim=1).indices
    # top5_indices = sample_outputs.topk(5, dim=1).indices
    #
    # print("top1_index: ", top1_indices)
    # print("top5_indices: ", top5_indices)

    # top1_labels = []
    # for i in range(top1_indices.size()[0]):
    #     row = []
    #     for j in range(top1_indices.size()[1]):
    #         row.append(get_label_from_index(int(top1_indices[i][j])))
    #     top1_labels.append(row)
    #
    # top5_labels = []
    # for i in range(top5_indices.size()[0]):
    #     row = []
    #     for j in range(top5_indices.size()[1]):
    #         row.append(get_label_from_index(int(top5_indices[i][j])))
    #     top5_labels.append(row)
    #
    # print("top 1 labels: ", top1_labels)
    # print("top 5 labels: ", top5_labels)

    # print("sample labels: ", sample_labels)

    visualize_image_predictions(images=sample_images , true_labels=sample_labels, model=network)

    #now find the top 1 and top 5 error rates over the whole test set, including test and validation.
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)

    full_labels = None
    full_predictions = None
    with torch.no_grad():
        for images, labels in testloader:
            predictions = network.forward(images)
            if full_labels == None:
                full_labels = labels
                pass
            else:
                full_labels = torch.cat((full_labels, labels), dim=0)
                pass
            if full_predictions == None:
                full_predictions = predictions
                pass
            else:
                full_predictions = torch.cat((full_predictions, predictions), dim=0)
                pass

    # print("shape of full_predictions: ", full_predictions.size())
    # print("shape of full_labels: ", full_labels.size())

    errors = topk_error_rates(full_predictions, full_labels, topk=(1,5))
    print("top 1 error rate: ", errors[0].item())
    print("top 5 error rate: ", errors[1].item())
