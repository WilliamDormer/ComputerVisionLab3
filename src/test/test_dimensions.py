from torchsummary import summary
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data

from ..model_definitions.default_encoder import encoder
from ..model_definitions.vanilla_classifier import VanillaClassifier
from ..model_definitions.vanilla_model import Vanilla_Model
from ..model_definitions.upgrade_model import UpgradedModel

encoder_path = 'model_parameters/encoder/default_encoder.pth'

encoder = encoder
# encoder.load_state_dict(torch.load(encoder_path))
#initialize the classifier module.
hidden_size_1 = 4000
hidden_size_2 = 2000
hidden_size_3 = 700

transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

print("getting first item in train set: ")
sample = torch.utils.data.Subset(trainset, [0])
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
encoded_sample = encoder(sample_image_cpu)
print("size of encoded sample", encoded_sample.size())
encoded_size = encoded_sample.size()
target_size = 100
classifier = VanillaClassifier(encoded_size, hidden_size_1, hidden_size_2, hidden_size_3, target_size)
classifier = classifier.sequential
network = Vanilla_Model(encoder, classifier)

#load the right state dict
network.load_state_dict(torch.load("model_parameters/classifier/VanillaModelV1.2_20231105_114052_20"))
print("sample_image_size: ", sample_image_cpu.size())
network = network.to("cpu")
network.eval()
# network.load_state_dict({k: v.to('cpu') for k, v in network.state_dict().items()})
print("summary for Vanilla model: ")
print(summary(network, sample_image_cpu.size(), device="cpu"))

network2 = UpgradedModel(image_dimensions=sample_image_cpu.size(), initial_encoder_path=encoder_path)
network2.load_state_dict(torch.load("model_parameters/classifier/UpgradeModelV1.1_20231105_104925_20"))
print("summary for Upgraded Model")
print(summary(network2, sample_image_cpu.size(), device='cpu'))
