import torch.nn as nn
import math
# classifier = nn.Sequential(
#     nn.Linear(in_features=512, out_features=10)
# )

class VanillaClassifier(nn.Module):
    def __init__(self, input_dim,hidden_dim_1, hidden_dim_2,hidden_dim_3, num_classes):
        super(VanillaClassifier, self).__init__()

        layers = []

        #calculate the number of input features
        num_features = input_dim[0] * input_dim[1] * input_dim[2]

        self.input_dim = num_features
        self.num_classes = num_classes
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.hidden_dim_3 = hidden_dim_3
        #add layers to the dynamic sequential
        #may need to add an AVGPOOL2D to convert to a 1x1 by channel depth before flattening.
        layers.append(nn.Flatten())
        layers.append(nn.Linear(num_features, self.hidden_dim_1))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim_1, self.hidden_dim_2))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim_2, self.hidden_dim_3))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim_3, self.num_classes))
        layers.append(nn.Softmax(dim=1))

        #create the sequential
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)

    def get_info(self):
        return {
            "input_dim" : self.input_dim,
            "hidden_dim_1" : self.hidden_dim_1,
            "hidden_dim_2": self.hidden_dim_2,
            "hidden_dim_3": self.hidden_dim_3,
            "output_dim" : self.num_classes
        }
