import torch.nn as nn

class Vanilla_Model(nn.Module):
    def __init__(self, encoder, classifier):
        super(Vanilla_Model, self).__init__()
        self.encoder = encoder
        #freeze the encoder weights for this version of the model, turning this off for now because i want to do 1 to 1 test.
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.classifier = classifier
        self.initialize_classifier_weights()
        self.loss_function = nn.CrossEntropyLoss()

    def initialize_classifier_weights(self):
        for layer in self.classifier:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                if hasattr(layer, 'weight'):
                    if len(layer.weight.size()) > 1:
                        #apply desired weight initialization method, we'll use the default
                        nn.init.xavier_normal_(layer.weight)

    def forward(self, images, labels=None):
        if self.training: #training
            bottleneck = self.encoder.forward(images)
            output = self.classifier.forward(bottleneck)
            loss = self.loss_function(output, labels)
            return loss
        else: #inference
            bottleneck = self.encoder.forward(images)
            output = self.classifier.forward(bottleneck)
            return output