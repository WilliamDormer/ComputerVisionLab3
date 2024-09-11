import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def visualize_image_predictions(images, true_labels, model):

    model.eval()  # Set the model to evaluation mode

    num_images = len(images)
    rows = int(np.ceil(num_images / 3))  # Adjust the number of rows and columns for the grid
    cols = 3

    plt.figure(figsize=(17, 5))
    for i in range(num_images):
        image = images[i].cpu()
        label = true_labels[i]

        # Make a prediction using the model
        with torch.no_grad():
            output = model(image.unsqueeze(0))
            #predicted_label = torch.argmax(output, dim=1).item()
            predicted_label = output.topk(5,dim=1).indices


        # Convert the image tensor to a NumPy array for visualization
        image_np = image.permute(1, 2, 0).numpy()
        image_np = np.clip(image_np, 0, 1)  # Clip values to [0, 1]

        # print("predicted label size: ", predicted_label.size())
        label_string = get_label_from_index(label.item())
        predicted_label_string = []
        for l in range(0, predicted_label.size()[0]):
            row = []
            for j in range(0, predicted_label.size()[1]):
                row.append(get_label_from_index(predicted_label[l][j].item()))
            predicted_label_string.append(row)

        # Display the image
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image_np)
        plt.axis('off')
        plt.title(f'True: {label_string}\nPredicted: {predicted_label_string}')

    plt.tight_layout()
    plt.show()

def get_label_from_index(index):
    #uses the index in the list of 100 classes to determine which label (string) to return.
    conversion_chart = {
    0: "apple",
    1: "aquarium_fish",
    2: "baby",
    3: "bear",
    4: "beaver",
    5: "bed",
    6: "bee",
    7: "beetle",
    8: "bicycle",
    9: "bottle",
    10: "bowl",
    11: "boy",
    12: "bridge",
    13: "bus",
    14: "butterfly",
    15: "camel",
    16: "can",
    17: "castle",
    18: "caterpillar",
    19: "cattle",
    20: "chair",
    21: "chimpanzee",
    22: "clock",
    23: "cloud",
    24: "cockroach",
    25: "couch",
    26: "cra",
    27: "crocodile",
    28: "cup",
    29: "dinosaur",
    30: "dolphin",
    31: "elephant",
    32: "flatfish",
    33: "forest",
    34: "fox",
    35: "girl",
    36: "hamster",
    37: "house",
    38: "kangaroo",
    39: "keyboard",
    40: "lamp",
    41: "lawn_mower",
    42: "leopard",
    43: "lion",
    44: "lizard",
    45: "lobster",
    46: "man",
    47: "maple_tree",
    48: "motorcycle",
    49: "mountain",
    50: "mouse",
    51: "mushroom",
    52: "oak_tree",
    53: "orange",
    54: "orchid",
    55: "otter",
    56: "palm_tree",
    57: "pear",
    58: "pickup_truck",
    59: "pine_tree",
    60: "plain",
    61: "plate",
    62: "poppy",
    63: "porcupine",
    64: "possum",
    65: "rabbit",
    66: "raccoon",
    67: "ray",
    68: "road",
    69: "rocket",
    70: "rose",
    71: "sea",
    72: "seal",
    73: "shark",
    74: "shrew",
    75: "skunk",
    76: "skyscraper",
    77: "snail",
    78: "snake",
    79: "spider",
    80: "squirrel",
    81: "streetcar",
    82: "sunflower",
    83: "sweet_pepper",
    84: "table",
    85: "tank",
    86: "telephone",
    87: "television",
    88: "tiger",
    89: "tractor",
    90: "train",
    91: "trout",
    92: "tulip",
    93: "turtle",
    94: "wardrobe",
    95: "whale",
    96: "willow_tree",
    97: "wolf",
    98: "woman",
    99: "worm"
    }

    result = conversion_chart[index]
    return result

# def topk_error_rates(true_labels, predicted_probs, k=1):
#
#     _, topk_indices = predicted_probs.topk(k, dim=1, largest=True, sorted=True)
#     correct_predictions = topk_indices.eq(true_labels.view(-1, 1).expand_as(topk_indices))
#     topk_accuracy = correct_predictions.float().sum(1)
#     error_rate = 1 - (topk_accuracy / k).mean().item()
#     return error_rate

# https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
def topk_error_rates(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # st()
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # st()
        # correct = pred.eq(target.view(1, -1).expand_as(pred))
        # correct = (pred == target.view(1, -1).expand_as(pred))
        correct = (pred == target.unsqueeze(dim=0)).expand_as(pred)

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(1-correct_k.mul_(1.0 / batch_size))
        return res
