from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F

transform_source = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

transform_target = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


def convert_score_to_label(scores, is_logit=True, threshold=0.5) -> torch.Tensor:
    if is_logit:
        p = F.sigmoid(scores)  # convert log(p) back into probabilty range [0, 1] using sigmoid function
    else:
        p = scores
    if threshold:
        p[p > threshold] = 1  # set it to either True or False based on threshold
        p[p < threshold] = 0
        p = p.type(torch.bool)
    return p


def convert_tensor_to_image(x: torch.Tensor) -> np.ndarray:
    # remember that we are outputsing tensors of shape [bs, 784]
    # in order to have them displayed as images we have to reshape them to the disared image format
    x = x.view(-1, 28, 28, 1) # numpy follows the format [bs, H, W, C] whereas torch is of format [bs, C, H, W]
                              # hence the reshape into (-1, 28, 28, 1) instead of (-1, 1, 28, 28)
    x = x.cpu().numpy()
    return x


def create_grid_plot(images, labels) -> plt.Figure:
    # we create a plut with 16 subplots (nrows * ncols)
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(56, 56))
    # we slices as many images as subplots we have
    images = convert_tensor_to_image(images)[:4 * 4]
    values = convert_score_to_label(labels)  # this line can be skipped if you want to display the logit
                                             # alternatively if you want to see p(x) instead of the logit
                                             # replace it with the line bellow
    # values = F.sigmoid(logits)
    for idx, image in enumerate(images):
        # we compute our current position in the subplot
        row = idx // 4
        col = idx % 4
        axes[row, col].axis("off")
        axes[row, col].set_title(str(values[idx].item()), size=72) # we plot the image label in the title field
                                                                   # such that each subplot will display its individual labeled value
        axes[row, col].imshow(image, cmap="gray", aspect="auto")
    return fig
