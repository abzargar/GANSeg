from torch import Tensor
import torch
import torch.nn.functional as F
from torch.nn import Module


class CombinedLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        # Compute Cross-Entropy (CE) loss
        ce_loss = F.cross_entropy(inputs, targets)

        # Compute Dice loss
        inputs = F.softmax(inputs, dim=1)[:, 1]
        targets = (targets == 1).float()  # One-hot encoding

        intersection = (inputs * targets).sum()
        dice_loss = 1 - ((2. * intersection + smooth) /
                         (inputs.sum() + targets.sum() + smooth))

        # Combine the losses
        combined_loss = ce_loss + dice_loss

        return combined_loss

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]