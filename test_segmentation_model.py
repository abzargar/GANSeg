# Import necessary libraries and packages
import os
import torch.utils.data as data
import transforms as transforms
import numpy as np
import argparse
import random
import torch.nn as nn
from models import Segmentation  # Import the Segmentation model
from data import BasicDataset  # Import the Dataset handling module
import torch
from evaluate import evaluate_segmentation  # Import the evaluation function

# Set up seeds for reproducible results
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# Define the testing function
def test(args, image_size=[512, 768], image_means=[0.5], image_stds=[0.5], batch_size=1):
    # Determine if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the transformation to be applied on the images
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_means, std=image_stds)
    ])

    # Load the test dataset and apply the transformations
    test_data = BasicDataset(os.path.join(args.test_set_dir, 'images'), os.path.join(args.test_set_dir, 'masks'),
                             transforms=test_transforms)

    # Create a dataloader for the test dataset
    test_iterator = data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Create an instance of the Segmentation model and load the trained model
    Seg = nn.DataParallel(Segmentation(n_channels=1, n_classes=2, bilinear=True)).to(device)
    Seg.load_state_dict(torch.load(args.seg_ckpt_dir))

    # Evaluate the model and calculate the dice score and average precision
    dice_score, test_avg_precision = evaluate_segmentation(Seg, test_iterator, device, len(test_data),
                                                           is_avg_prec=True, prec_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
                                                           output_dir=args.output_dir)

    # Print the dice score and average precision
    print('INFO: Dice score:', dice_score)
    print('INFO: Average precision at ordered thresholds:', test_avg_precision)


# Define the main function
if __name__ == "__main__":
    # Define the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_set_dir", required=True, type=str, help="path for the test dataset")
    ap.add_argument("--seg_ckpt_dir", required=True, type=str, help="path for the checkpoint of segmentation model to test")
    ap.add_argument("--output_dir", required=True, type=str, help="path for saving the test outputs")

    # Parse the command-line arguments
    args = ap.parse_args()

    # Check if the test set directory exists
    assert os.path.isdir(args.test_set_dir), 'No such file or directory: ' + args.test_set_dir

    # If output directory doesn't exist, create it
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'input_images'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'segmentation_predictions'), exist_ok=True)


    # Run the test function
    test(args)
