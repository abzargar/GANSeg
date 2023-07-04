# Import necessary libraries and packages
import argparse
from data import BasicDataset
import cv2
from models import StyleUnetGenerator, StyleVectorizer
from utils import set_requires_grad, mixed_list, noise_list, image_noise, latent_to_w, styles_def_to_tensor
import torch.utils.data as data
import transforms as transforms
import torch
import numpy as np
import os
import random
from tqdm import tqdm
from utils import calculate_fid

# Setting the seed for generating random numbers for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def normalize_image(image):
    # Normalize the image to a range of [0, 1]
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def save_images(args, real_img_list, fake_img_list, real_A_list):
    # Save real, generated and mask images
    for i, (real_img, gen_img, mask_img) in enumerate(zip(real_img_list, fake_img_list, real_A_list)):
        cv2.imwrite(os.path.join(args.output_dir, 'real_images', 'images_{:04d}.png'.format(i)), real_img * 255)
        cv2.imwrite(os.path.join(args.output_dir, 'gen_images', 'images_{:04d}.png'.format(i)), gen_img * 255)
        cv2.imwrite(os.path.join(args.output_dir, 'mask_images', 'images_{:04d}.png'.format(i)), mask_img * 255)


def reshape_and_repeat(images_list, image_size):
    # Reshape and repeat grayscale images for FID calculation
    num_images = len(images_list)
    images = np.array(images_list).reshape(num_images, 1, image_size[0], image_size[1])
    return np.repeat(images, 3, axis=1)  # Repeat grayscale channel 3 times

def test(args, image_size=[512, 768], image_means=[0.5], image_stds=[0.5], batch_size=1):
    # Using CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Applying transformations on the test data
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_means, std=image_stds)
    ])

    # Loading the test data
    test_data = BasicDataset(os.path.join(args.test_set_dir, 'images'), os.path.join(args.test_set_dir, 'masks'),
                             transforms=test_transforms)
    test_iterator = data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Load the models
    Gen = StyleUnetGenerator().to(device)
    Gen.load_state_dict(torch.load(args.gen_ckpt_dir))

    StyleNet = StyleVectorizer(128, depth=3, lr_mul=0.1).to(device)  # latent_dim set to 128
    StyleNet.load_state_dict(torch.load(args.style_ckpt_dir))

    # Set the models to evaluation mode
    Gen.eval()
    StyleNet.eval()

    real_img_list=[]
    real_A_list=[]
    fake_img_list=[]

    # Iterating over batches of test data
    for step, batch in enumerate(tqdm(test_iterator)):
        real_img = batch['image'].to(device=device, dtype=torch.float32)
        real_mask = batch['mask'].to(device=device, dtype=torch.float32)

        set_requires_grad(Gen, False)

        with torch.no_grad():
            style = mixed_list(real_img.shape[0], 5, 128, device=device) if random.random() < 0.9 else noise_list(real_img.shape[0], 5, 128, device=device)  # latent_dim set to 128
            im_noise = image_noise(real_mask.shape[0], image_size, device=device)
            w_styles = styles_def_to_tensor(latent_to_w(StyleNet, style))

            fake_img = Gen(real_mask, w_styles, im_noise)
            fake_img = normalize_image(fake_img.cpu().numpy()[0,0,:,:])
            real_img = normalize_image(real_img.cpu().numpy()[0, 0, :, :])
            real_mask = normalize_image(real_mask.cpu().numpy()[0, 0, :, :])

            real_img_list.append(real_img)
            real_A_list.append(real_mask)
            fake_img_list.append(fake_img)

    # Saving the real and generated images
    save_images(args, real_img_list, fake_img_list, real_A_list)

    # Reshape and repeat images for FID calculation
    real_images = reshape_and_repeat(real_img_list, image_size)
    generated_images = reshape_and_repeat(fake_img_list, image_size)

    # Calculate and print FID score
    fid_score = calculate_fid(real_images, generated_images, device)
    print('FID Score: %f' % fid_score)

if __name__ == "__main__":
    # Argument parsing
    ap = argparse.ArgumentParser()
    # ap.add_argument("--test_set_dir", required=True, type=str, help="path for the test dataset")
    # ap.add_argument("--gen_ckpt_dir", required=True, type=str, help="path for the generator model checkpoint")
    # ap.add_argument("--style_ckpt_dir", required=True, type=str, help="path for the style vectorizer model checkpoint")
    # ap.add_argument("--output_dir", required=True, type=str, help="path for saving the test outputs")

    args = ap.parse_args()
    args.test_set_dir = '/home/azargari/GAN_1/large_stem_cell_dataset/test/'

    args.gen_ckpt_dir = "/home/azargari/GAN_1/tmp/Gen.pth"
    args.style_ckpt_dir = "/home/azargari/GAN_1/tmp/StyleNet.pth"
    args.output_dir = '/home/azargari/GAN_1/test_results/'

    # Check if test dataset directory exists
    assert os.path.isdir(args.test_set_dir), 'No such file or directory: ' + args.test_set_dir

    # Create output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Call the test function
    test(args)
