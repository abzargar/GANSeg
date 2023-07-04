import numbers
import random
import numpy as np
import warnings
import cv2
from collections.abc import Sequence
from typing import Tuple, List, Optional
from PIL import Image
import torch
from torch import Tensor

try:
    import accimage
except ImportError:
    accimage = None


from torchvision.transforms import functional as F

class CLAHE:
    def __call__(self, img,mask):
        img=np.array(img)
        clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(3, 3))
        img = clahe.apply(img)
        img[img>255]=255
        return Image.fromarray(img.astype(np.uint8)),mask

    def __repr__(self):
        return self.__class__.__name__


class AddGaussianNoise:
    def __init__(self, mean=0., std_max=1.):
        self.std_max = std_max
        self.mean = mean

    def __call__(self, img,mask):
        img=np.array(img)
        std=random.uniform(.005, self.std_max)
        gaussian = np.random.normal(self.mean, std, (img.shape[0], img.shape[1]))
        img=img+255*gaussian
        img[img>255]=255
        return Image.fromarray(img.astype(np.uint8)),mask

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img,mask):
        for t in self.transforms:
            img,mask = t(img,mask)
        return img,mask

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor:
    def __call__(self, img,mask):
        return F.to_tensor(img),F.to_tensor(mask)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PILToTensor:
    def __call__(self, pic):
        return F.pil_to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ConvertImageDtype(torch.nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, image):
        return F.convert_image_dtype(image, self.dtype)


class ToPILImage:
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, img,mask):
        return F.to_pil_image(img, self.mode),F.to_pil_image(mask, self.mode)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.mode is not None:
            format_string += 'mode={0}'.format(self.mode)
        format_string += ')'
        return format_string


class Normalize(torch.nn.Module):
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, img: Tensor,mask: Tensor) -> Tensor:
        return F.normalize(img, self.mean, self.std, self.inplace),mask

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Resize(torch.nn.Module):
    def __init__(self, size, img_interpolation=Image.BILINEAR,mask_interpolation=Image.NEAREST):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size
        self.img_interpolation = img_interpolation
        self.mask_interpolation = mask_interpolation

    def forward(self, img,mask):
        return F.resize(img, self.size, self.img_interpolation),F.resize(mask, self.size, self.mask_interpolation)

    def __repr__(self):
        interpolate_str = self.interpolation.value
        return self.__class__.__name__ + '(size={0}, interpolation={1}, max_size={2}, antialias={3})'.format(
            self.size, interpolate_str, self.max_size, self.antialias)


class Scale(Resize):
    def __init__(self, *args, **kwargs):
        warnings.warn("The use of the transforms.Scale transform is deprecated, " +
                      "please use transforms.Resize instead.")
        super(Scale, self).__init__(*args, **kwargs)



class Pad(torch.nn.Module):
    def __init__(self, padding, fill=0, padding_mode="constant"):
        super().__init__()
        if not isinstance(padding, (numbers.Number, tuple, list)):
            raise TypeError("Got inappropriate padding arg")

        if not isinstance(fill, (numbers.Number, str, tuple)):
            raise TypeError("Got inappropriate fill arg")

        if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
            raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

        if isinstance(padding, Sequence) and len(padding) not in [1, 2, 4]:
            raise ValueError("Padding must be an int or a 1, 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        return F.pad(img, self.padding, self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.padding, self.fill, self.padding_mode)


class Lambda:
    def __init__(self, lambd):
        if not callable(lambd):
            raise TypeError("Argument lambd should be callable, got {}".format(repr(type(lambd).__name__)))
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomTransforms:
    def __init__(self, transforms):
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence")
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomApply(torch.nn.Module):
    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, img,mask):
        if self.p < torch.rand(1):
            return img,mask
        for t in self.transforms:
            img,mask = t(img,mask)
        return img,mask

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomOrder(RandomTransforms):
    def __call__(self, img,mask):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            img,mask = self.transforms[i](img,mask)
        return img,mask


class RandomChoice(RandomTransforms):
    def __call__(self, img,mask):
        t = random.choice(self.transforms)
        return t(img,mask)


class RandomCrop(torch.nn.Module):
    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        w, h = F.get_image_size(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return i, j, th, tw

    def __init__(self, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img,mask):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = F.get_image_size(img)
        self.size=[int(0.65*height),int(0.65*width)]
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w),F.crop(mask, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(self.size, self.padding)


class Pass(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img,mask):
        return img,mask

    def __repr__(self):
        return self.__class__.__name__

class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img,mask):
        if torch.rand(1) < self.p:
            return F.hflip(img),F.hflip(mask)
        return img,mask
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img,mask):
        if torch.rand(1) < self.p:
            return F.vflip(img),F.vflip(mask)
        return img,mask

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class LinearTransformation(torch.nn.Module):
    def __init__(self, transformation_matrix, mean_vector):
        super().__init__()
        if transformation_matrix.size(0) != transformation_matrix.size(1):
            raise ValueError("transformation_matrix should be square. Got " +
                             "[{} x {}] rectangular matrix.".format(*transformation_matrix.size()))

        if mean_vector.size(0) != transformation_matrix.size(0):
            raise ValueError("mean_vector should have the same length {}".format(mean_vector.size(0)) +
                             " as any one of the dimensions of the transformation_matrix [{}]"
                             .format(tuple(transformation_matrix.size())))

        if transformation_matrix.device != mean_vector.device:
            raise ValueError("Input tensors should be on the same device. Got {} and {}"
                             .format(transformation_matrix.device, mean_vector.device))

        self.transformation_matrix = transformation_matrix
        self.mean_vector = mean_vector

    def forward(self, tensor: Tensor) -> Tensor:
        shape = tensor.shape
        n = shape[-3] * shape[-2] * shape[-1]
        if n != self.transformation_matrix.shape[0]:
            raise ValueError("Input tensor and transformation matrix have incompatible shape." +
                             "[{} x {} x {}] != ".format(shape[-3], shape[-2], shape[-1]) +
                             "{}".format(self.transformation_matrix.shape[0]))

        if tensor.device.type != self.mean_vector.device.type:
            raise ValueError("Input tensor should be on the same device as transformation matrix and mean vector. "
                             "Got {} vs {}".format(tensor.device, self.mean_vector.device))

        flat_tensor = tensor.view(-1, n) - self.mean_vector
        transformed_tensor = torch.mm(flat_tensor, self.transformation_matrix)
        tensor = transformed_tensor.view(shape)
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + '(transformation_matrix='
        format_string += (str(self.transformation_matrix.tolist()) + ')')
        format_string += (", (mean_vector=" + str(self.mean_vector.tolist()) + ')')
        return format_string


class ColorJitter(torch.nn.Module):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness: Optional[List[float]],
                   contrast: Optional[List[float]],
                   saturation: Optional[List[float]],
                   hue: Optional[List[float]]
                   ) -> Tuple[Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def forward(self, img,mask):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)

        return img,mask

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

class RandomRotation(torch.nn.Module):
    def __init__(self, degrees, resample=Image.NEAREST, expand=False, center=None, fill=0):
        super().__init__()
        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2, ))

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2, ))

        self.center = center

        self.resample = resample
        self.expand = expand
        self.fill = fill

    @staticmethod
    def get_params(degrees: List[float]) -> float:
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        return angle

    def forward(self, img,mask):
        angle = self.get_params(self.degrees)
        return F.rotate(img, angle, self.resample, self.expand, self.center, self.fill),F.rotate(mask, angle, self.resample, self.expand, self.center, self.fill)


class Grayscale(torch.nn.Module):
    def __init__(self, num_output_channels=1):
        super().__init__()
        self.num_output_channels = num_output_channels

    def forward(self, img,mask):
        return F.rgb_to_grayscale(img, num_output_channels=self.num_output_channels),F.rgb_to_grayscale(mask, num_output_channels=self.num_output_channels)

    def __repr__(self):
        return self.__class__.__name__ + '(num_output_channels={0})'.format(self.num_output_channels)

class GaussianBlur(torch.nn.Module):
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        super().__init__()
        self.kernel_size = _setup_size(kernel_size, "Kernel size should be a tuple/list of two integers")
        for ks in self.kernel_size:
            if ks <= 0 or ks % 2 == 0:
                raise ValueError("Kernel size value should be an odd and positive number.")

        if isinstance(sigma, numbers.Number):
            if sigma <= 0:
                raise ValueError("If sigma is a single number, it must be positive.")
            sigma = (sigma, sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0. < sigma[0] <= sigma[1]:
                raise ValueError("sigma values should be positive and of the form (min, max).")
        else:
            raise ValueError("sigma should be a single number or a list/tuple with length 2.")

        self.sigma = sigma

    @staticmethod
    def get_params(sigma_min: float, sigma_max: float) -> float:
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()

    def forward(self, img: Tensor,mask: Tensor) -> Tensor:
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        return F.gaussian_blur(img, self.kernel_size, [sigma, sigma]),mask

    def __repr__(self):
        s = '(kernel_size={}, '.format(self.kernel_size)
        s += 'sigma={})'.format(self.sigma)
        return self.__class__.__name__ + s


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError("{} should be a sequence of length {}.".format(name, msg))
    if len(x) not in req_sizes:
        raise ValueError("{} should be sequence of length {}.".format(name, msg))


def _setup_angle(x, name, req_sizes=(2, )):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError("If {} is a single number, it must be positive.".format(name))
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]

class RandomAdjustSharpness(torch.nn.Module):
    def __init__(self, sharpness_factor, p=0.5):
        super().__init__()
        self.sharpness_factor = sharpness_factor
        self.p = p

    def forward(self, img,mask):
        if torch.rand(1).item() < self.p:
            return F.adjust_sharpness(img, self.sharpness_factor),mask
        return img,mask

    def __repr__(self):
        return self.__class__.__name__ + '(sharpness_factor={},p={})'.format(self.sharpness_factor, self.p)



