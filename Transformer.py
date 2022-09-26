import math
import numbers
import random
import warnings
from collections.abc import Sequence

import torch

try:
    import accimage
except ImportError:
    accimage = None

from torchvision.utils import _log_api_usage_once
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int
from typing import Tuple, List, Optional
from torch import Tensor

__all__ = [
    "Compose",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomRotation",
    "InterpolationMode",
]


class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, transforms):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            _log_api_usage_once(self)
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class RandomHorizontalFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        _log_api_usage_once(self)
        self.p = p

    def forward(self, img, label):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
            :param img:
            :param label:
        """
        if torch.rand(1) < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomVerticalFlip(torch.nn.Module):
    """Vertically flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        _log_api_usage_once(self)
        self.p = p

    def forward(self, img, label):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return F.vflip(img), F.vflip(label)
        return img, label

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomRotation(torch.nn.Module):
    """Rotate the image by angle.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image[.Resampling].NEAREST``) are still accepted,
            but deprecated since 0.13 and will be removed in 0.15. Please use InterpolationMode enum.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (sequence, optional): Optional center of rotation, (x, y). Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number): Pixel fill value for the area outside the rotated
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
        resample (int, optional):
            .. warning::
                This parameter was deprecated in ``0.12`` and will be removed in ``0.14``. Please use ``interpolation``
                instead.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(
            self, degrees, interpolation=InterpolationMode.NEAREST, expand=False, center=None, fill=0, resample=None
    ):
        super().__init__()
        _log_api_usage_once(self)
        if resample is not None:
            warnings.warn(
                "The parameter 'resample' is deprecated since 0.12 and will be removed 0.14. "
                "Please use 'interpolation' instead."
            )
            interpolation = _interpolation_modes_from_int(resample)

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument 'interpolation' of type int is deprecated since 0.13 and will be removed in 0.15. "
                "Please use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2,))

        self.center = center

        self.resample = self.interpolation = interpolation
        self.expand = expand

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill

    def get_params(self, degrees: List[float]) -> float:
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            float: angle parameter to be passed to ``rotate`` for random rotation.
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        return angle

    def forward(self, img, label):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
            :param img:
            :param label:
        """
        fill = self.fill
        channels, _, _ = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]
        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.resample, self.expand, self.center, fill), \
               F.rotate(label, angle, self.resample, self.expand, self.center, fill)

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(degrees={self.degrees}"
        format_string += f", interpolation={interpolate_str}"
        format_string += f", expand={self.expand}"
        if self.center is not None:
            format_string += f", center={self.center}"
        if self.fill is not None:
            format_string += f", fill={self.fill}"
        format_string += ")"
        return format_string


def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError(f"{name} should be a sequence of length {msg}.")
    if len(x) not in req_sizes:
        raise ValueError(f"{name} should be sequence of length {msg}.")


def _setup_angle(x, name, req_sizes=(2,)):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError(f"If {name} is a single number, it must be positive.")
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]
