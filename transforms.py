# core packages
from typing import Dict, List

import numpy as np

# related packages
import torch

import pandas as pd
from monai.transforms import (
    AsDiscrete,
    Compose,
    LoadImaged,
    MapTransform,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambda,
    CenterSpatialCropd,
    SpatialPadd,
    ConcatItemsd,
    DeleteItemsd,
)

from monai.transforms.transform import Transform, MapTransform

from monai.config.type_definitions import NdarrayOrTensor
from monai.config import KeysCollection
from monai.utils.enums import TransformBackends

class ReArrangeLabelClassesd(MapTransform):
    """
    Rearrange label classes according to RTSS extraction:
    Class 0 is background
    Class 1 is tumor
    Class 2 is skull or external
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]
            # if label.shape[0]>1: # this line here is a somewhat dirty fix on this dirty function. It allows for labels that are single channel.
            d[key] = label[1:2]
        return d

class ConvertToMultiChannelBasedOnBratsClassesCustom(Transform):
    """
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        result = [(img == 3), (img == 1) | (img == 4)]
        # result = [(img == 1) | (img == 4), (img == 1) | (img == 4) | (img == 2), img == 4] # original combination of classes. Note that commas separate the classes.
        # merge labels 1 (tumor non-enh) and 4 (tumor enh) and 2 (large edema) to WT
        # label 4 is ET
        labels = torch.stack(result, dim=0) if isinstance(img, torch.Tensor) \
            else np.stack(result, axis=0)
        return labels

class ReArrangeLabelClassesd(MapTransform):
    """
    Rearrange label classes according to RTSS extraction:
    Class 0 is background
    Class 1 is tumor
    Class 2 is skull or external

    self.keys here will typically be ['label'], and data is
    a torch dataset. This means that d[key] is the label, which 
    in the case of nifti files prepared by our data pipeline,
    will have three classes, according to the above description.
    Now, we are only interested in the tumour label, which has
    index 1. Thus, we redefine the label as only the labels
    corresponding to this index.
    
    --to be deprecated--
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]
            # if label.shape[0]>1: # this line here is a somewhat dirty fix on this dirty function. It allows for labels that are single channel.
            d[key] = label[1:2]
        return d

class ConvertToMultiChannelBasedOnBratsClasses_custom(Transform):
    """
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        result = [(img == 3), (img == 1) | (img == 4)]
        # result = [(img == 1) | (img == 4), (img == 1) | (img == 4) | (img == 2), img == 4] # original combination of classes. Note that commas separate the classes.
        # merge labels 1 (tumor non-enh) and 4 (tumor enh) and 2 (large edema) to WT
        # label 4 is ET
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)


class ChooseLabelClasses(MapTransform):
    """
    Combine multiple label channels to new classes according to preset
    definitions for different datasets.
    For CLVL the union of channels 0 and 2 and with 1 cut out correspond
    to background. Thus it is easier to let background be the inverse of
    the foreground.
    """
    
    def __init__(self, keys: KeysCollection, dataset_name: str, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.dataset_name = dataset_name

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                if self.dataset_name=='cleveland' or self.dataset_name.startswith('Brats'):
                        data[key][0] = torch.logical_not(data[key][1])
                        data[key][1] = data[key][1]
        return data

class to_onehot_conditionald(MapTransform):
    """
    """
    def __init__(self, keys: KeysCollection, highest_channel: int, dataset_name: str, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.highest_channel = highest_channel
        self.dataset_name = dataset_name

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                if data[key].shape[0] == 1:
                    if self.dataset_name.startswith('Brats'):
                        data[key] = 1.*ConvertToMultiChannelBasedOnBratsClasses_custom()(data[key])
                    else:
                        # label_onehot_channels = int(data[key].unique().as_tensor()[-1])+1
                        data[key] = AsDiscrete(to_onehot=self.highest_channel)(data[key])
            else:
                raise Exception
        return data

class select_channelsd(MapTransform):
    """
    Rearrange label classes according to RTSS extraction:
    Class 0 is background
    Class 1 is tumor

    self.keys here will typically be ['label'], and data is
    a torch dataset. This means that d[key] is the label, which 
    in the case of nifti files prepared by our data pipeline,
    will have three classes, according to the above description.
    Now, we are only interested in the tumour label, which has
    index 1. Thus, we redefine the label as only the labels
    corresponding to this index.
    """
    
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, ch_to_keep: List=[0,1]):
        super().__init__(keys, allow_missing_keys)
        self.ch_to_keep = ch_to_keep

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                if data[key].shape[0] > 1:
                    data[key] = data[key][self.ch_to_keep]
            else:
                raise Exception
        return data



def get_standard_transform_variable_size(hyperparameters: Dict) -> List:
    """Get set of base transforms that can be applied to list of dicts
    containing keys 'label' and 'image' for ground truth and images,
    respectively. 'image' may contain list of images, and these may in
    principle be of different sizes (though this would likely not be a
    very good idea).
    
    Resamples images and ground truths to an isotropic resolution of
    1.0x1.0x1.0 mm3 and crops and pads image to centre_crop_size.

    Args:
        hyperparameters (Dict): {..., n_input_modalities: n,
        centre_crop_size: (x, y, z), ...}

    Returns:
        List: List of transformations that are the same for training,
        validation, and inference transforms.
    """    
    modalities = [f'mod_{i}' for i in range(hyperparameters['n_input_modalities'])]
    centre_crop_size = hyperparameters['centre_crop_size']
    dataset_name = hyperparameters['site']
    label_ch_to_keep = hyperparameters['label_ch_to_keep']
    highest_channel = hyperparameters['highest_channel']
    
    data_loading_transforms = [
        Lambda(func=lambda x: dict({'label':x['label']}, **{m:x['image'][i] for i,m in enumerate(modalities)}, **{k: v for (k,v) in x.items() if k not in ['label', *modalities]}) if isinstance(x['image'], List) else \
                dict({'label':x['label'], modalities[0]:x['image']}, **{k: v for (k,v) in x.items() if k not in ['label', *modalities]})),
        LoadImaged(keys=['label', *modalities]),
        EnsureChannelFirstd(keys=['label', *modalities]),
        Spacingd(
            keys=['label', *modalities],
            pixdim=(1.0, 1.0, 1.0),
            mode=("nearest", *["bilinear" for i,_ in enumerate(modalities)]),
        ),
        CenterSpatialCropd(
            keys=['label', *modalities], roi_size=centre_crop_size
        ),
        to_onehot_conditionald(keys=['label'], highest_channel=highest_channel, dataset_name = dataset_name),
        ChooseLabelClasses(keys=['label'], dataset_name = dataset_name),
        select_channelsd(keys=['label'], ch_to_keep=label_ch_to_keep),
        SpatialPadd(
            keys=['label', *modalities], spatial_size=centre_crop_size
        ),
        ConcatItemsd(keys=modalities, name="image"),
        DeleteItemsd(keys=modalities),
        EnsureTyped(keys=["image", "label"]),
    ]
    
    return Compose(data_loading_transforms)