# A pretty minimalistic script to run an inference with one of our models

import torch
import monai

import numpy as np

from typing import List, Dict, Union, Optional, Tuple
from torch import jit

#########################################################################
# Define utility functions
#########################################################################

def get_predictor(net, inferer, device='cpu'):
    net.eval()
    # if cuda:
    net.to(device) # only use this option if the gpu can handle the memory requirement
    def predictor(samples, **kwargs):
        return inferer(
            inputs=samples, network=net, **kwargs
        )
    return predictor

def check_tensor(x: Union[torch.Tensor, np.array]) -> np.array:
    """
    Checks if the input is a PyTorch tensor. If it is, and if the tensor requires gradients 
    (i.e., it's part of a computation graph), the tensor is detached from the current graph, and 
    then converted to a NumPy array. If the tensor does not require gradients, it is directly 
    converted to a NumPy array. The conversion ensures the tensor is moved to CPU memory before 
    conversion, as NumPy cannot handle GPU tensors.

    Parameters:
    x (Union[torch.Tensor, np.ndarray]): The input tensor/array to be converted to a NumPy array.

    Returns:
    np.ndarray: The input tensor/array converted to a NumPy array.
    """
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()

    return x

def batch_to_sample(batch: Dict) -> Tuple[Union[torch.Tensor, torch.Tensor], Dict]:
    """Extract the input and kwargs from the batch.
    """
    return_tensors = []
    sample = batch.pop("image")
    return_tensors.append(sample)
    if 'label' in batch.keys():
        ground_truth = batch.pop("label")
        return_tensors.append(ground_truth)

    kwarg = dict()
    for key in batch.keys():
        kwarg[key] = check_tensor(batch[key])
        # kwarg[key] = batch[key]

    return (*return_tensors, kwarg)


def get_post_pred_transforms(n_classes: int, threshold: float=0.5) -> List[monai.transforms.Transform]:
    """
    Get post-prediction transforms for converting logits to probabilities and probabilities to discrete predictions.

    This function constructs a series of MONAI transforms to be applied on network outputs.
    It creates three sets of transforms: one for converting logits to probabilities, another for 
    converting probabilities to discrete predictions, and a third one that combines both for a complete 
    logits-to-predictions transformation.

    Parameters:
    n_classes (int): Number of classes in the classification task.
    threshold (float, optional): Threshold used for binarization in the discrete prediction transform. 
                                 Defaults to 0.5.

    Returns:
    List[monai.transforms.Transform]: A tuple containing three sets of transforms: 
                                      (logit-to-probability transforms, probability-to-prediction transforms, 
                                      combined logit-to-prediction transforms).
    """
    logit_to_prob_transforms = monai.transforms.Compose([
        monai.transforms.EnsureType(),
        monai.transforms.Activations(softmax=True)
    ])
    prob_to_pred_transforms = monai.transforms.Compose([
        monai.transforms.EnsureType(),
        monai.transforms.AsDiscrete(argmax=True, to_onehot=n_classes) # need to_onehot after argmax
    ])

    logit_to_pred_transforms = monai.transforms.Compose([
        logit_to_prob_transforms,
        prob_to_pred_transforms
    ])

    return logit_to_prob_transforms, prob_to_pred_transforms, logit_to_pred_transforms

def get_predictions_from_logits(logits: torch.Tensor, mode: str='discrete', threshold: float=0.5) -> List:
    """Takes logit coming directly from network, so dimensionality is
    [BCWH[D]].

    Returns list of discrete predictions or probabilities:
    dimensions [EBCWH[D]]
    """
    n_classes = logits.shape[1]
    logit_to_prob_transforms, _, logit_to_pred_transforms = get_post_pred_transforms(n_classes, threshold)
    if mode=='discrete':
        preds = [logit_to_pred_transforms(logit) for logit in logits]
    elif mode=='prob':
        preds = [logit_to_prob_transforms(logit) for logit in logits]
    return torch.stack(preds)

def get_predictions_from_probs(probs: torch.Tensor, threshold: Optional[float]=None) -> List:
    """
    Convert probability predictions to class predictions using specified post-prediction transforms.

    This function applies a series of transforms to convert raw probability predictions
    into discrete class predictions. It first retrieves the necessary transforms based on
    the number of classes and an optional threshold. It then applies these transforms to 
    each probability prediction in the input.

    Parameters:
    probs (torch.Tensor): Tensor of probability predictions
    threshold (Optional[float], optional): Threshold value used in the post-prediction transforms. 
                                           Defaults to None.

    Returns:
    torch.Tensor: Tensor of class predictions derived from the probability predictions.
    """
    n_classes = probs.shape[1]
    _, prob_to_pred_transforms, _ = get_post_pred_transforms(n_classes, threshold)
    preds = [prob_to_pred_transforms(prob) for prob in probs]
    return torch.stack(preds)

def get_post_label_transforms(n_classes: int, add_one_hot: bool = False) -> List[monai.transforms.Transform]:
    '''Transform label to one-hot format in case of multiple output
    channels.
    '''
    if n_classes==1:
        post_label_transform = monai.transforms.Compose([monai.transforms.EnsureType()])
    else:
        if add_one_hot:
            post_label_transform = monai.transforms.Compose([monai.transforms.EnsureType(), monai.transforms.AsDiscrete(to_onehot=n_classes)])
        else:
            post_label_transform = monai.transforms.Compose([monai.transforms.EnsureType()])

    return post_label_transform

def load_model(model_trace):
    net = jit.load(model_trace)
    net.eval();
    return net