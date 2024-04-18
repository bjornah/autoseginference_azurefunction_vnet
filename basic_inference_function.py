# A pretty minimalistic script to run an inference with one of our models

import logging
import os
import torch
import monai
import glob
import tempfile
import yaml

import pandas as pd
import numpy as np
import SimpleITK as sitk
import nibabel as nib

from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Tuple
from monai.inferers.inferer import SlidingWindowInferer
from monai.data import Dataset, decollate_batch
from monai.data.utils import list_data_collate
from torch.utils.data import DataLoader
from torch import jit


from niftidicomconverter import dicom2nifti, nifti2dicom # the only custom package required required to run this script
from niftidicomconverter.dicomhandling import load_dicom_images, save_itk_image_as_nifti_sitk
from niftidicomconverter.niftihandling import resample_nifti_to_new_spacing
from niftidicomconverter.utils import copy_nifti_header

from transforms import get_standard_transform_variable_size

#########################################################################
# Define utility functions
#########################################################################

def get_predictor(net, inferer, cuda=False):
    net.eval()
    if cuda:
        net.to('cuda:0') # only use this option if the gpu can handle the memory requirement
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

#########################################################################
# Set paths and create folders
#########################################################################
def setup_inference(manifest):
    '''
    Inference on a single stack of dicom images
    '''
    # TO DO: turn hard coded settings into yml file 

    ROI_SIZE = (128, 128, 96)
    OVERLAP = 0.5
    PRED_THRESHOLD = 0.5  # Threshold above which voxel is considered foreground

    SITE = 'cleveland' # NYU

    MODEL_PATH = 'traced_model.pt'
    NEW_IMAGE_SPACING = (1.0, 1.0, 1.0)
    STRUCTURE_MAP_FILE = 'structure_map.yml'

    if 'dicom_folder' in manifest.keys():
        DICOM_FOLDER = manifest['dicom_folder']
    else:
        DICOM_FOLDER = None
    if 'rtss_tmp' in manifest.keys():
        DICOM_GT = manifest['rtss_tmp']
    else:
        DICOM_GT = None
    PATIENT_ID = manifest['patient_ID']

    SAVEPATH_NAME = 'inference'
    BASE_DIR = manifest["base_temp_dir"]
    SAVEPATH_BASE = f'{BASE_DIR}/{SAVEPATH_NAME}'

    for SAVEPATH in [
        SAVEPATH_BASE,
    ]:
        os.makedirs(SAVEPATH, exist_ok=True)

    if 'nifti_file' in manifest.keys():
        NIFTI_IMAGE = manifest['nifti_file']
    else:
        NIFTI_IMAGE = os.path.abspath(f'{BASE_DIR}/nifti_image.nii.gz')

    # if (DICOM_GT is not None):
    NIFTI_GT = os.path.abspath(f'{SAVEPATH_BASE}/nifti_label.nii.gz')
    # else:
    #     NIFTI_GT = None

    NIFTI_RTSS = os.path.abspath(f'{SAVEPATH_BASE}/nifti_pred.nii.gz')
    DICOM_RTSS_OUTPUT = os.path.abspath(f'{SAVEPATH_BASE}/dicom_pred.dcm')

    DICE_PATH = os.path.abspath(f'{SAVEPATH_BASE}/dice.csv')
    CALIBRATION_PATH = os.path.abspath(f'{SAVEPATH_BASE}/model_calibration.csv')

    inference_settings = {
        'ROI_SIZE': ROI_SIZE,
        'OVERLAP': OVERLAP,
        'NEW_IMAGE_SPACING':NEW_IMAGE_SPACING,
        'PRED_THRESHOLD': PRED_THRESHOLD,
        'SAVEPATH_BASE': SAVEPATH_BASE,
        'MODEL_PATH': MODEL_PATH,
        'NIFTI_RTSS': NIFTI_RTSS,
        'NIFTI_IMAGE': NIFTI_IMAGE,
        'NIFTI_GT': NIFTI_GT,
        'DICOM_FOLDER': DICOM_FOLDER,
        'DICOM_RTSS_OUTPUT': DICOM_RTSS_OUTPUT,
        'DICE_PATH': DICE_PATH,
        'CALIBRATION_PATH': CALIBRATION_PATH,
    }

    #########################################################################
    # Convert dicom to nifti
    #########################################################################
    if DICOM_FOLDER is not None:
        dicom2nifti.convert_dicom_to_nifti(DICOM_FOLDER, NIFTI_IMAGE) # function to convert dicom to nifti, as the name makes pretty clear
        
        print('before resampling')
        nifti_image = nib.load(NIFTI_IMAGE)
        print(f'nifti_image.shape = {nifti_image.get_fdata().shape}')

        _ = resample_nifti_to_new_spacing(
            nifti_file_path=NIFTI_IMAGE,
            new_spacing=NEW_IMAGE_SPACING,
            save_path=NIFTI_IMAGE,
            interpolation_order=3
        ) # resample nifti to new spacing

        print('after resampling')
        nifti_image = nib.load(NIFTI_IMAGE)
        print(f'nifti_image.shape = {nifti_image.get_fdata().shape}')

    # then, if there are labels do those:
    if DICOM_GT is not None:
        structure_map = yaml.safe_load(Path(STRUCTURE_MAP_FILE).read_text())

        # some potentially useful debugging print statements
        # print(f"DICOM_FOLDER = {DICOM_FOLDER}")
        # print(f"DICOM_GT = {DICOM_GT}")
        # print(f"NIFTI_GT = {NIFTI_GT}")
        # print(f"os.path.exists(DICOM_GT) = {os.path.exists(DICOM_GT)}")
        # print(f"os.path.exists(DICOM_FOLDER) = {os.path.exists(DICOM_FOLDER)}")
        # print(f"os.path.exists(NIFTI_GT) = {os.path.exists(NIFTI_GT)}")

        try:
            dicom2nifti.convert_dicom_rtss_to_nifti(DICOM_FOLDER, DICOM_GT, NIFTI_GT, structure_map)

            print('before resampling')
            nifti_gt = nib.load(NIFTI_GT)
            print(f'nifti_gt.shape = {nifti_gt.get_fdata().shape}')

            _ = resample_nifti_to_new_spacing(
                nifti_file_path=NIFTI_GT,
                new_spacing=NEW_IMAGE_SPACING,
                save_path=NIFTI_GT,
                interpolation_order=0
            ) # resample nifti to new spacing

            print('after resampling')
            nifti_gt = nib.load(NIFTI_GT)
            print(f'nifti_gt.shape = {nifti_gt.get_fdata().shape}')

        except Exception as e:
            print(f'Error converting RTSS: {e}')
            raise e

        nifti_data = {
            'image':NIFTI_IMAGE,
            'label':NIFTI_GT
        }

    else:
        nifti_data = {
            'image':NIFTI_IMAGE,
        }

    #########################################################################
    # Create data loader
    #########################################################################
    data_dict = nifti_data
    data_dict['site'] = SITE
    data_dict['PATIENT_ID'] = PATIENT_ID
        
    if DICOM_GT is None:
        all_transforms = monai.transforms.Compose([
            monai.transforms.LoadImaged(keys=['image']),
            monai.transforms.EnsureChannelFirstd(keys=['image']),
            monai.transforms.Spacingd(
                keys=['image'],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear"),
            ),
            monai.transforms.ScaleIntensityRangePercentilesd(
                keys="image",
                lower=0.5,
                upper=99.5,
                b_min=0,
                b_max=1,
                clip=True,
                relative=False,
                channel_wise=True,
            ),
            monai.transforms.ToTensord(keys=["image"]),
            monai.transforms.EnsureTyped(keys=["image"]),
        ])

    else:
        data_loading_transforms = get_standard_transform_variable_size(hyperparameters={
                'centre_crop_size':[-1, -1, -1],
                'n_input_modalities':1,
                'site':data_dict['site'],
                'label_ch_to_keep':[0,1],
                'highest_channel':3
            }
        )
        normalisation_transforms = monai.transforms.Compose([
            monai.transforms.ScaleIntensityRangePercentilesd(
                keys="image",
                lower=0.5,
                upper=99.5,
                b_min=0,
                b_max=1,
                clip=True,
                relative=False,
                channel_wise=True,
            ),
            monai.transforms.ToTensord(keys=["image"]),
            monai.transforms.EnsureTyped(keys=["image"]),
        ])

        all_transforms = monai.transforms.Compose([data_loading_transforms,normalisation_transforms])

        # raise NotImplementedError

    ds = Dataset(
        data=[data_dict],
        transform=all_transforms,
    )
    dl = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            collate_fn = list_data_collate
        )

    return dl, inference_settings


#########################################################################
# Loop through data and perform inference and create plots and metrics
#########################################################################
def do_inference(dl, inference_settings):

    NIFTI_RTSS = inference_settings['NIFTI_RTSS']
    NIFTI_IMAGE = inference_settings['NIFTI_IMAGE']
    DICOM_RTSS_OUTPUT = inference_settings['DICOM_RTSS_OUTPUT']
    DICOM_FOLDER = inference_settings['DICOM_FOLDER']
    NEW_SPACING = inference_settings['NEW_IMAGE_SPACING']
    # SAVEPATH_BASE = inference_settings['SAVEPATH_BASE']
    MODEL_PATH = inference_settings['MODEL_PATH']
    DICE_PATH = inference_settings['DICE_PATH']
    CALIBRATION_PATH = inference_settings['CALIBRATION_PATH']

    dicom_list = [f for f in glob.glob(f'{DICOM_FOLDER}/*dcm')]

    net = load_model(MODEL_PATH)

    inferer = SlidingWindowInferer(
        roi_size=inference_settings['ROI_SIZE'],
        overlap=inference_settings['OVERLAP'],
        mode='gaussian',
        sw_device='cpu', # only use if the gpu can handle the memory requirement!
        device='cpu'
    )

    def infer(samples, net, inferer, cuda=False):

        predictor = get_predictor(
            net, inferer, cuda=cuda
        )

        with torch.no_grad():
            logits = predictor(samples)
        probs = get_predictions_from_logits(logits=logits, mode='prob')
        preds = get_predictions_from_probs(probs, threshold=inference_settings['PRED_THRESHOLD'])

        return probs, preds
    
    for idx, batch in tqdm(enumerate(dl)):

        batch_unpacked = batch_to_sample(batch)  # samples have shape (BCWHD)
        if len(batch_unpacked) == 2:
            samples, kwargs = batch_unpacked
            LABELS = False
        elif len(batch_unpacked) == 3:
            samples, labels, kwargs = batch_unpacked
            LABELS = True
            print(f"samples.shape = {samples.shape}")
            print(f"labels.shape = {labels.shape}")
        # labels = kwargs['label']
        PATIENT_ID = kwargs['PATIENT_ID']

        probs, preds = infer(samples, net, inferer, cuda=False)
        print(f"preds.shape = {preds.shape}")

        # save pred as nifti
        ## save tensor to file
        rtss = check_tensor(preds[0,1,...])
        rtss_nii = nib.Nifti1Image(rtss, affine=np.eye(4))

        if DICOM_FOLDER is not None:
            # this is to get the header of the original dicom, incl affine matrix
            # with tempfile.TemporaryDirectory() as tmp_dir:
            #     tmpfile = os.path.join(tmp_dir, 'image.nii')
            #     dicom_image_sitk = load_dicom_images(DICOM_FOLDER, new_spacing=NEW_SPACING)
            #     save_itk_image_as_nifti_sitk(dicom_image_sitk, tmpfile)
            #     nifti_image_src = nib.load(tmpfile)
            #     nib_rtss = copy_nifti_header(nifti_image_src, rtss_nii)
            #     # save file 'to disk'
            #     nib.save(nib_rtss, NIFTI_RTSS)

            # the following could be used instead when you know that the nifti image already has the correct affine matrix, ie sampling was done on a file level, and not as a transformation during training
            nifti_image_src = nib.load(NIFTI_IMAGE) # this already carries the correct affine matrix
            nib_rtss = copy_nifti_header(nifti_image_src, rtss_nii)
            nib.save(nib_rtss, NIFTI_RTSS)

            # save pred as dicom rtss
            nifti2dicom.nifti_rtss_to_dicom_rtss(
                NIFTI_RTSS,
                DICOM_FOLDER,
                DICOM_RTSS_OUTPUT,
                inference_threshold=inference_settings['PRED_THRESHOLD'], # note that input is already binarised,
                new_spacing='original',
                dicom_list=dicom_list
            )
        else:
            # get affine from nifti image
            nifti_image_src = nib.load(NIFTI_IMAGE)
            nib_rtss = copy_nifti_header(nifti_image_src, rtss_nii)
            nib.save(nib_rtss, NIFTI_RTSS)

        # after this point there would be code to save results as nifti 
        # files, convert to dicom rtss, calculate metrics, and plot figures.
        if LABELS:
            try:
                dice = get_dice_scores(preds, labels)
                df_dice = pd.DataFrame({'PATIENT_ID':PATIENT_ID,
                                        'bkg':dice[:,0], 
                                        'fg':dice[:,1],
                                        })
                df_dice.to_csv(
                    DICE_PATH,
                    index=False, header=True
                )

                # calculate model calibration
                calibration_average = calculate_binned_model_calibration(
                    probs[0,1,...], labels[0,1,...], delta_p=0.1
                )
                df_calibration = pd.DataFrame(
                    calibration_average, columns=['prob_interval_upper', 'model accuracy']
                ).round(decimals=2)
                # always write new file (since we update the contents of calibration_res)
                df_calibration.to_csv(
                    CALIBRATION_PATH,
                    index=False, header=True
                )
            except Exception as e:
                print(f'Error calculating metrics: {e}')

def get_dice_scores(
    pred: torch.Tensor,
    ground_truth: torch.Tensor,
    add_one_hot: bool = False,
    # input_logits_prob: str,
    # threshold=None
) -> torch.Tensor:
    """Dice scores per batch and per class

    Args:
        pred (torch.Tensor): prediction [BC]
        ground_truth (torch.Tensor): _description_
        input_logits_prob (str): either 'logit' or 'prob'
        threshold (_type_, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: Dice scores per batch and per class, (shape [batch_size, num_classes]
    """
    # n_classes = pred.shape[1]

    # post_label_transform = get_post_label_transforms(n_classes, add_one_hot=add_one_hot)

    # ground_truth = [post_label_transform(gt) for gt in decollate_batch(ground_truth)]

    dice_metric = monai.metrics.DiceMetric(include_background=True, reduction="mean", get_not_nans=False) #obs dice for bkg is calculated as its own class, it does not affect the dice of the other classes! 
    # logging.info(f'type(ground_truth) = {type(ground_truth)}')
    # logging.info(f'type(ground_truth[0]) = {type(ground_truth[0])}')
    # logging.info(f'ground_truth.shape = {ground_truth.shape}')

    # logging.info(f'type(pred) = {type(pred)}')
    # logging.info(f'type(pred[0]) = {type(pred[0])}')

    return dice_metric(y_pred=pred, y=ground_truth)

def calculate_binned_model_calibration(prob, ground_truth, delta_p=0.1) -> List:
    p_i = 0
    calibration_results = []
    while p_i < 1-delta_p:
        mask = (prob > p_i) & (prob < p_i+delta_p)
        if mask.sum()>0:
            p_frac = ground_truth[mask].sum()/mask.sum()
        else:
            p_frac = np.nan
        # print(f'{(p_i):.1f} < p < {(p_i+delta_p):.1f} has a fraction {p_frac:.2f} foreground voxels')
        p_i += delta_p
        calibration_results.append([p_i, p_frac]) # p_i is now upper bound of interval
    return np.array(calibration_results)