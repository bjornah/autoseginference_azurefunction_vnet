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


from niftidicomconverter import dicom2nifti, nifti2dicom, dicomhandling_pydicom # the only custom package required required to run this script
# from niftidicomconverter.dicomhandling import load_dicom_images, save_itk_image_as_nifti_sitk
from niftidicomconverter.niftihandling import resample_nifti_to_new_spacing, reorient_nifti, resample_nifti_to_new_spacing_sitk
from niftidicomconverter.utils import axes_swapping

from metrics_and_loss import calculate_metrics

from transforms import get_standard_transform_variable_size

from inference_help_functions import get_predictor, check_tensor, batch_to_sample, load_model, get_predictions_from_logits, get_predictions_from_probs

import logger_config  # Import your logger configuration


# Get the logger
logger = logging.getLogger('local_logger')


#########################################################################
# Set paths and create folders
#########################################################################
def setup_inference(manifest):
    '''
    Inference on a single stack of dicom images
    '''

    OVERLAP = 0.5
    if 'device' in manifest.keys():
        DEVICE = manifest['device']
    else:
        DEVICE = 'cpu'
    
    if 'prediction_threshold' in manifest.keys():
        PRED_THRESHOLD = manifest['prediction_threshold']
    else:
        PRED_THRESHOLD = 0.5 # Threshold above which voxel is considered foreground
    if ('detection_threshold' in manifest.keys()) and ('binarization_threshold' in manifest.keys()):
        DETECTION_THRESHOLD = manifest['detection_threshold']
        BINARIZATION_THRESHOLD = manifest['binarization_threshold']
    else:
        DETECTION_THRESHOLD = 0.5
        BINARIZATION_THRESHOLD = 0.5

    SITE = 'CLVL' # the CLVL option is also compatible with Brats and NYU and PAH

    # MODEL_PATH = 'traced_model.pt'
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
    os.makedirs(SAVEPATH_BASE, exist_ok=True)

    if 'nifti_file' in manifest.keys():
        NIFTI_IMAGE = manifest['nifti_file']
    else:
        NIFTI_IMAGE = os.path.abspath(f'{BASE_DIR}/nifti_image.nii.gz')

    # if (DICOM_GT is not None):
    NIFTI_GT = os.path.abspath(f'{SAVEPATH_BASE}/nifti_label.nii.gz')
    NIFTI_GT_PROCESSED = os.path.abspath(f'{SAVEPATH_BASE}/nifti_label_processed.nii.gz')
    # else:
    #     NIFTI_GT = None

    NIFTI_RTSS_BIN = os.path.abspath(f'{SAVEPATH_BASE}/nifti_pred_binarised.nii.gz')
    NIFTI_RTSS_PROB = os.path.abspath(f'{SAVEPATH_BASE}/nifti_pred_prob.nii.gz')
    DICE_PATH_FINAL = os.path.abspath(f'{SAVEPATH_BASE}/dice.csv')
    CALIBRATION_PATH_FINAL = os.path.abspath(f'{SAVEPATH_BASE}/model_calibration.csv')
    METRICS_CSV_FINAL = os.path.abspath(f'{SAVEPATH_BASE}/metrics.csv')

    NIFTI_RTSS_BIN_list = []
    NIFTI_RTSS_PROB_list = []
    DICE_PATH_list = []
    CALIBRATION_PATH_list = []
    SAVEPATH_PARTS = []
    MODEL_PATHS = []
    ROI_SIZES = []
    CENTRE_CROP_SIZES = []

    model_settings = manifest['model_settings']

    for i,model_dict in enumerate(model_settings['models']):
        logger.info(f'collecting information about model {model_dict["name"]}')
        MODEL_PATH = model_dict['path']
        MODEL_PATHS.append(MODEL_PATH)

        ROI_SIZE = model_dict['patch_size']
        ROI_SIZES.append(ROI_SIZE)

        SAVEPATH_PART = os.path.abspath(f'{SAVEPATH_BASE}/inference_{i}')
        SAVEPATH_PARTS.append(SAVEPATH_PART)
        os.makedirs(SAVEPATH_PART, exist_ok=True)

        CENTRE_CROP_SIZE = model_dict['centre_crop_size']
        CENTRE_CROP_SIZES.append(CENTRE_CROP_SIZE)

        NIFTI_RTSS_BIN_i = os.path.abspath(f'{SAVEPATH_PART}/nifti_pred_binarised_{i}.nii.gz')
        NIFTI_RTSS_PROB_i = os.path.abspath(f'{SAVEPATH_PART}/nifti_pred_prob_{i}.nii.gz')
        DICE_PATH_i = os.path.abspath(f'{SAVEPATH_PART}/dice_{i}.csv')
        CALIBRATION_PATH_i = os.path.abspath(f'{SAVEPATH_PART}/model_calibration_{i}.csv')
        
        NIFTI_RTSS_BIN_list.append(NIFTI_RTSS_BIN_i)
        NIFTI_RTSS_PROB_list.append(NIFTI_RTSS_PROB_i)
        DICE_PATH_list.append(DICE_PATH_i)
        CALIBRATION_PATH_list.append(CALIBRATION_PATH_i)

    DICOM_RTSS_OUTPUT = os.path.abspath(f'{SAVEPATH_BASE}/dicom_pred.dcm')

    # DICE_FILE_NAME = 'dice.csv'
    # CALIBRATION_FILE_NAME = 'model_calibration.csv'

    inference_settings = {
        'ROI_SIZES': ROI_SIZES,
        'CENTRE_CROP_SIZES': CENTRE_CROP_SIZES,
        'OVERLAP': OVERLAP,
        'NEW_IMAGE_SPACING':NEW_IMAGE_SPACING,
        'PRED_THRESHOLD': PRED_THRESHOLD,
        'DETECTION_THRESHOLD': DETECTION_THRESHOLD,
        'BINARIZATION_THRESHOLD': BINARIZATION_THRESHOLD,
        'SAVEPATH_BASE': SAVEPATH_BASE,
        'SAVEPATH_PARTS': SAVEPATH_PARTS, 
        'MODEL_PATHS': MODEL_PATHS,
        'NIFTI_RTSS_BIN': NIFTI_RTSS_BIN,
        'NIFTI_RTSS_PROB': NIFTI_RTSS_PROB,
        'NIFTI_RTSS_BIN_list': NIFTI_RTSS_BIN_list,
        'NIFTI_RTSS_PROB_list': NIFTI_RTSS_PROB_list,
        'model_settings':model_settings,
        'NIFTI_IMAGE': NIFTI_IMAGE,
        'NIFTI_GT': NIFTI_GT,
        'NIFTI_GT_PROCESSED': NIFTI_GT_PROCESSED,
        'DICOM_FOLDER': DICOM_FOLDER,
        'DICOM_RTSS_OUTPUT': DICOM_RTSS_OUTPUT,
        'DICE_PATHS': DICE_PATH_list,
        'DICE_PATH_FINAL': DICE_PATH_FINAL,
        'CALIBRATION_PATHS': CALIBRATION_PATH_list,
        'CALIBRATION_PATH_FINAL': CALIBRATION_PATH_FINAL,
        'METRICS_CSV_FINAL': METRICS_CSV_FINAL,
        'DEVICE': DEVICE,
    }

    #########################################################################
    # Convert dicom to nifti
    #########################################################################
    if DICOM_FOLDER is not None:
        dicom2nifti.convert_dicom_to_nifti(DICOM_FOLDER, NIFTI_IMAGE) # function to convert dicom to nifti, as the name makes pretty clear
        
        logger.debug('before resampling')
        nifti_image = nib.load(NIFTI_IMAGE)
        logger.debug(f'nifti_image.shape = {nifti_image.get_fdata().shape}')
        logger.debug(f'nifti_image.affine = {nifti_image.affine}')

        # _ = resample_nifti_to_new_spacing(
        #     nifti_file_path=NIFTI_IMAGE,
        #     new_spacing=NEW_IMAGE_SPACING,
        #     save_path=NIFTI_IMAGE,
        #     interpolation_order=3,
        #     debug=True
        # ) # resample nifti to new spacing

        resample_nifti_to_new_spacing_sitk(
            nifti_file_path=NIFTI_IMAGE,
            new_spacing=NEW_IMAGE_SPACING,
            save_path=NIFTI_IMAGE,
        ) # resample nifti to new spacing

        # this reorients the nifti image to the standard dicom-style orientation. However, it deteriorates the quality of the predictions
        # nifti_nib = reorient_nifti(input_data=NIFTI_IMAGE, target_orientation=('A', 'R', 'S'), print_debug=True)
        # nib.save(nifti_nib, NIFTI_IMAGE)

        logger.debug('after resampling')
        nifti_image = nib.load(NIFTI_IMAGE)
        logger.debug(f'nifti_image.shape = {nifti_image.get_fdata().shape}')
        # print out the affine matrix of the nifti image
        logger.debug(f'nifti_image.affine = {nifti_image.affine}')

    # then, if there are labels do those:
    if DICOM_GT is not None:
        logger.info('converting DICOM RTSS to nifti')
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
            # dicom2nifti.dicom_rtss_to_nifti(rtss_path=DICOM_GT, reference_image_dir=DICOM_FOLDER, output_path=NIFTI_GT, structure_map=structure_map)

            logger.debug('before resampling')
            nifti_gt = nib.load(NIFTI_GT)
            logger.debug(f'nifti_gt.shape = {nifti_gt.get_fdata().shape}')
            logger.debug(f'nifti_gt.affine = {nifti_gt.affine}')
            # logger.debug(f'nifti header = {list(nifti_gt.header.items())}')

            # _ = resample_nifti_to_new_spacing(
            #     nifti_file_path=NIFTI_GT,
            #     new_spacing=NEW_IMAGE_SPACING,
            #     save_path=NIFTI_GT,
            #     interpolation_order=0,
            #     debug=True
            # ) # resample nifti to new spacing

            resample_nifti_to_new_spacing_sitk(
                nifti_file_path=NIFTI_GT,
                new_spacing=NEW_IMAGE_SPACING,
                save_path=NIFTI_GT,
            ) # resample nifti to new spacing

            # this reorients the labels to the standard dicom-style orientation. This should only be done if you also transpose the image. However, it deteriorates the quality of the predictions
            # nifti_nib = reorient_nifti(input_data=NIFTI_GT, target_orientation=('A', 'R', 'S'), print_debug=True)
            # nib.save(nifti_nib, NIFTI_GT)

            logger.debug('after resampling')
            nifti_gt = nib.load(NIFTI_GT)
            logger.debug(f'nifti_gt.shape = {nifti_gt.get_fdata().shape}')
            logger.debug(f'nifti_gt.affine = {nifti_gt.affine}')
            # logger.debug(f'nifti header = {list(nifti_gt.header.items())}')

        except Exception as e:
            logger.exception(f'Error converting RTSS: {e}')
            raise e

        nifti_data = {
            'image':NIFTI_IMAGE,
            'label':NIFTI_GT
        }

    else:
        logger.info('No DICOM RTSS to convert to nifti. No ground truth in data.')
        nifti_data = {
            'image':NIFTI_IMAGE,
        }

    #########################################################################
    # Create data loader
    #########################################################################
    data_dict = nifti_data
    data_dict['site'] = SITE
    data_dict['PATIENT_ID'] = PATIENT_ID
    DATALOADERS = []

    for i,model_dict in enumerate(model_settings['models']):
        CENTRE_CROP_SIZE = model_dict['centre_crop_size']
        # print(f'CENTRE_CROP_SIZE = {CENTRE_CROP_SIZE}')

        if DICOM_GT is None:
            print(nifti_data)
            all_transforms = monai.transforms.Compose([
                monai.transforms.LoadImaged(keys=['image']),
                monai.transforms.EnsureChannelFirstd(keys=['image']),
                # monai.transforms.Spacingd(
                #     keys=['image'],
                #     pixdim=(1.0, 1.0, 1.0),
                #     mode=("bilinear"),
                # ),
                monai.transforms.ScaleIntensityRangePercentilesd(
                    keys=["image"],
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
                    'centre_crop_size':CENTRE_CROP_SIZE,
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
        DATALOADERS.append(dl)

    inference_settings['dataloaders'] = DATALOADERS

    return inference_settings

#########################################################################
# Loop through data and perform inference and create plots and metrics
#########################################################################

def do_single_inference(dl, inference_settings, MODEL_PATH, ROI_SIZE, NIFTI_RTSS_BIN, NIFTI_RTSS_PROB, DICE_PATH, CALIBRATION_PATH):

    NIFTI_IMAGE = inference_settings['NIFTI_IMAGE']
    DICOM_RTSS_OUTPUT = inference_settings['DICOM_RTSS_OUTPUT']
    DICOM_FOLDER = inference_settings['DICOM_FOLDER']
    # NEW_SPACING = inference_settings['NEW_IMAGE_SPACING']
    DEVICE = inference_settings['DEVICE']

    dicom_list = [f for f in glob.glob(f'{DICOM_FOLDER}/*dcm')]

    net = load_model(MODEL_PATH)

    inferer = SlidingWindowInferer(
        roi_size=ROI_SIZE,
        overlap=inference_settings['OVERLAP'],
        mode='gaussian',
        sw_device=DEVICE, # only use if the gpu can handle the memory requirement!
        device='cpu'
    )

    def infer(samples, net, inferer, device='cpu'):

        predictor = get_predictor(
            net, inferer, device=device
        )

        with torch.no_grad():
            logits = predictor(samples)
        logits = check_tensor(logits)
        probs = get_predictions_from_logits(logits=logits, mode='prob')
        preds = get_predictions_from_probs(probs, threshold=inference_settings['PRED_THRESHOLD'])

        return probs, preds
    
    for idx, batch in tqdm(enumerate(dl)):

        batch_unpacked = batch_to_sample(batch)  # samples have shape (BCWHD)
        if len(batch_unpacked) == 2:
            samples, kwargs = batch_unpacked
            labels = None
        elif len(batch_unpacked) == 3:
            samples, labels, kwargs = batch_unpacked
            # labels = check_tensor(labels)
            logger.info(f"samples.shape = {samples.shape}")
            logger.info(f"labels.shape = {labels.shape}")
        else:
            raise ValueError(f"batch_unpacked has length {len(batch_unpacked)}")
        
        PATIENT_ID = kwargs['PATIENT_ID']

        # samples = samples.to(torch.device(DEVICE))
        probs, preds = infer(samples, net, inferer, device=DEVICE)
        logger.debug(f"preds.shape = {preds.shape}")

        # save pred and prob as nifti files
        for res_tensor, NIFTI_FILE in zip([preds, probs],[NIFTI_RTSS_BIN, NIFTI_RTSS_PROB]):

            rtss = check_tensor(res_tensor[0,1,...])
            # get affine and header from nifti image
            nifti_image_src = nib.load(NIFTI_IMAGE)
            rtss_nii = nib.Nifti1Image(rtss, affine=nifti_image_src.affine, header=nifti_image_src.header)

            # print('during do_single_inference, when copying header for nifti files:')
            # print(f'nifti_image_src.affine = {nifti_image_src.affine}')
            # print(f'rts_nii.affine = {rtss_nii.affine}')
            # nib_rtss = copy_nifti_header(nifti_image_src, rtss_nii)
            # nib_rtss.affine = nifti_image_src.affine
            # print(f'nib_rtss.affine = {nib_rtss.affine}')
            # print('nib_rtss should have the same header and affine as nifti_image_src, and the same data as rtss_nii')
            nib.save(rtss_nii, NIFTI_FILE)

            logger.debug(f'during do_single_inference for PatientID = {PATIENT_ID}:')
            logger.debug(f'rtss_nii.affine = {rtss_nii.affine}')
            logger.debug(f'rtss.shape = {rtss.shape}')

        # commenting this out temporarily 2024-04-17 since the nifti to dicom rtss conversion does not seem to want to run.
        # It is complaining about 'ITK ERROR: ITK only supports orthonormal direction cosines. No orthonormal definition found!'
        # when trying to read the nifti rtss file.
        # if DICOM_FOLDER is not None:
            # this is to get the header of the original dicom, incl affine matrix
            # with tempfile.TemporaryDirectory() as tmp_dir:
            #     tmpfile = os.path.join(tmp_dir, 'image.nii')
            #     dicom_image_sitk = load_dicom_images(DICOM_FOLDER)
            #     save_itk_image_as_nifti_sitk(dicom_image_sitk, tmpfile)
            #     nifti_image_src = nib.load(tmpfile)
            #     nib_rtss = copy_nifti_header(nifti_image_src, rtss_nii)
            #     # save file 'to disk'
            #     nib.save(nib_rtss, NIFTI_RTSS_BIN)
                # print('during do_single_inference, when using dicom stack to get affine right:')
                # print(f'nifti_image_src.affine = {nifti_image_src.affine}')
                # print(f'nib_rtss.affine = {nib_rtss.affine}')

            # save pred as dicom rtss # do not do this for intermittent steps
            # print(f'in do_single_inference for PATIENT_ID = {PATIENT_ID}, nifti2dicom.nifti_rtss_to_dicom_rtss saving NIFTI_RTSS_BIN into DICOM_RTSS_OUTPUT')
            # try:
            #     nifti2dicom.nifti_rtss_to_dicom_rtss(
            #         NIFTI_RTSS_BIN,
            #         DICOM_FOLDER,
            #         DICOM_RTSS_OUTPUT,
            #         inference_threshold=inference_settings['PRED_THRESHOLD'], # note that input is already binarised,
            #         new_spacing='original',
            #         dicom_list=dicom_list
            #     )
            # except Exception as e:
            #     print(f'Error converting nifti to dicom RTSS for patient {PATIENT_ID}: {e}')

        # if labels is not None:
        #     try:
        #         # preds = check_tensor(nib.load(NIFTI_RTSS_BIN).get_fdata())
        #         # probs = check_tensor(nib.load(NIFTI_RTSS_PROB).get_fdata())
        #         logger.info(f'inside dice calculation for PATIENT_ID={PATIENT_ID} inside do_single_inference')
                
        #         print(f'labels.shape = {labels.shape}')
        #         print(f'preds.shape = {preds.shape}')

        #         dice = get_dice_scores(preds, labels)
        #         print(f'dice = {dice}')
        #         df_dice = pd.DataFrame({'PATIENT_ID':PATIENT_ID,
        #                                 'bkg':dice[:,0], 
        #                                 'fg':dice[:,1],
        #                                 })
        #         df_dice.to_csv(
        #             DICE_PATH,
        #             index=False, header=True
        #         )

        #         # calculate model calibration
        #         calibration_average = calculate_binned_model_calibration(
        #             probs[0,1,...], labels[0,1,...], delta_p=0.1
        #         )
        #         df_calibration = pd.DataFrame(
        #             calibration_average, columns=['prob_interval_upper', '% of correct foreground predictions']
        #         ).round(decimals=2)
        #         # always write new file (since we update the contents of calibration_res)
        #         df_calibration.to_csv(
        #             CALIBRATION_PATH,
        #             index=False, header=True
        #         )
        #     except Exception as e:
        #         print(f'Error calculating metrics in do_single_inference: {e}')


    return labels, PATIENT_ID
##################################################################################################


#########################################################################
# Loop through data and perform inference and create plots and metrics
#########################################################################
def do_inference(inference_settings):

    dataloaders = inference_settings['dataloaders']

    NIFTI_RTSS_PROB = inference_settings['NIFTI_RTSS_PROB']
    NIFTI_RTSS_BIN = inference_settings['NIFTI_RTSS_BIN']

    NIFTI_IMAGE = inference_settings['NIFTI_IMAGE']
    NIFTI_GT_PROCESSED = inference_settings['NIFTI_GT_PROCESSED']
    DICOM_RTSS_OUTPUT = inference_settings['DICOM_RTSS_OUTPUT']
    DICOM_FOLDER = inference_settings['DICOM_FOLDER']

    SAVEPATH_BASE = inference_settings['SAVEPATH_BASE']
    SAVEPATH_PARTS = inference_settings['SAVEPATH_PARTS']
    DICE_PATH_list = inference_settings['DICE_PATHS']
    CALIBRATION_PATH_list = inference_settings['CALIBRATION_PATHS']
    DICE_PATH_FINAL = inference_settings['DICE_PATH_FINAL']
    CALIBRATION_PATH_FINAL = inference_settings['CALIBRATION_PATH_FINAL']
    METRICS_CSV_FINAL = inference_settings['METRICS_CSV_FINAL']
    # DICE_FILE_NAME = inference_settings['DICE_FILE_NAME']
    # DICE_PATH_FINAL = os.path.abspath(SAVEPATH_BASE, DICE_FILE_NAME)
    # CALIBRATION_FILE_NAME = inference_settings['CALIBRATION_FILE_NAME']
    # CALIBRATION_PATH_FINAL = os.path.abspath(SAVEPATH_BASE, CALIBRATION_FILE_NAME)

    dicom_list = [f for f in glob.glob(f'{DICOM_FOLDER}/*dcm')]

    NIFTI_RTSS_BIN = inference_settings['NIFTI_RTSS_BIN']
    NIFTI_RTSS_PROB = inference_settings['NIFTI_RTSS_PROB']
    # model_settings = inference_settings['model_settings']
    NIFTI_RTSS_BIN_list = inference_settings['NIFTI_RTSS_BIN_list']
    NIFTI_RTSS_PROB_list = inference_settings['NIFTI_RTSS_PROB_list']
    MODEL_PATHS = inference_settings['MODEL_PATHS']
    ROI_SIZES = inference_settings['ROI_SIZES']

    # logger.info('check length of all lists that we iterate over for each inference step')
    # for l in [dataloaders, MODEL_PATHS, ROI_SIZES, NIFTI_RTSS_BIN_list, NIFTI_RTSS_PROB_list, DICE_PATH_list, CALIBRATION_PATH_list]:
    #     logger.info(len(l))
    logger.info('starting inference loop')
    for (dl, MODEL_PATH, ROI_SIZE, NIFTI_RTSS_BIN_i, NIFTI_RTSS_PROB_i, DICE_PATH, CALIBRATION_PATH) in zip(dataloaders, MODEL_PATHS, ROI_SIZES, NIFTI_RTSS_BIN_list, NIFTI_RTSS_PROB_list, DICE_PATH_list, CALIBRATION_PATH_list):
        logger.info(f'performing inference for model {MODEL_PATH}')
        labels, PATIENT_ID = do_single_inference(
            dl=dl,
            inference_settings=inference_settings,
            MODEL_PATH=MODEL_PATH,
            ROI_SIZE=ROI_SIZE,
            NIFTI_RTSS_BIN=NIFTI_RTSS_BIN_i,
            NIFTI_RTSS_PROB=NIFTI_RTSS_PROB_i,
            DICE_PATH=DICE_PATH,
            CALIBRATION_PATH=CALIBRATION_PATH
        )

    # save labels as processed nifti_gt file
    if labels is not None:
        rtss = check_tensor(labels[0,1,...])
        # get affine and header from nifti image
        nifti_image_src = nib.load(NIFTI_IMAGE)
        rtss_nii = nib.Nifti1Image(rtss, affine=nifti_image_src.affine, header=nifti_image_src.header)

        nib.save(rtss_nii, NIFTI_GT_PROCESSED)

    # next we load all tensors from all NIFTI_RTSS_PROB_list into a single tensor
    
    # probs = torch.tensor(np.array([nib.load(f).get_fdata() for f in NIFTI_RTSS_PROB_list]))
    # probs = torch.stack(probs)
    # preds = torch.stack([nib.load(f).get_fdata() for f in NIFTI_RTSS_BIN_list])

    # Load the NIFTI files and get the data arrays
    prob_arrays = [nib.load(file).get_fdata() for file in NIFTI_RTSS_PROB_list]
    # Convert the data arrays to PyTorch tensors
    prob_tensors = [torch.from_numpy(data) for data in prob_arrays]
    # Stack the tensors along the first dimension
    probs = torch.stack(prob_tensors, dim=0)

    # average over the probabilities from each model in the probs tensor
    probs_avg = probs.mean(dim=0)
    
    # save the average probabilities as a nifti file
    rtss = check_tensor(probs_avg)
    # rtss_nii = nib.Nifti1Image(rtss, affine=np.eye(4))
    # get affine and header from nifti image
    nifti_image_src = nib.load(NIFTI_IMAGE)
    rtss_nii = nib.Nifti1Image(rtss, affine=nifti_image_src.affine, header=nifti_image_src.header)
    # nib_rtss = copy_nifti_header(nifti_image_src, rtss_nii)
    # nib_rtss.affine = nifti_image_src.affine
    logger.debug('during do_inference (probs_avg):')
    logger.debug(f'rtss_nii.affine = {rtss_nii.affine}')
    logger.debug(f'probs_avg.shape = {probs_avg.shape}')
    
    nib.save(rtss_nii, NIFTI_RTSS_PROB)

    # binarise combined probabilities and save to nifti file
    # preds_avg = get_predictions_from_probs(probs_avg, threshold=inference_settings['PRED_THRESHOLD'])
    preds_avg = monai.transforms.AsDiscrete(threshold=inference_settings['PRED_THRESHOLD'])(probs_avg)
    rtss = check_tensor(preds_avg)
    # get affine and header from nifti image
    nifti_image_src = nib.load(NIFTI_IMAGE)
    rtss_nii = nib.Nifti1Image(rtss, affine=nifti_image_src.affine, header=nifti_image_src.header)
    # logger.debug(f'\n\nnifti header = {list(nifti_image_src.header.items())}\n\n')
    # rtss_nii = nib.Nifti1Image(rtss, affine=np.eye(4))
    # get affine from nifti image and save results
    # nifti_image_src = nib.load(NIFTI_IMAGE)
    # nib_rtss = copy_nifti_header(nifti_image_src, rtss_nii)
    # nib_rtss.affine = nifti_image_src.affine
    nib.save(rtss_nii, NIFTI_RTSS_BIN)

    logger.debug('during do_inference (preds_avg):')
    logger.debug(f'rtss_nii.affine = {rtss_nii.affine}')
    logger.debug(f'preds_avg.shape = {preds_avg.shape}')
    # print(f'nifti_image_src.affine = {nifti_image_src.affine}')
    # print(f'nib_rtss.affine = {nib_rtss.affine}')
    # print('nib_rtss should have the same affine as nifti_image_src and the same data as rtss_nii')

    if DICOM_FOLDER is not None:
        # this is to get the header of the original dicom, incl affine matrix
        # with tempfile.TemporaryDirectory() as tmp_dir:
        #     tmpfile = os.path.join(tmp_dir, 'image.nii')
        #     dicom_image_sitk = load_dicom_images(DICOM_FOLDER)
        #     save_itk_image_as_nifti_sitk(dicom_image_sitk, tmpfile)
        #     nifti_image_src = nib.load(tmpfile)
        #     nib_rtss = copy_nifti_header(nifti_image_src, rtss_nii)
        #     # save file 'to disk'
        #     nib.save(nib_rtss, NIFTI_RTSS_BIN)

        # save pred as dicom rtss
        logger.info(f'in do_inference for PATIENT_ID = {PATIENT_ID}, nifti2dicom.nifti_rtss_to_dicom_rtss saving NIFTI_RTSS_BIN into DICOM_RTSS_OUTPUT')
        try:
            nifti2dicom.nifti_rtss_to_dicom_rtss(
                NIFTI_RTSS_BIN,
                DICOM_FOLDER,
                DICOM_RTSS_OUTPUT,
                inference_threshold=inference_settings['PRED_THRESHOLD'], # note that input is already binarised,
                new_spacing='original',
                dicom_list=dicom_list,
                roi_colors=[255, 0, 0] # all red contours
            )
            # nifti2dicom.prob_map_to_dicom_rtss(
            #     NIFTI_RTSS_PROB, # OBS, using probs_avg instead of preds_avg so that we can use detection and binarization thresholds instead of a flat threshold
            #     DICOM_FOLDER,
            #     DICOM_RTSS_OUTPUT,
            #     detection_threshold=inference_settings['DETECTION_THRESHOLD'],
            #     binarization_threshold=inference_settings['BINARIZATION_THRESHOLD'],
            #     roi_colors=[255, 0, 0] # all red contours
            # )


            # dicomhandling_pydicom.update_contours_in_dicom_file(
            #     DICOM_RTSS_OUTPUT,
            #     window_length=9,
            #     polyorder=3,
            #     upsample_factor=3,
            #     upsampling_kind='linear'
            # )


        except Exception as e:
            logger.exception(f'Error converting combined nifti rtss to dicom RTSS for patient {PATIENT_ID}: {e}')


    if labels is not None:
        try:
            # preds = check_tensor(nib.load(NIFTI_RTSS_BIN).get_fdata())
            # probs = check_tensor(nib.load(NIFTI_RTSS_PROB).get_fdata())
            # labels = check_tensor(nib.load(inference_settings['NIFTI_GT']).get_fdata())
            logger.debug(f'inside dice calculation for PATIENT_ID={PATIENT_ID} inside do_inference')
            logger.debug('check if labels have been smoothed or not')
            logger.debug(f"labels.min = {labels.min()}")
            logger.debug(f"labels.max = {labels.max()}")
            
            # add a channel and batch dimension to labels and preds_avg
            
            preds_avg = preds_avg.unsqueeze(0).unsqueeze(0) # turn preds_avg back into [BCHWD]. Note that labels is still in this dimension
            labels = labels[0,1,...] # choose channel 1, which is the only channel remaining in preds_avg
            labels = labels.unsqueeze(0).unsqueeze(0) # turn labels back into [BCHWD]. 
            logger.debug(f'preds_avg.shape = {preds_avg.shape}')
            logger.debug(f'labels.shape = {labels.shape}')
            save_metrics(preds_avg, labels, METRICS_CSV_FINAL, PATIENT_ID)

            dice = get_dice_scores(preds_avg, labels).squeeze().numpy()
            logger.info(f'patient = {PATIENT_ID}\ndice = {dice}')
            df_dice = pd.DataFrame({'PATIENT_ID':PATIENT_ID,
                                    'fg':dice,
                                    })
            df_dice.to_csv(
                DICE_PATH_FINAL,
                index=False, header=True
            )

            # calculate model calibration
            calibration_average = calculate_binned_model_calibration(
                probs_avg.squeeze(), labels.squeeze(), delta_p=0.1
            )
            df_calibration = pd.DataFrame(
                calibration_average, columns=['prob_interval_upper', '% of correct foreground predictions']
            ).round(decimals=2)
            # always write new file (since we update the contents of calibration_res)
            df_calibration.to_csv(
                CALIBRATION_PATH_FINAL,
                index=False, header=True
            )
        except Exception as e:
            logger.exception(f'Error calculating metrics in do_inferece: {e}')


def save_metrics(pred, gt, output_csv_path, patient_id):
    dice, cf = calculate_metrics(pred, gt)
    cf = list(cf)
    # Sum the confusion matrices
    confusion_matrices = np.array(cf)

    # Save the results to a DataFrame and then to a CSV file
    precision = confusion_matrices[0] / (confusion_matrices[0] + confusion_matrices[1])
    df = pd.DataFrame({
        'patient_id': patient_id,
        'Dice Score': dice,
        'numLesions': confusion_matrices[0] + confusion_matrices[2],
        'TP': confusion_matrices[0],
        'FP': confusion_matrices[1],
        'FN': confusion_matrices[2],
        'TPR/sensitivity/recall': confusion_matrices[3],
        'FNR/miss_rate': confusion_matrices[4],
        'Recall': confusion_matrices[3],
        'Precision': precision,
        'F1': (2 * precision * confusion_matrices[3]) / (precision + confusion_matrices[3])
    })

    # Round all elements of df to 2dp
    df = df.round(2)

    df.to_csv(output_csv_path, index=False)

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

    dice_metric = monai.metrics.DiceMetric(include_background=True, reduction="mean", get_not_nans=False) #obs dice for bkg is calculated as its own class, it does not affect the dice of the other classes! 

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