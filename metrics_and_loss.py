# %% Libraries

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from medpy.metric.binary import hd, hd95, assd

# %% 

def calculate_metrics(pred, mask):
    metric = Metrics(pred, mask, mask_channels=[0], threshold=True)
    dice = metric.dice()[0].item()
    cf = metric.per_lesion_cf()
    return dice, cf

class Metrics:
    """
    Calculate metrics between a batch of ground truth and prediction masks.

    Parameters
    ----------
    pred : tensor
        Prediction masks batch of shape (B, C, H, W, D)
    mask : tensor
        Ground truth masks batch of shape (B, C, H, W, D)
    mask_channels : list
        List of mask channels.
    threshold : bool, optional
        If True, the prediction and ground truth masks are thresholded to 0 and 1.
        The default is False.
    exclude_bg_from_mean : bool, optional
        If True, the background channel is excluded from the mean dice calculation.
        The default is False.

    Methods
    -------
    dice()
        Calculate the dice coefficient for all mask channels.
    hd95()
        Calculate the 95th percentile Hausdorff distance for all mask channels.
    msd()
        Calculate the mean symmetric distance for all mask channels.
    per_lesion_cf()
        Calculate the per-lesion confusion matrix.
    count_overlapping_masks()
        Count the number of overlapping masks.
    """
    def __init__(
        self, 
        pred, 
        mask, 
        mask_channels=None, 
        threshold=False, 
        exclude_bg_from_mean=False,
        dimensions='3D'
    ):
        self.pred = pred
        self.mask = mask
        self.mask_channels = mask_channels
        self.threshold = threshold
        self.exclude_bg_from_mean=exclude_bg_from_mean
        self.dimensions = dimensions

        # If numpy convert to torch
        if isinstance(self.pred, np.ndarray):
            self.pred = torch.from_numpy(self.pred)
        if isinstance(self.mask, np.ndarray):
            self.mask = torch.from_numpy(self.mask)
        
    def dice(self): 
        """
        Calculate the dice coefficient for all mask channels

        Returns
        -------
        dice_mean : float
            Mean dice coefficient across the mask_channels. If exclude_bg_from_mean
            is True, the background channel is excluded from the mean dice calculation.
        dice_all_channels : list
            List of dice coefficients for each mask_channel.
        dice_channels_std : float
            Standard deviation of dice coefficients across all mask_channels.
            Can be used for error bars in a bar chart.
        """ 
        # Define initial variables
        EPSILON = 1e-6
        dice_all_channels = []

        if self.dimensions == '2D' and self.pred.dim() == 2:
            self.pred = self.pred.unsqueeze(0).unsqueeze(0)
        if self.dimensions == '2D' and self.mask.dim() == 2:
            self.mask = self.mask.unsqueeze(0).unsqueeze(0)

        # Loop through channels
        for c in range(len(self.mask_channels)):

            # Extract a single channel from all batches
            if self.dimensions == '3D':
                pred_ch = self.pred[:, c, :, :, :]
                mask_ch = self.mask[:, c, :, :, :]
            elif self.dimensions == '2D':
                pred_ch = self.pred[:, c, :, :]
                mask_ch = self.mask[:, c, :, :]

            # Threshold
            if self.threshold == True:
                pred_ch = torch.where(pred_ch>0.5, 1, 0)
                mask_ch = torch.where(mask_ch>0.5, 1, 0)

            # Calculate intersection
            intersection = torch.sum(pred_ch * mask_ch)

            # Calculate union
            union = torch.sum(pred_ch) + torch.sum(mask_ch)

            # Calculate dice coefficient
            dice_coeff = (2*intersection + EPSILON) / (union + EPSILON)

            # Append dice coefficient to list of dice coefficients for each channel across all batches
            dice_all_channels.append(dice_coeff)

        # Calculate mean dice across all mask_channels
        # self.exclude_bg_from_mean=False $ For BRATS
        if self.exclude_bg_from_mean:
            dice_mean = sum(dice_all_channels[1:]) / (len(self.mask_channels) - 1)
        else:
            dice_mean = sum(dice_all_channels) / len(self.mask_channels)

        # Round dice coefficients to 3 decimal places
        dice_all_channels = [round(x.item(), 3) for x in dice_all_channels]

        # Calculate standard deviation of dice coefficients across all mask_channels
        dice_channels_std = torch.std(torch.tensor(dice_all_channels)).item()

        return dice_mean, dice_all_channels, dice_channels_std
            
    def hd95(self, voxel_spacing, hd95=True):

        # Threshold
        self.pred[self.pred<=0.5] = 0
        self.pred[self.pred>0.5] = 1

        batch_size = self.pred.size(0)
        hd_channels = np.array([0]*(len(self.mask_channels)-1), dtype=np.float32)

        if self.exclude_bg_from_mean == True: # 0 to include background, 1 to exclude background
            start = 1
        else:
            start = 0   

        for total_batches, b in enumerate(range(batch_size), 1):

            hd_channels_batch = []
            for c in range(start, len(self.mask_channels)):

                # Convert tensors to numpy arrays
                output_np = self.pred[b, c].cpu().numpy()
                target_np = self.mask[b, c].cpu().numpy()

                # Convert to Boollean
                output_np = output_np.astype(bool)
                target_np = target_np.astype(bool)

                # Compute Hausdorff distance                    
                if np.sum(output_np) > 0 and np.sum(target_np) > 0:
                    # Calculate directed Hausdorff distances
                    if hd95 == False:
                        hd1 = hd(result = output_np, reference = target_np, voxelspacing=voxel_spacing)
                        hd2 = hd(result = target_np, reference = output_np, voxelspacing=voxel_spacing)
                    else:
                        hd1 = hd95(result = output_np, reference = target_np, voxelspacing=voxel_spacing)
                        hd2 = hd95(result = target_np, reference = output_np, voxelspacing=voxel_spacing)
                    # Take the maximum of the two directed distances
                    hd_distance = max(hd1, hd2)
                    hd_channels_batch.append(hd_distance)
                else:
                    hd_channels_batch.append(0)

            hd_channels += np.array(hd_channels_batch, dtype=np.float32)

        hd_channels = hd_channels/total_batches
        mean_hd = np.mean(hd_channels)

        return mean_hd, hd_channels

    def msd(self, gpu=True, voxel_spacing=(1,1,1)):
        if gpu == True:
                pred, mask = self.pred.to('cuda'), self.mask.to('cuda')
        else:
            pred, mask = self.pred.cpu(), self.mask.cpu()

        # Threshold
        pred[pred<=0.5] = 0
        pred[pred>0.5] = 1

        batch_size = pred.size(0)
        assd_channels = np.array([0]*(len(self.mask_channels)-1), dtype=np.float32)

        if self.exclude_bg_from_mean == True: # 0 to include background, 1 to exclude background
            start = 1
        else:
            start = 0   

        for total_batches, b in enumerate(range(batch_size), 1):

            assd_channels_batch = []
            for c in range(start, len(self.mask_channels)):

                # Convert tensors to numpy arrays
                output_np = pred[b, c].cpu().numpy()
                target_np = mask[b, c].cpu().numpy()

                # Convert to Boollean
                output_np = output_np.astype(bool)
                target_np = target_np.astype(bool)

                # Compute Hausdorff distance                    
                if np.sum(output_np) > 0 and np.sum(target_np) > 0:
                    # Calculate directed Hausdorff distances
                    assd1 = assd(output_np, target_np, voxelspacing=voxel_spacing)
                    assd_channels_batch.append(assd1)
                else:
                    assd_channels_batch.append(0)

            assd_channels += np.array(assd_channels_batch, dtype=np.float32)

        assd_channels = assd_channels/total_batches
        mean_assd = np.mean(assd_channels)

        return mean_assd, assd_channels
    
    def per_lesion_cf(self):

        TP = 0
        FP = 0
        FN = 0
        total_true_masks = 0
        total_pred_masks = 0

        for i in range(0, len(self.pred)):
            
            mask1_data = self.mask[i]
            mask2_data = self.pred[i]
            
            # Calculate the number of overlapping ROIs
            num_rois1, num_rois2, num_overlap_rois = self.count_overlapping_masks(mask1_data, mask2_data)

            total_true_masks += num_rois1
            total_pred_masks += num_rois2

            TP += num_overlap_rois
            FP += num_rois2 - num_overlap_rois 
            FN += num_rois1 - num_overlap_rois

        # Calculate the TP and FP rates
        TP_rate = TP / (TP + FN) # predicted and real
        FN_rate = FN / (FN + TP) # not predicted but real

        # Print the results
        print(f'Total ground truth masks: {total_true_masks}')
        print(f'Total predicted masks: {total_pred_masks}')
        print(f'True positives: {TP}')
        print(f'False positives: {FP}')
        print(f'False negatives: {FN}')
        print(f'TP rate (sensitivity): {TP_rate:.3f}')
        print(f'FN rate: {FN_rate:.3f}')

        # Return cf
        return TP, FP, FN, TP_rate, FN_rate

    def count_overlapping_masks(self, mask1, mask2):

        # Combine the masks
        combined_masks = mask1 + mask2
        combined_masks[combined_masks != 2] = 0
        combined_masks[combined_masks == 2] = 1
        
        # Count the rois in each mask and the overlapping mask
        _, num_rois1 = ndimage.label(mask1)
        _, num_rois2 = ndimage.label(mask2)
        _, num_overlap_rois = ndimage.label(combined_masks)
        
        return num_rois1, num_rois2, num_overlap_rois
    

# Dice loss (non-regularised)    
class DiceLoss(nn.Module):
    """
    Dice loss for a single mask_channel.

    Parameters
    ----------
    dice : float
        Dice coefficient.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, dice):

        # Calculate dice loss
        dice_loss = 1 - dice
        
        return dice_loss