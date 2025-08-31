import os
import io
import blobfile as bf
import torch as th
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from ddpm import Unet3D, GaussianDiffusion_Nolatent
from Dataset.TS_Dataset import get_TS_dataloader
from Dataset.MMWHS_Dataset import get_MMWHS_dataloader
import torchio as tio
from omegaconf import DictConfig
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
import atexit
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage as ndimage
from scipy.ndimage import distance_transform_edt

def dev(device):
    if device is None:
        if th.cuda.is_available():
            return th.device(f"cuda")
        return th.device("cpu")
    return th.device(device)


def load_state_dict(path, backend=None, **kwargs):
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)

try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass

def get_dice(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.reshape(preds.shape[0], -1)
    target = labels.reshape(labels.shape[0], -1)
    if np.sum(target) == 0 and np.sum(predict) == 0:
        return 1.0
    else:
        num = np.sum(np.multiply(predict, target), axis=1)
        den = np.sum(predict, axis=1) + np.sum(target, axis=1)
        dice = 2 * num / den
        return dice.mean()

def ignore_background(y_pred: torch.Tensor, y: torch.Tensor):
    return y_pred[:, 1:], y[:, 1:]

def prepare_spacing(spacing, batch_size, img_dim):
    if spacing is None:
        spacing = tuple([1.0] * img_dim)
    if isinstance(spacing, (int, float)):
        spacing = tuple([float(spacing)] * img_dim)
    elif isinstance(spacing, (tuple, list)):
        if len(spacing) == 1:
            spacing = tuple([float(spacing[0])] * img_dim)
        elif len(spacing) == img_dim:
            spacing = tuple(float(s) for s in spacing)
        else:
            raise ValueError("spacing should be a number or sequence of numbers matching image dimensions")
    return [spacing] * batch_size

def get_edge_surface_distance(pred, gt, distance_metric="euclidean", spacing=None, use_subvoxels=False, symmetric=True, class_index=None):
    # Convert tensors to numpy arrays and ensure boolean type
    pred = pred.cpu().numpy().astype(bool)
    gt = gt.cpu().numpy().astype(bool)
    
    # Get surface voxels using boolean operations
    edges_pred = ndimage.binary_dilation(pred).astype(bool) ^ pred
    edges_gt = ndimage.binary_dilation(gt).astype(bool) ^ gt
    
    # Compute distance transforms
    if distance_metric == "euclidean":
        dt_pred = distance_transform_edt(~edges_pred, sampling=spacing)
        dt_gt = distance_transform_edt(~edges_gt, sampling=spacing)
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")
    
    # Get surface distances
    distances_pred_gt = dt_gt[edges_pred]
    distances_gt_pred = dt_pred[edges_gt]
    
    if use_subvoxels:
        areas = None  # Simplified version without subvoxel precision
    else:
        areas = None
        
    return (edges_pred, edges_gt), (distances_pred_gt, distances_gt_pred), areas


def compute_surface_dice(y_pred, y, class_thresholds, include_background=False, 
                       distance_metric="euclidean", spacing=None, use_subvoxels=False):
    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)

    if not isinstance(y_pred, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise ValueError("y_pred and y must be PyTorch Tensor.")

    if y_pred.ndimension() not in (4, 5) or y.ndimension() not in (4, 5):
        raise ValueError("y_pred and y should be one-hot encoded: [B,C,H,W] or [B,C,H,W,D].")

    if y_pred.shape != y.shape:
        raise ValueError(
            f"y_pred and y should have same shape, but instead, shapes are {y_pred.shape} (y_pred) and {y.shape} (y)."
        )

    batch_size, n_class = y_pred.shape[:2]
    img_dim = y_pred.ndim - 2
    spacing_list = prepare_spacing(spacing=spacing, batch_size=batch_size, img_dim=img_dim)
    nsd = torch.empty((batch_size, n_class), device=y_pred.device, dtype=torch.float)

    for b, c in np.ndindex(batch_size, n_class):
        (edges_pred, edges_gt), (distances_pred_gt, distances_gt_pred), areas = get_edge_surface_distance(
            y_pred[b, c],
            y[b, c],
            distance_metric=distance_metric,
            spacing=spacing_list[b],
            use_subvoxels=use_subvoxels,
            symmetric=True,
            class_index=c,
        )
        
        boundary_complete = len(distances_pred_gt) + len(distances_gt_pred)
        boundary_correct = torch.sum(torch.tensor(distances_pred_gt <= class_thresholds[c])) + \
                         torch.sum(torch.tensor(distances_gt_pred <= class_thresholds[c]))

        if boundary_complete == 0:
            nsd[b, c] = torch.tensor(float('nan'))
        else:
            nsd[b, c] = boundary_correct / boundary_complete

    return nsd

class NSDMetric(nn.Module):
    def __init__(self, n_classes, percentile=95):
        super(NSDMetric, self).__init__()
        self.n_classes = n_classes
        self.class_thresholds = [1.0] * n_classes  # 1mm threshold for all classes

    def forward(self, inputs, target, spacing=(1.0, 1.0, 1.0), softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
            
        # Convert to one-hot encoding
        inputs = F.one_hot(inputs, num_classes=self.n_classes).permute(0, 4, 1, 2, 3).float()
        target = F.one_hot(target, num_classes=self.n_classes).permute(0, 4, 1, 2, 3).float()
        
        nsd_scores = compute_surface_dice(
            inputs, 
            target,
            class_thresholds=self.class_thresholds,
            include_background=False,
            spacing=spacing
        )
        
        # return torch.nanmean(nsd_scores)  # Average over batch and classes, ignoring NaN values
        return nsd_scores[0]  # Average over batch and classes, ignoring NaN values
    

class Tee:
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  
    
    def flush(self):
        for f in self.files:
            f.flush()


@hydra.main(config_path='confs', config_name='infer', version_base=None)
def main(conf: DictConfig):
    print(OmegaConf.to_container(conf, resolve=True))

    log_dir = "log_inference"
    filename = os.path.join(log_dir, conf.dir_name,  str(conf.model_num) + ".log")
    os.makedirs(os.path.join(log_dir, conf.dir_name), exist_ok=True)
    log_file = open(filename, 'w', encoding='utf-8')  
    sys.stdout = Tee(sys.stdout, log_file)
    atexit.register(lambda: log_file.close())
    
    visdir = "inference_visualization"
    vis_dir_name = os.path.join(visdir, conf.dir_name, str(conf.model_num))

    device = dev(conf.get('device'))

    model = Unet3D(
        dim=conf.diffusion_img_size,
        dim_mults=conf.dim_mults,
        channels=conf.diffusion_num_channels,
        cond_dim=16,
    )

    diffusion = GaussianDiffusion_Nolatent(
        model,
        image_size=conf.diffusion_img_size,
        num_frames=conf.diffusion_depth_size,
        channels=conf.diffusion_num_channels,
        timesteps=conf.timesteps,
        loss_type=conf.loss_type,
    )
    diffusion.to(device)

    model_path = os.path.join(conf.model_path, f"model-{conf.model_num}.pt")
    weights_dict = {}
    for k, v in (load_state_dict(os.path.expanduser(model_path), map_location="cpu")["model"].items()):
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    diffusion.load_state_dict(weights_dict)

    model.eval()

    print("sampling...")
    
    if conf.dataset == 'MMWHS':
        dataloader = get_MMWHS_dataloader(root_dir=conf.root_dir, mode=conf.mode, data_type=conf.data_type) 
    elif conf.dataset == 'TS' :
        dataloader = get_TS_dataloader(root_dir=conf.root_dir, mode=conf.mode)
    else :
        raise ValueError ("No Such Dataset")
    
    idx = 0
    dice_total = [0, 0, 0, 0, 0]
    nsd_total = [0, 0, 0, 0, 0]
    for batch in iter(dataloader):
        idx += 1
        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)
                affine = batch['affine'].squeeze(0).cpu()

        real_image = batch["img"]
        real_mask = batch.get('mask').cpu()
        real_mask_sdf = batch.get('mask_sdf').cpu()
        gt_name = batch['name'][0]

        gt_name = gt_name.split('_image')[0]
        gt_name = gt_name.split('-image')[0]
        
        print(idx,":", gt_name)

        dice = [0, 0, 0, 0, 0]
        nsd = [0, 0, 0, 0, 0]
        seed_num = 1
        for _ in range(seed_num):
            seed = th.randint(0, 10000, (1,)).item()
            print("     seed:", seed)
            th.manual_seed(seed)
            th.cuda.manual_seed(seed)
            th.cuda.manual_seed_all(seed)
            th.backends.cudnn.deterministic = True
            th.backends.cudnn.benchmark = False

            sample_fn = diffusion.p_sample_loop

            result = sample_fn(
                shape_image = real_image.size(),
                shape_mask = real_mask_sdf.size(),
                device=device,
                image=real_image,
            )

            gen_image = result[:,0,:,:,:]
            gen_image = gen_image.cpu()
            gen_mask = result[:,1:(result.size()[1]),:,:,:]

            real_img_to_save = tio.ScalarImage(tensor=real_image.squeeze(0).cpu(), channels_last=False, affine=affine)
            os.makedirs(os.path.join(vis_dir_name, 'Image'), exist_ok=True)
            real_img_to_save.save(os.path.join(vis_dir_name, 'Image', f"{gt_name}-image-real.nii.gz"))
            
            for i in range(gen_mask.size()[1]):
                gen_mask_i = gen_mask[:,i,:,:,:]
                gen_mask_i = gen_mask_i.cpu()
                gen_mask_i_de_sdf = torch.where(gen_mask_i < 0.0, torch.tensor(1.0), torch.tensor(0.0))
                
                gen_mask_sdf_to_save = tio.LabelMap(tensor=gen_mask_i, channels_last=False, affine=affine)
                os.makedirs(os.path.join(vis_dir_name, 'Label'), exist_ok=True)
                gen_mask_sdf_to_save.save(os.path.join(vis_dir_name, 'Label', f"{gt_name}-{seed}-label-sdf-{i+1}-gen.nii.gz"))
                
                gen_mask_de_sdf_to_save = tio.LabelMap(tensor=gen_mask_i_de_sdf, channels_last=False, affine=affine)
                os.makedirs(os.path.join(vis_dir_name, 'Label'), exist_ok=True)
                gen_mask_de_sdf_to_save.save(os.path.join(vis_dir_name, 'Label', f"{gt_name}-{seed}-label-de-sdf-{i+1}-gen.nii.gz"))

                real_mask_sdf_to_save = tio.LabelMap(tensor=real_mask_sdf[:,i,:,:,:], channels_last=False, affine=affine)
                os.makedirs(os.path.join(vis_dir_name, 'Label'), exist_ok=True)
                real_mask_sdf_to_save.save(os.path.join(vis_dir_name, 'Label', f"{gt_name}-{seed}-label-sdf-{i+1}-real.nii.gz"))
                
                real_mask_de_sdf_to_save = tio.LabelMap(tensor=real_mask[:,i,:,:,:], channels_last=False, affine=affine)
                os.makedirs(os.path.join(vis_dir_name, 'Label'), exist_ok=True)
                real_mask_de_sdf_to_save.save(os.path.join(vis_dir_name, 'Label', f"{gt_name}-{seed}-label-de-sdf-{i+1}-real.nii.gz"))

                real_mask_i = real_mask[:,i,:,:,:]
                Dice = get_dice(real_mask_i.numpy(), gen_mask_i_de_sdf.numpy())
                print(f"        {i+1}_dice:", Dice)
                dice[i] += Dice
            
            gen_mask_de_sdf = torch.where(gen_mask < 0.0, torch.tensor(1.0), torch.tensor(0.0))
            background_mask = torch.ones((1, 1, 64, 64, 64), dtype=torch.float16)
            background_mask[0, 0, gen_mask_de_sdf[0].sum(dim=0) > 0] = 0  
            gen_mask_togather = torch.cat((background_mask.cpu(), gen_mask_de_sdf.cpu()), dim=1) 
            gen_mask_togather = gen_mask_togather.squeeze(0)
            gen_mask_togather = torch.argmax(gen_mask_togather, dim=0) 
            gen_mask_togather=gen_mask_togather.unsqueeze(0)
            gen_mask_togather_to_save = tio.LabelMap(tensor=gen_mask_togather.cpu().int(), channels_last=False, affine=affine)
            os.makedirs(os.path.join(vis_dir_name, 'Label'), exist_ok=True)
            gen_mask_togather_to_save.save(os.path.join(vis_dir_name, 'Label', f"{gt_name}-{seed}-label-together-gen.nii.gz"))

            get_nsd = NSDMetric(n_classes=6)
            real_mask_togather = real_mask
            background_mask = torch.ones((1, 1, 64, 64, 64), dtype=torch.float16)
            background_mask[0, 0, real_mask_togather[0].sum(dim=0) > 0] = 0  
            real_mask_togather_ = torch.cat((background_mask.cpu(),real_mask_togather.cpu()), dim=1) 
            real_mask_togather_ = real_mask_togather_.squeeze(0)
            real_mask_togather_ = torch.argmax(real_mask_togather_, dim=0) 
            nnsd = get_nsd(inputs=gen_mask_togather.long(), target=real_mask_togather_.unsqueeze(0).long())
            for i in range(0, 5):
                nsd[i] += nnsd[i]
            print(f"        {nnsd}")

            th.random.seed() 
            th.cuda.seed()  
            th.backends.cudnn.deterministic = False
            th.backends.cudnn.benchmark = True

        dice_avg = [item / seed_num for item in dice]
        nsd_avg = [item / seed_num for item in nsd]
        print("     average:")
        print(f"        dice:", dice_avg)
        print(f"        nsd:", nsd_avg)
        for i in range(5):
            dice_total[i] += dice_avg[i]
            nsd_total[i] += nsd_avg[i]

    dice_total_avg = [item / idx for item in dice_total]
    nsd_total_avg = [item / idx for item in nsd_total]
    print("total average:")
            
    print(f"    dice:", dice_total_avg)
    print(f"    nsd:", nsd_total_avg)


if __name__ == "__main__":
    
    main()