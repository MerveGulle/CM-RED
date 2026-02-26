import torch
import numpy as np
import sigpy as sp
from skimage.metrics import structural_similarity

def clear(x):
    x = x.clone().detach().cpu().squeeze().numpy()
    return x


def mask_gen_non_uniform(ACS_length, Nro, Npe, R, seed=0):
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

    ACS_start = int((Npe - ACS_length) / 2) - 1
    ACS_end = ACS_start + ACS_length

    mask = torch.zeros(size=(Nro, Npe), dtype=torch.complex64)
    mask[:, ACS_start:ACS_end] = 1  # ACS lines

    all_lines = set(range(Npe))
    acs_lines = set(range(ACS_start, ACS_end))

    # if conf.data_type == 'AXT2':
    no_signal_lines = set()
    # else:
    #     no_signal_lines = set(range(0, 17)) | set(range(352, Npe))

    candidate_lines = list(all_lines - acs_lines - no_signal_lines)
    num_sampled_lines = int(len(candidate_lines) / R)
    
    sampled_indices = torch.randperm(len(candidate_lines), generator=g if seed is not None else None)[:num_sampled_lines]
    selected_pe_lines = [candidate_lines[i] for i in sampled_indices.tolist()]

    # Vectorized masking
    mask[:, selected_pe_lines] = 1
    if no_signal_lines:
        mask[:, list(no_signal_lines)] = 1

    return mask  # .permute(2, 0, 1).unsqueeze(0)  # shape: (1, Nch, Nro, Npe)


def normalize(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img = img - torch.min(img)
    img = img / torch.max(img)
    return img
     
            
def normalize_complex(img):
    """ normalizes the magnitude of complex-valued image to range [0, 1] """
    abs_img = normalize(torch.abs(img))
    ang_img = torch.angle(img)
    return abs_img * torch.exp(1j * ang_img)


def get_mvue(kspace, s_maps):
    """ Get mvue estimate from coil measurements """
    return np.sum(
        sp.ifft(kspace, axes=(-1, -2)) * np.conj(s_maps), axis=1
    )


def cal_PSNR(ref, recon):
    mse = np.sum(np.square(np.abs(ref - recon))) / ref.size
    return 20 * np.log10(ref.max() / (np.sqrt(mse) + 1e-10))


def nchw_comp_to_real(x):
    """
    [1, 1, H, W] comp --> [1, 2, H, W] real
    """
    x = torch.view_as_real(x)
    x = x.squeeze(dim=1)
    x = x.permute(0, 3, 1, 2)
    return x


def real_to_nchw_comp(x):
    """
    [1, 2, H, W] real --> [1, 1, H, W] comp
    """
    if len(x.shape) == 4:
        x = x[:, 0:1, :, :] + x[:, 1:2, :, :] * 1j
    elif len(x.shape) == 3:
        x = x[0:1, :, :] + x[1:2, :, :] * 1j
    return x

   
def cal_SSIM(ref, recon):
    return structural_similarity(ref, recon, data_range=recon.max() - recon.min())

def mask_generator(Nro, Npe, R, ACS, mask_type, seed=0):
    
    mask = np.zeros((Nro, Npe), dtype = np.float32)
    
    if mask_type=='equidistant':
        mask[:, 0::R] = 1
        ACS_start = (Npe - ACS)//2 - 1
        ACS_end = ACS_start + ACS
        mask[:, ACS_start:ACS_end] = 1
        
    else:
        
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(seed)
            np.random.seed(seed)
        
        if mask_type == 'gaussian1d':
            
            Nsamp = Npe // R - ACS
            ACS_start = (Npe - ACS)//2 - 1
            ACS_end = ACS_start + ACS
            lines = np.arange(0, Npe)
            mu = Npe//2 - 0.5
            std = Npe/4
            weights = np.exp(-0.5 * ((lines - mu) / std) ** 2)
            weights[ACS_start:ACS_end] = 0
            weights /= weights.sum()
            
            sampled_lines = np.random.choice(lines, size=Nsamp, replace=False, p=weights)
            mask[:,sampled_lines] = 1
            mask[:,ACS_start:ACS_end] = 1
            
        elif mask_type == 'uniformrandom1d':
            Nsamp = Npe // R - ACS
            ACS_start = (Npe - ACS)//2 - 1
            ACS_end = ACS_start + ACS
            lines = np.arange(0, Npe-ACS)            
            
            sampled_lines = np.random.choice(lines, size=Nsamp, replace=False)
            sampled_lines[sampled_lines>ACS_start] += ACS
            sampled_lines = np.concatenate([sampled_lines, np.arange(ACS_start, ACS_end)])
            sampled_lines = np.sort(sampled_lines)
            
            mask[:,sampled_lines] = 1
        
        elif mask_type == 'gaussian2d':
            
            rows = np.arange(0, Nro)
            cols = np.arange(0, Npe)

            rr, cc = np.meshgrid(rows, cols, indexing='ij')
            
            row_start = (Nro - ACS) // 2 - 1
            row_end   = row_start + ACS
            col_start = (Npe - ACS) // 2 - 1
            col_end   = col_start + ACS

            rows = np.arange(row_start, row_end)
            cols = np.arange(col_start, col_end)

            rr_acs, cc_acs = np.meshgrid(rows, cols, indexing='ij')
            
            mu_r, mu_c = Nro // 2 - 0.5, Npe // 2 - 0.5
            std_r, std_c = Nro/4, Npe/4

            weights = np.exp(-0.5 * (((rr - mu_r)/std_r)**2 + ((cc - mu_c)/std_c)**2))
            weights[rr_acs, cc_acs] = 0.0
            weights /= weights.sum()  
            
            linear_indices = np.ravel_multi_index((rr.flatten(), cc.flatten()), (Nro, Npe))
            
            Nsamp = (Nro * Npe) // R - ACS**2

            sampled_pixels = np.random.choice(linear_indices, size=Nsamp, replace=False, p=weights.flatten())
            
            sampled_rows, sampled_cols = np.unravel_index(sampled_pixels, (Nro, Npe))
            
            mask[sampled_rows, sampled_cols] = 1
            mask[rr_acs, cc_acs] = 1
        
        elif mask_type == 'uniformrandom2d':
            rows = np.arange(0, Nro)
            cols = np.arange(0, Npe)

            rr, cc = np.meshgrid(rows, cols, indexing='ij')
            
            row_start = (Nro - ACS) // 2 - 1
            row_end   = row_start + ACS
            col_start = (Npe - ACS) // 2 - 1
            col_end   = col_start + ACS

            rows = np.arange(row_start, row_end)
            cols = np.arange(col_start, col_end)

            rr_acs, cc_acs = np.meshgrid(rows, cols, indexing='ij')
            
            mu_r, mu_c = Nro // 2 - 0.5, Npe // 2 - 0.5
            std_r, std_c = Nro/4, Npe/4

            weights = np.ones_like(mask)
            weights[rr_acs, cc_acs] = 0.0
            weights /= weights.sum()  
            
            linear_indices = np.ravel_multi_index((rr.flatten(), cc.flatten()), (Nro, Npe))
            
            Nsamp = (Nro * Npe) // R - ACS**2

            sampled_pixels = np.random.choice(linear_indices, size=Nsamp, replace=False, p=weights.flatten())
            
            sampled_rows, sampled_cols = np.unravel_index(sampled_pixels, (Nro, Npe))
            
            mask[sampled_rows, sampled_cols] = 1
            mask[rr_acs, cc_acs] = 1
        
        else:
            NotImplementedError(f'Mask type {type} is currently not supported.')
    
    return mask