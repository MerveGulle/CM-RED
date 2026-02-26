import os
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.utils as tvu
import tqdm

from datasets import get_dataset
from guided_diffusion.script_util import create_model
from skimage.metrics import structural_similarity

loss_fn_mse = torch.nn.MSELoss()

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class CM_RED_Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)

    def sample(self, logger):
        config_dict = vars(self.config.model)
        config_dict.update({"image_size": self.config.data.image_size})
        model = create_model(**config_dict)
        ckpt = os.path.join(self.args.exp, f"logs/{self.args.model_ckpt}")
        if not os.path.exists(ckpt):
            raise ValueError(f"Model ckpt not found in: {ckpt}. Please refer to https://github.com/MerveGulle/CM-RED.git"
                             f"to see where the models that were used in the paper can be found.")

        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        model.to(self.device)
        if self.config.model.use_fp16:
            model.convert_to_fp16()
        model.eval()
        model = torch.nn.DataParallel(model, device_ids=[self.args.device_ids])

        print('Running Testing with Database Trained Coefficients.',
              f'Dataset: {self.config.data.dataset}',
              f'Task: {self.args.deg}.',
              )
        self.cm_red_wrapper(model, logger)
            
    def cm_red_wrapper(self, model, logger):
        from functions.cm_red_scheme import cm_red_restoration

        args, config = self.args, self.config
        
        test_dataset = get_dataset(args, config)

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Test dataset has size {len(test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)

        test_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            shuffle=False,
        )
        
        # Set deltas
        if args.deltas == "":
            deltas = [0.0] * config.sampling.T_sampling
        else:
            deltas = [float(x.strip()) for x in args.deltas.split(',')]
            if len(deltas) != config.sampling.T_sampling:
                raise ValueError(
                    f"deltas length must be equal to T_sampling. Got {len(deltas)} deltas, but {config.sampling.T_sampling} sampling steps.")

        # Set kappas
        if args.kappas == "":
            kappas = [0.5,5.0]
        else:
            kappas = [float(x.strip()) for x in args.kappas.split(',')]
            if len(kappas) != 2:
                raise ValueError(
                    f"kappas length must be equal to 2, but got {len(kappas)} kappas.")

        # Get undersampling mask
        deg = args.deg
        if deg == 'fast_mri':
            from functions.mri_function import MulticoilMRI
            from functions.util_mri import real_to_nchw_comp, nchw_comp_to_real, cal_PSNR, get_mvue, normalize_complex, clear
            
            acceleration_mask_full_path = os.path.join(args.exp, "acceleration_masks", f"{args.pattern}_R{args.acc_rate}_mask.npy")
            if os.path.exists(acceleration_mask_full_path):
                mask = np.load(acceleration_mask_full_path)
            else:
                from functions.util_mri import mask_generator
                mask = mask_generator(
                    Nro=config.data.image_size, 
                    Npe=config.data.image_size, 
                    R=args.acc_rate, 
                    ACS=args.acs_lines,
                    mask_type=args.pattern
                    )
            
            mask = torch.from_numpy(mask).to(self.device).unsqueeze(0).unsqueeze(0)
            tvu.save_image(mask, os.path.join(self.args.image_folder, "acceleration_mask.png"))
            
        # Start sampling  
        test_bar = tqdm.tqdm(test_loader, ncols=100)
        
        os.makedirs(os.path.join(self.args.image_folder, "samples"), exist_ok=True)
        os.makedirs(os.path.join(self.args.image_folder, "labels"), exist_ok=True)
        os.makedirs(os.path.join(self.args.image_folder, "zero_filled"), exist_ok=True)
        
        logger.info("----------------------------------------------")    
        img_ind = -1
        psnr_arr = []
        ssim_arr = []
        
        for kspace, coils, file_name in test_bar:
            
            img_ind += 1

            ''' Loading Coil Sens. Map and Raw k-space data '''
            kspace = kspace.to(self.device)             # fully sampled k-space
            coils = coils.to(self.device)               # coil sensitivity maps
            file_name = file_name[0]                        
            
            ''' Define forward operator '''
            A_funcs = MulticoilMRI(config.data.image_size, mask, coils)
            
            ''' Sense1 Map '''
            mvue = get_mvue((kspace.clone().detach().cpu().numpy()),
                            (coils.clone().detach().cpu().numpy()))[0]
            sense1 = torch.from_numpy(mvue.astype(np.complex64)).to(self.device)
            norm_factor = np.max(np.abs(clear(sense1)))
            x_orig = normalize_complex(sense1)  
              
            if args.save_observed_img:
                tvu.save_image(x_orig.abs().flipud(), os.path.join(self.args.image_folder, "labels", f"{file_name}.png"))

            ''' Undersampling k-space '''
            y = kspace * mask / norm_factor         # normalized undersampled kspace measurement
            
            ''' Zerofilled image'''
            ATy = A_funcs.At(y)                     # normalized zero-filled image
            if args.save_observed_img:
                tvu.save_image(ATy.squeeze().abs().flipud(), os.path.join(self.args.image_folder, "zero_filled", f"{file_name}.png"))
                
            ATy = nchw_comp_to_real(ATy) # B, 2, H, W
            
            with torch.no_grad():
                x = cm_red_restoration(
                    ATy=ATy,
                    model=model,
                    A_funcs=A_funcs,
                    betas=self.betas,
                    iN=args.iN,
                    gamma=args.gamma,
                    deltas=deltas,
                    kappas=kappas,
                    rho=args.rho,
                    mu=args.mu,
                    cg_iter=args.cg_iter,
                    classes=None,
                    config=config
                )
            
            rssq_sens_map = torch.sqrt(torch.sum(coils[0]**2, dim=0)).to(self.device)
            x_sv = (rssq_sens_map.abs()!=0) * real_to_nchw_comp(x).squeeze()
            x_orig_sv = (rssq_sens_map.abs()!=0) * x_orig
            
            tvu.save_image(x_sv.abs().flipud(), os.path.join(self.args.image_folder, "samples", f"{file_name}.png"))
            
            psnr = cal_PSNR(np.abs(clear(x_orig_sv)), np.abs(clear(x_sv)))
            psnr_arr.append(psnr)
            ssim = structural_similarity(np.abs(clear(x_orig_sv)), np.abs(clear(x_sv)), data_range=1.0)
            ssim_arr.append(ssim)

            logger.info("%s, PSNR: %.2f -- SSIM: %.3f", file_name, psnr, ssim)
        
        logger.info("----------------------------------------------")  
        logger.info("DATABASE REULST:")
        logger.info("Avg. PSNR (mean ± std): %.2f ± %.2f", np.mean(psnr_arr), np.std(psnr_arr))
        logger.info("Avg. SSIM (mean ± std): %.3f ± %.3f", np.mean(ssim_arr), np.std(ssim_arr))
        logger.info("----------------------------------------------")  