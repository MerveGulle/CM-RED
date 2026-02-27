import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np


torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    # General Setting
    parser.add_argument("--config", type=str, default='fast_mri_320.yml', help="Path to the config file")
    parser.add_argument("--seed", type=int, default=1234, help="Set different seeds for diverse results")
    parser.add_argument("--device_ids", type=int, default=0, help="cuda=?")

    # Model
    parser.add_argument("--model_ckpt", type=str, default='fast_mri/ema_0.9999432189950708_700000_cm_knee.pt', help="Name of the model checkpoint") # e.g.,  "fast_mri/ema_0.9999432189950708_1050000_cm_brain.pt"
    

    # Save / Log
    parser.add_argument("--exp", type=str, default="exp", help="Path for saving running related data.")
    parser.add_argument("--save_y", dest="save_observed_img", action="store_true")
    parser.add_argument("-i", "--image_folder",  type=str,  default="demo", help="The folder name of samples")
    parser.add_argument("--verbose", type=str, default="info", help="Verbose level: info | debug | warning | critical")
    parser.add_argument("--ni", action="store_false", help="No interaction. Suitable for Slurm Job launcher")

    # Degredation
    parser.add_argument("--deg", type=str, default='fast_mri')
    
    # Data
    parser.add_argument('--subset_start', type=int, default=-1)
    parser.add_argument('--subset_end', type=int, default=-1)
    parser.add_argument("--acc_rate", type=int, default=4, help="Acceleration rate")
    parser.add_argument("--acs_lines", type=int, default=24, help="Number of ACS lines")
    parser.add_argument("--data_type", type=str, default=4, help="PD, PDFS, AXT1PRE, AXT1POST, AXT2, AXFLAIR")
    parser.add_argument("--pattern", type=str, default="equidistant", help="equidistant, gaussian1d")


    # Hyperparameters
    parser.add_argument("--iN", type=int, default=250, help="iN hyperparameter")
    parser.add_argument("--gamma", type=float, default=0.7, help="Gamma hyperparameter")
    parser.add_argument("--deltas", type=str, default="", help="A comma separated list of the delta hyperparameters, sigma_cm[n]=sigma[n]*(1+delta[n])")
    parser.add_argument("--kappas", type=str, default="", help="A comma separated list of the kappa hyperparameters, CM out scale parameters")
    parser.add_argument("--rho", type=float, default=-2.0, help="Rho (penalty) hyperparameter")
    parser.add_argument("--mu", type=float, default=0.9, help="Mu (initial momentum) hyperparameter") 
    parser.add_argument("--cg_iter", type=int, default=10, help="Number of iterations used in CG sense")   
    
    
    args = parser.parse_args()

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    
    os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
    args.image_folder = os.path.join(
        args.exp, "image_samples", args.image_folder
    )
    
        
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input(
                f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
            )
            if response.upper() == "Y":
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    log_path = os.path.join(args.image_folder, '0_logs.log')
    fh = logging.FileHandler(log_path)  # , mode='a')
    fh.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fh)
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    # add device
    device = torch.device(f"cuda:{args.device_ids}") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config, logger


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config, logger = parse_args_and_config()
    
    try:
        from guided_diffusion.diffusion_cm_red import CM_RED_Diffusion
        runner = CM_RED_Diffusion(args, config, config.device)
        runner.sample(logger)
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())