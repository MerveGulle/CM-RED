def get_dataset(args, config):
    if config.data.dataset == 'fast_mri':
        from datasets.fast_mri import KneeMVU_MatDataset
        dataset_dir = f"exp/datasets/fastMRI/{args.data_type}" 
        test_dataset = KneeMVU_MatDataset(dataset_dir)

    else:
        test_dataset = None

    return test_dataset

