### This repository contains the official code for the ISBI 2026 paper: ###
### **"CM-RED : Consistency Models for Fast MRI Using Regularization by Denoising"**. ###


<!-- ![Example Output](figs/qualitative_result.jpg) -->
<p align="center">
  <img src="figs/recon_example.jpg" width="800">
</p>

---

### 1. Create Conda Environment and Install Requirements

```bash
conda env create -f -n {env_name}
conda activate {env_name}

cd CM-RED
pip install requirement.txt
```

### 2. Pre-trained CM Models

We provide pre-trained CM models for the **fastMRI knee** and **brain** datasets. \
Pre-trained CM models are available at the following here [link](https://www.dropbox.com/scl/fo/l5q06udyq1zbg2rhjbvvm/AFSAMaZbHmJNG1Nd1qyJ-Ko?rlkey=h2np6dpba8tnnv3pc66ew5o7x&dl=0). \
Please download and place the pre-trained models under: ./exp/logs/fast_mri/
```bash
./exp/logs/fast_mri/
├── cm_ckpt (knee)
└── cm_ckpt (brain)
```

### 3. Dataset

Please download the fastMRI dataset from [fastMRI](https://fastmri.med.nyu.edu/) after agreeing to the data use agreement.

We use the following validation sets for evaluations **knee_multicoil_val** and **brain_multicoil_val_batch_i**
Coil sensitivity maps are generated using the `sigpy.mri.app.EspiritCalib` function.
The preprocessed dataset should be placed under ./exp/datasets/fast_MRI/ :

```bash
./exp/datasets/fast_MRI/
├── PD 
└── PDFS
└── AXT1PRE
└── ...
```
`./datasets/fast_mri.py` loads the raw k-space data and the corresponding coil sensitivity maps.


### 4. Run MRI Reconstrcution Code
```bash
sh evaluate.sh
```
You can configure the dataset, model, and number of iterations (NFEs) in `configs/fast_mri_320.yml`.
The hyperparameters corresponding to the settings reported in the paper are implemented as defaults in `evaluate.sh`.
For additional customization or hyperparameter tuning, please edit evaluate.sh accordingly.

## Acknowledgements

This codebase is mainly built upon [CM4IR](https://github.com/tirer-lab/CM4IR) repository.

## 📝 Citation
If you use this code, please cite our paper:
```bibtex

@inproceedings{???,
  title={Consistency Models for Fast MRI Using Regularization by Denoising},
  author={G{\"u}lle, Merve and Yun, Junno and Al{\c{c}}alar, Ya{\c{s}}ar Utku and Ak{\c{c}}akaya, Mehmet},
  booktitle={Proc. IEEE Int. Symp. Biomed. Imag.},
  year = {2026}
}

```
