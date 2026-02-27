#####################
#        R=4        #
#####################

# Data & Acquisition
data="PD";                 # {PD, PDFS, AXT1PRE, AXT1POST, AXT2, AXFLAIR}
acc_rate=4;                  # Acceleration rate 
pattern="gaussian1d";       # {equidistant, gaussian1d} 

# Hyperparameters
iN=50;
gamma=0.1;
deltas="0.4,3.0,3.0,2.5";
kappas="0.5,5.0";
rho=-2.0;
mu=0.9;
cg_iter=10;

device_ids=0;
add_name="-expname";          # If you add name for expriment name (dir_path)

cm_ckpt="fast_mri/ema_0.9999432189950708_700000_cm_knee.pt";
# cm_ckpt="fast_mri/ema_0.9999432189950708_1050000_cm_brain.pt";

python main.py --config fast_mri_320.yml --deg fast_mri \
    -i $data-ACC_$acc_rate-$pattern-iN_$iN-gamma_$gamma-deltas_$deltas-kappas_$kappas-rho_$rho-mu_$mu-cg_$cg_iter$add_name  \
    --data_type=$data --model_ckpt $cm_ckpt --acc_rate $acc_rate --pattern=$pattern \
    --iN=$iN --gamma=$gamma --deltas=$deltas --kappas=$kappas --rho=$rho --mu=$mu --cg_iter=$cg_iter \
    --device_ids=$device_ids \
    --save_y


#####################
#        R=8        #
#####################

# Data & Acquisition
data="AXT1PRE";            # {PD, PDFS, AXT1PRE, AXT1POST, AXT2, AXFLAIR}
acc_rate=8;                 # Acceleration rate 
pattern="equidistant";      # {equidistant, gaussian1d} 

# Hyperparameters
iN=50;
gamma=0.1;
deltas="0.5,7.0,6.0,3.5";
kappas="1.0,2.5";
rho=-3.0;
mu=0.9;
cg_iter=10;

device_ids=0;
add_name="-expname";         # If you add name for expriment name (dir_path)

# cm_ckpt="fast_mri/ema_0.9999432189950708_700000_cm_knee.pt";
cm_ckpt="fast_mri/ema_0.9999432189950708_1050000_cm_brain.pt";

python main.py --config fast_mri_320.yml --deg fast_mri \
    -i $data-ACC_$acc_rate-$pattern-iN_$iN-gamma_$gamma-deltas_$deltas-kappas_$kappas-rho_$rho-mu_$mu-cg_$cg_iter$add_name  \
    --data_type=$data --model_ckpt $cm_ckpt --acc_rate $acc_rate --pattern=$pattern \
    --iN=$iN --gamma=$gamma --deltas=$deltas --kappas=$kappas --rho=$rho --mu=$mu --cg_iter=$cg_iter \
    --device_ids=$device_ids \
    --save_y