



EXP_DIR=//media/storage/eustinova/hieroglyphs/experiments//

LOG_DIR=logs

LIST=(hyerog_bhattacharyya_KS_asm_calc_grad_100d_layer_1e-2_0 hyerog_bhattacharyya_KS_asm_calc_grad_100d_layer_1e-2_0_balance_lw cuhk03_KS_calc_grad_layer_1e-5_wd_0.001 cuhk03_KS_calc_grad_layer_1e-3_wd_0.001/ ) #hyerog_KS_smooth_calc_grad_100d_layer_1e-6_0 hyerog_KS_smooth_calc_grad_100d_layer_1e-7_0 cuhk03_KS_calc_grad_layer_1e-5 hyerog_KS_smooth_calc_grad_100d_layer_1e-3_0 hyerog_KS_smooth_calc_grad_100d_layer_1e-4_0/) #cuhk03_KS_calc_grad_layer_1e-2 hyerog_KS_asm_calc_grad_100d_layer_1e-2_0_wd_0.005 hyerog_KS_asm_calc_grad_100d_layer_1e-2_0_wd_0.05 )   #hyerog_bhattacharyya_calc_grad_100d_layer_1e-2_0 /hyerog_bhattacharyya_100d_layer_1e-2 hyerog_KS_asm_calc_grad_100d_layer_1e-2_0/) #hyerog_KS_asm_calc_grad_100d_layer_1e-2_0) #cuhk03_bhattacharyya_calc_grad_layer_1e-2_withh_pretr500_adam) #cuhk03_bhattacharyya_calc_grad_layer_1e-2_withh_pretr500 hyerog_KS_asm_calc_grad_100d_layer_1e-2_0 cuhk03_bin_dev_1e-3_aug) # cuhk03_bhattacharyya_calc_grad_layer_1e-2_0_aug hyerog_bin_dev_c_1_100d_layer_1e-2 )  #hyerog_bhattacharyya_calc_grad_100d_layer_1e-2_0.2_1 hyerog_bhattacharyya_calc_grad_100d_layer_1e-2_0 hyerog_bin_dev_100d_layer_1e-3 hyerog_bin_dev_100d_layer_1e-2 hyerog_bhattacharyya_calc_grad_100d_layer_1e-2_0.2) #cuhk03_bhattacharyya_calc_grad_layer_1e-2_0.1 hyerog_bhattacharyya_calc_grad_100d_layer_1e-2_0.1) #cuhk03_bin_dev_truncate_1e-3 ) #cuhk03_bin_dev_calc_grad_truncate_1e-3 cuhk03_bhattacharyya_1e-2_2  cuhk03_bhattacharyya_1e-1_2 cuhk03_bhattacharyya_1e-4_2) #/cuhk03_bhattacharyya_1e-3 /hyerog_bin_dev_batch_100d_layer_1e-3 cuhk03_pair_contrastive_1e-3 cuhk03_pair_bin_dev_1e-3 hyerog_bhattacharyya_label_batch_100d_layer_1e-3  /hyerog_bin_dev_100d_layer_1e-3 hyerog_bhattacharyya_100d_layer_1e-3 hyerog_bhattacharyya_100d_hist_layer_1e-3/ hyerog_bhattacharyya_100d_hist_layer_1e-4/ hyerog_bhattacharyya_100d_hist_layer_1e-5/)
for VALUE in "${LIST[@]}"
do
  echo ${VALUE}
  python parse_log.py ${EXP_DIR}/${VALUE}/${LOG_DIR}/log.txt ${EXP_DIR}/${VALUE}/ 
  
done

python plot_log.py  -f loss1 -f loss1 -f loss2 -f loss2 -p train -p test -p train -p test  \
${EXP_DIR}/hyerog_bhattacharyya_KS_asm_calc_grad_100d_layer_1e-2_0/log.txt.train \
${EXP_DIR}/hyerog_bhattacharyya_KS_asm_calc_grad_100d_layer_1e-2_0/log.txt.test \
${EXP_DIR}/hyerog_bhattacharyya_KS_asm_calc_grad_100d_layer_1e-2_0/log.txt.train \
${EXP_DIR}/hyerog_bhattacharyya_KS_asm_calc_grad_100d_layer_1e-2_0/log.txt.test \

python plot_log.py  -f loss1 -f loss1 -f loss2 -f loss2 -p train -p test -p train -p test  \
${EXP_DIR}/hyerog_bhattacharyya_KS_asm_calc_grad_100d_layer_1e-2_0_balance_lw/log.txt.train \
${EXP_DIR}/hyerog_bhattacharyya_KS_asm_calc_grad_100d_layer_1e-2_0_balance_lw/log.txt.test \
${EXP_DIR}/hyerog_bhattacharyya_KS_asm_calc_grad_100d_layer_1e-2_0_balance_lw/log.txt.train \
${EXP_DIR}/hyerog_bhattacharyya_KS_asm_calc_grad_100d_layer_1e-2_0_balance_lw/log.txt.test \


python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/cuhk03_KS_calc_grad_layer_1e-5/log.txt.train \
${EXP_DIR}/cuhk03_KS_calc_grad_layer_1e-5/log.txt.test \

python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/cuhk03_KS_calc_grad_layer_1e-5_wd_0.001/log.txt.train \
${EXP_DIR}/cuhk03_KS_calc_grad_layer_1e-5_wd_0.001/log.txt.test \


python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/cuhk03_KS_calc_grad_layer_1e-3_wd_0.001/log.txt.train \
${EXP_DIR}/cuhk03_KS_calc_grad_layer_1e-3_wd_0.001/log.txt.test \




python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_KS_smooth_calc_grad_100d_layer_1e-6_0/log.txt.train \
${EXP_DIR}/hyerog_KS_smooth_calc_grad_100d_layer_1e-6_0/log.txt.test \

python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_KS_smooth_calc_grad_100d_layer_1e-7_0/log.txt.train \
${EXP_DIR}/hyerog_KS_smooth_calc_grad_100d_layer_1e-7_0/log.txt.test \

python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_KS_smooth_calc_grad_100d_layer_1e-2_0/log.txt.train \
${EXP_DIR}/hyerog_KS_smooth_calc_grad_100d_layer_1e-2_0/log.txt.test \


python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_KS_smooth_calc_grad_100d_layer_1e-3_0/log.txt.train \
${EXP_DIR}/hyerog_KS_smooth_calc_grad_100d_layer_1e-3_0/log.txt.test \

python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_KS_smooth_calc_grad_100d_layer_1e-4_0/log.txt.train \
${EXP_DIR}/hyerog_KS_smooth_calc_grad_100d_layer_1e-4_0/log.txt.test \

python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_KS_smooth_calc_grad_100d_layer_1e-5_0/log.txt.train \
${EXP_DIR}/hyerog_KS_smooth_calc_grad_100d_layer_1e-5_0/log.txt.test \






python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/cuhk03_KS_calc_grad_layer_1e-5/log.txt.train \
${EXP_DIR}/cuhk03_KS_calc_grad_layer_1e-5/log.txt.test \






python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/cuhk03_KS_calc_grad_layer_1e-2/log.txt.train \
${EXP_DIR}/cuhk03_KS_calc_grad_layer_1e-2/log.txt.test \


python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_KS_asm_calc_grad_100d_layer_1e-2_0_wd_0.005/log.txt.train \
${EXP_DIR}/hyerog_KS_asm_calc_grad_100d_layer_1e-2_0_wd_0.005/log.txt.test \

python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_KS_asm_calc_grad_100d_layer_1e-2_0_wd_0.05/log.txt.train \
${EXP_DIR}/hyerog_KS_asm_calc_grad_100d_layer_1e-2_0_wd_0.05/log.txt.test \


python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_bhattacharyya_calc_grad_100d_layer_1e-2_0/log.txt.train \
${EXP_DIR}/hyerog_bhattacharyya_calc_grad_100d_layer_1e-2_0/log.txt.test \



python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_bhattacharyya_100d_layer_1e-2/log.txt.train \
${EXP_DIR}/hyerog_bhattacharyya_100d_layer_1e-2/log.txt.test \




python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_KS_asm_calc_grad_100d_layer_1e-2_0/log.txt.train \
${EXP_DIR}/hyerog_KS_asm_calc_grad_100d_layer_1e-2_0/log.txt.test \


python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/cuhk03_bhattacharyya_calc_grad_layer_1e-2_withh_pretr500_adam/log.txt.train \
${EXP_DIR}/cuhk03_bhattacharyya_calc_grad_layer_1e-2_withh_pretr500_adam/log.txt.test \


python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/cuhk03_bhattacharyya_calc_grad_layer_1e-2_withh_pretr500/log.txt.train \
${EXP_DIR}/cuhk03_bhattacharyya_calc_grad_layer_1e-2_withh_pretr500/log.txt.test \

python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_KS_asm_calc_grad_100d_layer_1e-2_0/log.txt.train \
${EXP_DIR}/hyerog_KS_asm_calc_grad_100d_layer_1e-2_0/log.txt.test \


python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/cuhk03_bin_dev_1e-3_aug/log.txt.train \
${EXP_DIR}/cuhk03_bin_dev_1e-3_aug/log.txt.test \


python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/cuhk03_bhattacharyya_1e-2_2/log.txt.train \
${EXP_DIR}/cuhk03_bhattacharyya_1e-2_2/log.txt.test \


python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/cuhk03_bhattacharyya_calc_grad_layer_1e-2_0_aug/log.txt.train \
${EXP_DIR}/cuhk03_bhattacharyya_calc_grad_layer_1e-2_0_aug/log.txt.test \


python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_bin_dev_c_1_100d_layer_1e-2/log.txt.train \
${EXP_DIR}/hyerog_bin_dev_c_1_100d_layer_1e-2/log.txt.test \


python plot_log.py  -f loss -f loss  -f loss -f loss  -p train_m0 -p test_m0  -p train -p test  \
${EXP_DIR}/hyerog_bhattacharyya_calc_grad_100d_layer_1e-2_0/log.txt.train \
${EXP_DIR}/hyerog_bhattacharyya_calc_grad_100d_layer_1e-2_0/log.txt.test \
${EXP_DIR}/hyerog_bhattacharyya_calc_grad_100d_layer_1e-2_0.2_1/log.txt.train \
${EXP_DIR}/hyerog_bhattacharyya_calc_grad_100d_layer_1e-2_0.2_1/log.txt.test \




python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_bin_dev_100d_layer_1e-3/log.txt.train \
${EXP_DIR}/hyerog_bin_dev_100d_layer_1e-3/log.txt.test \

python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_bin_dev_100d_layer_1e-2/log.txt.train \
${EXP_DIR}/hyerog_bin_dev_100d_layer_1e-2/log.txt.test \


python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_bhattacharyya_calc_grad_100d_layer_1e-2_0.2/log.txt.train \
${EXP_DIR}/hyerog_bhattacharyya_calc_grad_100d_layer_1e-2_0.2/log.txt.test \


python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/cuhk03_bhattacharyya_calc_grad_layer_1e-2_0.1/log.txt.train \
${EXP_DIR}/cuhk03_bhattacharyya_calc_grad_layer_1e-2_0.1/log.txt.test \


python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_bhattacharyya_calc_grad_100d_layer_1e-2_0.1/log.txt.train \
${EXP_DIR}/hyerog_bhattacharyya_calc_grad_100d_layer_1e-2_0.1/log.txt.test \




python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/cuhk03_bin_dev_truncate_1e-3/log.txt.train \
${EXP_DIR}/cuhk03_bin_dev_truncate_1e-3/log.txt.test \


python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/cuhk03_bin_dev_calc_grad_truncate_1e-3/log.txt.train \
${EXP_DIR}/cuhk03_bin_dev_calc_grad_truncate_1e-3/log.txt.test \


python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/cuhk03_bin_dev_calc_grad_truncate_1e-3/log.txt.train \
${EXP_DIR}/cuhk03_bin_dev_calc_grad_truncate_1e-3/log.txt.test \



python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/cuhk03_bhattacharyya_1e-2_2//log.txt.train \
${EXP_DIR}/cuhk03_bhattacharyya_1e-2_2/log.txt.test \


python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/cuhk03_bhattacharyya_1e-1_2//log.txt.train \
${EXP_DIR}/cuhk03_bhattacharyya_1e-1_2/log.txt.test \




python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/cuhk03_bhattacharyya_1e-4_2//log.txt.train \
${EXP_DIR}/cuhk03_bhattacharyya_1e-4_2/log.txt.test \


python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/cuhk03_bhattacharyya_1e-3///log.txt.train \
${EXP_DIR}/cuhk03_bhattacharyya_1e-3/log.txt.test \

python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_bin_dev_batch_100d_layer_1e-3///log.txt.train \
${EXP_DIR}/hyerog_bin_dev_batch_100d_layer_1e-3//log.txt.test \



python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/cuhk03_pair_contrastive_1e-3//log.txt.train \
${EXP_DIR}/cuhk03_pair_contrastive_1e-3/log.txt.test \


python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_bin_dev_100d_layer_1e-3///log.txt.train \
${EXP_DIR}/hyerog_bin_dev_100d_layer_1e-3//log.txt.test \

python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/cuhk03_pair_bin_dev_1e-3/log.txt.train \
${EXP_DIR}/cuhk03_pair_bin_dev_1e-3/log.txt.test \


python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_bhattacharyya_label_batch_100d_layer_1e-3/log.txt.train \
${EXP_DIR}/hyerog_bhattacharyya_label_batch_100d_layer_1e-3/log.txt.test \






python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_bhattacharyya_100d_hist_layer_1e-3///log.txt.train \
${EXP_DIR}/hyerog_bhattacharyya_100d_hist_layer_1e-3//log.txt.test \

python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_bhattacharyya_100d_hist_layer_1e-4///log.txt.train \
${EXP_DIR}/hyerog_bhattacharyya_100d_hist_layer_1e-4//log.txt.test \

python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_bhattacharyya_100d_hist_layer_1e-5//log.txt.train \
${EXP_DIR}/hyerog_bhattacharyya_100d_hist_layer_1e-5//log.txt.test \

python plot_log.py  -f loss -f loss  -p train -p test  \
${EXP_DIR}/hyerog_bhattacharyya_100d_layer_1e-3/log.txt.train \
${EXP_DIR}/hyerog_bhattacharyya_100d_layer_1e-3/log.txt.test \
