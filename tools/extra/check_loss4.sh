

EXP_DIR3=/media/storage/eustinova/segmentation/mean_color_prediction_viper/3
LOG_DIR3=logs
LIST=(larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_RMSProp larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_scld larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_Adam baseline_batchnorm_eliminate_pos_same_camera grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000 grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000 larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_1e-4 larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_x100 larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000 larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init )
for VALUE in "${LIST[@]}"
do
  echo ${VALUE}
  python parse_log.py ${EXP_DIR3}/${VALUE}/${LOG_DIR3}/log.txt ${EXP_DIR3}/${VALUE}/ 
  
done



python plot_log.py -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors   -f loss_mean_colors -f loss_mean_colors  -f loss_mean_colors   -f loss_mean_colors  -f loss_mean_colors  -f loss_mean_colors  -f loss_mean_colors -f loss_mean_colors   \
-p baseline_batchnorm_eliminate_pos_same_camera_test  \
-p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000_test \
-p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_test  \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_1e-4_test \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_x100_test  \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_test  \
-p larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init_test  \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_RMSProp_test  \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_scld_test \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_Adam_test \
${EXP_DIR3}/baseline_batchnorm_eliminate_pos_same_camera/log.txt.test \
${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000/log.txt.test \
${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000/log.txt.test \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_1e-4/log.txt.test \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_x100/log.txt.test \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000/log.txt.test \
${EXP_DIR3}/larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init/log.txt.test \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_RMSProp/log.txt.test \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_scld/log.txt.test \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_Adam/log.txt.test \

python plot_log.py -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors   -f loss_mean_colors -f loss_mean_colors  -f loss_mean_colors   -f loss_mean_colors  -f loss_mean_colors  -f loss_mean_colors  -f loss_mean_colors -f loss_mean_colors   \
-p baseline_batchnorm_eliminate_pos_same_camera_test  \
-p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000_train \
-p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_train  \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_1e-4_train \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_x100_train  \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_train  \
-p larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init_train  \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_RMSProp_train  \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_scld_train \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_Adam_train \
${EXP_DIR3}/baseline_batchnorm_eliminate_pos_same_camera/log.txt.train \
${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000/log.txt.train \
${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000/log.txt.train \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_1e-4/log.txt.train \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_x100/log.txt.train \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000/log.txt.train \
${EXP_DIR3}/larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init/log.txt.train \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_RMSProp/log.txt.train \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_scld/log.txt.train \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_Adam/log.txt.train \

# python plot_log.py -f loss_mean_colors  -f loss_mean_colors   -f loss_mean_colors  -f loss_mean_colors  -f loss_mean_colors  -f loss_mean_colors -f loss_mean_colors  -f loss_mean_colors  -f loss_mean_colors  -f loss_mean_colors  -f loss_mean_colors  -f loss_mean_colors  -f loss_mean_colors  -f loss_mean_colors \
# -p baseline_batchnorm_eliminate_pos_same_camera_test -p baseline_batchnorm_eliminate_pos_same_camera_train \
# -p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000_test -p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000_train \
# -p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_test -p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_train \
# -p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_1e-4_test -p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_1e-4_train \
# -p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_x100_test -p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_x100_train \
# -p larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_test -p larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_train \
# -p larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init_test -p larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init_train \
# ${EXP_DIR3}/baseline_batchnorm_eliminate_pos_same_camera/log.txt.test \
# ${EXP_DIR3}/baseline_batchnorm_eliminate_pos_same_camera/log.txt.train \
# ${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000/log.txt.test \
# ${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000/log.txt.train \
# ${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000/log.txt.test \
# ${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000/log.txt.train \
# ${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_1e-4/log.txt.test \
# ${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_1e-4/log.txt.train \
# ${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_x100/log.txt.test \
# ${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_x100/log.txt.train \
# ${EXP_DIR3}/larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000/log.txt.test \
# ${EXP_DIR3}/larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000/log.txt.train \
# ${EXP_DIR3}/larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init/log.txt.test \
# ${EXP_DIR3}/larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init/log.txt.train \


# python plot_log.py -f loss  -f loss   -f loss -f loss  -f loss  -f loss -f loss -f loss  -f loss  -f loss -f loss  -f loss  -f loss -f loss \
# -p baseline_batchnorm_eliminate_pos_same_camera_test -p baseline_batchnorm_eliminate_pos_same_camera_train \
# -p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000_test -p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000_train \
# -p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_test -p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_train \
# -p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_1e-4_test -p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_1e-4_train \
# -p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_x100_test -p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_x100_train \
# -p larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_test -p larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_train \
# -p larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init_test -p larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init_train \
# ${EXP_DIR3}/baseline_batchnorm_eliminate_pos_same_camera/log.txt.test \
# ${EXP_DIR3}/baseline_batchnorm_eliminate_pos_same_camera/log.txt.train \
# ${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000/log.txt.test \
# ${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000/log.txt.train \
# ${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000/log.txt.test \
# ${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000/log.txt.train \
# ${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_1e-4/log.txt.test \
# ${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_1e-4/log.txt.train \
# ${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_x100/log.txt.test \
# ${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_x100/log.txt.train \
# ${EXP_DIR3}/larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000/log.txt.test \
# ${EXP_DIR3}/larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000/log.txt.train \
# ${EXP_DIR3}/larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init/log.txt.test \
# ${EXP_DIR3}/larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init/log.txt.train \


python plot_log.py -f loss  -f loss   -f loss -f loss  -f loss  -f loss -f loss  -f loss  -f loss -f loss \
-p baseline_batchnorm_eliminate_pos_same_camera_test  \
-p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000_test \
-p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_test  \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_1e-4_test \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_x100_test  \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_test  \
-p larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init_test  \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_RMSProp_test  \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_scld_test \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_Adam_test \
${EXP_DIR3}/baseline_batchnorm_eliminate_pos_same_camera/log.txt.test \
${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000/log.txt.test \
${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000/log.txt.test \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_1e-4/log.txt.test \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_x100/log.txt.test \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000/log.txt.test \
${EXP_DIR3}/larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init/log.txt.test \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_RMSProp/log.txt.test \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_scld/log.txt.test \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_Adam/log.txt.test \


python plot_log.py -f loss  -f loss   -f loss -f loss  -f loss  -f loss -f loss -f loss  -f loss -f loss \
-p baseline_batchnorm_eliminate_pos_same_camera_train  \
-p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000_train \
-p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_train  \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_1e-4_train \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_x100_train  \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_train  \
-p larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init_train  \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_RMSProp_train  \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_scld_train \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_Adam_train \
${EXP_DIR3}/baseline_batchnorm_eliminate_pos_same_camera/log.txt.train \
${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000/log.txt.train \
${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000/log.txt.train \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_1e-4/log.txt.train \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_x100/log.txt.train \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000/log.txt.train \
${EXP_DIR3}/larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init/log.txt.train \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_RMSProp/log.txt.train \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_init_scld/log.txt.train \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_Adam/log.txt.train \