EXP_DIR2=/media/storage/eustinova/segmentation/mean_color_prediction_viper/2
LOG_DIR2=logs
LIST=(mt_grl_batchnorm  test_baseline_batchnorm  mt_grl_batchnorm_1e-4 mt_batchnorm mt_grl_batchnorm_1e-4_adam mt_grl_batchnorm_1e-4_mt_init)
for VALUE in "${LIST[@]}"
do
  echo ${VALUE}
  python parse_log.py ${EXP_DIR2}/${VALUE}/${LOG_DIR2}/log.txt ${EXP_DIR2}/${VALUE}/ 
  
done


EXP_DIR3=/media/storage/eustinova/segmentation/mean_color_prediction_viper/3
LOG_DIR3=logs
LIST=(baseline_batchnorm_eliminate_pos_same_camera mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000)
for VALUE in "${LIST[@]}"
do
  echo ${VALUE}
  python parse_log.py ${EXP_DIR3}/${VALUE}/${LOG_DIR3}/log.txt ${EXP_DIR3}/${VALUE}/ 
  
done




EXP_DIR1=/media/storage/eustinova/segmentation/mean_color_prediction_viper/1


python plot_log.py  -f loss_mean_colors  -f loss_mean_colors -f loss_mean_colors  -f loss_mean_colors   -f loss_mean_colors  -f loss_mean_colors  -f loss_mean_colors  -f loss_mean_colors \
-p mt_grl_batchnorm_test -p mt_grl_batchnorm_train \
-p mt_grl_batchnorm_1e-4_test -p mt_grl_batchnorm_1e-4_train \
-p mt_grl_batchnorm_1e-4_adam_test -p mt_grl_batchnorm_1e-4_adam_train \
-p mt_grl_batchnorm_1e-4_mt_init_test -p mt_grl_batchnorm_1e-4_mt_init_train \
${EXP_DIR2}/mt_grl_batchnorm/log.txt.test \
${EXP_DIR2}/mt_grl_batchnorm/log.txt.train \
${EXP_DIR2}/mt_grl_batchnorm_1e-4/log.txt.test \
${EXP_DIR2}/mt_grl_batchnorm_1e-4/log.txt.train \
${EXP_DIR2}/mt_grl_batchnorm_1e-4_adam/log.txt.test \
${EXP_DIR2}/mt_grl_batchnorm_1e-4_adam/log.txt.train \
${EXP_DIR2}/mt_grl_batchnorm_1e-4_mt_init/log.txt.test \
${EXP_DIR2}/mt_grl_batchnorm_1e-4_mt_init/log.txt.train \


python plot_log.py -f loss -f loss -f loss -f loss -f loss -f loss  -f loss -f loss -f loss -f loss -f loss -f loss  -f loss -f loss  -f loss -f loss  -f loss -f loss \
-p multitask_mean_colors_viper_0_1_1e-3_test -p multitask_mean_colors_viper_0_1_1e-3_train \
-p baseline_batchnorm_test -p baseline_batchnorm_train \
-p mt_batchnorm_test -p mt_batchnorm_train \
-p mt_grl_batchnorm_test -p mt_grl_batchnorm_train \
-p mt_grl_batchnorm_1e-4_test -p mt_grl_batchnorm_1e-4_train \
-p mt_grl_batchnorm_1e-4_adam_test -p mt_grl_batchnorm_1e-4_adam_train \
-p mt_grl_batchnorm_1e-4_mt_init_test -p mt_grl_batchnorm_1e-4_mt_init_train \
-p baseline_batchnorm_eliminate_pos_same_camera_test -p baseline_batchnorm_eliminate_pos_same_camera_train \
-p mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000_test -p mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000_train \
${EXP_DIR1}/multitask_mean_colors_viper_0_1_1e-3/log.txt.test \
${EXP_DIR1}/multitask_mean_colors_viper_0_1_1e-3/log.txt.train \
${EXP_DIR2}/test_baseline_batchnorm/log.txt.test \
${EXP_DIR2}/test_baseline_batchnorm/log.txt.train \
${EXP_DIR2}/mt_batchnorm/log.txt.test \
${EXP_DIR2}/mt_batchnorm/log.txt.train \
${EXP_DIR2}/mt_grl_batchnorm/log.txt.test \
${EXP_DIR2}/mt_grl_batchnorm/log.txt.train \
${EXP_DIR2}/mt_grl_batchnorm_1e-4/log.txt.test \
${EXP_DIR2}/mt_grl_batchnorm_1e-4/log.txt.train \
${EXP_DIR2}/mt_grl_batchnorm_1e-4_adam/log.txt.test \
${EXP_DIR2}/mt_grl_batchnorm_1e-4_adam/log.txt.train \
${EXP_DIR2}/mt_grl_batchnorm_1e-4_mt_init/log.txt.test \
${EXP_DIR2}/mt_grl_batchnorm_1e-4_mt_init/log.txt.train \
${EXP_DIR3}/baseline_batchnorm_eliminate_pos_same_camera/log.txt.test \
${EXP_DIR3}/baseline_batchnorm_eliminate_pos_same_camera/log.txt.train \
${EXP_DIR3}/mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000/log.txt.test \
${EXP_DIR3}/mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000/log.txt.train \