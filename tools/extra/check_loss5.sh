LOG_DIR=logs

EXP_DIR1=/media/storage/eustinova/segmentation/mean_color_prediction_viper/1

LIST=(multitask_mean_colors_viper_1_1_1e-3_grl_50000)
for VALUE in "${LIST[@]}"
do
  echo ${VALUE}
  python parse_log.py ${EXP_DIR1}/${VALUE}/${LOG_DIR}/log.txt ${EXP_DIR1}/${VALUE}/ 
  
done


EXP_DIR2=/media/storage/eustinova/segmentation/mean_color_prediction_viper/2

LIST=(mt_grl_batchnorm)
for VALUE in "${LIST[@]}"
do
  echo ${VALUE}
  python parse_log.py ${EXP_DIR2}/${VALUE}/${LOG_DIR}/log.txt ${EXP_DIR2}/${VALUE}/ 
  
done


EXP_DIR3=/media/storage/eustinova/segmentation/mean_color_prediction_viper/3

LIST=(mt_grl_batchn_eliminate_pos_same_camera_grl_50000_mean_colors_only baseline_batchnorm_eliminate_pos_same_camera \
	grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000 \
	grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000 \
	larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init \
	larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_coef_0.1 \ 
	larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_coef_0.001)
for VALUE in "${LIST[@]}"
do
  echo ${VALUE}
  python parse_log.py ${EXP_DIR3}/${VALUE}/${LOG_DIR}/log.txt ${EXP_DIR3}/${VALUE}/ 
  
done

python plot_log.py -f loss_mean_colors -f loss_mean_colors -p train -p test \
{EXP_DIR3}/mt_grl_batchn_eliminate_pos_same_camera_grl_50000_mean_colors_only/log.txt.train \
{EXP_DIR3}/mt_grl_batchn_eliminate_pos_same_camera_grl_50000_mean_colors_only/log.txt.test \

# python plot_log.py -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors  -f loss_mean_colors \
# -p baseline_batchnorm_eliminate_pos_same_camera  \
# -p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000 \
# -p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000 \
# -p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_Adam \
# -p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_RMSProp \
# -p larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init \
# -p mt_grl_batchnorm \
# -p multitask_mean_colors_viper_1_1_1e-3_grl_50000_wo_bn  ${EXP_DIR3}/baseline_batchnorm_eliminate_pos_same_camera/log.txt.test \
# ${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000/log.txt.test \
# ${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000/log.txt.test \
# ${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_Adam/log.txt.test \
# ${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_RMSProp/log.txt.test \
# ${EXP_DIR3}/larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init/log.txt.test \
# ${EXP_DIR2}/mt_grl_batchnorm/log.txt.test \
# ${EXP_DIR1}/multitask_mean_colors_viper_1_1_1e-3_grl_50000/log.txt.test \


# python plot_log.py -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors  -f loss_mean_colors \
# -p baseline_batchnorm_eliminate_pos_same_camera  \
# -p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000 \
# -p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000 \
# -p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_Adam \
# -p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_RMSProp \
# -p larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init \
# -p mt_grl_batchnorm \
# -p multitask_mean_colors_viper_1_1_1e-3_grl_50000_wo_bn  ${EXP_DIR3}/baseline_batchnorm_eliminate_pos_same_camera/log.txt.train \
# ${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000/log.txt.train \
# ${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000/log.txt.train \
# ${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_Adam/log.txt.train \
# ${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_RMSProp/log.txt.train \
# ${EXP_DIR3}/larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init/log.txt.train \
# ${EXP_DIR2}/mt_grl_batchnorm/log.txt.train \
# ${EXP_DIR1}/multitask_mean_colors_viper_1_1_1e-3_grl_50000/log.txt.train \

python plot_log.py -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors  -f loss_mean_colors \
-p baseline_batchnorm_eliminate_pos_same_camera  \
-p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000 \
-p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000 \
-p larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init \
-p mt_grl_batchnorm \
-p multitask_mean_colors_viper_1_1_1e-3_grl_50000_wo_bn -p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_coef_0.1 -p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_coef_0.001 \
${EXP_DIR3}/baseline_batchnorm_eliminate_pos_same_camera/log.txt.test \
${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000/log.txt.test \
${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000/log.txt.test \
${EXP_DIR3}/larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init/log.txt.test \
${EXP_DIR2}/mt_grl_batchnorm/log.txt.test \
${EXP_DIR1}/multitask_mean_colors_viper_1_1_1e-3_grl_50000/log.txt.test \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_coef_0.1/log.txt.test \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_coef_0.001/log.txt.test \

python plot_log.py -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors -f loss_mean_colors \
-p baseline_batchnorm_eliminate_pos_same_camera  \
-p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000 \
-p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000 \
-p larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init \
-p mt_grl_batchnorm \
-p multitask_mean_colors_viper_1_1_1e-3_grl_50000_wo_bn \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_coef_0.1 \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_coef_0.001 \
${EXP_DIR3}/baseline_batchnorm_eliminate_pos_same_camera/log.txt.train \
${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000/log.txt.train \
${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000/log.txt.train \
${EXP_DIR3}/larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init/log.txt.train \
${EXP_DIR2}/mt_grl_batchnorm/log.txt.train \
${EXP_DIR1}/multitask_mean_colors_viper_1_1_1e-3_grl_50000/log.txt.train \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_coef_0.1/log.txt.train \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_coef_0.001/log.txt.train \


python plot_log.py -f loss -f loss -f loss -f loss -f loss -f loss -f loss  -f loss \
-p baseline_batchnorm_eliminate_pos_same_camera  \
-p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000 \
-p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000 \
-p larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init \
-p mt_grl_batchnorm \
-p multitask_mean_colors_viper_1_1_1e-3_grl_50000_wo_bn -p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_coef_0.1 -p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_coef_0.001 \
${EXP_DIR3}/baseline_batchnorm_eliminate_pos_same_camera/log.txt.test \
${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000/log.txt.test \
${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000/log.txt.test \
${EXP_DIR3}/larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init/log.txt.test \
${EXP_DIR2}/mt_grl_batchnorm/log.txt.test \
${EXP_DIR1}/multitask_mean_colors_viper_1_1_1e-3_grl_50000/log.txt.test \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_coef_0.1/log.txt.test \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_coef_0.001/log.txt.test \

python plot_log.py -f loss -f loss -f loss -f loss -f loss -f loss -f loss -f loss \
-p baseline_batchnorm_eliminate_pos_same_camera  \
-p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000 \
-p grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000 \
-p larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init \
-p mt_grl_batchnorm \
-p multitask_mean_colors_viper_1_1_1e-3_grl_50000_wo_bn \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_coef_0.1 \
-p larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_coef_0.001 \
${EXP_DIR3}/baseline_batchnorm_eliminate_pos_same_camera/log.txt.train \
${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_1000/log.txt.train \
${EXP_DIR3}/grl_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000/log.txt.train \
${EXP_DIR3}/larger_descr_mt_grl_batchnorm_eliminate_pos_same_camera_grl_50000_init/log.txt.train \
${EXP_DIR2}/mt_grl_batchnorm/log.txt.train \
${EXP_DIR1}/multitask_mean_colors_viper_1_1_1e-3_grl_50000/log.txt.train \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_coef_0.1/log.txt.train \
${EXP_DIR3}/larger_descr_mt_grl_batchn_eliminate_pos_same_camera_grl_50000_coef_0.001/log.txt.train \
