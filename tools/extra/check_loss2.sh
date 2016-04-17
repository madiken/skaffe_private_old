




EXP_DIR1=/media/storage/eustinova/segmentation/multitask3_viper/

LOG_DIR=logs

# LIST=(grl_multitask_viper_1_1_reid_500_segm_500_128_1e-4 grl_multitask_viper_1_1_reid_500_500_segm_500_128_1e-4 multitask_viper_1_1_reid_500_500_segm_500_128_1e-4 multitask_viper_1_1_reid_500_segm_500_128_1e-4 baseline_500_1e-3 baseline_500_500_1e-3 multitask_viper_1_0_128_1e-3 multitask_viper_1_0_500_128_1e-4 multitask_viper_1_0_500_128_1e-3 multitask_viper_1_0_2048_1e-3 multitask_viper_1_0_2048_1e-4 multitask_viper_1_0_10000_128_1e-4 multitask_viper_1_0_10000_128_1e-3 multitask_viper_1_1_reid_500_500_segm_500_128_1e-3 multitask_viper_1_1_reid_500_segm_128_1e-3)
# for VALUE in "${LIST[@]}"
# do
#   echo ${VALUE}
#   python parse_log.py ${EXP_DIR1}/${VALUE}/${LOG_DIR}/log.txt ${EXP_DIR}/${VALUE}/ 
  
# done

EXP_DIR=/media/storage/eustinova/segmentation/mean_color_prediction_viper/

LIST=(mt_grl_mean_colors_pretr/ _debug_multitask_mean_colors_viper_1_0_1e-3_grl_5000 multitask_mean_colors_viper_1_1_1e-3_grl_1000 multitask_mean_colors_viper_1_1_1e-3_grl_5000 multitask_mean_colors_viper_0_1_1e-3 multitask_mean_colors_viper_1_1_1e-3 multitask_mean_colors_viper_1_1_1e-3_grl_50000)
for VALUE in "${LIST[@]}"
do
  echo ${VALUE}
  python parse_log.py ${EXP_DIR}/${VALUE}/${LOG_DIR}/log.txt ${EXP_DIR}/${VALUE}/ 
  
done


python plot_log.py  -f loss -f loss -f loss  -f loss  -f loss -f loss  -f loss  -f loss  -f loss  -f loss  -f loss  -f loss  -f loss  -f loss  -f loss  -f loss \
-p baseline_500_1e-3_test -p baseline_500_1e-3_train -p baseline_500_500_1e-3_test -p baseline_500_500_1e-3_train \
-p multitask_mean_colors_viper_0_1_1e-3_test -p multitask_mean_colors_viper_0_1_1e-3_train \
-p multitask_mean_colors_viper_1_1_1e-3_test -p multitask_mean_colors_viper_1_1_1e-3_train \
-p multitask_mean_colors_viper_1_1_1e-3_grl_50000_test -p multitask_mean_colors_viper_1_1_1e-3_grl_50000_train \
-p multitask_mean_colors_viper_1_1_1e-3_grl_5000_test -p multitask_mean_colors_viper_1_1_1e-3_grl_5000_train \
-p multitask_mean_colors_viper_1_1_1e-3_grl_1000_test -p multitask_mean_colors_viper_1_1_1e-3_grl_1000_train \
-p mt_grl_mean_colors_pretr_test -p mt_grl_mean_colors_pretr_train \
${EXP_DIR1}/baseline_500_1e-3/log.txt.test \
${EXP_DIR1}/baseline_500_1e-3/log.txt.train \
${EXP_DIR1}/baseline_500_500_1e-3/log.txt.test \
${EXP_DIR1}/baseline_500_500_1e-3/log.txt.train \
${EXP_DIR}/multitask_mean_colors_viper_0_1_1e-3/log.txt.test \
${EXP_DIR}/multitask_mean_colors_viper_0_1_1e-3/log.txt.train \
${EXP_DIR}/multitask_mean_colors_viper_1_1_1e-3/log.txt.test \
${EXP_DIR}/multitask_mean_colors_viper_1_1_1e-3/log.txt.train \
${EXP_DIR}/multitask_mean_colors_viper_1_1_1e-3_grl_50000/log.txt.test \
${EXP_DIR}/multitask_mean_colors_viper_1_1_1e-3_grl_50000/log.txt.train \
${EXP_DIR}/multitask_mean_colors_viper_1_1_1e-3_grl_5000/log.txt.test \
${EXP_DIR}/multitask_mean_colors_viper_1_1_1e-3_grl_5000/log.txt.train \
${EXP_DIR}/multitask_mean_colors_viper_1_1_1e-3_grl_1000/log.txt.test \
${EXP_DIR}/multitask_mean_colors_viper_1_1_1e-3_grl_1000/log.txt.train \
${EXP_DIR}/mt_grl_mean_colors_pretr/log.txt.test \
${EXP_DIR}/mt_grl_mean_colors_pretr/log.txt.train \




python plot_log.py  -f loss_mean_colors  -f loss_mean_colors -f loss_mean_colors  -f loss_mean_colors -f loss_mean_colors  -f loss_mean_colors  -f loss_mean_colors  -f loss_mean_colors \
-p multitask_mean_colors_viper_1_1_1e-3_grl_50000_test -p multitask_mean_colors_viper_1_1_1e-3_grl_50000_train \
-p multitask_mean_colors_viper_1_1_1e-3_grl_5000_test -p multitask_mean_colors_viper_1_1_1e-3_grl_5000_train \
-p multitask_mean_colors_viper_1_1_1e-3_grl_1000_test -p multitask_mean_colors_viper_1_1_1e-3_grl_1000_train \
-p _debug_multitask_mean_colors_viper_1_0_1e-3_grl_5000_test -p _debug_multitask_mean_colors_viper_1_0_1e-3_grl_5000_train \
${EXP_DIR}/multitask_mean_colors_viper_1_1_1e-3_grl_50000/log.txt.test \
${EXP_DIR}/multitask_mean_colors_viper_1_1_1e-3_grl_50000/log.txt.train \
${EXP_DIR}/multitask_mean_colors_viper_1_1_1e-3_grl_5000/log.txt.test \
${EXP_DIR}/multitask_mean_colors_viper_1_1_1e-3_grl_5000/log.txt.train \
${EXP_DIR}/multitask_mean_colors_viper_1_1_1e-3_grl_1000/log.txt.test \
${EXP_DIR}/multitask_mean_colors_viper_1_1_1e-3_grl_1000/log.txt.train \
${EXP_DIR}/_debug_multitask_mean_colors_viper_1_0_1e-3_grl_5000/log.txt.test \
${EXP_DIR}/_debug_multitask_mean_colors_viper_1_0_1e-3_grl_5000/log.txt.train \