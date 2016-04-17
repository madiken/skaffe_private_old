EXP_DIR3=/media/storage/eustinova/segmentation/mean_color_prediction_viper/3
LOG_DIR=logs
LIST=(mt_grl_batchn_eliminate_pos_same_camera_grl_50000_mean_colors_only)
for VALUE in "${LIST[@]}"
do
  echo ${VALUE}
  python parse_log.py ${EXP_DIR3}/${VALUE}/${LOG_DIR}/log.txt ${EXP_DIR3}/${VALUE}/ 
  
done

python plot_log.py -f loss_mean_colors -f loss_mean_colors -p train -p test \
/media/storage/eustinova/segmentation/mean_color_prediction_viper/3/mt_grl_batchn_eliminate_pos_same_camera_grl_50000_mean_colors_only/log.txt.train \
/media/storage/eustinova/segmentation/mean_color_prediction_viper/3/mt_grl_batchn_eliminate_pos_same_camera_grl_50000_mean_colors_only/log.txt.test \
