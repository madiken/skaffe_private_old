



# python parse_log.py /media/storage/eustinova/segmentation/multitask3_viper/multitask_viper_1_0/logs/log.txt /media/storage/eustinova/segmentation/multitask3_viper/multitask_viper_1_0/ 
# python parse_log.py /media/storage/eustinova/segmentation/multitask3_viper/simple_multitask_viper_1_0/logs/log.txt /media/storage/eustinova/segmentation/multitask3_viper/simple_multitask_viper_1_0/ 
# python parse_log.py /media/hpc2_storage/eustinova/segmentation/multitask3_viper/multitask_viper_1_0_plus_fc_10000_128/logs/log.txt /media/hpc2_storage/eustinova/segmentation/multitask3_viper/multitask_viper_1_0_plus_fc_10000_128/ 
# python parse_log.py /media/hpc2_storage/eustinova/segmentation/multitask3_viper/multitask_viper_1_0_plus_fc_10000_128/logs/log.txt /media/hpc2_storage/eustinova/segmentation/multitask3_viper/multitask_viper_1_0_plus_fc_10000_128/ 
# python parse_log.py /media/hpc2_storage/eustinova/segmentation/multitask3_viper/multitask_viper_1_0_plus_fc_10000_128/logs/log.txt /media/hpc2_storage/eustinova/segmentation/multitask3_viper/multitask_viper_1_0_plus_fc_10000_128/ 
# python parse_log.py /media/hpc2_storage/eustinova/segmentation/multitask3_viper/simple_multitask_viper_1_0_2048/logs/log.txt /media/hpc2_storage/eustinova/segmentation/multitask3_viper/simple_multitask_viper_1_0_2048/ 
# python parse_log.py /media/hpc2_storage/eustinova/segmentation/multitask3_viper/multitask_viper_1_0_plus_fc_10000_128_less_lr_1e-4/logs/log.txt /media/hpc2_storage/eustinova/segmentation/multitask3_viper/multitask_viper_1_0_plus_fc_10000_128_less_lr_1e-4/ 
# python parse_log.py /media/hpc2_storage/eustinova/segmentation/multitask3_viper/simple_multitask_viper_1_0_2048_less_lr_1e-4/logs/log.txt /media/hpc2_storage/eustinova/segmentation/multitask3_viper/simple_multitask_viper_1_0_2048_less_lr_1e-4/ 

# python plot_log.py  -f loss_segm -f loss_segm  -f loss_segm -f loss_segm -f loss_segm -f loss_segm -f loss_segm -f loss_segm  -p loss_segm_test -p loss_segm_train -p simple_loss_segm_test -p simple_loss_segm_train -p simple_loss_segm_test_2048_less_lr_1e-4 -p simple_loss_segm_train_2048_less_lr_1e-4 -p simple_loss_segm_test_10000_128_less_lr_1e-4 -p simple_loss_segm_train_10000_128_less_lr_1e-4 \
# /media/storage/eustinova/segmentation/multitask3_viper/multitask_viper_1_0/log.txt.test \
# /media/storage/eustinova/segmentation/multitask3_viper/multitask_viper_1_0/log.txt.train \
# /media/storage/eustinova/segmentation/multitask3_viper/simple_multitask_viper_1_0/log.txt.test \
# /media/storage/eustinova/segmentation/multitask3_viper/simple_multitask_viper_1_0/log.txt.train \
# /media/storage/eustinova/segmentation/multitask3_viper/simple_multitask_viper_1_0_2048_less_lr_1e-4/log.txt.test \
# /media/storage/eustinova/segmentation/multitask3_viper/simple_multitask_viper_1_0_2048_less_lr_1e-4/log.txt.train \
# /media/storage/eustinova/segmentation/multitask3_viper/multitask_viper_1_0_plus_fc_10000_128_less_lr_1e-4/log.txt.test \
# /media/storage/eustinova/segmentation/multitask3_viper/multitask_viper_1_0_plus_fc_10000_128_less_lr_1e-4/log.txt.train

# /media/storage/eustinova/segmentation/multitask3_viper/simple_multitask_viper_1_0_2048/log.txt.test \
# /media/storage/eustinova/segmentation/multitask3_viper/simple_multitask_viper_1_0_2048/log.txt.train \
# /media/storage/eustinova/segmentation/multitask3_viper/multitask_viper_1_0_plus_fc_10000_128/log.txt.test \
# /media/storage/eustinova/segmentation/multitask3_viper/multitask_viper_1_0_plus_fc_10000_128/log.txt.train \

  

EXP_DIR=/media/storage/eustinova/segmentation/multitask3_viper/

LOG_DIR=logs

LIST=(grl_multitask_viper_1_1_reid_500_segm_500_128_1e-4 grl_multitask_viper_1_1_reid_500_500_segm_500_128_1e-4 multitask_viper_1_1_reid_500_500_segm_500_128_1e-4 multitask_viper_1_1_reid_500_segm_500_128_1e-4 baseline_500_1e-3 baseline_500_500_1e-3 multitask_viper_1_0_128_1e-3 multitask_viper_1_0_500_128_1e-4 multitask_viper_1_0_500_128_1e-3 multitask_viper_1_0_2048_1e-3 multitask_viper_1_0_2048_1e-4 multitask_viper_1_0_10000_128_1e-4 multitask_viper_1_0_10000_128_1e-3 multitask_viper_1_1_reid_500_500_segm_500_128_1e-3 multitask_viper_1_1_reid_500_segm_128_1e-3)
for VALUE in "${LIST[@]}"
do
  echo ${VALUE}
  python parse_log.py ${EXP_DIR}/${VALUE}/${LOG_DIR}/log.txt ${EXP_DIR}/${VALUE}/ 
  
done

python plot_log.py  -f loss_segm  -f loss_segm  -f loss_segm -f loss_segm  -f loss_segm -f loss_segm -f loss_segm -f loss_segm \
-p multitask_viper_1_0_2048_1e-4_test -p multitask_viper_1_0_2048_1e-4_train \
-p multitask_viper_1_0_10000_128_1e-4_test -p multitask_viper_1_0_10000_128_1e-4_train \
-p multitask_viper_1_0_128_1e-3_test -p multitask_viper_1_0_128_1e-3_train  -p multitask_viper_1_0_500_128_1e-4_test -p multitask_viper_1_0_500_128_1e-4_train \
${EXP_DIR}/multitask_viper_1_0_2048_1e-4/log.txt.test \
${EXP_DIR}/multitask_viper_1_0_2048_1e-4/log.txt.train \
${EXP_DIR}/multitask_viper_1_0_10000_128_1e-4/log.txt.test \
${EXP_DIR}/multitask_viper_1_0_10000_128_1e-4/log.txt.train \
${EXP_DIR}/multitask_viper_1_0_128_1e-3/log.txt.test \
${EXP_DIR}/multitask_viper_1_0_128_1e-3/log.txt.train \
${EXP_DIR}/multitask_viper_1_0_500_128_1e-4/log.txt.test \
${EXP_DIR}/multitask_viper_1_0_500_128_1e-4/log.txt.train \
#${EXP_DIR}/multitask_viper_1_0_500_128_1e-3/log.txt.test \
#${EXP_DIR}/multitask_viper_1_0_500_128_1e-3/log.txt.train \


python plot_log.py  -f loss  -f loss  -f loss -f loss \
-p baseline_500_1e-3_test -p baseline_500_1e-3_train -p baseline_500_500_1e-3_test -p baseline_500_500_1e-3_train \
${EXP_DIR}/baseline_500_1e-3/log.txt.test \
${EXP_DIR}/baseline_500_1e-3/log.txt.train \
${EXP_DIR}/baseline_500_500_1e-3/log.txt.test \
${EXP_DIR}/baseline_500_500_1e-3/log.txt.train \


python plot_log.py  -f loss  -f loss  -f loss -f loss -f loss -f loss -f loss -f loss \
-p multitask_viper_1_1_reid_500_500_segm_500_128_1e-3_test -p multitask_viper_1_1_reid_500_500_segm_500_128_1e-3_train -p multitask_viper_1_1_reid_500_500_segm_500_128_1e-4_test -p multitask_viper_1_1_reid_500_500_segm_500_128_1e-4_train -p multitask_viper_1_1_reid_500_segm_128_1e-3_test -p multitask_viper_1_1_reid_500_segm_128_1e-3_train -p multitask_viper_1_1_reid_500_segm_500_128_1e-4_test -p multitask_viper_1_1_reid_500_segm_500_128_1e-4_train \
${EXP_DIR}/multitask_viper_1_1_reid_500_500_segm_500_128_1e-3/log.txt.test \
${EXP_DIR}/multitask_viper_1_1_reid_500_500_segm_500_128_1e-3/log.txt.train \
${EXP_DIR}/multitask_viper_1_1_reid_500_500_segm_500_128_1e-4/log.txt.test \
${EXP_DIR}/multitask_viper_1_1_reid_500_500_segm_500_128_1e-4/log.txt.train \
${EXP_DIR}/multitask_viper_1_1_reid_500_segm_128_1e-3/log.txt.test \
${EXP_DIR}/multitask_viper_1_1_reid_500_segm_128_1e-3/log.txt.train \
${EXP_DIR}/multitask_viper_1_1_reid_500_segm_500_128_1e-4/log.txt.test \
${EXP_DIR}/multitask_viper_1_1_reid_500_segm_500_128_1e-4/log.txt.train 


python plot_log.py  -f loss  -f loss  -f loss -f loss \
-p grl_multitask_viper_1_1_reid_500_500_segm_500_128_1e-4_test -p grl_multitask_viper_1_1_reid_500_500_segm_500_128_1e-4_train -p grl_multitask_viper_1_1_reid_500_segm_500_128_1e-4_test -p grl_multitask_viper_1_1_reid_500_segm_500_128_1e-4_train  \
${EXP_DIR}/grl_multitask_viper_1_1_reid_500_500_segm_500_128_1e-4/log.txt.test \
${EXP_DIR}/grl_multitask_viper_1_1_reid_500_500_segm_500_128_1e-4/log.txt.train \
${EXP_DIR}/grl_multitask_viper_1_1_reid_500_segm_500_128_1e-4/log.txt.test \
${EXP_DIR}/grl_multitask_viper_1_1_reid_500_segm_500_128_1e-4/log.txt.train \

