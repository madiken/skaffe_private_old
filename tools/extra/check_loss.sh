# EXP_DIR=//media/storage/eustinova/segmentation/multitask1/


# python plot_log.py  -f loss -f loss  -f loss  -p 0_test_loss  -p 1e_2_test_loss  -p 1_test_loss \
# ${EXP_DIR}/multitask_0_num_out_64_64/log.txt.test \
# ${EXP_DIR}/multitask_1e-2_num_out_64_64/log.txt.test \
# ${EXP_DIR}/multitask_1_num_out_64_64/log.txt.test \

# python plot_log.py  -f loss1 -f loss1  -f loss1  -p 0_test_loss_segm  -p 1e_2_test_loss_segm  -p 1_test_loss_segm \
# ${EXP_DIR}/multitask_0_num_out_64_64/log.txt.test \
# ${EXP_DIR}/multitask_1e-2_num_out_64_64/log.txt.test \
# ${EXP_DIR}/multitask_1_num_out_64_64/log.txt.test \

# python plot_log.py  -f loss -f loss  -f loss  -p 0_train_loss  -p 1e_2_train_loss  -p 1_train_loss \
# ${EXP_DIR}/multitask_0_num_out_64_64/log.txt.train \
# ${EXP_DIR}/multitask_1e-2_num_out_64_64/log.txt.train \
# ${EXP_DIR}/multitask_1_num_out_64_64/log.txt.train \



EXP_DIR=/media/storage/eustinova/segmentation/multitask1_grl_cuhk03/

LOG_DIR=logs

LIST=(multitask_segm_grl_5000 multitask_segm_grl_50000 multitask_segm_grl_500000 baseline_check baseline_check_1 multitask_segm_1_1  multitask_segm_1e-2_1 multitask_segm_grl_50000_1_1e-2 multitask_segm_grl_50000_1_1 multitask_segm_grl_50000_1e-2_1 multitask_segm_grl_500000_1e-2_1  )
for VALUE in "${LIST[@]}"
do
  echo ${VALUE}
  python parse_log.py ${EXP_DIR}/${VALUE}/${LOG_DIR}/log.txt ${EXP_DIR}/${VALUE}/ 
  
done




python parse_log.py /media/storage/eustinova/segmentation/multitask3_cuhk03/cuhk03_labeled_split1_sanity_check_norm//${LOG_DIR}/log.txt /media/storage/eustinova/segmentation/multitask3_cuhk03/cuhk03_labeled_split1_sanity_check_norm/ 

python plot_log.py  -f loss -f loss -f loss   -f loss -p baseline -p baseline_1 -p bs_initial -p bs_norm \
${EXP_DIR}/baseline_check/log.txt.test \
${EXP_DIR}/baseline_check_1/log.txt.test \
/media/storage/eustinova/segmentation/multitask3_cuhk03/cuhk03_160_60_PPSS_pca_masks_prediction_pca_64_64_64_500_pool2_split_0_1/log.txt.test \
/media/storage/eustinova/segmentation/multitask3_cuhk03/cuhk03_labeled_split1_sanity_check_norm/log.txt.test

python plot_log.py  -f loss -f loss  -f loss  -f loss  -p baseline -p baseline_1 -p baseline_initial -p bs_norm \
${EXP_DIR}/baseline_check/log.txt.train  \
${EXP_DIR}/baseline_check_1/log.txt.train \
/media/storage/eustinova/segmentation/multitask3_cuhk03/cuhk03_160_60_PPSS_pca_masks_prediction_pca_64_64_64_500_pool2_split_0_1/log.txt.train \
/media/storage/eustinova/segmentation/multitask3_cuhk03/cuhk03_labeled_split1_sanity_check_norm/log.txt.train

#############  multitask with grl
python plot_log.py  -f loss -f loss  -f loss  -f loss  -f loss -f loss  -f loss  -f loss  -p 5000_1e-2_1e-2_test_loss -p 50000_1e-2_1e-2_test_loss -p 50000_1_1e-2_test_loss  -p 50000_1e-2_1_test_loss -p 50000_1_1_test_loss  -p 500000_1e-2_1e-2_test_loss -p 500000_1e-2_1_test_loss  -p baseline \
${EXP_DIR}/multitask_segm_grl_5000/log.txt.test \
${EXP_DIR}/multitask_segm_grl_50000/log.txt.test \
${EXP_DIR}/multitask_segm_grl_50000_1_1e-2/log.txt.test \
${EXP_DIR}/multitask_segm_grl_50000_1e-2_1/log.txt.test \
${EXP_DIR}/multitask_segm_grl_50000_1_1/log.txt.test \
${EXP_DIR}/multitask_segm_grl_500000/log.txt.test \
${EXP_DIR}/multitask_segm_grl_500000_1e-2_1/log.txt.test \
${EXP_DIR}/baseline_check/log.txt.test \

python plot_log.py  -f loss1 -f loss1  -f loss1  -f loss1 -f loss1 -f loss1  -f loss1  -f loss1 -p 5000_1e-2_1e-2_test_loss -p 50000_1e-2_1e-2_test_loss -p 50000_1_1e-2_test_loss -p 50000_1e-2_1_test_loss  -p 50000_1_1_test_loss  -p 500000_1e-2_1e-2_test_loss -p 500000_1e-2_1_test_loss -p baseline \
${EXP_DIR}/multitask_segm_grl_5000/log.txt.test \
${EXP_DIR}/multitask_segm_grl_50000/log.txt.test \
${EXP_DIR}/multitask_segm_grl_50000_1_1e-2/log.txt.test \
${EXP_DIR}/multitask_segm_grl_50000_1e-2_1/log.txt.test \
${EXP_DIR}/multitask_segm_grl_50000_1_1/log.txt.test \
${EXP_DIR}/multitask_segm_grl_500000/log.txt.test \
${EXP_DIR}/multitask_segm_grl_500000_1e-2_1/log.txt.test \
${EXP_DIR}/baseline_check/log.txt.test \


python plot_log.py  -f loss -f loss  -f loss  -f loss  -f loss -f loss  -f loss -f loss  -p 5000_1e-2_1e-2_train_loss -p 50000_1e-2_1e-2_train_loss -p 50000_1_1e-2_train_loss -p 50000_1e-2_1_train_loss  -p 50000_1_1_train_loss  -p 500000_1e-2_1e-2_train_loss -p 500000_1e-2_1_train_loss -p baseline \
${EXP_DIR}/multitask_segm_grl_5000/log.txt.train \
${EXP_DIR}/multitask_segm_grl_50000/log.txt.train \
${EXP_DIR}/multitask_segm_grl_50000_1_1e-2/log.txt.train \
${EXP_DIR}/multitask_segm_grl_50000_1e-2_1/log.txt.train \
${EXP_DIR}/multitask_segm_grl_50000_1_1/log.txt.train \
${EXP_DIR}/multitask_segm_grl_500000/log.txt.train \
${EXP_DIR}/multitask_segm_grl_500000_1e-2_1/log.txt.train \
${EXP_DIR}/baseline_check/log.txt.train \



python plot_log.py  -f loss1 -f loss1  -f loss1  -f loss1  -f loss1 -f loss1 -f loss1  -f loss1  -p 5000_train_loss -p 50000_train_loss_1e-2_1e-2 -p 50000_train_loss_1_1e-2 -p 50000_train_loss_1e-2_1  -p 50000_train_loss_1_1  -p 500000_train_loss -p baseline \
${EXP_DIR}/multitask_segm_grl_5000/log.txt.train \
${EXP_DIR}/multitask_segm_grl_50000/log.txt.train \
${EXP_DIR}/multitask_segm_grl_50000_1_1e-2/log.txt.train \
${EXP_DIR}/multitask_segm_grl_50000_1e-2_1/log.txt.train \
${EXP_DIR}/multitask_segm_grl_50000_1_1/log.txt.train \
${EXP_DIR}/multitask_segm_grl_500000/log.txt.train \
${EXP_DIR}/baseline_check/log.txt.train \

###### multitask
python plot_log.py  -f loss -f loss  -f loss  -p baseline_check_test -p multitask_segm_1_1_test -p multitask_segm_1e_2_1_test \
${EXP_DIR}/baseline_check/log.txt.test \
${EXP_DIR}/multitask_segm_1_1/log.txt.test \
${EXP_DIR}/multitask_segm_1e-2_1/log.txt.test \


python plot_log.py  -f loss1 -f loss1  -f loss1  -p baseline_check_test_segm -p multitask_segm_1_1_test_segm -p multitask_segm_1e_2_1_test_segm \
${EXP_DIR}/baseline_check/log.txt.test \
${EXP_DIR}/multitask_segm_1_1/log.txt.test \
${EXP_DIR}/multitask_segm_1e-2_1/log.txt.test \

python plot_log.py  -f loss -f loss  -f loss  -p baseline_check_train -p multitask_segm_1_1_train -p multitask_segm_1e_2_1_train \
${EXP_DIR}/baseline_check/log.txt.train \
${EXP_DIR}/multitask_segm_1_1/log.txt.train \
${EXP_DIR}/multitask_segm_1e-2_1/log.txt.train \
