LOG_DIR=logs

EXP_DIR=//media/storage/eustinova/CUHK03_bn/

LIST=(cuhk03_labeled_split1_bn_with_scale_and_bias/ cuhk03_labeled_split2_bn cuhk03_labeled_split3_bn cuhk03_labeled_split3_bn cuhk03_labeled_split2_bn_wd_5e-05  cuhk03_labeled_split2_bn_wo_bn_first cuhk03_labeled_split1_bn_wd_5e-05) 
for VALUE in "${LIST[@]}"
do
  echo ${VALUE}
  python parse_log.py ${EXP_DIR}/${VALUE}/${LOG_DIR}/log.txt ${EXP_DIR}/${VALUE}/ 
  
done

EXP_DIR_OLD=/media/storage/eustinova/segmentation/multitask3_cuhk03/

LIST=(cuhk03_split2_160_60_64_64_500 cuhk03_split3_160_60_64_64_500) 
for VALUE in "${LIST[@]}"
do
  echo ${VALUE}
  python parse_log.py ${EXP_DIR_OLD}/${VALUE}/${LOG_DIR}/log.txt ${EXP_DIR_OLD}/${VALUE}/ 
done

python plot_log.py -f loss -f loss -f loss -f loss -f loss -f loss -f loss -f loss -f loss -f loss \
-p cuhk03_labeled_split2_bn_test -p cuhk03_labeled_split2_bn_train  \
-p cuhk03_labeled_split2_bn_wd_5e-05_test -p cuhk03_labeled_split2_bn_wd_5e-05_train \
-p cuhk03_labeled_split2_bn_wo_bn_first_test -p cuhk03_labeled_split2_bn_wo_bn_first_train \
-p cuhk03_split2_160_60_64_64_500_test -p cuhk03_split2_160_60_64_64_500_train \
-p cuhk03_labeled_split1_bn_with_scale_and_bias_test -p cuhk03_labeled_split1_bn_with_scale_and_bias_train \
${EXP_DIR}/cuhk03_labeled_split2_bn/log.txt.test \
${EXP_DIR}/cuhk03_labeled_split2_bn/log.txt.train \
${EXP_DIR}/cuhk03_labeled_split2_bn_wd_5e-05/log.txt.test \
${EXP_DIR}/cuhk03_labeled_split2_bn_wd_5e-05/log.txt.train \
${EXP_DIR}/cuhk03_labeled_split2_bn_wo_bn_first/log.txt.test \
${EXP_DIR}/cuhk03_labeled_split2_bn_wo_bn_first/log.txt.train \
${EXP_DIR_OLD}/cuhk03_split2_160_60_64_64_500/log.txt.test \
${EXP_DIR_OLD}/cuhk03_split2_160_60_64_64_500/log.txt.train \
${EXP_DIR}/cuhk03_labeled_split1_bn_with_scale_and_bias/log.txt.test \
${EXP_DIR}/cuhk03_labeled_split1_bn_with_scale_and_bias/log.txt.train \

python plot_log.py -f loss \
-p test \
${EXP_DIR}/cuhk03_labeled_split1_bn_with_scale_and_bias/log.txt.test \

python plot_log.py -f loss -f loss -f loss -f loss  \
-p cuhk03_labeled_split3_bn_test -p cuhk03_labeled_split3_bn_train  \
-p cuhk03_split2_160_60_64_64_500_test -p cuhk03_split2_160_60_64_64_500_train \
${EXP_DIR}/cuhk03_labeled_split3_bn/log.txt.test \
${EXP_DIR}/cuhk03_labeled_split3_bn/log.txt.train \
${EXP_DIR_OLD}/cuhk03_split3_160_60_64_64_500/log.txt.test \
${EXP_DIR_OLD}/cuhk03_split3_160_60_64_64_500/log.txt.train \
  









