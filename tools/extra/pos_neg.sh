 

EXP_DIR=//media/storage/eustinova/CUHK03_bn/

LOG_DIR=logs

# LIST=(cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_0.5 cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_1 cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000 cuhk03_labeled_split1_batch_pos_neg_128_hnm_1_45000 cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_20000 cuhk03_labeled_split1_batch_pos_neg_128_hnm_4 cuhk03_labeled_split1_batch_pos_neg_128_hnm_1 cuhk03_labeled_split1_batch_pos_neg_192 cuhk03_labeled_split1_batch_pos_neg_64 cuhk03_labeled_split1_batch_pos_neg_128)
# for VALUE in "${LIST[@]}"
# do
#   echo ${VALUE}
#   python parse_log.py ${EXP_DIR}/${VALUE}/${LOG_DIR}/log.txt ${EXP_DIR}/${VALUE}/ 
  
# done

LIST=(cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_update_5000) #cuhk03_labeled_split1_batch_pos_neg_128_wo_too_hard_0.85 cuhk03_labeled_split1_batch_pos_neg_128 cuhk03_labeled_split1_batch_pos_neg_128_wo_too_hard_0.6 ) #cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_100_rand_pos_fix cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0.4_rand_pos cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0_fix_1 /cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0.4_fix_elim_possame/ cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0_fix_ cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0.4_fix //cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0.6_fix /cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_1_semi_16_random cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0.6_semi_16_random cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0.6_semi_16_hard cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0.7 cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0.6 cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0.95 cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0.85)
for VALUE in "${LIST[@]}"
do
  echo ${VALUE}
  python parse_log.py ${EXP_DIR}/${VALUE}/${LOG_DIR}/log.txt ${EXP_DIR}/${VALUE}/ 
  
done

e=cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_update_5000
python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e}/log.txt.test \
${EXP_DIR}/${e}/log.txt.test  \
${EXP_DIR}/${e}/log.txt.train \
${EXP_DIR}/${e}/log.txt.train \


e=/cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_100_rand_pos_fix
python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e}/log.txt.test \
${EXP_DIR}/${e}/log.txt.test  \
${EXP_DIR}/${e}/log.txt.train \
${EXP_DIR}/${e}/log.txt.train \


e=cuhk03_labeled_split1_batch_pos_neg_128_wo_too_hard_0.85
python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e}/log.txt.test \
${EXP_DIR}/${e}/log.txt.test  \
${EXP_DIR}/${e}/log.txt.train \
${EXP_DIR}/${e}/log.txt.train \


e=cuhk03_labeled_split1_batch_pos_neg_128
python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e}/log.txt.test \
${EXP_DIR}/${e}/log.txt.test  \
${EXP_DIR}/${e}/log.txt.train \
${EXP_DIR}/${e}/log.txt.train \


e=cuhk03_labeled_split1_batch_pos_neg_128_wo_too_hard_0.6
python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e}/log.txt.test \
${EXP_DIR}/${e}/log.txt.test  \
${EXP_DIR}/${e}/log.txt.train \
${EXP_DIR}/${e}/log.txt.train \

e=cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_100_rand_pos_fix
python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e}/log.txt.test \
${EXP_DIR}/${e}/log.txt.test  \
${EXP_DIR}/${e}/log.txt.train \
${EXP_DIR}/${e}/log.txt.train \


e=cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0.4_rand_pos
python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e}/log.txt.test \
${EXP_DIR}/${e}/log.txt.test  \
${EXP_DIR}/${e}/log.txt.train \
${EXP_DIR}/${e}/log.txt.train \


e=/cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0_fix_1/
python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e}/log.txt.test \
${EXP_DIR}/${e}/log.txt.test  \
${EXP_DIR}/${e}/log.txt.train \
${EXP_DIR}/${e}/log.txt.train \


e=/cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0.4_fix_elim_possame/
python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e}/log.txt.test \
${EXP_DIR}/${e}/log.txt.test  \
${EXP_DIR}/${e}/log.txt.train \
${EXP_DIR}/${e}/log.txt.train \


e=/cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0_fix_
python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e}/log.txt.test \
${EXP_DIR}/${e}/log.txt.test  \
${EXP_DIR}/${e}/log.txt.train \
${EXP_DIR}/${e}/log.txt.train \


e=/cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0.4_fix
python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e}/log.txt.test \
${EXP_DIR}/${e}/log.txt.test  \
${EXP_DIR}/${e}/log.txt.train \
${EXP_DIR}/${e}/log.txt.train \


e=/cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0.6_fix
python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e}/log.txt.test \
${EXP_DIR}/${e}/log.txt.test  \
${EXP_DIR}/${e}/log.txt.train \
${EXP_DIR}/${e}/log.txt.train \

e=/cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_1_semi_16_random
python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e}/log.txt.test \
${EXP_DIR}/${e}/log.txt.test  \
${EXP_DIR}/${e}/log.txt.train \
${EXP_DIR}/${e}/log.txt.train \


e=cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0.6_semi_16_hard
python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e}/log.txt.test \
${EXP_DIR}/${e}/log.txt.test  \
${EXP_DIR}/${e}/log.txt.train \
${EXP_DIR}/${e}/log.txt.train \

e=/cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0.6_semi_16_random
python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e}/log.txt.test \
${EXP_DIR}/${e}/log.txt.test  \
${EXP_DIR}/${e}/log.txt.train \
${EXP_DIR}/${e}/log.txt.train \

e=/cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0.7
python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e}/log.txt.test \
${EXP_DIR}/${e}/log.txt.test  \
${EXP_DIR}/${e}/log.txt.train \
${EXP_DIR}/${e}/log.txt.train \

e=/cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0.6
python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e}/log.txt.test \
${EXP_DIR}/${e}/log.txt.test  \
${EXP_DIR}/${e}/log.txt.train \
${EXP_DIR}/${e}/log.txt.train \

e=/cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0.85
python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e}/log.txt.test \
${EXP_DIR}/${e}/log.txt.test  \
${EXP_DIR}/${e}/log.txt.train \
${EXP_DIR}/${e}/log.txt.train \

e=/cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_thr_0.95
python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e}/log.txt.test \
${EXP_DIR}/${e}/log.txt.test  \
${EXP_DIR}/${e}/log.txt.train \
${EXP_DIR}/${e}/log.txt.train \

e=/cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_0.5
python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e}/log.txt.test \
${EXP_DIR}/${e}/log.txt.test  \
${EXP_DIR}/${e}/log.txt.train \
${EXP_DIR}/${e}/log.txt.train \


e=/cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000_1
python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e}/log.txt.test \
${EXP_DIR}/${e}/log.txt.test  \
${EXP_DIR}/${e}/log.txt.train \
${EXP_DIR}/${e}/log.txt.train \


e=/cuhk03_labeled_split1_batch_pos_neg_128_hnm_4_45000
python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e}/log.txt.test \
${EXP_DIR}/${e}/log.txt.test  \
${EXP_DIR}/${e}/log.txt.train \
${EXP_DIR}/${e}/log.txt.train \


e=/cuhk03_labeled_split1_batch_pos_neg_128_hnm_1_45000
python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e}/log.txt.test \
${EXP_DIR}/${e}/log.txt.test  \
${EXP_DIR}/${e}/log.txt.train \
${EXP_DIR}/${e}/log.txt.train \

e1=cuhk03_labeled_split1_batch_pos_neg_192
 python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e1}/log.txt.test \
${EXP_DIR}/${e1}/log.txt.test  \
${EXP_DIR}/${e1}/log.txt.train \
${EXP_DIR}/${e1}/log.txt.train \



e2=cuhk03_labeled_split1_batch_pos_neg_64
 python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e2}/log.txt.test \
${EXP_DIR}/${e2}/log.txt.test  \
${EXP_DIR}/${e2}/log.txt.train \
${EXP_DIR}/${e2}/log.txt.train \



e3=cuhk03_labeled_split1_batch_pos_neg_128
 python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e3}/log.txt.test \
${EXP_DIR}/${e3}/log.txt.test  \
${EXP_DIR}/${e3}/log.txt.train \
${EXP_DIR}/${e3}/log.txt.train \

e4=cuhk03_labeled_split1_batch_pos_neg_128_hnm_4
 python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e4}/log.txt.test \
${EXP_DIR}/${e4}/log.txt.test  \
${EXP_DIR}/${e4}/log.txt.train \
${EXP_DIR}/${e4}/log.txt.train \


e5=cuhk03_labeled_split1_batch_pos_neg_128_hnm_1
 python plot_log.py -f loss_pos -f loss_neg -f loss_pos -f loss_neg  -p los_pos_test -p loss_neg_test -p los_pos -p loss_neg  \
${EXP_DIR}/${e5}/log.txt.test \
${EXP_DIR}/${e5}/log.txt.test  \
${EXP_DIR}/${e5}/log.txt.train \
${EXP_DIR}/${e5}/log.txt.train \
