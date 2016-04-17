
python parse_log.py /media/storage/eustinova/CUHK03_bn/market_with_add_data_128_28_separate_300_conf09_300/logs/log.txt  /media/storage/eustinova/CUHK03_bn/market_with_add_data_128_28_separate_300_conf09_300/

python parse_log.py /media/storage/eustinova/CUHK03_bn/market_market_128_28_separate/logs/log.txt  /media/storage/eustinova/CUHK03_bn/market_market_128_28_separate/


python parse_log.py /media/storage/eustinova/segmentation/multitask3_cuhk03/market_1501_160_60_wo_pretr//logs/log.txt  /media/storage/eustinova/segmentation/multitask3_cuhk03/market_1501_160_60_wo_pretr/


python plot_log.py -f loss -f loss -f loss -p market_with_add_data_128_28_separate_300_conf09_300 -p market_market_128_28_separate -p market_1501_160_60_wo_pretr \
/media/storage/eustinova/CUHK03_bn/market_with_add_data_128_28_separate_300_conf09_300/log.txt.test \
/media/storage/eustinova/CUHK03_bn/market_market_128_28_separate/log.txt.test \
/media/storage/eustinova/segmentation/multitask3_cuhk03/market_1501_160_60_wo_pretr//log.txt.test 