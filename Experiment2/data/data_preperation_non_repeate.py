import sys
import os
import pandas as pd
import numpy as np

ROOT = os.path.abspath('../../')
print(ROOT)
sys.path.append(ROOT)
DATA_DIR = os.path.join(ROOT, 'data')
OUTPUT_DIR = os.path.abspath('./')

EN_data = pd.read_csv(
    DATA_DIR+'\\jigsaw-toxic-comment-train-processed-seqlen128.csv.zip')
TR_data = pd.read_csv(
    DATA_DIR+'\\compressed_jigsaw-toxic-comment-train_tr_clean.csv.zip')
RU_data = pd.read_csv(
    DATA_DIR+'\\compressed_jigsaw-toxic-comment-train_ru_clean.csv.zip')
IT_data = pd.read_csv(
    DATA_DIR+'\\compressed_jigsaw-toxic-comment-train_it_clean.csv.zip')
FR_data = pd.read_csv(
    DATA_DIR+'\\compressed_jigsaw-toxic-comment-train_fr_clean.csv.zip')
PT_data = pd.read_csv(
    DATA_DIR+'\\compressed_jigsaw-toxic-comment-train_pt_clean.csv.zip')
ES_data = pd.read_csv(
    DATA_DIR+'\\compressed_jigsaw-toxic-comment-train_es_clean.csv.zip')

total_train_size = 85536
toxic_ratio = 0.25  # toxic: 21384


def non_repeat_datasets_random(datasets, tot_size, toxic_ratio, seed=1):
    num_datasets = len(datasets)
    non_toxic_num = int(tot_size*(1-toxic_ratio))
    toxic_num = int(tot_size - non_toxic_num)
    non_toxic_idx = np.random.shuffle(np.arange(0, non_toxic_num, 1))
    toxic_idx = np.random.shuffle(np.arange(0, toxic_num, 1))
    non_toxic_sep = non_toxic_num//num_datasets
    toxic_sep = toxic_num//num_datasets
    output_datasets = []
    for idx, dataset in enumerate(datasets):
        toxic = dataset[['comment_text', 'toxic']][dataset.toxic == 1].sample(
            toxic_num, random_state=seed).iloc[toxic_sep*idx:toxic_sep*(idx+1)]
        non_toxic = dataset[['comment_text', 'toxic']][dataset.toxic == 0].sample(
            non_toxic_num, random_state=seed).iloc[non_toxic_sep*idx:non_toxic_sep*(idx+1)]
        output = pd.concat((toxic, non_toxic), axis=0)
        output_datasets.append(output)
    return output_datasets
# all repeat

# random-repeat

# non-repeate


if __name__ == '__main__':
    # generate {RU,FR,PT} dataset for test 8
    ru_subdata, fr_subdata, pt_subdata = non_repeat_datasets_random(
        [RU_data, FR_data, PT_data], total_train_size, toxic_ratio)

    test8_dataset = pd.concat((ru_subdata, fr_subdata, pt_subdata))
    del(ru_subdata, fr_subdata, pt_subdata)

    # generate {TR,IT,ES} dataset for test 9
    tr_subdata, it_subdata, es_subdata = non_repeat_datasets_random(
        [TR_data, IT_data, ES_data], total_train_size, toxic_ratio)
    test9_dataset = pd.concat((tr_subdata, it_subdata, es_subdata))
    del(tr_subdata, it_subdata, es_subdata)

    # generate {RU,FR,PT,TR,IT,ES} dataset for test 10
    tr_subdata, it_subdata, es_subdata, ru_subdata, fr_subdata, pt_subdata = non_repeat_datasets_random(
        [TR_data, IT_data, ES_data, RU_data, FR_data, PT_data], total_train_size, toxic_ratio)
    test10_dataset = pd.concat(
        (ru_subdata, fr_subdata, pt_subdata, tr_subdata, it_subdata, es_subdata))
    del(tr_subdata, it_subdata, es_subdata, ru_subdata, fr_subdata, pt_subdata)

    # save all dataset to output
    dataset_list = [test8_dataset, test9_dataset, test10_dataset]

    for idx, d in enumerate(dataset_list):
        d.to_csv(OUTPUT_DIR+'\\test{0}_training_data.csv'.format(idx+8))
