import sys
import os
import pandas as pd

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


def create_dataset_random(data, tot_size, toxic_ratio, seed=1):
    non_toxic_num = int(tot_size*(1-toxic_ratio))
    toxic_num = int(tot_size - non_toxic_num)
    toxic = data[['toxic', 'comment_text']][data.toxic ==
                                            1].sample(toxic_num, random_state=seed)
    non_toxic = data[['toxic', 'comment_text']][data.toxic ==
                                                0].sample(non_toxic_num, random_state=seed)
    output = pd.concat((toxic, non_toxic), axis=0)
    return output


# all repeat

# random-repeat

# non-repeate

if __name__ == '__main__':

    # Sort and align EN_dataset as others
    temp = FR_data.join(EN_data[['id', 'comment_text', 'toxic']].set_index(
        'id'), lsuffix=('_fr'), on='id')
    EN_data_aligned = temp[['comment_text', 'toxic']]

    # generate monolingual dataset for test 1
    en_dataset = create_dataset_random(
        EN_data_aligned, total_train_size, toxic_ratio)
    tr_dataset = create_dataset_random(TR_data, total_train_size, toxic_ratio)
    ru_dataset = create_dataset_random(RU_data, total_train_size, toxic_ratio)
    it_dataset = create_dataset_random(IT_data, total_train_size, toxic_ratio)
    fr_dataset = create_dataset_random(FR_data, total_train_size, toxic_ratio)
    es_dataset = create_dataset_random(PT_data, total_train_size, toxic_ratio)
    pt_dataset = create_dataset_random(ES_data, total_train_size, toxic_ratio)

    # generate {RU,FR,PT} dataset for test 2
    subdataset_num = 85536//3
    ru_subdata = create_dataset_random(ru_dataset, subdataset_num, toxic_ratio)
    fr_subdata = create_dataset_random(fr_dataset, subdataset_num, toxic_ratio)
    pt_subdata = create_dataset_random(pt_dataset, subdataset_num, toxic_ratio)
    test2_dataset = pd.concat((ru_subdata, fr_subdata, pt_subdata))
    del(ru_subdata, fr_subdata, pt_subdata)

    # generate (EN,FR,RU,PT) dataset for test 3
    subdataset_num = 85536//4
    en_subdata = create_dataset_random(en_dataset, subdataset_num, toxic_ratio)
    ru_subdata = create_dataset_random(ru_dataset, subdataset_num, toxic_ratio)
    fr_subdata = create_dataset_random(fr_dataset, subdataset_num, toxic_ratio)
    pt_subdata = create_dataset_random(pt_dataset, subdataset_num, toxic_ratio)
    test3_dataset = pd.concat((en_subdata, ru_subdata, fr_subdata, pt_subdata))
    del(ru_subdata, fr_subdata, pt_subdata)

    # generate {TR,IT,ES} dataset for test 4
    subdataset_num = 85536//3
    tr_subdata = create_dataset_random(tr_dataset, subdataset_num, toxic_ratio)
    it_subdata = create_dataset_random(it_dataset, subdataset_num, toxic_ratio)
    es_subdata = create_dataset_random(es_dataset, subdataset_num, toxic_ratio)
    test4_dataset = pd.concat((tr_subdata, it_subdata, es_subdata))
    del(tr_subdata, it_subdata, es_subdata)

    # generate {EN,TR,IT,ES} datasetfor test 5
    subdataset_num = 85536//4
    tr_subdata = create_dataset_random(tr_dataset, subdataset_num, toxic_ratio)
    it_subdata = create_dataset_random(it_dataset, subdataset_num, toxic_ratio)
    es_subdata = create_dataset_random(es_dataset, subdataset_num, toxic_ratio)
    test5_dataset = pd.concat((en_subdata, tr_subdata, it_subdata, es_subdata))
    del(tr_subdata, it_subdata, es_subdata)

    # generate {RU,FR,PT,TR,IT,ES} dataset for test 6
    subdataset_num = 85536//6
    tr_subdata = create_dataset_random(tr_dataset, subdataset_num, toxic_ratio)
    it_subdata = create_dataset_random(it_dataset, subdataset_num, toxic_ratio)
    es_subdata = create_dataset_random(es_dataset, subdataset_num, toxic_ratio)
    ru_subdata = create_dataset_random(ru_dataset, subdataset_num, toxic_ratio)
    fr_subdata = create_dataset_random(fr_dataset, subdataset_num, toxic_ratio)
    pt_subdata = create_dataset_random(pt_dataset, subdataset_num, toxic_ratio)

    test6_dataset = pd.concat(
        (ru_subdata, fr_subdata, pt_subdata, tr_subdata, it_subdata, es_subdata))
    del(tr_subdata, it_subdata, es_subdata, ru_subdata, fr_subdata, pt_subdata)

    # generate {EN,RU,FR,PT,TR,IT,ES} dataset for test 7
    subdataset_num = 85536//7
    tr_subdata = create_dataset_random(tr_dataset, subdataset_num, toxic_ratio)
    it_subdata = create_dataset_random(it_dataset, subdataset_num, toxic_ratio)
    es_subdata = create_dataset_random(es_dataset, subdataset_num, toxic_ratio)
    ru_subdata = create_dataset_random(ru_dataset, subdataset_num, toxic_ratio)
    fr_subdata = create_dataset_random(fr_dataset, subdataset_num, toxic_ratio)
    pt_subdata = create_dataset_random(pt_dataset, subdataset_num, toxic_ratio)
    en_subdata = create_dataset_random(en_dataset, subdataset_num, toxic_ratio)
    test7_dataset = pd.concat(
        (en_subdata, ru_subdata, fr_subdata, pt_subdata, tr_subdata, it_subdata, es_subdata))
    del(tr_subdata, it_subdata, es_subdata, ru_subdata, fr_subdata, pt_subdata)

    # save all dataset to output
    dataset_list = [en_dataset, test2_dataset, test3_dataset,
                    test4_dataset, test5_dataset, test6_dataset, test7_dataset]

    for idx, d in enumerate(dataset_list):
        d.to_csv(OUTPUT_DIR+'\\test{0}_training_data.csv'.format(idx+1))
