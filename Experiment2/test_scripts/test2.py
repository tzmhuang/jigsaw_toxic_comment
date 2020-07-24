import sys
import os
import itertools
import re
import numpy as np
import pandas as pd
import transformers
import tensorflow as tf
import datetime
ROOT = os.path.abspath('../')
sys.path.append(ROOT)
from util import *
import models
"""
==========Configs==========
"""

TEST_NAME = 'test2'
EXP_DIR = os.path.abspath('./')
DATA_DIR = os.path.join(EXP_DIR, 'data')
LOG_DIR = os.path.join(EXP_DIR, 'logs')
CHECKPT_DIR = os.path.join(LOG_DIR, 'checkpoints')
ENCODE_DIR = os.path.join(LOG_DIR, 'encodings')
FIT_LOG_DIR = os.path.join(LOG_DIR, 'fit')
MODEL = 'distilbert-base-multilingual-cased'
N_EPOCHS = 8
BATCH_SIZE = 32
LOAD_ENCODING = False
SAVE_ENCODING = True
CLEAN = True
FILE_NAME = ENCODE_DIR + \
    '\\{0}_encoding_clean3.npy'.format(MODEL.replace('/', '-'))
TRAIN_DIR = os.path.join(
    DATA_DIR, '{0}_training_data.csv'.format(TEST_NAME))
VALIDATE_DIR = os.path.join(DATA_DIR, 'validation.csv.zip')
VALIDATE_2_DIR = os.path.join(DATA_DIR, 'compressed_validation_en.csv.zip')
TEST_DIR = os.path.join(DATA_DIR, 'test.csv.zip')

tf.keras.backend.clear_session()
print('Config loaded... Tokenizing...')

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)

x_train, y_train, x_validate, y_validate,x_validate_2, y_validate_2, x_test = encoding(
    enc_dir=FILE_NAME, load_encoding=LOAD_ENCODING, save_encoding=SAVE_ENCODING, clean=CLEAN, second_validation=True,
    train_data_path=TRAIN_DIR, validation_data_path=VALIDATE_DIR, validation_data_2_path=VALIDATE_2_DIR,
    test_data_path=TEST_DIR, tokenizer=tokenizer, max_len=models.MAX_LEN, clean_func=clean_text_3)
print('Preparing dataset...')

train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                 .repeat(1)
                 .shuffle(2048)
                 .batch(BATCH_SIZE)
                 .prefetch(-1))

validate_dataset = (tf.data.Dataset.from_tensor_slices((x_validate, y_validate))
                    .batch(BATCH_SIZE)
                    .prefetch(-1))

en_validate_dataset = (tf.data.Dataset.from_tensor_slices((x_validate_2, y_validate_2))
                    .batch(BATCH_SIZE)
                    .prefetch(-1))


# test_dataset = (tf.data.Dataset.from_tensor_slices(x_test)
#                 .batch(BATCH_SIZE)
#                 .prefetch(-1))
# Free Space
del(x_train, y_train, x_validate, y_validate, x_test)
print('Building model...')

config = transformers.AutoConfig.from_pretrained(
    MODEL, output_hidden_states=True)
transformer = transformers.TFAutoModel.from_pretrained(MODEL, config=config)
model = models.distilbert_hs_mean_max_min_dense(transformer)
model.compile(tf.keras.optimizers.Adam(5e-6), tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
              metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])

# tensorboard setup
log_dir = FIT_LOG_DIR + '\\'+TEST_NAME+ '_'+\
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = AddtionalValidationTB(additional_validation_sets=[(
    en_validate_dataset, 'validation_EN')], verbose=0, log_dir=log_dir, histogram_freq=1, update_freq='batch')

# training
train_hist = model.fit(train_dataset, validation_data=validate_dataset,
                       epochs=N_EPOCHS, callbacks=[tensorboard_callback])
# save model weights
ckpt_name = MODEL.replace('/', '-') + '-' + model.__class__.__name__
model.save_weights(CHECKPT_DIR+'\\'+ckpt_name + '\\'+TEST_NAME +
                   datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
