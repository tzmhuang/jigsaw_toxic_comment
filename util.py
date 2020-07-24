import sys
import os
import zipfile
import gc
import re
import itertools
import tensorflow.compat.v1
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1.keras.backend import clear_session
from tensorflow.compat.v1.keras.backend import get_session
import tensorflow as tf
import pandas as pd
import numpy as np


def batch_encode(input, tokenizer, max_len):
    enc = tokenizer.batch_encode_plus(
        input, pad_to_max_length=True, max_length=max_len)
    return enc['input_ids']

# Reset Keras Session


def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del classifier  # this is from global space - change this as you need
    except:
        pass

    # if it's done something you should see a number being outputted
    print(gc.collect())

    # use the same config as you used to create the session
    config = tensorflow.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.compat.v1.Session(config=config))

def clean_text_1(text):
    place_holder = '<placeholder>'
    # remove strange \xcc\.. \xcd\.. etc. character
    # if total number \xcc \xcd exceed certain thresh hold
    # remove all \xc.. by converteing to ascii and back
    n_schar = str(text.encode()).count('\\xc')
    n_char = len(text)
    if n_schar/n_char >= 0.5:
        text = text.encode('ascii', 'ignore').decode()
    # remove http
    text = re.sub('https?://\S+|www\.\S+', '', text)
    # remove nextline
    text = text.replace('\n', ' ')
    text = text.replace('""', place_holder)
    # remove " at behining and end
    text = text.replace('"', ' ')
    # replace "" with "
    text = text.replace(place_holder, '\"')
    text = text.replace('==', ' ')
    text = text.replace('-=', ' ')
    text = text.replace('=-', ' ')
    text = re.sub('#', '', text)
    text = re.sub('©', '', text)
    text = re.sub('&', '', text)
    # word with number inbetween
    text = re.sub('\w*\d\w*', '', text)
    # remove ':' at start of sentence
    text = re.sub('^:*', '', text)
    return text


def clean_text_2(text):
    place_holder = '<placeholder>'
    # remove strange \xcc\.. \xcd\.. etc. character
    # if total number \xcc \xcd exceed certain thresh hold
    # remove all \xc.. by converteing to ascii and back
    n_schar = str(text.encode()).count('\\xc')
    n_char = len(text)
    if n_schar/n_char >= 0.5:
        text = text.encode('ascii', 'ignore').decode()
    # remove http
    text = re.sub('https?://\S+|www\.\S+', '', text)
    # limit all character repetition to 5
    text = ''.join(''.join(i)[:5] for _, i in itertools.groupby(text))
    # remove nextline
    text = text.replace('\n', ' ')
    text = text.replace('""', place_holder)
    # remove " at behining and end
    text = text.replace('"', ' ')
    # replace "" with "
    text = text.replace(place_holder, '\"')
    text = text.replace('==', ' ')
    text = text.replace('-=', ' ')
    text = text.replace('=-', ' ')
    text = re.sub('#', '', text)
    text = re.sub('©', '', text)
    text = re.sub('!', '', text)
    text = re.sub('&', '', text)
    # word with number inbetween
    text = re.sub('\w*\d\w*', '', text)
    # remove ':' at start of sentence
    text = re.sub('^:*', '', text)
    # convert all to lower case
    text = text.lower()
    return text


def clean_text_3(text):
    place_holder = '<placeholder>'
    # remove strange \xcc\.. \xcd\.. etc. character
    # if total number \xcc \xcd exceed certain thresh hold
    # remove all \xc.. by converteing to ascii and back
    n_schar = str(text.encode()).count('\\xc')
    n_char = len(text)
    if n_schar/n_char >= 0.5:
        text = text.encode('ascii', 'ignore').decode()
    # remove http
    text = re.sub('https?://\S+|www\.\S+', '', text)
    # remove nextline
    text = text.replace('\n', ' ')
    text = text.replace('""', place_holder)
    # remove " at behining and end
    text = text.replace('"', ' ')
    # replace "" with "
    text = text.replace(place_holder, '\"')
    # remove excessive !,?,",:,;,/,<space>,,
    s = '!?":;/ ,()%'
    text = ''.join(''.join(i) if i in s else ''.join(j)
                   for i, j in itertools.groupby(text))
    # remove special character
    text = text.replace('=', '')
    text = text.replace('-', '')
    text = re.sub('\w*[@#&*|\\`~©^]\w*', '', text)
    # word with number inbetween
    text = re.sub('\w*\d\w*', '', text)
    # remove ':' at start of sentence
    text = re.sub('\s:\w*', '', text)
    return text

def encoding(enc_dir, load_encoding=True, clean=True, save_encoding=False, second_validation=False, **kwargs):
    if load_encoding:
        print('Loading: ', enc_dir)
        with open(enc_dir, 'rb') as f:
            x_train = np.load(f)
            y_train = np.load(f)
            x_validate = np.load(f)
            y_validate = np.load(f)
            x_test = np.load(f)
            if second_validation:
                x_validate_2 = np.load(f)
                y_validate_2 = np.load(f)
                return (x_train, y_train, x_validate, y_validate, x_validate_2, y_validate_2, x_test)
        return (x_train, y_train, x_validate, y_validate, x_test)
    else:
        assert('train_data_path' in kwargs)
        assert('validation_data_path' in kwargs)
        assert('test_data_path' in kwargs)
        assert('tokenizer' in kwargs)
        assert('max_len' in kwargs)
        train = pd.read_csv(kwargs['train_data_path']).sample(frac=1, random_state=2020)
        # train = pd.concat([train_raw.query('toxic==0').sample(60000, random_state=2020),
        #                    train_raw.query('toxic==1')]).sample(frac=1, random_state=2020)
        validation_data = pd.read_csv(kwargs['validation_data_path'])
        test_data = pd.read_csv(kwargs['test_data_path'])
        if clean:
            assert('clean_func' in kwargs)
            train.comment_text = train.comment_text.apply(kwargs['clean_func'])
            test_data.content = test_data.content.apply(kwargs['clean_func'])
            validation_data.comment_text = validation_data.comment_text.apply(
                kwargs['clean_func'])
        # Data encoding
        tokenizer = kwargs['tokenizer']
        x_train = batch_encode(train.comment_text.values,
                               tokenizer, max_len=kwargs['max_len'])
        x_validate = batch_encode(
            validation_data.comment_text.values, tokenizer, max_len=kwargs['max_len'])
        x_test = batch_encode(test_data.content.values,
                              tokenizer, max_len=kwargs['max_len'])
        y_train = train.toxic.values
        y_validate = validation_data.toxic.values
        if save_encoding:
            x_train_enc = np.array(x_train)
            y_train_enc = np.array(y_train)
            x_validate_enc = np.array(x_validate)
            y_validate_enc = np.array(y_validate)
            x_test_enc = np.array(x_test)
            with open(enc_dir, 'wb') as f:
                np.save(f, x_train_enc)
                np.save(f, y_train_enc)
                np.save(f, x_validate_enc)
                np.save(f, y_validate_enc)
                np.save(f, x_test_enc)
        if second_validation:
            validation_data_2 = pd.read_csv(kwargs['validation_data_2_path'])
            if clean:
                validation_data_2.comment_text = validation_data_2.comment_text.apply(
                    kwargs['clean_func'])
            x_validate_2 = batch_encode(
                validation_data_2.comment_text.values, tokenizer, max_len=kwargs['max_len'])
            y_validate_2 = validation_data_2.toxic.values
            if save_encoding:
                x_validate_2_enc = np.array(x_validate_2)
                y_validate_2_enc = np.array(y_validate_2)
                with open(enc_dir, 'ab') as f:
                    np.save(f, x_validate_2_enc)
                    np.save(f, y_validate_2_enc)
            return (x_train, y_train, x_validate, y_validate, x_validate_2, y_validate_2, x_test)
        return (x_train, y_train, x_validate, y_validate, x_test)


class AddtionalValidationTB(tf.keras.callbacks.TensorBoard):
    """
    Param:
        additional_validation_sets: a list of tuple (tf.data.Dataset object, validation_name)
    """

    def __init__(self,
                 additional_validation_sets,
                 verbose,
                 log_dir='logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False,
                 update_freq='epoch',
                 profile_batch=2,
                 embeddings_freq=0,
                 embeddings_metadata=None,
                 **kwargs):
        super(AddtionalValidationTB, self).__init__(log_dir=log_dir,
                                                    histogram_freq=histogram_freq,
                                                    write_graph=write_graph,
                                                    write_images=write_images,
                                                    update_freq=update_freq,
                                                    profile_batch=profile_batch,
                                                    embeddings_freq=embeddings_freq,
                                                    embeddings_metadata=embeddings_metadata,
                                                    **kwargs)
        self.additional_validation_sets = additional_validation_sets
        self.verbose = verbose
        for v in self.additional_validation_sets:
            if len(v) != 2:
                raise ValueError()

    def on_epoch_end(self, epoch, logs=None):
        """Runs metrics and histogram summaries at epoch end."""
        logs = logs or {}
        validation_prefix = 'val_'
        for validate_dataset, validation_name in self.additional_validation_sets:
            self.reset_metrics()
            results = self.model.evaluate(
                x=validate_dataset, verbose=self.verbose, return_dict=True)
            [logs.setdefault(validation_prefix+validation_name+'_'+name, val)
             for name, val in results.items()]

        self._log_metrics(logs, prefix='epoch_', step=epoch)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_weights(epoch)

        if self.embeddings_freq and epoch % self.embeddings_freq == 0:
            self._log_embeddings(epoch)

    def reset_metrics(self):
        for m in self.model.metrics:
            m.reset_states()
