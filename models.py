import tensorflow as tf

MAX_LEN = 128

# RoBERTa Models


def build_roberta_dense(base_model, optimizer, loss, metric):
    input_layer = tf.keras.layers.Input(
        shape=(MAX_LEN,), dtype=tf.int32, name='roberta_dense_input')
    transformer_layer_out = base_model(input_layer, training=False)[0]
    cls_token = transformer_layer_out[:, 0, :]
    dense_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    output = dense_layer(cls_token)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer, loss=loss, metrics=['accuracy'])
    return model


def build_roberta_concat_dense(base_model, optimizer, loss, metric):
    input_layer = tf.keras.layers.Input(
        shape=(MAX_LEN,), dtype=tf.int32, name='roberta_dense_input')
    transformer_output = base_model(input_layer, training=False)
    hidden_states = transformer_output[2]
    pooling_layer = tf.concat(
        hidden_states[-4:], axis=-1, name='pooling_hidden_states')
    dense_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    dense_layer1 = tf.keras.layers.Dense(1, activation='sigmoid')
    dense_output = dense_layer(pooling_layer)[:, :, 0]
    output = dense_layer1(dense_output)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer, loss=loss, metrics=['accuracy'])
    return model


def build_roberta_mean_pooling_dense(base_model, optimizer, loss, metric):
    input_layer = tf.keras.layers.Input(
        shape=(MAX_LEN,), dtype=tf.int32, name='roberta_dense_input')
    base_model.config.output_hidden_states = True
    transformer_output = base_model(input_layer, training=True)
    hidden_states = trainsformer_output[2]
    pooling_layer = tf.reduce_mean(
        hidden_states[-4:], axis=0, name='mean_pooling')
    dense_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    output = dense_layer(concat_layer)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer, loss=loss, metrics=['accuracy'])
    return model


class roberta_dense(tf.keras.Model):
    def __init__(self, transformer):
        super(roberta_dense,self).__init__()
        self.transformer = transformer
        self.dense_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x, training):
        x = self.transformer(x,training=training)[0][:,0,:]
        out = self.dense_layer(x)
        return out
    
    def get_hidden_states(self,x):
        return self.transformer(x, training=False)[2]
    
    def get_embedding(self, x):
        return self.transformer(x,training=False)[0][:,0,:]

class distilbert_hs_pooling_dense(tf.keras.Model):
    def __init__(self, transformer):
        super(distilbert_hs_pooling_dense,self).__init__()
        self.transformer = transformer
        self.dense_layer1 = tf.keras.layers.Dense(1)
        self.dense_layer2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x, training):
        x =  self.transformer(x,training=training)[1]
        x = tf.concat(x[-4:], axis=-1, name='pooling_hidden_states')
        x = tf.transpose(x, perm=[0,2,1])
        x = self.dense_layer1(x)[:,:,0]
        x = self.dense_layer2(x)
        return x
    
    def get_hidden_states(self,x):
        return self.transformer(x, training=False)[2]
    
    def get_embedding(self, x):
        return self.transformer(x,training=False)[0][:,0,:]


class distilbert_hs_mean_max_dense(tf.keras.Model):
    def __init__(self, transformer):
        super(distilbert_hs_mean_max_dense,self).__init__()
        self.transformer = transformer
        self.dense_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, x, training):
        x =  self.transformer(x,training=training)[0]
        avg_pool = tf.math.reduce_mean(x, axis=1, name='mean_pool')
        max_pool = tf.math.reduce_max(x,axis=1,name='max_pool')
        x = tf.concat((avg_pool,max_pool), axis=-1, name='concat')
        x = self.dropout(x,training=training)
        x = self.dense_layer(x)
        return x
    
    def get_hidden_states(self,x):
        return self.transformer(x, training=False)[2]
    
    def get_embedding(self, x):
        return self.transformer(x,training=False)[0][:,0,:]

class distilbert_hs_mean_max_min_dense(tf.keras.Model):
    def __init__(self, transformer):
        super(distilbert_hs_mean_max_min_dense,self).__init__()
        self.transformer = transformer
        self.dense_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, x, training):
        x =  self.transformer(x,training=training)[0]
        avg_pool = tf.math.reduce_mean(x, axis=1, name='mean_pool')
        max_pool = tf.math.reduce_max(x,axis=1,name='max_pool')
        min_pool = tf.math.reduce_min(x, axis=1,name='min_pool')
        x = tf.concat((avg_pool,max_pool,min_pool), axis=-1, name='concat')
        x = self.dropout(x,training=training)
        x = self.dense_layer(x)
        return x
    
    def get_hidden_states(self,x):
        return self.transformer(x, training=False)[2]
    
    def get_embedding(self, x):
        return self.transformer(x,training=False)[0][:,0,:]
# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    # def __init__(self, warmup_step, )