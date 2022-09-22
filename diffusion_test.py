# %% [markdown]
# Based on tensorflow starter code from https://www.kaggle.com/alexozerin/end-to-end-baseline-tf-estimator-lb-0-72

# %%
#%pip install pandas
#%pip install keras_tqdm
#%pip install tensorflow-addons
#%pip install tensorflow-io
#%pip install numba
#%pip install tqdm
#%pip install joblib
#%pip install scipy


# %%

#import array 
#from sklearn.utils import shuffle

import tensorflow as tf

import tensorflow.keras as keras
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Conv2D, Flatten, MaxPooling2D, Conv1D, MaxPooling1D, Add, Concatenate, LocallyConnected1D
from keras.layers import Activation, BatchNormalization, GlobalMaxPooling1D, GlobalMaxPool2D, GlobalAveragePooling1D
from keras.layers import Dense, Dropout, Reshape, LSTM, Layer, LayerNormalization, InputLayer, Permute, GRU, Cropping1D
from keras.layers import TimeDistributed, Conv2DTranspose, UpSampling2D, MultiHeadAttention, Embedding, Rescaling, Masking
from keras.layers import ZeroPadding1D, ZeroPadding2D, GaussianNoise, DepthwiseConv2D, Cropping2D, RepeatVector, RNN, AveragePooling2D
from keras.regularizers import l1, l2
from keras import activations, losses
from keras.constraints import max_norm

import tensorflow_addons as tfa
from tensorflow_addons.optimizers import LAMB 

from keras import optimizers
from keras.losses import CategoricalCrossentropy
import numpy as np


import re
import os

from my_layers import ScaleNorm, L1Norm, TransformerBlock, DenseBlock, PositionEmbedding, RandomMask, RestoreUnmaskedTokens
from my_layers import FourierEmbeddingLayer, LinearPositionEmbedding, AugmentAmplitude, ConstantLayer, InvertedResidual2D

from random import *
import math


#from tqdm.notebook import tqdm, trange

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN
import keras.backend as K
import matplotlib.pyplot as plt

import gc as gc

import pandas as pd
import datetime
from datetime import datetime as dt
import time

from collections import deque

from data import get_dataset
from utils import ObjectView


projDir = ''


model_dtype = "float32"

K.set_floatx(model_dtype)
K.set_epsilon(1e-6)
#tf.keras.mixed_precision.experimental.set_policy(model_dtype)

tf.config.run_functions_eagerly(False)
#tf.compat.v1.disable_eager_execution()


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

tf.config.set_visible_devices([], 'GPU') 

DEVICE = "/device:CPU:0"
print("Done")





# %%
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


x_train = tf.cast(x_train, tf.float32) / 255.0

tf.print(x_train.shape, x_train.dtype)
tf.print(y_train.shape)
tf.print(y_test.shape)



max_steps = 800  

beta_min = 0.0001
beta_max = 0.02

alphas = np.zeros([max_steps], np.float32)

def linear_schedule(t):
    return tf.cast(i, tf.float32) / tf.cast(max_steps, tf.float32) * (beta_max - beta_min) + beta_min

def cosine_schedule(t):
    return (tf.math.cos(tf.cast(i, tf.float32) / tf.cast(max_steps, tf.float32) * math.pi) * 0.5 + 0.5) * (beta_max - beta_min) + beta_min




alpha_t = 1.0
for i in tf.range(max_steps):
    alpha_t *= 1.0 - cosine_schedule(i)
    alphas[i] = alpha_t 

#alpha_t = 1.0
#for i in tf.range(max_steps):
#    alphas[i] = (tf.math.cos(tf.cast(i, tf.float32) / tf.cast(max_steps, tf.float32) * math.pi) * 0.5 + 0.5) * (std_max - std_min)
#    alphas[i] = std_min + alphas[i]
    
alphas = tf.convert_to_tensor(alphas)

print("Alphas:", alphas[0], alphas[-1], "\n\n")
print("Alphas:", alphas, "\n\n")



@tf.function(jit_compile=False) # random sequencing doesn't work with JIT
def get_random_seq_diffusion(seq : tf.Tensor) -> tf.Tensor:      

    no_embed_seq = seq
    seq = (seq - tf.reduce_mean(seq)) / tf.math.reduce_std(seq)
    
    step = tf.random.uniform([], minval=0, maxval=max_steps-1, dtype=tf.int32)
    
    #step = 199.0

    alpha_t = alphas[step]          
    noise = tf.random.normal(tf.shape(seq), mean=0.0, stddev = 1.0, dtype=tf.float32)
    
    X = tf.sqrt(alpha_t) * seq + tf.sqrt(1.0 - alpha_t) * noise
    
    t = tf.cast(step, tf.float32) / tf.cast(max_steps, tf.float32) 
    
    #X = tf.clip_by_value(X, -8.0, 8.0)
    
    #min_val = tf.math.reduce_min(X)
    #max_val = tf.math.reduce_max(X)
    #X = (X - min_val) / (max_val - min_val + 1e-6) * 2.0 - 1.0
    
    #min_val = tf.math.reduce_min(noise)
    #max_val = tf.math.reduce_max(noise)
    #noise = (noise - min_val) / (max_val - min_val + 1e-6) * 2.0 - 1.0
    
    return X, noise, t, seq, no_embed_seq





diff_dataset = tf.data.Dataset.from_tensor_slices(x_train)

diff_dataset = (
    diff_dataset.repeat()
    .map(get_random_seq_diffusion, num_parallel_calls=16)       # get_random_seq_diffusion, get_random_trade
)

print("take ", diff_dataset.take(1))




X, noise, step, seq, no_embed_seq = get_random_seq_diffusion(x_train[7])

tf.print(X.shape, X.dtype, step)

#fig = plt.figure(figsize=(20, 12))
#plt.imshow(X)


# %%
from keras.callbacks import TensorBoard
exp_name = "diffusion_script_2"


class My_Weight_Saver(keras.callbacks.Callback):
    def __init__(self, interval=5):
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0:
            model.save_weights(projDir + 'weights/{exp_name}_{epoch:04d}'.format(exp_name=exp_name, epoch=epoch))
            #model.save(projDir + 'models/{exp_name}_{epoch:04d}'.format(exp_name=exp_name, epoch=epoch))


                
callbacks = [EarlyStopping(monitor='loss',
                           patience=400,
                           verbose=1,
                           mode='min'),
             ReduceLROnPlateau(monitor='loss',
                               factor=0.1,
                               patience=100,
                               verbose=1,
                               min_delta=0.00001,
                               mode='min'),
             TerminateOnNaN(),
             My_Weight_Saver(interval=5)]

#%load_ext tensorboard
#%tensorboard --logdir gdrive/Shareddrives/edu_VAD/Anton/VAD_project/logs





# %%

    
################################ encoder
# Unet

embed_dim = 32
patch_size = 7
patch_count = 28**2 // (patch_size**2)


inp = Input(shape=[28, 28])
step = Input(shape=[1])             # <0, 1>

x = inp  




#y = Embedding(input_dim=max_steps, output_dim=16)(step)

y = Reshape([1, 1])(step)
y = FourierEmbeddingLayer(embed_dim=8)(y)
y = Reshape([8])(y)
              
              
x = Reshape([28, 28, 1])(x)
x_inp = x

x128 = Conv2D(embed_dim, 3, activation="ReLU", padding="same")(x)

x64 = MaxPooling2D((2, 2))(x128)
x64 = ScaleNorm()(x64)
x64 = InvertedResidual2D(embed_dim, 1, expansion_factor=4)(x64)
x64 = InvertedResidual2D(embed_dim, 1, expansion_factor=4)(x64)

x32 = MaxPooling2D((2, 2))(x64)
x32 = ScaleNorm()(x32)
x32 = InvertedResidual2D(embed_dim, 1, expansion_factor=4)(x32)
x32 = InvertedResidual2D(embed_dim, 1, expansion_factor=4)(x32)

x16 = MaxPooling2D((2, 2))(x32)
x16 = ScaleNorm()(x16)
x16 = InvertedResidual2D(embed_dim, 1, expansion_factor=4)(x16)
x16 = InvertedResidual2D(embed_dim, 1, expansion_factor=4)(x16)

x16 = Flatten()(x16)
x16 = Concatenate(axis=-1)([x16, y]) 

x32 = Dense(7*7*8, activation = 'ReLU')(x16)
#x32 = DenseBlock(7*7*4)(x32)
#x32 = DenseBlock(7*7*4)(x32)
y32 = Reshape([7, 7, 8])(x32)

y32 = InvertedResidual2D(embed_dim, 1, expansion_factor=8)(y32)
y32 = InvertedResidual2D(embed_dim, 1, expansion_factor=8)(y32)
y64 = Conv2DTranspose(embed_dim, kernel_size=4, strides=2, padding="SAME")(y32)        # upsample
y64 = y64 + x64

#y64 = x64

y64 = InvertedResidual2D(embed_dim, 1, expansion_factor=4)(y64)
y64 = InvertedResidual2D(embed_dim, 1, expansion_factor=4)(y64)
y128 = Conv2DTranspose(embed_dim, kernel_size=4, strides=2, padding="SAME")(y64)        # upsample
y128 = y128 + x128

y128 = InvertedResidual2D(embed_dim, 1, expansion_factor=4)(y128)
y128 = InvertedResidual2D(embed_dim, 1, expansion_factor=4)(y128)


x = Dense(1, activation = 'linear')(y128)

x = Reshape([28, 28])(x)

outp = x







encoder = Model([inp, step], outp, name="encoder")

opt = LAMB(learning_rate=0.001)

#loss = CategoricalCrossentropy(label_smoothing=0.0)

#encoder.compile(optimizer=opt, loss="MSE")     # tf.keras.losses.MSE
#encoder.build(input_shape=[None, sequence_length])


encoder.summary()



# %%
class DiffusionModel(keras.Model):

    def __init__(self, encoder):
        super(DiffusionModel, self).__init__()
        self.model = encoder
        self.loss_tracker = keras.metrics.Mean(name="loss")


    @property
    def metrics(self):
        return [self.loss_tracker]
    
    
    
    def call(self, inp):
        
        pred = self.model(inp, training=False)
        
        return pred         
    


    def train_step(self, batch):

        x, noise, step, seq, no_emb_seq = batch                            #self.aug(batch)

        with tf.GradientTape() as tape:
            pred = self.model([x, step], training=True)
            loss = self.loss(pred, noise)                       # prediting noise
  

        grads_model = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(grads_model, self.model.trainable_variables))
        
        self.loss_tracker.update_state(loss)

        
        return {"loss": self.loss_tracker.result()}
    

model = DiffusionModel(encoder)

#opt = LAMB(learning_rate=0.001)
opt = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)


model.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError())     # "binary_crossentropy", tf.keras.losses.MSE, tf.keras.losses.MeanAbsoluteError(), FFT_loss_1D
model.build(input_shape=[[None, 28, 28], [None, 1]])

#model.load_weights("test_diff")




# %%
# train
gc.collect()

K.set_value(model.optimizer.learning_rate, 0.001)
batch_size = 64

try:
    batched_DS = diff_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    hist = model.fit(batched_DS, callbacks=callbacks,
                    batch_size = batch_size, epochs = 4000, steps_per_epoch = 200)  # initial_epoch , validation_data=(valX[:32], valX[:32])
    plt.plot(hist.history["loss"])
    plt.title("Loss curve")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.show()
except KeyboardInterrupt:
    print("Interrupted")





# %%

# eval prediction

BS = 16
#noise = tf.random.normal([BS, 28, 28], mean=0.0, stddev=1.0, dtype=tf.float32)

batched_DS = diff_dataset.batch(BS).take(1)
noise, _, _, _, _ = next(iter(batched_DS))


#noise = tf.convert_to_tensor(noise)
#noise = tf.expand_dims(seq, axis=0)

print("SHAPE", noise.shape)

tf.print("Start")
prediction = noise      #tf.expand_dims(noise, axis=0)


T = max_steps     # steps


for i in range(1, T-2):                         # reversed()

    
    beta_t = cosine_schedule(i)               #i / tf.cast(T, tf.float32)     # * std_mod
    alpha_t = 1.0 - beta_t
    
    t = tf.cast(i, tf.float32) / tf.cast(T, tf.float32)
    t = tf.expand_dims(tf.expand_dims(t, axis=0), axis=0)
    
    t = tf.tile(t, [BS, 1])


    inv_noise = model([prediction, t], training=False) * (beta_t / tf.sqrt(1.0 - alphas[i]))    # predict noise + amplitude
    prediction = (prediction - inv_noise) / tf.sqrt(alpha_t)   # apply denoising
    
    prediction = (prediction - tf.reduce_mean(prediction, axis=0)) / tf.math.reduce_std(prediction, axis=0)
    prediction = tf.clip_by_value(prediction, -8.0, 8.0)
    
    #stats
    min_val = tf.math.reduce_min(prediction)
    max_val = tf.math.reduce_max(prediction)
    std = tf.math.reduce_std(prediction)
    
    if i % 10 == 1:
        tf.print("Iter: ", i, "min:", min_val, "max:", max_val, "std:", std, "noise amp", (beta_t / tf.sqrt(1.0 - alphas[i])))
        #tf.print("Iter: ", i)
        
        

    
# show result
min_val = tf.math.reduce_min(prediction, axis=0)
max_val = tf.math.reduce_max(prediction, axis=0)
prediction = (prediction - min_val) / (max_val - min_val + 1e-6)

tf.print("End")



for i in range(BS):
    fig = plt.figure(figsize=(20, 12))
    plt.imshow(prediction[i])
    
    fig = plt.figure(figsize=(20, 12))
    plt.imshow(noise[i])


plt.show()






