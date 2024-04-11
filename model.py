import numpy as np
from tensorflow import keras
from keras import models
from keras.models import Model
from keras.models import Sequential
from keras.layers import *
from keras.activations import *
from keras.initializers import RandomNormal
from keras.losses import BinaryCrossentropy
import tensorflow as tf

LOSS = BinaryCrossentropy()
INIT = RandomNormal(mean = 0, stddev = 0.02)
# generator 
class generator(Model):
    def __init__(self):
        super().__init__()
        self.Dense1 = Dense(units=(8*8*512),input_dim=100,activation='relu')
        self.reshape1 = Reshape((8,8,512))
        # self.CNNTranspose1 = Conv2DTranspose(256,4,2,'same',activation='relu',use_bias=False,kernel_initializer=INIT)
        # self.batch_normalizer1 = BatchNormalization()
        self.CNNTranspose2 = Conv2DTranspose(256,4,2,'same',activation='relu',use_bias=False,kernel_initializer=INIT)
        self.batch_normalizer2 = BatchNormalization()
        self.CNNTranspose3 = Conv2DTranspose(128,4,2,'same',activation='relu',use_bias=False,kernel_initializer=INIT)
        self.batch_normalizer3 = BatchNormalization()
        self.CNNTranspose4 = Conv2DTranspose(3,4,2,'same',activation='tanh')

    def call(self,input):
        x = self.Dense1(input)
        x = self.reshape1(x)
        # x = self.CNNTranspose1(x)
        # x = self.batch_normalizer1(x)
        x = self.CNNTranspose2(x)
        x = self.batch_normalizer2(x)
        x = self.CNNTranspose3(x)
        x = self.batch_normalizer3(x)
        x = self.CNNTranspose4(x)
        return x
    

def gen_loss(dis_output):
    return LOSS(tf.ones_like(dis_output),dis_output)

def dis_loss(dis_real,dis_fake):
    dis_noise_real = -0.05*tf.random.uniform(tf.shape(dis_real))
    dis_noise_fake = 0.05*tf.random.uniform(tf.shape(dis_fake))
    disy_real = tf.ones_like(dis_real)
    disy_fake = tf.zeros_like(dis_fake)
    disy_realfake = tf.concat([disy_real,disy_fake],axis=0)
    dis_noise_realfake = tf.concat([dis_noise_real,dis_noise_fake],axis=0)
    disy_realfake += dis_noise_realfake
    dis_realfake = tf.concat([dis_real,dis_fake],axis =0)
    return LOSS(disy_realfake,dis_realfake)

# discriminator
class discriminator(Model):
    def __init__(self):
        super().__init__()
        self.CNN1 = Conv2D(128,4,2,'same',activation=LeakyReLU(0.2),use_bias=False,kernel_initializer=INIT)
        self.CNN2 = Conv2D(256,4,2,'same',activation=LeakyReLU(0.2),use_bias=False,kernel_initializer=INIT)
        self.batch_normalizer1 = BatchNormalization()
        # self.CNN3 = Conv2D(256,4,2,'same',activation=LeakyReLU(0.2),use_bias=False,kernel_initializer=INIT)
        # self.batch_normalizer2 = BatchNormalization()
        self.CNN4 = Conv2D(512,4,2,'same',activation=LeakyReLU(0.2),use_bias=False,kernel_initializer=INIT)
        self.batch_normalizer3 = BatchNormalization()
        self.CNN5 = Conv2D(1,4,2,'valid',activation='sigmoid',use_bias=False,kernel_initializer=INIT)
    
    def call(self,input):
        x = self.CNN1(input)
        x = self.CNN2(x)
        x = self.batch_normalizer1(x)
        # x = self.CNN3(x)
        # x = self.batch_normalizer2(x)
        x = self.CNN4(x)
        x = self.batch_normalizer3(x)
        x = self.CNN5(x)
        return x


if __name__ == "main":
    gen_input_layer = Input(shape=(100,))
    gen_result = generator()(gen_input_layer)
    Gen = Model(inputs=gen_input_layer,outputs=gen_result)
    # print(Gen.summary())
    dis_input_layer =Input(shape=(64,64,3,))
    dis_result = discriminator()(dis_input_layer)
    dis = Model(inputs=dis_input_layer,outputs=dis_result)
    # print(dis.summary())
    pass
