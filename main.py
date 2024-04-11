from model import generator,discriminator,gen_loss,dis_loss
from keras.models import Model
from keras.layers import *
from plotting import plot
import tensorflow as tf
import cv2
import os
from PIL.Image import fromarray
import numpy as np
LR = 0.0002
BATCH_SIZE = 32

class AnimeGan(Model):
    def __init__(self,gen,dis,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.gen = gen
        self.dis = dis
    
    def compile(self, gen_optimizer,dis_optimizer,gen_loss,dis_loss,**kwargs):
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.gen_loss = gen_loss
        self.dis_loss = dis_loss
    
    @tf.function
    def training_batch(self,image):
        
        real_images = image
        # random_input = tf.random.normal([BATCH_SIZE,100],seed=1)
        fake_images = self.gen(tf.random.normal([BATCH_SIZE,100],seed=1))

        with tf.GradientTape() as d_tape:
            
            d_fake = self.dis(fake_images,training=True)
            d_real = self.dis(real_images,training=True)
            
            d_loss = self.dis_loss(d_real,d_fake)
        
        dgrad = d_tape.gradient(d_loss,self.dis.trainable_variables)
        self.dis_optimizer.apply_gradients(zip(dgrad, self.dis.trainable_variables))
        with tf.GradientTape() as g_tape:
            fake_images = self.gen(tf.random.normal([BATCH_SIZE,100],seed=1))
            d_fake = self.dis(fake_images,training=True)
            g_loss = self.gen_loss(d_fake)

        ggrad = g_tape.gradient(g_loss,self.gen.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(ggrad, self.gen.trainable_variables))

        # with tf.GradientTape(persistent=True) as tape:
        #     fake_images = self.gen(random_input)
        #     d_fake = self.dis(fake_images,training=True)
        #     d_real = self.dis(real_images,training=True)
        #     d_loss = self.dis_loss(d_real,d_fake)
        #     g_loss = self.gen_loss(d_fake)

        # ggrad = tape.gradient(g_loss,self.gen.trainable_variables)
        # dgrad = tape.gradient(d_loss,self.dis.trainable_variables)
        
        # self.dis_optimizer.apply_gradients(zip(dgrad, self.dis.trainable_variables))
        # self.gen_optimizer.apply_gradients(zip(ggrad, self.gen.trainable_variables))

#you need to change for yourself if you want to run the code!!!!
PATH = "C:\\Users\\User\\Desktop\\GAN_Anime\\images"   
def data_processing():
    for _,__,files in os.walk('C:\\Users\\User\\Desktop\\GAN_Anime\\images'):
        return files


FILES = data_processing()
Images = []
def image_data(start,end):
    global Images
    Images = []
    for file in FILES[start:end]:
        img = tf.keras.utils.img_to_array(fromarray(cv2.cvtColor(cv2.imread(os.path.join(PATH,file)),cv2.COLOR_BGR2RGB)))
        img = (img-127.5)/127.5
        Images.append(img)
    Images = np.array(Images)
    train_data = tf.data.Dataset.from_tensor_slices(Images).shuffle(1000).batch(BATCH_SIZE,drop_remainder=True)
    return train_data

def fits(epochs):
    gen_input_layer = Input(shape=(100,))
    gen_result = generator()(gen_input_layer)
    Gen = Model(inputs=gen_input_layer,outputs=gen_result)
    Gen.load_weights('./GEN')
    dis_input_layer = Input(shape=(64,64,3,))
    dis_result = discriminator()(dis_input_layer)
    Dis = Model(inputs=dis_input_layer,outputs=dis_result)
    Dis.load_weights('./DIS')
    gen_optimizer = tf.keras.optimizers.Adam(0.0002)
    dis_optimizer = tf.keras.optimizers.Adam(0.0002)
    DCGan = AnimeGan(Gen,Dis)
    DCGan.compile(gen_optimizer,dis_optimizer,gen_loss,dis_loss)
    for _ in range(epochs):
        print("current epoch: ", _+1)
        images = 0
        for x in range(60):
            images = image_data(1000*x,1000*(x+1))
            print("#", end="")
            for image_batch in images:
                DCGan.training_batch(image_batch)
        
        if _%2 ==0:
            result = DCGan.gen(tf.random.normal([4,100],seed=0))
            plot(result)
    DCGan.gen.save_weights('./GEN')
    DCGan.dis.save_weights('./DIS')

if __name__ == '__main__':
    # Bringing in tensorflow
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus: 
        tf.config.experimental.set_memory_growth(gpu, True)
    fits(20)
        





