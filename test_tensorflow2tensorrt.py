from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 1

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('tf_savedmodel',save_format='tf')
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

from tensorflow.keras.datasets import mnist
import time

import time
import tensorflow as tf
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定只是用第三块GPU


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype('float32')
x_test = x_test.reshape(10000, 784)
x_test /= 255

# frame = cv2.imread('/modeldata/jupyterhub/multi-user/zhouxuanchen/notebook/OCR/1656552746_7f8680cef81411ecb9c400163e16ff5d.jpg')
# print(frame.shape)
# x_test = frame

model = tf.saved_model.load('tf_savedmodel')

t=time.time()
# a = model.signatures["serving_default"]
output = model(tf.constant(x_test))[0].numpy()
print(time.time()-t)
print((output.argmax(-1)==y_test).mean())





from tensorflow.python.compiler.tensorrt import trt_convert as trt
params=trt.DEFAULT_TRT_CONVERSION_PARAMS
params._replace(precision_mode=trt.TrtPrecisionMode.FP32)
converter = trt.TrtGraphConverterV2(input_saved_model_dir='tf_savedmodel',conversion_params=params)
converter.convert()#完成转换,但是此时没有进行优化,优化在执行推理时完成
converter.save('test_trt_savedmodel')



saved_model_loaded = tf.saved_model.load(
    "trt_savedmodel", tags=[trt.tag_constants.SERVING])#读取模型
# graph_func = saved_model_loaded.signatures[
#     trt.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]#获取推理函数,也可以使用saved_model_loaded.signatures['serving_default']
# frozen_func = trt.convert_to_constants.convert_variables_to_constants_v2(
#     graph_func)#将模型中的变量变成常量,这一步可以省略,直接调用graph_func也行



t=time.time()
output = saved_model_loaded(tf.constant(x_test))[0].numpy()
print(time.time()-t)
print((output.argmax(-1)==y_test).mean())

