import tensorflow as tf
import numpy as np

print(f"tensorflow的版本是 :{tf.__version__}")

# 测试加减运算
a = tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]])
b = tf.Variable(tf.random.normal(shape = (2,3)))
d = a+b
print(d.numpy())
#---------------------------------------------------------------------------------------------------

# 下面测试卷积网络与循环神经网络
with tf.device("/gpu:2"):
    # 测试二维卷积运算[filter_height, filter_width, in_channels, out_channels]
    x_image = tf.random.normal(shape = [5, 10, 10, 3])    # 定义输入图片(n,w,h,c)格式
    # 定义 filter：[filter_height, filter_width, in_channels, out_channels]
    conv2d = tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3),data_format="channels_last")
    y = conv2d(x_image)
    print(tf.shape(y))   # (5,8,8,10)
    
    #---------------------------------------------------------------------------------------------------
    # 下面测试循环神经网络
    inputs = np.random.random([32, 10, 8]).astype(np.float32)
    lstm = tf.keras.layers.LSTM(4)
    output = lstm(inputs)  
    print(tf.shape(output))  # The output has shape  [32, 4].

