import tensorflow as tf
import numpy as np

print(f"tensorflow的版本是 :{tf.__version__}")

# 测试加减运算
a = tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]])
b = tf.Variable(tf.random.normal(shape = (2,3)))
c = tf.placeholder(tf.float32, shape=(2, 3))
d = a+b+c
#---------------------------------------------------------------------------------------------------
# 测试卷积运算与循环神经网络运算
with tf.device("/gpu:2"):
    # 测试二维卷积运算[filter_height, filter_width, in_channels, out_channels]
    x_image = tf.placeholder(tf.float32, shape = [1, 10, 10, 1])    # 定义输入图片(n,w,h,c)格式
    # 定义 filter：[filter_height, filter_width, in_channels, out_channels]
    W_cpu = np.array([[1, 1, 1], [0, -1, 0], [0, -1 , 1]], dtype = np.float32)
    W= tf.Variable(W_cpu)
    W = tf.reshape(W, [3, 3, 1, 1])
 
    # 定义步长以及填充方式，卷积类型
    strides = [1, 1, 1, 1]
    padding = 'VALID'
    y = tf.nn.conv2d(x_image, W, strides, padding)
    y_shape = tf.shape(y)      # (1,8,8,1)
    
    #---------------------------------------------------------------------------------------------------
    # 循环神经网络
    inputs = tf.placeholder(tf.float32, shape = [5, 10, 8])
    lstm = tf.keras.layers.LSTM(4)
    output = lstm(inputs)
    output_shape = tf.shape(output)  # (5,4)


# 下面构建会话，构建三个输入
c_1 = np.random.randint(low=1, high=10, size=(2,3), dtype=int)
x_image_1 = np.random.random(size=(1,10,10,1))
inputs_1 = np.random.random(size=(5,10,8))

with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    result1,result2,result3,y_shape_,output_shape_ = sess.run([d,y,output,y_shape,output_shape],feed_dict={c:c_1,x_image:x_image_1,inputs:inputs_1})
    
    print(result1)
    print("=====================================================")
    print(y_shape_)
    print(output_shape_)
    
    
'''
[[ 3.0697515  5.709157   4.130986 ]
 [ 5.8225546  9.108072  11.701527 ]]
=====================================================
[1 8 8 1]
[5 4]
'''