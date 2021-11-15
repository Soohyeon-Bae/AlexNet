import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout


input_img = (224, 224, 3)

num_of_classes = 1000

def AlexNet():
    inputs = keras.Input(shape=(224, 224, 3))
    # 1번째 레이어
    # input = input_img
    # 96 kernels of size 11x11x3, stride = 4
    # tf.keras.layers.Conv2D(
    #     filters, kernel_size, strides=(1, 1), padding='valid',
    #     data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
    #     use_bias=True, kernel_initializer='glorot_uniform',
    #     bias_initializer='zeros', kernel_regularizer=None,
    #     bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    #     bias_constraint=None, **kwargs
    # )
    # padding = 'valid' : padding을 하지 않음 (이미지 크기 작아짐)
    # padding = 'same' : output image 크기와 input image의 크기와 동일해지도록 padding
    conv1 = Conv2D(filters=96, kernel_size=(11, 11), strides=4)(inputs)
    # The ReLU non-linearity is applied to the output of every convolutional and fully-connected layer.
    conv1 = Activation('relu')(conv1)
    # tf.nn.local_response_normalization(
    #     input, depth_radius=5, bias=1, alpha=1, beta=0.5, name=None
    # )
    conv1 = tf.nn.local_response_normalization(conv1)

    # output_size = (((n+2p-f)/s)+1, ((n+sp-f)/s)+1, c')
    # 1번째 레이어 출력 크기 = (55, 55, 96)가 3채널

    # Overlapping pooling
    # s(stride)=2, z(pooling을 통해 출력되는 특징맵 크기)=3
    # Max-pooling layer follow both response-normalization layers as well as the fifth convolutional layer.
    # tf.keras.layers.MaxPooling2D(
    #     pool_size=(2, 2), strides=None, padding="valid", data_format=None, **kwargs
    # )
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=2)(conv1)

    # output_size = (((n-ps)/s)+1, ((n-ps)/s)+1)
    # 1번째 레이어 output = (27, 27, 96)


    # 2번째 레이어
    # input = (27, 27, 96)
    conv2 = Conv2D(filters=256, kernel_size=(5, 5))(pool1)
    conv2 = Activation('relu')(conv2)
    conv2 = tf.nn.local_response_normalization(input=conv2)

    pool2 = MaxPooling2D(pool_size=(3, 3), strides=2)(conv2)

    # 2번째 레이어 output = (13, 13, 256)


    # 3번째 레이어
    # input = (13, 13, 256)
    conv3 = Conv2D(filters=384, kernel_size=(3, 3))(pool2)
    conv3 = Activation('relu')(conv3)

    # 3번째 레이어 output = (13, 13, 384)


    # 4번째 레이어
    # input = (13, 13, 384)
    conv4 = Conv2D(filters=384, kernel_size=(3, 3))(conv3)
    conv4 = Activation('relu')(conv4)

    # 4번째 레이어 output = (13, 13, 384)


    # 5번째 레이어
    # input = (13, 13, 384)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3))(conv4)
    conv5 = Activation('relu')(conv5)
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=2)(conv5)

    # 5번째 레이어 output = (6, 6, 256)


    # 6번째 레이어(FC)
    # FC layer는 layer의 뉴런 수와 동일한 길이의 벡터 출력
    fc1 = Flatten()(pool3)
    # The fully-connected layers have 4096 neurons each.
    fc1 = Dense(4096)(fc1)
    fc1 = Activation('relu')(fc1)
    fc1 = Dropout(0.5)(fc1)

    # 7번째 레이어(FC)
    fc2 = Dense(4096)(fc1)
    fc2 = Activation('relu')(fc2)
    fc2 = Dropout(0.5)(fc2)

    # 8번째 레이어(FC)
    # The output of the last fully-connected layer is fed to a 1000-way softmax.
    # 1000 = num of classes
    fc3 = Dense(1000)(fc2)
    outputs = Activation('softmax')(fc3)

    return keras.Model(inputs=inputs, outputs=outputs)

model = AlexNet()
model.summary()
