from keras import backend as K
from keras.layers import Input
from keras.layers import Conv2D, Activation, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense, Dropout
from keras.initializers import TruncatedNormal
from keras.regularizers import l2
from keras.models import Model

def inception_resnet_block(X, scale, block_type, block_idx, activation='relu'):
    if block_type == 'block35':
        branch_0 = conv2d_bn(X, 32, 1)
        branch_1 = conv2d_bn(X, 32, 1)
        branch_1 = conv2d_bn(branch_1, 32, 3)
        branch_2 = conv2d_bn(X, 32, 1)
        branch_2 = conv2d_bn(branch_2, 48, 3)
        branch_2 = conv2d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_bn(X, 192, 1)
        branch_1 = conv2d_bn(X, 128, 1)
        branch_1 = conv2d_bn(branch_1, 160, [1, 7])
        branch_1 = conv2d_bn(branch_1, 192, [7, 1])
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_bn(X, 192, 1)
        branch_1 = conv2d_bn(X, 192, 1)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn(branch_1, 256, [3, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError('unknown block type')

    block_name = block_type + '_' + str(block_idx)
    mixed = Concatenate(axis=3, name=block_name + '_mixed')(branches)
    up = Conv2D(K.int_shape(X)[3], 1, kernel_initializer=TruncatedNormal(stddev=0.1), kernel_regularizer=l2(0.0001),name=block_name + '_conv', padding='same')(mixed)
    X = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale, output_shape=K.int_shape(X)[1:],arguments={'scale': scale}, name=block_name+"_Lambda")([X, up])
    if activation is not None:
        X = Activation(activation, name=block_name + '_ac')(X)
    return X

def conv2d_bn(X,filters,kernel_size,strides=1,padding='same',activation='relu',use_bias=False,name=None):
    X = Conv2D(filters,kernel_size,strides=strides,padding=padding,use_bias=use_bias, kernel_initializer=TruncatedNormal(stddev=0.1), kernel_regularizer=l2(0.0001), name=name)(X)
    if not use_bias:
        bn_axis = 3
        bn_name = None if name is None else name + '_bn'
        X = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(X)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        X = Activation(activation, name=ac_name)(X)
    return X

def InceptionResnetV2(image_shape, embedding_size=512, dropout=0.2):
    # input image
    X_input = Input((image_shape[0], image_shape[1], image_shape[2]), name="vector_input")
    # 299 X 299 X 3  ->   # 149 X 149 X 32
    X = conv2d_bn(X_input, 32, 3, strides=2, padding='valid')
    # 149 X 149 X 32   ->  # 147 x 147 X 32
    X = conv2d_bn(X, 32, 3, padding='valid')
    # 147 x 147 X 32   ->    # 147 X 147 X 64
    X = conv2d_bn(X, 64, 3)
    # 147 X 147 X 64   ->    # 73 X 73 X 64
    X = MaxPooling2D(3, strides=2)(X)
    # 73 X 73 X 64    ->    # 73 X 73 X 80
    X = conv2d_bn(X, 80, 1, padding='valid')
    # 73 X 73 X 80    ->    # 71 X 71 X 192
    X = conv2d_bn(X, 192, 3, padding='valid')
    # 71 X 71 X 192  ->  # 35 X 35 X 192
    X = MaxPooling2D(3, strides=2)(X)

    # 35 X 35 X 192 -> 35 X 35 X 96
    branch_0 = conv2d_bn(X, 96, 1)

    # 35 X 35 X 192 -> 35 X 35 X 64
    branch_1 = conv2d_bn(X, 48, 1)
    branch_1 = conv2d_bn(branch_1, 64, 5)

    # 35 X 35 X 192 -> 35 X 35 X 96
    branch_2 = conv2d_bn(X, 64, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3)

    # 35 X 35 X 192 -> 35 X 35 X 64
    branch_pool = AveragePooling2D(3, strides=1, padding='same')(X)
    branch_pool = conv2d_bn(branch_pool, 64, 1)

    branches = [branch_0, branch_1, branch_2, branch_pool]
    X = Concatenate(axis=3, name='mixed_5b')(branches)  # 35 X 35 X 320

    # 10x block35
    for block_idx in range(1, 11):
        X = inception_resnet_block(X, scale=0.17, block_type='block35', block_idx=block_idx)

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(X, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn(X, 256, 1)
    branch_1 = conv2d_bn(branch_1, 256, 3)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='valid')
    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(X)
    branches = [branch_0, branch_1, branch_pool]
    X = Concatenate(axis=3, name='mixed_6a')(branches)

    for block_idx in range(1, 21):
        X = inception_resnet_block(X, scale=0.1, block_type='block17', block_idx=block_idx)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(X, 256, 1)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn(X, 256, 1)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='valid')
    branch_2 = conv2d_bn(X, 256, 1)
    branch_2 = conv2d_bn(branch_2, 288, 3)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='valid')
    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(X)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    X = Concatenate(axis=3, name='mixed_7a')(branches)

    for block_idx in range(1, 10):
        X = inception_resnet_block(X, scale=0.2, block_type='block8', block_idx=block_idx)

    X = inception_resnet_block(X, scale=1., activation=None, block_type='block8', block_idx=10)

    # Final convolution block: 8 x 8 x 1536
    X = conv2d_bn(X, 1536, 1, name='conv_7b')

    X = AveragePooling2D(K.int_shape(X)[1:3], strides=1, padding='valid')(X)
    X = Flatten()(X)
    
    X = Dropout(dropout)(X)

    X = Dense(embedding_size, name='dense_layer')(X)

    X = Lambda(lambda x: K.l2_normalize(x, axis=1))(X)

    return Model(inputs=X_input, outputs=X)

