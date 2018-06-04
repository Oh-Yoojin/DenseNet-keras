from keras.models import Model
from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.regularizers import l2
from keras.layers.merge import Concatenate
from keras.engine import Layer, InputSpec

try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations




def conv_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu 3x3, Conv2D, optional dropout
    Args:
        input: Input keras tensor
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added
    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(input)
    x = Activation('relu')(x)
    x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same",
                      use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)

    return x




def transition_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional dropout and Maxpooling2D
    Args:
        input: keras tensor
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(input)
    x = Activation('relu')(x)
    x = Convolution2D(nb_filter, (1, 1), kernel_initializer="he_uniform", padding="same", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(x)

    x = AveragePooling2D((2, 2), strides=(2, 2),padding='same')(x)


    return x




def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with nb_layers of conv_block appended
    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    feature_list = [x]


    for i in range(nb_layers):
        x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                               beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = Convolution2D(nb_filter, (1,1), kernel_initializer='he_uniform', padding='same', use_bias=False,
                         kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                               beta_regularizer=l2(weight_decay))(x)
        x = conv_block(x, growth_rate, dropout_rate, weight_decay)
        feature_list.append(x)
        x = Concatenate(axis=concat_axis)(feature_list)
        nb_filter += growth_rate

    return x, nb_filter




def DenseNet(nb_classes, img_dim, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=16, dropout_rate=None,
             weight_decay=1E-4, verbose=True):
    model_input = Input(shape=img_dim)

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(model_input)
    x = Activation('relu')(x)
    x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", name="initial_conv2D", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Add dense blocks - 1
    for block_idx in range(nb_dense_block):
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)

    # add transition_block - 1
    x = transition_block(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

    # Add dense blocks - 2
    for block_idx in range(nb_dense_block):
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)

    # add transition_block - 2
    x = transition_block(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

    # Add dense blocks - 3
    for block_idx in range(nb_dense_block):
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)

    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(
        x)

    densenet = Model(inputs=model_input, outputs=x)

    if verbose:
        print("DenseNet-%d-%d created." % (depth, growth_rate))

    return densenet



