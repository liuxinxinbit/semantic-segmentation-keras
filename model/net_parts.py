from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Lambda, Layer, BatchNormalization, Activation,concatenate,LeakyReLU,\
    AveragePooling2D,DepthwiseConv2D,SeparableConv2D,Dropout

import tensorflow as tf
# from tensorflow import keras
def BN(name=""):
    return BatchNormalization(momentum=0.95, epsilon=1e-5)

def build_conv2D_block( inputs, filters, kernel_size, strides,dilation_rate=(1, 1),use_bias=True):
    conv2d = Conv2D(filters = filters, kernel_size=kernel_size,strides=strides, padding='same',dilation_rate=dilation_rate,use_bias=use_bias)(inputs)
    conv2d = BatchNormalization(momentum=0.95, epsilon=1e-5)(conv2d)
    conv2d_output = Activation(LeakyReLU(alpha=0.1))(conv2d)
    return conv2d_output

def build_conv2Dtranspose_block( inputs, filters, kernel_size, strides,use_bias=True):
    conv2d = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, use_bias=use_bias,bias_initializer='zeros', padding='same')(inputs)
    conv2d = BatchNormalization(momentum=0.95, epsilon=1e-5)(conv2d)
    conv2d_deconv = Activation(LeakyReLU(alpha=0.1))(conv2d)
    return conv2d_deconv

def build_DepthwiseConv2D_block(inputs, filters):
    Depthwiseconv2d = DepthwiseConv2D(filters, padding="same")(inputs)
    Depthwiseconv2d = BatchNormalization(momentum=0.95, epsilon=1e-5)(Depthwiseconv2d)
    Depthwiseconv2d = Activation(LeakyReLU(alpha=0.1))(Depthwiseconv2d)
    return Depthwiseconv2d

def build_SeparableConv2D_block(inputs, filters,kernel_size,strides):
    Separableconv2d = SeparableConv2D(filters,kernel_size,strides, padding="same")(inputs)
    Separableconv2d = BatchNormalization(momentum=0.95, epsilon=1e-5)(Separableconv2d)
    Separableconv2d = Activation(LeakyReLU(alpha=0.1))(Separableconv2d)
    return Separableconv2d

def pyramid_pooling(input_tensor, sub_region_sizes):
    """This class implements the Pyramid Pooling Module

    WARNING: This function uses eager execution, so it only works with
        Tensorflow 2.0 backend.

    Args:
        input_tensor: Tensor with shape: (batch, rows, cols, channels)
        sub_region_sizes: A list containing the size of each region for the
            sub-region average pooling. The default value is [1, 2, 3, 6]

    Returns:
        output_tensor: Tensor with shape: (batch, rows, cols, channels * 2)
    """
    _, input_height, input_width, input_channels = input_tensor.shape
    feature_maps = [input_tensor]
    for i in sub_region_sizes:
        curr_feature_map = AveragePooling2D(
            pool_size=(input_height // i, input_width // i),
            strides=(input_height // i, input_width // i))(input_tensor)
        curr_feature_map = Conv2D(
            filters=int(input_channels) // len(sub_region_sizes),
            kernel_size=3,
            padding='same')(curr_feature_map)
        curr_feature_map = Lambda(
            lambda x: tf.image.resize(
                x, (input_height, input_width)))(curr_feature_map)
        feature_maps.append(curr_feature_map)

    output_tensor = concatenate(feature_maps, axis=-1)

    output_tensor = Conv2D(
        filters=128, kernel_size=3, strides=1, padding="same")(
        output_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = Activation("relu")(output_tensor)
    return output_tensor


def bottleneck(input_tensor, filters, strides, expansion_factor):
    """Implementing Bottleneck.

    This class implements the bottleneck module for Fast-SCNN.
    Layer structure:
        ----------------------------------------------------------------
        |  Input shape   |  Block  |  Kernel | Stride |  Output shape  |
        |                |         |   size  |        |                |
        |----------------|---------|---------|--------|----------------|
        |   h * w * c    |  Conv2D |    1    |    1   |   h * w * tc   |
        |----------------|---------|---------|--------|----------------|
        |   h * w * tc   |  DWConv |    3    |    s   | h/s * w/s * tc |
        |----------------|---------|---------|--------|----------------|
        | h/s * w/s * tc |  Conv2D |    1    |    1   | h/s * w/s * c` |
        |--------------------------------------------------------------|

        Designations:
            h: input height
            w: input width
            c: number of input channels
            t: expansion factor
            c`: number of output channels
            DWConv: depthwise convolution

    Args:
        input_tensor: Tensor with shape: (batch, rows, cols, channels)
        filters: Output filters
        strides: Stride used in depthwise convolution layer
        expansion_factor: hyperparameter

    Returns:
        output_tensor: Tensor with shape: (batch, rows // stride,
            cols // stride, new_channels)
    """
    _, input_height, input_width, input_channels = input_tensor.shape
    tensor = Conv2D(filters=input_channels * expansion_factor,
        kernel_size=1,
        strides=1,
        padding="same",
        activation="relu")(input_tensor)
    tensor = BatchNormalization()(tensor)
    tensor = Activation('relu')(tensor)

    tensor = DepthwiseConv2D(kernel_size=3,
                                          strides=strides,
                                          padding="same")(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = Activation('relu')(tensor)

    tensor = Conv2D(filters=filters,
                                 kernel_size=1,
                                 strides=1,
                                 padding="same")(tensor)
    tensor = BatchNormalization()(tensor)
    output_tensor = Activation('relu')(tensor)
    return output_tensor
def Interp(x, shape):
    ''' 对图片做一个放缩，配合Keras的Lambda层使用'''
    # from keras.backend import tf as ktf
    new_height, new_width = shape
    resized = tf.image.resize(x, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR)
    return resized
def _ASPPModule(tensor, filters, kernel_size=(1, 1), strides=(1, 1), padding="same", dilation=(1, 1)):
    tensor = Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,dilation_rate=dilation)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = Activation('relu')(tensor)
    return tensor
def ASPP(tensor, output_stride):
    if output_stride == 16:
        dilations = [1, 6, 12, 18]
    elif output_stride == 8:
        dilations = [1, 12, 24, 36]
    else:
        raise NotImplementedError
    aspp1 = _ASPPModule(tensor, filters=256, kernel_size=(1, 1), dilation=(dilations[0], dilations[0]))
    aspp2 = _ASPPModule(tensor, filters=256, kernel_size=(3, 3), dilation=(dilations[1], dilations[1]))
    aspp3 = _ASPPModule(tensor, filters=256, kernel_size=(3, 3), dilation=(dilations[2], dilations[2]))
    aspp4 = _ASPPModule(tensor, filters=256, kernel_size=(3, 3), dilation=(dilations[3], dilations[3]))
    global_avg_pool = AveragePooling2D(pool_size=(tensor.shape[1],tensor.shape[2]))(tensor)
    global_avg_pool = build_conv2D_block( global_avg_pool, filters=256, kernel_size=(1, 1), strides=(1, 1))
    global_avg_pool = Interp(global_avg_pool,[aspp4.shape[1],aspp4.shape[2]])

    x = concatenate([aspp1,aspp2,aspp3,aspp4,global_avg_pool],axis=-1)

    x = build_conv2D_block( x, filters=256, kernel_size=(1, 1), strides=(1, 1),use_bias=False)
    x = Dropout(0.5)(x)
    return x
def Decoder(x, low_level_feat,num_classes):
    low_level_feat = build_conv2D_block(low_level_feat, filters=48, kernel_size=(1, 1), strides=(1, 1),use_bias=False)
    x = Interp(x,[low_level_feat.shape[1],low_level_feat.shape[2]])
    x = concatenate([x, low_level_feat],axis=-1)

    x = build_conv2D_block(x, filters=256, kernel_size=(3, 3), strides=(1, 1),use_bias=False)
    x = Dropout(0.5)(x)
    x = build_conv2D_block(x, filters=256, kernel_size=(3, 3), strides=(1, 1),use_bias=False)
    x = Dropout(0.5)(x)
    x = Conv2D(filters=num_classes,kernel_size=(1, 1), strides=(1, 1))(x)
    return x