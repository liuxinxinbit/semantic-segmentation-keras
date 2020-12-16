import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #当前程序上上一级目录，这里为mycompany
sys.path.append(BASE_DIR)
import math
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Lambda, Layer, BatchNormalization, Activation,concatenate,\
    LeakyReLU,AveragePooling2D,DepthwiseConv2D,SeparableConv2D,Add
from model.net_parts import build_conv2D_block, build_conv2Dtranspose_block,bottleneck,pyramid_pooling,build_SeparableConv2D_block,build_DepthwiseConv2D_block
from tensorflow.keras import Model, Input


def Bottleneck(inputs, filters, stride=1, dilation=1, downsample=None,use_bias=True):
    residual = inputs
    conv2d = Conv2D(filters = filters, kernel_size=1,strides=1, padding='same',dilation_rate=dilation,use_bias=use_bias)(inputs)
    conv2d = BatchNormalization(momentum=0.95, epsilon=1e-5)(conv2d)
    conv2d = Activation(LeakyReLU(alpha=0.1))(conv2d)

    conv2d = Conv2D(filters = filters, kernel_size=3,strides=stride, padding='same',dilation_rate=dilation,use_bias=use_bias)(conv2d)
    conv2d = BatchNormalization(momentum=0.95, epsilon=1e-5)(conv2d)
    conv2d = Activation(LeakyReLU(alpha=0.1))(conv2d)

    conv2d = Conv2D(filters = filters, kernel_size=3,strides=1, padding='same',dilation_rate=dilation,use_bias=use_bias)(conv2d)
    conv2d = BatchNormalization(momentum=0.95, epsilon=1e-5)(conv2d)
    
    if downsample:
        residual = Conv2D(filters = filters, kernel_size=1,strides=stride, padding='same',dilation_rate=dilation,use_bias=False)(residual)
        residual = BatchNormalization(momentum=0.95, epsilon=1e-5)(residual)
    out = Add()([conv2d,residual])
    out = Activation(LeakyReLU(alpha=0.1))(out)
    return out
def _make_layer(inputs, filters, blocks, stride=1, dilation=1):
    downsample = False
    if stride != 1 or inputs.shape[0] != filters * 4:
        downsample=True

    output = Bottleneck(inputs, filters, stride=stride, downsample=downsample)
    for i in range(1, blocks):
        output = Bottleneck(output, filters, dilation=dilation, downsample=downsample)
    return output
def _make_MG_unit(inputs, filters, blocks, stride=1, dilation=1):
    downsample = False
    if stride != 1 or inputs.shape[0] != filters * 4:
        downsample=True

    output = Bottleneck(inputs, filters, stride=stride, downsample=downsample)
    for i in range(1, blocks):
        output = Bottleneck(output, filters, dilation=blocks*dilation, downsample=downsample)
    return output

def ResNet(inputdata, layers, output_stride, pretrained=True):
    # blocks = [1, 2, 4]
    if output_stride == 16:
        strides = [1, 2, 2, 1]
        dilations = [1, 1, 1, 2]
    elif output_stride == 8:
        strides = [1, 2, 1, 1]
        dilations = [1, 1, 2, 4]
    else:
        raise NotImplementedError
    cnv1 = build_conv2D_block(inputdata, filters=64, kernel_size=3, strides=1)
    cnv1 = build_conv2D_block(cnv1, filters=64, kernel_size=3, strides=1)
    cnv1 = build_conv2D_block(cnv1, filters=64, kernel_size=3, strides=1)
    res = MaxPooling2D(pool_size=(3, 3), padding='same', strides=(2, 2))(cnv1)

    low_level_feat = _make_layer(inputs = res, filters = 64, blocks = layers[0], stride=strides[0], dilation=dilations[0])
    res = _make_layer(inputs = low_level_feat, filters = 128, blocks =layers[1], stride=strides[1], dilation=dilations[1])
    res = _make_layer(inputs = res, filters = 256, blocks =layers[2], stride=strides[2], dilation=dilations[2])
    res = _make_MG_unit(inputs = res, filters = 512, blocks =layers[3], stride=strides[3], dilation=dilations[3])
    return res, low_level_feat


# if __name__ == "__main__":
#     inputdata = Input(shape=(512, 512, 3))
#     res, low_level_feat = ResNet101(inputdata, layers = [3, 4, 23, 3], output_stride=8)

#     print(res.shape)
#     print(low_level_feat.shape)