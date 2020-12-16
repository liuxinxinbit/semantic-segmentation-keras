import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))  # 当前程序上上一级目录，这里为mycompany
sys.path.append(BASE_DIR)

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Lambda, Layer, BatchNormalization, Activation, concatenate,\
    LeakyReLU, AveragePooling2D, DepthwiseConv2D, SeparableConv2D, Add,Dropout,UpSampling2D
from tensorflow.keras.optimizers import SGD
import math
from model.net_parts import build_conv2D_block, build_conv2Dtranspose_block, bottleneck, pyramid_pooling, build_SeparableConv2D_block, build_DepthwiseConv2D_block, non_local_block
from tensorflow.keras import Model, Input

def _bn_relu(x):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(momentum=0.95, epsilon=1e-5)(x)
    return Activation(LeakyReLU(alpha=0.1))(norm)

def _bn_relu_conv(inputs, filters, kernel_size=1, stride=1, dilation=1, downsample=None,use_bias=True):
    """Helper to build a BN -> relu -> conv residual unit with full pre-activation function.
    This is the ResNet v2 scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    x = _bn_relu(inputs)
    x = Conv2D(filters = filters, kernel_size=kernel_size,strides=1, padding='same',dilation_rate=dilation,use_bias=use_bias)(x)
    return x

def BN():
    return BatchNormalization(momentum=0.95, epsilon=1e-5)

def residual_conv(prev, level, modify_stride=False):
    if modify_stride is False:
        prev = build_conv2D_block(prev, filters=64 * level, kernel_size=1, strides=1)
    elif modify_stride is True:
        prev = build_conv2D_block(prev, filters=64 * level, kernel_size=1, strides=2)
    prev = build_conv2D_block(prev, filters=64 * level, kernel_size=3, strides=1,dilation_rate=(1, 1))
    prev = Conv2D(256 * level, (1, 1), strides=(1, 1), use_bias=False)(prev)
    prev = BN()(prev)
    return prev


def short_convolution_branch(prev, level, modify_stride=False):

    if modify_stride is False:
        prev = Conv2D(256 * level, (1, 1), strides=(1, 1),
                      use_bias=False)(prev)
    elif modify_stride is True:
        prev = Conv2D(256 * level, (1, 1), strides=(2, 2),
                      use_bias=False)(prev)
    prev = BN()(prev)
    return prev


def empty_branch(prev):
    return prev


def residual_short(prev_layer, level, modify_stride=False):
    prev_layer = Activation('relu')(prev_layer)
    block_1 = residual_conv(prev_layer, level,
                            modify_stride=modify_stride)

    block_2 = short_convolution_branch(prev_layer, level,
                                       modify_stride=modify_stride)
    added = Add()([block_1, block_2])
    return added

def residual_empty(prev_layer, level):
    prev_layer = Activation('relu')(prev_layer)
    block_1 = residual_conv(prev_layer, level)
    block_2 = empty_branch(prev_layer)
    added = Add()([block_1, block_2])
    if added.shape[3] >= 256:
        print("Filters : ", added.shape[3], "Adding Non Local Blocks")
        added = non_local_block(added, mode='embedded', compression=2)
    return added

def Interp(x, shape):
    ''' 对图片做一个放缩，配合Keras的Lambda层使用'''
    # from keras.backend import tf as ktf
    new_height, new_width = shape
    resized = tf.image.resize(x, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR)
    return resized
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
def ResNet(inputdata, layers):
    cnv1 = build_conv2D_block(inputdata, filters=64, kernel_size=3, strides=2)
    cnv1 = build_conv2D_block(cnv1, filters=64, kernel_size=3, strides=1)
    cnv1 = build_conv2D_block(cnv1, filters=64, kernel_size=3, strides=1)
    res = MaxPooling2D(pool_size=(3, 3), padding='same', strides=(2, 2))(cnv1)

    # 2_1- 2_3
    res = residual_short(res, 1)
    for i in range(2):
        res = residual_empty(res, 1)
    # 3_1 - 3_3
    res = residual_short(res, 2, modify_stride=True)
    for i in range(3):
        res = residual_empty(res, 2)
    if layers == 50:
        # 4_1 - 4_6
        res = residual_short(res, 4)
        for i in range(5):
            res = residual_empty(res, 4)
    elif layers == 101:
        # 4_1 - 4_23
        res = residual_short(res, 4)
        for i in range(22):
            res = residual_empty(res, 4)
    else:
        print("This ResNet is not implemented")

    # 5_1 - 5_3
    res = residual_short(res, 8)
    for i in range(2):
        res = residual_empty(res, 8)
    res = Activation('relu')(res)
    return res

class non_local_pspnet:
    def __init__(self,  print_summary=False,image_size=(512, 512, 3),num_class=3):
        self.num_class = num_class
        self.build(print_summary=print_summary,image_size=image_size)
        self.batch_generator =  None
        
    def predict(self, image):
        return self.model.predict(np.array([image]))
    
    def save(self, file_path='pspnet_model.h5'):
        self.model.save_weights(file_path)
        
    def load(self, file_path='pspnet_model.h5'):
        self.model.load_weights(file_path)
            
    def train(self, epochs=10, steps_per_epoch=50,batch_size=32):
        self.model.fit(self.batch_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)

    def interp_block(self, prev_layer, level, feature_map_shape, input_shape):
        if input_shape == (input_shape[0], input_shape[1]):
            kernel_strides_map = {1: 64,
                                  2: 32,
                                  4: 16,
                                  8: 8}
        else:
            print("Pooling parameters for input shape ",
                input_shape, " are not defined.")
            exit(1)
        kernel = (kernel_strides_map[level], kernel_strides_map[level])
        strides = (kernel_strides_map[level], kernel_strides_map[level])
        prev_layer = AveragePooling2D(kernel, strides=strides)(prev_layer)
        prev_layer = Conv2D(512, (1, 1), strides=(1, 1),use_bias=False)(prev_layer)
        prev_layer = BN()(prev_layer)
        prev_layer = Activation('relu')(prev_layer)
        prev_layer = UpSampling2D(size=(int(feature_map_shape[0]/level),int(feature_map_shape[1]/level)))(prev_layer)
        # prev_layer = Interp(feature_map_shape)(prev_layer)
        return prev_layer

    def build_pyramid_pooling_module(self, res, input_shape):
        """Build the Pyramid Pooling Module."""
        # ---PSPNet concat layers with Interpolation
        feature_map_size = tuple(int(ceil(input_dim / 8.0))
                                 for input_dim in input_shape)
        print("PSP module will interpolate to a final feature map size of %s" %
            (feature_map_size, ))

        interp_block1 = self.interp_block(res, 1, feature_map_size, input_shape)
        interp_block2 = self.interp_block(res, 2, feature_map_size, input_shape)
        interp_block3 = self.interp_block(res, 4, feature_map_size, input_shape)
        interp_block6 = self.interp_block(res, 8, feature_map_size, input_shape)

        res = concatenate([res,
                        interp_block6,
                        interp_block3,
                        interp_block2,
                        interp_block1],-1)
        return res


    def build(self, print_summary=False,image_size=(512, 512, 3), num_classes=3,resnet_layers=101):
        inputdata = Input(shape=image_size)
        res = ResNet(inputdata, layers=resnet_layers)
        psp = self.build_pyramid_pooling_module(res, (image_size[0],image_size[1]))
        x = build_conv2D_block(psp,512, (3, 3), strides=(1, 1),use_bias=False)
        x = Dropout(0.1)(x)
        x = Conv2D(num_classes, (1, 1), strides=(1, 1))(x)
        x = Lambda(Interp, arguments={'shape': (image_size[0], image_size[1])})(x) # 使用Lambda层放缩到原图片大小

        x = Activation('softmax')(x)            
        self.model = Model(inputs=inputdata, outputs=x)
        if print_summary:
            print(self.model.summary())
        # # ~ parallel_model = multi_gpu_model(self.model, gpus=1)
        # Solver
        sgd = SGD(lr=1e-3 , momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
# PSPNet = non_local_pspnet(image_size = (512, 512, 3),num_class=3,print_summary=True)
