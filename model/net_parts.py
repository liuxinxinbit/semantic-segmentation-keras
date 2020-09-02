from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Lambda, Layer, BatchNormalization, Activation,concatenate,LeakyReLU



def build_conv2D_block( inputs, filters, kernel_size, strides):
    conv2d = Conv2D(filters = filters, kernel_size=kernel_size,strides=strides, padding='same')(inputs)
    conv2d = BatchNormalization()(conv2d)
    conv2d_output = Activation(LeakyReLU(alpha=0.1))(conv2d)
    return conv2d_output

def build_conv2Dtranspose_block( inputs, filters, kernel_size, strides):
    conv2d = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, use_bias=True,bias_initializer='zeros', padding='same')(inputs)
    conv2d = BatchNormalization()(conv2d)
    conv2d_deconv = Activation(LeakyReLU(alpha=0.1))(conv2d)
    return conv2d_deconv