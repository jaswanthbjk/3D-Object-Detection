""" Wrapper functions for TensorFlow layers.

Author: Charles R. Qi
Date: November 2017
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
    with tf.compat.v1.device("/cpu:0"):
        dtype = tf.compat.v1.float16 if use_fp16 else tf.compat.v1.float32
        var = tf.compat.v1.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
    """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

  Returns:
    Variable Tensor
  """
    if use_xavier:
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        initializer = tf.compat.v1.truncated_normal_initializer(stddev=stddev)
    var = _variable_on_cpu(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.compat.v1.multiply(tf.compat.v1.nn.l2_loss(var), wd, name='weight_loss')
        tf.compat.v1.add_to_collection('losses', weight_decay)
    return var


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=1,
           padding='SAME',
           data_format='NHWC',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.compat.v1.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
    """ 1D convolution with non-linear operation.

  Args:
    inputs: 3-D tensor variable BxLxC
    num_output_channels: int
    kernel_size: int
    scope: string
    stride: int
    padding: 'SAME' or 'VALID'
    data_format: 'NHWC' or 'NCHW'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch_size norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
    with tf.compat.v1.variable_scope(scope) as sc:
        assert (data_format == 'NHWC' or data_format == 'NCHW')
        if data_format == 'NHWC':
            num_in_channels = inputs.get_shape()[-1].value
        elif data_format == 'NCHW':
            num_in_channels = inputs.get_shape()[1].value
        # kernel_shape = [kernel_size,num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_size,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        outputs = tf.compat.v1.nn.conv1d(inputs, kernel,
                                         stride=stride,
                                         padding=padding,
                                         data_format=data_format)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.compat.v1.constant_initializer(0.0))
        outputs = tf.compat.v1.nn.bias_add(outputs, biases, data_format=data_format)

        if bn:
            outputs = batch_norm_for_conv1d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn',
                                            data_format=data_format)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           data_format='NHWC',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.compat.v1.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
    """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    data_format: 'NHWC' or 'NCHW'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch_size norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
    with tf.compat.v1.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        assert (data_format == 'NHWC' or data_format == 'NCHW')
        if data_format == 'NHWC':
            num_in_channels = inputs.get_shape()[-1].value
        elif data_format == 'NCHW':
            num_in_channels = inputs.get_shape()[1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_h, stride_w = stride
        outputs = tf.compat.v1.nn.conv2d(inputs, kernel,
                                         [1, stride_h, stride_w, 1],
                                         padding=padding,
                                         data_format=data_format)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.compat.v1.constant_initializer(0.0))
        outputs = tf.compat.v1.nn.bias_add(outputs, biases, data_format=data_format)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn',
                                            data_format=data_format)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def conv_flipout(inputs,
                   num_output_channels,
                   kernel_size,
                   scope,
                   stride=[1, 1],
                   padding='SAME',
                   data_format='NHWC',
                   use_xavier=True,
                   stddev=1e-3,
                   weight_decay=None,
                   activation_fn=tf.compat.v1.nn.relu,
                   bn=False,
                   bn_decay=None,
                   is_training=None, prior_fn=1):
    """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    data_format: 'NHWC' or 'NCHW'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch_size norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
    with tf.compat.v1.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        assert (data_format == 'NHWC' or data_format == 'NCHW')
        if data_format == 'NHWC':
            num_in_channels = inputs.get_shape()[-1].value
        elif data_format == 'NCHW':
            num_in_channels = inputs.get_shape()[1].value
        # kernel_shape = [kernel_h, kernel_w,num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_size,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_h, stride_w = stride
        outputs = tfp.layers.Convolution2DFlipout(inputs, kernel,
                                                  strides=[1, stride_h, stride_w, 1],
                                                  padding=padding,
                                                  data_format=data_format, kernel_prior_fn=prior_fn)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.compat.v1.constant_initializer(0.0))
        outputs = tf.compat.v1.nn.bias_add(outputs, biases, data_format=data_format)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn',
                                            data_format=data_format)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def conv2d_flipout(inputs, num_output_channels, kernel_size, scope, stride=[1, 1], padding='SAME', data_format='NHWC',
                   activation_fn=tf.compat.v1.nn.relu, bn=False):
    kernel_posterior_scale_mean = -9.0,
    kernel_posterior_scale_stddev = 0.1,
    kernel_posterior_scale_constraint = 0.2

    def _untransformed_scale_constraint(t):
        return tf.clip_by_value(t, -1000,
                                tf.math.log(kernel_posterior_scale_constraint))

    kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn(untransformed_scale_initializer=
    tf.compat.v1.initializers.random_normal(
        mean=kernel_posterior_scale_mean,
        stddev=kernel_posterior_scale_stddev),
        untransformed_scale_constraint=
        _untransformed_scale_constraint)

    with tf.compat.v1.variable_scope(scope) as sc:
        output = tfp.layers.Convolution2DFlipout(num_output_channels, kernel_size=kernel_size, padding=padding,
                                                 strides=stride, data_format='channels_last',
                                                 kernel_posterior_fn=kernel_posterior_fn)(inputs)
        if bn:
            output = tf.keras.layers.BatchNormalization()(output)

        if activation_fn is not None:
            output = tf.keras.layers.Activation('relu')(output)

    return output


def conv2d_transpose(inputs,
                     num_output_channels,
                     kernel_size,
                     scope,
                     stride=[1, 1],
                     padding='SAME',
                     use_xavier=True,
                     stddev=1e-3,
                     weight_decay=None,
                     activation_fn=tf.compat.v1.nn.relu,
                     bn=False,
                     bn_decay=None,
                     is_training=None):
    """ 2D convolution transpose with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch_size norm
    bn_decay: float or float tensor variable in [0,1]

  Returns:
    Variable tensor

  Note: conv2d(conv2d_transpose(a, num_out, ksize, stride), a.shape[-1], ksize, stride) == a
  """
    with tf.compat.v1.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_output_channels, num_in_channels]  # reversed to conv2d
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_h, stride_w = stride

        # from slim.convolution2d_transpose
        def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
            dim_size *= stride_size

            if padding == 'VALID' and dim_size is not None:
                dim_size += max(kernel_size - stride_size, 0)
            return dim_size

        # caculate output shape
        batch_size = inputs.get_shape()[0].value
        height = inputs.get_shape()[1].value
        width = inputs.get_shape()[2].value
        out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
        out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
        output_shape = [batch_size, out_height, out_width, num_output_channels]

        outputs = tf.compat.v1.nn.conv2d_transpose(inputs, kernel, output_shape,
                                                   [1, stride_h, stride_w, 1],
                                                   padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.compat.v1.constant_initializer(0.0))
        outputs = tf.compat.v1.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def conv3d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.compat.v1.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
    """ 3D convolution with non-linear operation.

  Args:
    inputs: 5-D tensor variable BxDxHxWxC
    num_output_channels: int
    kernel_size: a list of 3 ints
    scope: string
    stride: a list of 3 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch_size norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
    with tf.compat.v1.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_d, kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_d, stride_h, stride_w = stride
        outputs = tf.compat.v1.nn.conv3d(inputs, kernel,
                                         [1, stride_d, stride_h, stride_w, 1],
                                         padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.compat.v1.constant_initializer(0.0))
        outputs = tf.compat.v1.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv3d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=None,
                    activation_fn=tf.compat.v1.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None):
    """ Fully connected layer with non-linear operation.
  
  Args:
    inputs: 2-D tensor BxN
    num_outputs: int
  
  Returns:
    Variable tensor of size B x num_outputs.
    :param use_xavier:
  """
    with tf.compat.v1.variable_scope(scope) as sc:
        num_input_units = inputs.get_shape()[-1].value
        weights = _variable_with_weight_decay('weights',
                                              shape=[num_input_units, num_outputs],
                                              use_xavier=use_xavier,
                                              stddev=stddev,
                                              wd=weight_decay)
        outputs = tf.compat.v1.matmul(inputs, weights)
        biases = _variable_on_cpu('biases', [num_outputs],
                                  tf.compat.v1.constant_initializer(0.0))
        outputs = tf.compat.v1.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def dense_flipout(inputs,
                  num_outputs,
                  scope,
                  use_xavier=True,
                  stddev=1e-3,
                  weight_decay=None,
                  activation_fn=tf.compat.v1.nn.relu,
                  bn=False,
                  bn_decay=None,
                  is_training=None):
    """ Fully connected layer with non-linear operation.

  Args:
    inputs: 2-D tensor BxN
    num_outputs: int

  Returns:
    Variable tensor of size B x num_outputs.
    :param is_training:
    :param use_xavier:
  """
    with tf.compat.v1.variable_scope(scope) as sc:
        outputs = tfp.layers.DenseFlipout(num_outputs, activation='relu')

        # if bn:
        #     outputs = tf.keras.layers.BatchNormalization(outputs)

        # if activation_fn is not None:
        #     outputs = tf.keras.layers.Activation('relu')(outputs)
        return outputs


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
    """ 2D max pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
    :param inputs:
    :param padding:
  """
    with tf.compat.v1.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.compat.v1.nn.max_pool(inputs,
                                           ksize=[1, kernel_h, kernel_w, 1],
                                           strides=[1, stride_h, stride_w, 1],
                                           padding=padding,
                                           name=sc.name)
        return outputs


def avg_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
    """ 2D avg pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
    with tf.compat.v1.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.compat.v1.nn.avg_pool(inputs,
                                           ksize=[1, kernel_h, kernel_w, 1],
                                           strides=[1, stride_h, stride_w, 1],
                                           padding=padding,
                                           name=sc.name)
        return outputs


def max_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
    """ 3D max pooling.

  Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints
  
  Returns:
    Variable tensor
  """
    with tf.compat.v1.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        stride_d, stride_h, stride_w = stride
        outputs = tf.compat.v1.nn.max_pool3d(inputs,
                                             ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                                             strides=[1, stride_d, stride_h, stride_w, 1],
                                             padding=padding,
                                             name=sc.name)
        return outputs


def avg_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
    """ 3D avg pooling.

  Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints
  
  Returns:
    Variable tensor
  """
    with tf.compat.v1.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        stride_d, stride_h, stride_w = stride
        outputs = tf.compat.v1.nn.avg_pool3d(inputs,
                                             ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                                             strides=[1, stride_d, stride_h, stride_w, 1],
                                             padding=padding,
                                             name=sc.name)
        return outputs


def batch_norm_template_unused(inputs, is_training, scope, moments_dims, bn_decay):
    """ NOTE: this is older version of the util func. it is deprecated.
  Batch normalization on convolutional maps and beyond...
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  
  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.compat.v1.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
  Return:
      normed:        batch_size-normalized maps
  """
    with tf.compat.v1.variable_scope(scope) as sc:
        num_channels = inputs.get_shape()[-1].value
        beta = _variable_on_cpu(name='beta', shape=[num_channels],
                                initializer=tf.compat.v1.constant_initializer(0))
        gamma = _variable_on_cpu(name='gamma', shape=[num_channels],
                                 initializer=tf.compat.v1.constant_initializer(1.0))
        batch_mean, batch_var = tf.compat.v1.nn.moments(inputs, moments_dims, name='moments')
        decay = bn_decay if bn_decay is not None else 0.9
        ema = tf.compat.v1.train.ExponentialMovingAverage(decay=decay)
        # Operator that maintains moving averages of variables.
        # Need to set reuse=False, otherwise if reuse, will see moments_1/mean/ExponentialMovingAverage/ does not exist
        # https://github.com/shekkizh/WassersteinGAN.tensorflow/issues/3
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=False):
            ema_apply_op = tf.compat.v1.cond(is_training,
                                             lambda: ema.apply([batch_mean, batch_var]),
                                             lambda: tf.compat.v1.no_op())

        # Update moving average and return current batch_size's avg and var.
        def mean_var_with_update():
            with tf.compat.v1.control_dependencies([ema_apply_op]):
                return tf.compat.v1.identity(batch_mean), tf.compat.v1.identity(batch_var)

        # ema.average returns the Variable holding the average of var.
        mean, var = tf.compat.v1.cond(is_training,
                                      mean_var_with_update,
                                      lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.compat.v1.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
    return normed


def batch_norm_template(inputs, is_training, scope, moments_dims_unused, bn_decay, data_format='NHWC'):
    """ Batch normalization on convolutional maps and beyond...
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  
  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.compat.v1.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
      data_format:   'NHWC' or 'NCHW'
  Return:
      normed:        batch_size-normalized maps
  """
    bn_decay = bn_decay if bn_decay is not None else 0.9
    return tf.contrib.layers.batch_norm(inputs,
                                        center=True, scale=True,
                                        is_training=is_training, decay=bn_decay, updates_collections=None,
                                        scope=scope,
                                        data_format=data_format)


def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
    """ Batch normalization on FC data.
  
  Args:
      inputs:      Tensor, 2D BxC input
      is_training: boolean tf.compat.v1.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch_size-normalized maps
  """
    return batch_norm_template(inputs, is_training, scope, [0, ], bn_decay)


def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope, data_format):
    """ Batch normalization on 1D convolutional maps.
  
  Args:
      inputs:      Tensor, 3D BLC input maps
      is_training: boolean tf.compat.v1.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
      data_format: 'NHWC' or 'NCHW'
  Return:
      normed:      batch_size-normalized maps
  """
    return batch_norm_template(inputs, is_training, scope, [0, 1], bn_decay, data_format)


def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope, data_format):
    """ Batch normalization on 2D convolutional maps.
  
  Args:
      inputs:      Tensor, 4D BHWC input maps
      is_training: boolean tf.compat.v1.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
      data_format: 'NHWC' or 'NCHW'
  Return:
      normed:      batch_size-normalized maps
  """
    return batch_norm_template(inputs, is_training, scope, [0, 1, 2], bn_decay, data_format)


def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 3D convolutional maps.
  
  Args:
      inputs:      Tensor, 5D BDHWC input maps
      is_training: boolean tf.compat.v1.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch_size-normalized maps
  """
    return batch_norm_template(inputs, is_training, scope, [0, 1, 2, 3], bn_decay)


def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
    """ Dropout layer.

  Args:
    inputs: tensor
    is_training: boolean tf.compat.v1.Variable
    scope: string
    keep_prob: float in [0,1]
    noise_shape: list of ints

  Returns:
    tensor variable
  """
    with tf.compat.v1.variable_scope(scope) as sc:
        outputs = tf.compat.v1.cond(is_training,
                                    lambda: tf.compat.v1.nn.dropout(inputs, noise_shape, rate=keep_prob),
                                    lambda: inputs)
        return outputs
