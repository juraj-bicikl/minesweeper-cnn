import tensorflow as tf
import numpy as np
from tf_graph import TfGraph


class SimpleCnn:
    """ Constructs simple convolutional neural networks.

        The network predicts probabilities of finding a mine at any field.
    """

    def __init__(self,
                 input_height: int,
                 input_width: int,
                 num_hidden_layers: int,
                 num_hidden_channels: int,
                 kernel_width: int,
                 l2_penalty: float,
                 optimizer: tf.train.Optimizer,
                 include_layer_norm: bool,
                 seed: int = 123):

        assert kernel_width % 2 == 1  # we allow only odd kernel widths

        self.input_height = input_height
        self.input_width = input_width
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_channels = num_hidden_channels
        self.kernel_width = kernel_width
        self.l2_penalty = l2_penalty
        self.optimizer = optimizer
        self.include_layer_norm = include_layer_norm
        self.seed = seed
        self.__build_seed = None

    def build(self) -> TfGraph:
        self.__build_seed = self.seed

        tf.get_default_graph().finalize()
        graph = tf.Graph()

        with graph.as_default():
            raw_input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_height, self.input_width], name="input")
            raw_target = tf.placeholder(dtype=tf.float32, shape=[None, self.input_height, self.input_width], name="target")
            out = self.process_input(raw_input)
            target = self.process_target(raw_target)

            layer_penalties = []
            # adding hidden layers
            for i in range(self.num_hidden_layers):
                out, layer_penalty = self.add_cnn_layer(out, self.kernel_width, self.num_hidden_channels, apply_activation=True,
                                                        include_layer_norm=self.include_layer_norm, name="cnn_layer_%d" % (i+1))
                layer_penalties.append(layer_penalty)

            # final layer
            out, layer_penalty = self.add_cnn_layer(out, 1, 2, apply_activation=False, include_layer_norm=False, name="final_layer")
            layer_penalties.append(layer_penalty)

            # total penalty
            with tf.variable_scope("penalty"):
                total_penalty = tf.multiply(tf.constant(self.l2_penalty, dtype=np.float), tf.add_n(layer_penalties), name="total_penalty")

            # cost
            with tf.variable_scope("cross_entropy_cost"):
                cross_entropy_cost = tf.reduce_mean(- tf.multiply(target, tf.nn.log_softmax(out)), name="error_cost")

            # training cost and optimization
            training_cost = tf.add(cross_entropy_cost, total_penalty, "total_cost")
            train = self.optimizer.minimize(training_cost, name="train")

            final_output = tf.nn.softmax(out, axis=3, name="output")

            summary = tf.summary.scalar("error_cost_summary", cross_entropy_cost)
            tf.variables_initializer(tf.global_variables(), name='init')
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        graph.finalize()
        return TfGraph(graph,
                       saver,
                       train_name='train',
                       init_name='init',
                       input_name=raw_input.name,
                       target_name=raw_target.name,
                       output_name=final_output.name,
                       error_cost_name=cross_entropy_cost.name,
                       total_cost_name=training_cost.name,
                       summary_name=summary.name)

    def add_cnn_layer(self, input_tensor, kernel_width, out_channels, apply_activation, include_layer_norm, name="cnn_layer") -> (tf.Tensor, tf.Tensor):
        """ Adds convolutional layer on top of 'input_tensor'.

            Args:
                 input_tensor (tf.Tensor): input tensor
                 kernel_width (int): kernel width
                 out_channels (int): number of channels in output tensor
                 apply_activation (bool): whether to apply 'relu' activation at the end
                 include_layer_norm (bool): whether to include layer normalization
                 name (str): name of the layer

            Returns:
                (tf.Tensor, tf.Tensor): pair of (output tensor, penalty tensor)
        """
        with tf.variable_scope(name):
            in_channels = input_tensor.get_shape().dims[3].value
            kernel_shape = [kernel_width, kernel_width, in_channels, out_channels]
            kernel_init = tf.random_uniform(np.array(kernel_shape), -1e-2, 1e-2, tf.float32, seed=self.__next_seed())
            kernel = tf.Variable(kernel_init, name="kernel")
            bias = tf.Variable(tf.zeros([out_channels], dtype=np.float32), name="bias")
            penalty_node = tf.nn.l2_loss(kernel, name="l2_loss") if self.l2_penalty != 0.0 else None

            # padding input tensor
            padding_depth = kernel_width // 2
            if padding_depth > 0:
                out = tf.pad(input_tensor,
                             paddings=[[0, 0], [padding_depth, padding_depth], [padding_depth, padding_depth], [0, 0]],
                             mode='CONSTANT',
                             name="padding",
                             constant_values=-0.5)
            else:
                out = input_tensor

            out = tf.nn.conv2d(out, kernel, strides=[1, 1, 1, 1], padding="VALID")
            out = tf.nn.bias_add(out, bias)
            if include_layer_norm:
                out = self.add_layer_normalization(out)
            if apply_activation:
                out = tf.nn.relu(out)  # 'relu' activation

        return out, penalty_node

    def __next_seed(self):
        self.__build_seed += 1
        return self.__build_seed

    def add_layer_normalization(self, input_tensor) -> tf.Tensor:
        inputs_shape = input_tensor.get_shape()
        params_shape = inputs_shape[-1:]
        epsilon = 1e-12

        with tf.variable_scope("layer_norm"):
            inputs_rank = inputs_shape.ndims
            axis = list(range(1, inputs_rank))

            beta = tf.Variable(tf.zeros(params_shape, np.float32), name="beta")
            gamma = tf.Variable(tf.ones(params_shape, np.float32), name="gamma")
            mean, variance = tf.nn.moments(input_tensor, axis, keep_dims=True)

            outputs = tf.nn.batch_normalization(input_tensor, mean, variance, beta, gamma, epsilon)
            outputs.set_shape(inputs_shape)
            return outputs

    def process_input(self, raw_input) -> tf.Tensor:
        with tf.variable_scope("input_processing"):
            # we first transform input tensor to shape: [num_examples, height, width, 1] after which we can add
            # convolutional layers
            out = tf.reshape(raw_input, shape=[-1, self.input_height, self.input_width, 1], name="reshape_to_nhwc")

            # we add 1 to all the input values (i.e. we shift the encoding by 1 since we use -0.5 value for padding)
            return tf.add(out, tf.constant(1.0))

    def process_target(self, raw_target) -> tf.Tensor:
        with tf.variable_scope("target_processing"):
            raw_target_prime = tf.subtract(tf.constant(1.0, dtype=np.float32), raw_target)
            return tf.concat([tf.reshape(raw_target, shape=[-1, self.input_height, self.input_width, 1]),
                              tf.reshape(raw_target_prime, shape=[-1, self.input_height, self.input_width, 1])],
                             axis=3, name="processed_target")
