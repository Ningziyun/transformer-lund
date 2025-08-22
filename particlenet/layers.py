import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tfk = tf.keras
K = tfk.backend

@tf.function
def getDistanceMatrix(x):
    """ Compute pairwise distance matrix for a point cloud

    Input:
    point_cloud: tensor (batch_size, n_points, n_features)

    Returns:
    dists: tensor (batch_size, n_points, n_points) pairwise distances
    """
    part1 = -2 * K.batch_dot(x, K.permute_dimensions(x, (0, 2, 1)))
    part2 = K.permute_dimensions(K.expand_dims(K.sum(x**2, axis=2)), (0, 2, 1))
    part3 = K.expand_dims(K.sum(x**2, axis=2))
    dists = part1 + part2 + part3
    max_dist = tf.math.reduce_max(dists)
    diag = tf.linalg.diag(tf.fill((x.shape[1],), max_dist+1.))
    diag = K.expand_dims(diag, axis=0)
    dists = tf.math.add(dists, diag)
    return dists


@tf.function
def getKnearest(dists, k):
    """Get indices of k nearest neighbors from distance tensor
    Input:
    dists: (batch_size, n_points, n_points) pairwise distances
    Returns:
    knn_idx: (batch_size, n_points, k) nearest neighbor indices
    """
    _, knn_idx = tf.math.top_k(-dists, k=k)
    return knn_idx


@tf.function
def getEdgeFeature(point_cloud, nn_idx):
    """Construct the input for the edge convolution
    Input:
    point_cloud: (batch_size, n_points, n_features)
    nn_idx: (batch_size, n_points, n_neighbors)
    Returns:
    edge_features: (batch_size, n_points, k, n_features*2)
    """
    k = nn_idx.get_shape()[-1]

    point_cloud_shape = tf.shape(point_cloud)
    batch_size = point_cloud_shape[0]
    n_points = point_cloud_shape[1]
    n_features = point_cloud_shape[2]

    # Prepare indices to match neighbors in flattened cloud
    idx = K.arange(0, stop=batch_size, step=1) * n_points
    idx = K.reshape(idx, [-1, 1, 1])

    # Flatten cloud and gather neighbors
    flat_cloud = K.reshape(point_cloud, [-1, n_features])
    neighbors = K.gather(flat_cloud, nn_idx+idx)

    # Expand centers to (batch_size, n_points, k, n_features)
    cloud_centers = K.expand_dims(point_cloud, axis=-2)
    cloud_centers = K.tile(cloud_centers, [1, 1, k, 1])

    edge_features = K.concatenate([cloud_centers, neighbors-cloud_centers],
        axis=-1)
    return edge_features


class EdgeConv(tfk.layers.Layer):
    """ Keras layer to perform EdgeConvolutions (1801.07829)
    From a point cloud as input generate graph with connections between k
    nearest neighbors and perform a convolution over these local patches
    """
    def __init__(self, k, n_channel_out, activation,
                 kernel_size=1, debug=False, **kwargs):
        # Possibility to calculate proximity in given dimensions
        self.k = k
        self.debug = debug
        self.kernel_size = kernel_size
        self.n_channel_out = n_channel_out
        self.activation = activation
        super(EdgeConv, self).__init__(**kwargs)

    def build(self, input_shape):
        coordinate_shape = input_shape[0]
        input_shape = input_shape[1]

        kernel_initializer = 'truncated_normal'
        bias_initializer = 'zeros'

        if self.debug:
            tf.random.set_seed(0)
            kernel_initializer = 'ones'

        n_channel_in = input_shape[-1] * 2

        kernel_shape = [self.kernel_size, self.kernel_size,
                        n_channel_in, self.n_channel_out[0]]
        # Start array for convolution weights
        self.kernel = [self.add_weight(name='kernel0',
                                    shape=kernel_shape,
                                    initializer=kernel_initializer,
                                    trainable=True)]
        # Start array for convolution biases
        self.bias = [self.add_weight(name='bias0',
                                    shape=[kernel_shape[-1]],
                                    initializer=bias_initializer,
                                    trainable=True)]

        self.gamma = [self.add_weight(name='gamma0',
                                    shape=[kernel_shape[-1]],
                                    initializer='ones',
                                    trainable=True)]
        self.beta = [self.add_weight(name='beta0',
                                    shape=[kernel_shape[-1]],
                                    initializer='zeros',
                                    trainable=True)]
        self.moving_mean = [self.add_weight(name='moving_mean0',
                                    shape=[kernel_shape[-1]],
                                    initializer='zero',
                                    trainable=False)]
        self.moving_var = [self.add_weight(name='moving_var0',
                                    shape=[kernel_shape[-1]],
                                    initializer='one',
                                    trainable=False)]

        #Adding weights for muore convolutions before constructing a new graph
        for i in range(1, len(self.n_channel_out)):
            kernel_shape = [self.kernel_size, self.kernel_size,
                self.n_channel_out[i-1], self.n_channel_out[i]]
            self.kernel.append(self.add_weight(name='kernel{}'.format(i),
                                    shape=kernel_shape,
                                    initializer=kernel_initializer,
                                    trainable=True))
            self.bias.append(self.add_weight(name='bias{}'.format(i),
                                    shape=[kernel_shape[-1]],
                                    initializer=bias_initializer,
                                    trainable=True))
            self.gamma.append(self.add_weight(name='gamma{}'.format(i),
                                    shape=[kernel_shape[-1]],
                                    initializer='ones',
                                    trainable=True))
            self.beta.append(self.add_weight(name='beta{}'.format(i),
                                    shape=[kernel_shape[-1]],
                                    initializer='zeros',
                                    trainable=True))
            self.moving_mean.append(self.add_weight(
                                    name='moving_mean{}'.format(i),
                                    shape=[kernel_shape[-1]],
                                    initializer='zero',
                                    trainable=False))
            self.moving_var.append(self.add_weight(
                                    name='moving_var{}'.format(i),
                                    shape=[kernel_shape[-1]],
                                    initializer='one',
                                    trainable=False))
        super(EdgeConv, self).build(input_shape)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        def batch_norm(inputs, gamma, beta, dims, ind):
            """ Normalize batch and update moving averages for mean and std
            Input:
              inputs: (batchsize, n_points, k, n_features * 2) - edge_features
              gamma: weight - gamma for batch normalization
              beta: weight - beta for batch normalization
              dims: list - dimensions along which to normalize
              ind: int - indicating which weights to use
            Returns:
             During training:
              normed: (batchsize, n_points, k, n_features * 2) - normalized
                            batch of data using actual batch for normalization
             Else:
              normed_moving: same, but using the updated average values
            """

            # Calculate normalized data, mean and std for batch
            normed, batch_mean, batch_var = K.normalize_batch_in_training(
                                                x=inputs,
                                                gamma=gamma,
                                                beta=beta,
                                                reduction_axes=dims)
            # Update the moving averages
            if training:
                self.add_update(
                    [K.moving_average_update(self.moving_mean[ind], batch_mean, 0.9),
                     K.moving_average_update(self.moving_var[ind], batch_var, 0.9)])
                return normed

            else:
                # Calculate normalization using the averages
                normed_moving = K.batch_normalization(
                                                    x=inputs,
                                                    mean=self.moving_mean[ind],
                                                    var=self.moving_var[ind],
                                                    beta=beta,
                                                    gamma=gamma)
                return normed_moving


        coordinates = inputs[0]
        point_cloud = inputs[1]

        if not mask is None:
            coordinates = coordinates + tf.expand_dims(
                tf.cast(mask, tf.float32) - 1, -1) * 999.

        dists = getDistanceMatrix(coordinates)

        knn_idx = getKnearest(dists, self.k)
        edge_features = getEdgeFeature(point_cloud, knn_idx)

        # Create first convolutional block
        output = K.conv2d(edge_features, self.kernel[0], (1, 1), padding='same')
        output = K.bias_add(output, self.bias[0])
        output = batch_norm(output, self.gamma[0], self.beta[0], [0, 1, 2], 0)
        output = self.activation(output)

        # Additional convolutional blocks
        for i in range(1, len(self.n_channel_out)):
            output = K.conv2d(output, self.kernel[i], (1, 1), padding='same')
            output = K.bias_add(output, self.bias[i])
            output = batch_norm(output,
                                    self.gamma[i], self.beta[i], [0, 1, 2], i)
            output = self.activation(output)

        output = K.mean(output, axis=-2)

        if not mask is None:
            output = output * tf.expand_dims(
                tf.cast(mask, tf.float32), -1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.n_channel_out[-1])

    def get_config(self):
        # Store values necessary to load model later
        base_config = super(EdgeConv, self).get_config()
        base_config['k'] = self.k
        base_config['kernel_size'] = self.kernel_size
        base_config['n_channel_out'] = self.n_channel_out
        base_config['nearest_ind'] = self.n_ind
        return base_config


if __name__ == '__main__':
    pass