from layers import EdgeConv
import tensorflow as tf

tfk = tf.keras


class GraphNet(tfk.Model):
    def __init__(self, k, channels, classifier, activation, dropout, static, **kwargs):
        super(GraphNet, self).__init__(**kwargs)
        self.static = static
        self.edge_blocks = []
        self.concatenate = tfk.layers.Concatenate()
        self.global_pooling = tfk.layers.GlobalAveragePooling1D()
        self.classifier = []
        self.softmax = tfk.layers.Softmax()

        for chan in channels:
            self.edge_blocks.append(
                EdgeConv(k=k, n_channel_out=chan, activation=activation)
            )
        for nodes in classifier[:-1]:
            if nodes != 2:
                self.classifier.append(tfk.layers.Dropout(dropout))
                self.classifier.append(tfk.layers.Dense(nodes, activation=activation))

        self.classifier.append(
            tfk.layers.Dense(
                classifier[-1],
            )
        )

    @tf.function
    def call(
        self,
        inputs,
        training=None,
    ):
        coordinates = inputs[0]
        point_cloud = inputs[1]
        mask = inputs[2] if len(inputs) == 3 else None

        x = [point_cloud]
        # If static, don't use new coordinates for graph construction
        if self.static:
            for edgeblock in self.edge_blocks:
                x.append(edgeblock([coordinates, x[-1]], training=training, mask=mask))
        # I not static, dynamically update graph after each edge block
        else:
            # Calculate first edge block
            x.append(
                self.edge_blocks[0](
                    [coordinates, x[-1]],
                    training=training,
                    mask=mask,
                )
            )
            # Use previous output for consecutive edge blocks
            for edgeblock in self.edge_blocks[1:]:
                x.append(edgeblock([x[-1], x[-1]], mask=mask))

        x = self.concatenate(x[1:])
        x = self.global_pooling(x)

        for dense in self.classifier:
            x = dense(x)

        x = self.softmax(x)
        return x


if __name__ == "__main__":
    pass
