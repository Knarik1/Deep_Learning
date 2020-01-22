from models.BaseNN import *
from tensorflow.contrib import rnn


class RNN(BaseNN):

    def network(self):
        self.X_tf = tf.reshape(self.X_tf, [-1, self.data_loader.config["img_h"], self.data_loader.config["img_w"]])
        X_tf_unstacked = tf.unstack(self.X_tf, self.data_loader.config["img_h"], 1)

        # 4 layers
        rnn_layers = [rnn.LSTMCell(size) for size in [32, 64, 128, 256]]
        multi_rnn_cell = rnn.MultiRNNCell(rnn_layers)

        outputs, state = rnn.static_rnn(cell=multi_rnn_cell, inputs=X_tf_unstacked, dtype=tf.float32)

        Z = tf.layers.dense(outputs[-1], self.data_loader.config["num_cls"], activation=None)
        return Z

    def metrics(self):
        cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_preds_tf, labels=self.y_tf)

        return cost
