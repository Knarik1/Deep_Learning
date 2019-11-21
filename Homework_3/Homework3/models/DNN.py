from models.BaseNN import *


class DNN(BaseNN):

    def network(self):
        # Layer 1
        Z1 = tf.layers.dense(self.X_tf, units=self.config["num_cls"], activation=tf.nn.softmax)

        return Z1

    def metrics(self):
        cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_preds_tf, labels=self.y_tf)

        return cost
