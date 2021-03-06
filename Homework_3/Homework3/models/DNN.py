from models.BaseNN import *


class DNN(BaseNN):

    def network(self):
        self.X_tf = tf.reshape(self.X_tf, [-1, self.data_loader.config["img_h"] * self.data_loader.config["img_w"] * self.data_loader.config["img_chls"]])

        # Layer 1
        Z1 = tf.layers.dense(self.X_tf, units=self.data_loader.config["num_cls"], activation=None)

        return Z1

    def metrics(self):
        cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_preds_tf, labels=self.y_tf)

        return cost
