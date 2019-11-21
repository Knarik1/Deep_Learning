from models.BaseNN import *


class CNN(BaseNN):

    def network(self):
        tf.set_random_seed(1)

        # parameters
        W1 = tf.get_variable('W1', shape=[3, 3, 1, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        W2 = tf.get_variable('W2', shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        W3 = tf.get_variable('W3', shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer(seed=0))

        # Layer 1
        Z1 = tf.nn.conv2d(self.X_tf, W1, strides=[1, 1, 1, 1], padding='SAME')
        Z1 = tf.layers.batch_normalization(Z1, name='layer_1_norm', training=self.training_flag, momentum=0.9)
        #Z1 = tf.nn.dropout(Z1, 0.2)
        A1 = tf.nn.relu(Z1)
        P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Layer 2
        Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
        Z2 = tf.layers.batch_normalization(Z2, name='layer_2_norm', training=self.training_flag, momentum=0.9)
        #Z2 = tf.nn.dropout(Z2, 0.2)
        A2 = tf.nn.relu(Z2)
        P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Layer 3
        Z3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='SAME')
        Z3 = tf.layers.batch_normalization(Z3, name='layer_3_norm', training=self.training_flag, momentum=0.9)
        #Z3 = tf.nn.dropout(Z3, 0.2)
        A3 = tf.nn.relu(Z3)
        P3 = tf.nn.max_pool(A3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # FC
        Z4 = tf.contrib.layers.flatten(P3)
        Z4 = tf.contrib.layers.fully_connected(Z4, 1024, activation_fn=tf.nn.relu)
        Z4 = tf.layers.batch_normalization(Z4, name='layer_4_norm', training=self.training_flag, momentum=0.9)
        Z4 = tf.nn.dropout(Z4, 0.2)
        # softmax
        Z4 = tf.contrib.layers.fully_connected(Z4, self.config["num_cls"], activation_fn=None)

        return Z4

    def metrics(self):
        cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_preds_tf, labels=self.y_tf)

        return cost
