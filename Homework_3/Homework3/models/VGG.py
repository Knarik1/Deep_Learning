from models.BaseNN import *


class VGG(BaseNN):

    def network(self):
        tf.set_random_seed(1)

        # parameters
        W1 = tf.get_variable('W1', shape=[3, 3, 1, 64], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        W2 = tf.get_variable('W2', shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        W3 = tf.get_variable('W3', shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        W4 = tf.get_variable('W4', shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        W5 = tf.get_variable('W5', shape=[3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        W6 = tf.get_variable('W6', shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer(seed=0))

        # Layer 1
        Z1 = tf.nn.conv2d(self.X_tf, W1, strides=[1, 1, 1, 1], padding='SAME')
        Z1 = tf.layers.batch_normalization(Z1, name='layer_1_norm', training=self.training_flag, momentum=0.9)
        Z1 = tf.nn.dropout(Z1, 0.4)
        A1 = tf.nn.relu(Z1)

        # Layer 2
        Z2 = tf.nn.conv2d(A1, W2, strides=[1, 1, 1, 1], padding='SAME')
        Z2 = tf.layers.batch_normalization(Z2, name='layer_2_norm', training=self.training_flag, momentum=0.9)
        Z2 = tf.nn.dropout(Z2, 0.4)
        A2 = tf.nn.relu(Z2)

        P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Layer 3
        Z3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='SAME')
        Z3 = tf.layers.batch_normalization(Z3, name='layer_3_norm', training=self.training_flag, momentum=0.9)
        Z3 = tf.nn.dropout(Z3, 0.4)
        A3 = tf.nn.relu(Z3)

        # Layer 4
        Z4 = tf.nn.conv2d(A3, W4, strides=[1, 1, 1, 1], padding='SAME')
        Z4 = tf.layers.batch_normalization(Z4, name='layer_4_norm', training=self.training_flag, momentum=0.9)
        Z4 = tf.nn.dropout(Z4, 0.4)
        A4 = tf.nn.relu(Z4)

        P4 = tf.nn.max_pool(A4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Layer 5
        Z5 = tf.nn.conv2d(P4, W5, strides=[1, 1, 1, 1], padding='SAME')
        Z5 = tf.layers.batch_normalization(Z5, name='layer_5_norm', training=self.training_flag, momentum=0.9)
        Z5 = tf.nn.dropout(Z5, 0.4)
        A5 = tf.nn.relu(Z5)

        # Layer 6
        Z6 = tf.nn.conv2d(A5, W6, strides=[1, 1, 1, 1], padding='SAME')
        Z6 = tf.layers.batch_normalization(Z6, name='layer_6_norm', training=self.training_flag, momentum=0.9)
        Z6 = tf.nn.dropout(Z6, 0.4)
        A6 = tf.nn.relu(Z6)

        P6 = tf.nn.max_pool(A6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # FC
        P7 = tf.contrib.layers.flatten(P6)
        # FC-1
        Z7 = tf.contrib.layers.fully_connected(P7, 1024, activation_fn=tf.nn.relu)
        Z7 = tf.nn.dropout(Z7, 0.4)
        # FC-2
        Z8 = tf.contrib.layers.fully_connected(Z7, 1024, activation_fn=tf.nn.relu)
        Z8 = tf.nn.dropout(Z8, 0.4)
        # softmax
        Z8 = tf.contrib.layers.fully_connected(Z8, self.config["num_cls"], activation_fn=None)

        return Z8

    def metrics(self):
        cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_preds_tf, labels=self.y_tf)

        return cost
