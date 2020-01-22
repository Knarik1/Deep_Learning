import tensorflow as tf
from data_loader import *
import numpy as np
from abc import abstractmethod

class BaseNN:
    def __init__(self, train_images_dir, val_images_dir, test_images_dir, num_epochs, train_batch_size,
                 val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels, 
                 num_classes, learning_rate, base_dir, max_to_keep, model_name, model):

        self.data_loader = DataLoader(train_images_dir, val_images_dir, test_images_dir, train_batch_size,
                                      val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels, num_classes, model)

        self.config = {
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "base_dir": base_dir,
            "max_to_keep": max_to_keep,
            "model_name": model_name
        }

        save_path = os.path.join(self.config["base_dir"], self.config["model_name"])
        self.checkpoints_path = os.path.join(save_path, "checkpoints")
        self.summaries_path = os.path.join(save_path, "summaries")

        # check if checkpoints' directory exists, otherwise create it.
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)

        # check if summaries' directory exists, otherwise create it.
        if not os.path.exists(self.summaries_path):
            os.makedirs(self.summaries_path)

    def create_network(self):
        # clears the default graph stack and resets the global default graph.
        tf.reset_default_graph()

        self.create_placeholders()
        self.y_preds_tf = self.network()
        self.compute_cost()
        self.compute_accuracy()
        self.create_optimizer()

    def init_saver(self):
        # initialize Saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config["max_to_keep"])

    def init_fwriter(self):
        with tf.name_scope('summaries_train'):
            tf.summary.scalar("train_accuracy", self.accuracy)
            tf.summary.scalar("train_cost", self.cost)
            self.train_summary_merged = tf.summary.merge_all()

        with tf.name_scope('summaries_val'):
            tf.summary.scalar("val_accuracy", self.accuracy)
            tf.summary.scalar("val_cost", self.cost)
            self.val_summary_merged = tf.summary.merge_all()

        # initialize FileWriter to save summaries
        self.train_writer = tf.summary.FileWriter(self.summaries_path + '/train', self.sess.graph)
        self.val_writer = tf.summary.FileWriter(self.summaries_path + '/val', self.sess.graph)

    def create_placeholders(self):
        self.X_tf = tf.placeholder(tf.float32, shape=[None, self.data_loader.config["img_h"], self.data_loader.config["img_w"], self.data_loader.config["img_chls"]], name='images')
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.data_loader.config["num_cls"]], name='labels')
        self.training_flag = tf.placeholder_with_default(False, shape=[], name='training_flag')

    def initialize_network(self):
        # opening Session
        self.sess = tf.Session()

        # initializing Saver
        self.init_saver()

        # initializing FileWriter
        self.init_fwriter()

        ckpt = tf.train.get_checkpoint_state(self.checkpoints_path)

        # restore/init weights
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("----Restore params----")
        else:
            self.sess.run(tf.global_variables_initializer())
            print("----Init params----")

    def compute_cost(self):
        loss = self.metrics()
        self.cost = tf.reduce_mean(loss)

    def compute_accuracy(self):
        # Calculate the correct predictions
        predict_op = tf.argmax(self.y_preds_tf, axis=1)
        ground_truth_op = tf.argmax(self.y_tf, axis=1)
        correct_prediction = tf.equal(predict_op, ground_truth_op)

        # Calculate accuracy
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float") * 100)

    def create_optimizer(self):
        # update ops for batch normalization
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config["learning_rate"]).minimize(self.cost)

        self.optimizer = optimizer

    def train_model(self, display_step, validation_step, checkpoint_step, summary_step):
        # number of iterations
        num_iter_train = np.ceil(self.data_loader.config["m_train"] / self.data_loader.config["train_batch_size"])
        num_iter_val = np.ceil(self.data_loader.config["m_val"] / self.data_loader.config["val_batch_size"])

        for epoch in range(self.config["num_epochs"]):
            # ------------------------------------- Train ---------------------------------------
            minibatch_cost_sum_train = 0
            minibatch_acc_sum_train = 0

            # getting new sequence for every epoch to shuffle entire train dataset
            perm = np.random.permutation(self.data_loader.config["m_train"])

            for iter in range(int(num_iter_train)):
                X_batch, y_batch, y_cls_batch = self.data_loader.train_data_loader(iter, perm=perm)

                feed_dict_train = {
                    self.X_tf: X_batch,
                    self.y_tf: y_batch,
                    self.training_flag: True
                }

                _, minibatch_cost, minibatch_acc, train_summary = self.sess.run([self.optimizer, self.cost, self.accuracy, self.train_summary_merged], feed_dict=feed_dict_train)

                minibatch_cost_sum_train += minibatch_cost
                minibatch_acc_sum_train += minibatch_acc

                if iter % summary_step == 0:
                    # train summary
                    self.train_writer.add_summary(train_summary, epoch * self.data_loader.config["train_batch_size"] + iter)

            # mean cost and accuracy for each epoch
            minibatch_cost_mean_train = minibatch_cost_sum_train / num_iter_train
            minibatch_acc_mean_train = minibatch_acc_sum_train / num_iter_train


            # ------------------------------------- Validation -------------------------------------
            minibatch_cost_sum_val = 0
            minibatch_acc_sum_val = 0

            # getting new sequence for every epoch to shuffle entire val dataset
            perm_val = np.random.permutation(self.data_loader.config["m_val"])

            for iter in range(int(num_iter_val)):
                X_val_batch, y_val_batch, y_val_cls_batch = self.data_loader.val_data_loader(iter, perm=perm_val)

                feed_dict_val = {
                    self.X_tf: X_val_batch,
                    self.y_tf: y_val_batch,
                    self.training_flag: False
                }

                minibatch_cost, minibatch_acc, val_summary = self.sess.run([self.cost, self.accuracy, self.val_summary_merged], feed_dict=feed_dict_val)

                minibatch_cost_sum_val += minibatch_cost
                minibatch_acc_sum_val += minibatch_acc

                if iter % summary_step == 0:
                    # val summary
                    self.val_writer.add_summary(val_summary, epoch * self.data_loader.config["val_batch_size"] + iter)

            # mean cost and accuracy for each epoch
            minibatch_cost_mean_val = minibatch_cost_sum_val / num_iter_val
            minibatch_acc_mean_val = minibatch_acc_sum_val / num_iter_val


            if epoch % display_step == 0:
                print('Epoch %d: Train Cost = %.2f Train accuracy = %.2f' % (epoch, minibatch_cost_mean_train, minibatch_acc_mean_train))

            if epoch % validation_step == 0:
                print('Epoch %d: Val Cost = %.2f Val accuracy = %.2f' % (epoch, minibatch_cost_mean_val, minibatch_acc_mean_val))

            if epoch % checkpoint_step == 0:
                self.saver.save(self.sess, self.checkpoints_path + "/checkpoint_" + str(epoch)+".ckpt")

        self.train_writer.close()
        self.val_writer.close()

    def test_model(self):
        np.random.seed(2)
        minibatch_cost_sum = 0
        minibatch_acc_sum = 0

        perm = np.random.permutation(self.data_loader.config["m_test"])
        num_iter = np.ceil(self.data_loader.config["m_test"] / self.data_loader.config["test_batch_size"])

        for iter in range(int(num_iter)):
            X_test_batch, y_test_batch, y_test_cls_batch = self.data_loader.test_data_loader(iter, perm=perm)

            feed_dict_test = {
                self.X_tf: X_test_batch,
                self.y_tf: y_test_batch,
                self.training_flag: False
            }

            minibatch_cost, minibatch_acc = self.sess.run([self.cost, self.accuracy], feed_dict=feed_dict_test)

            minibatch_cost_sum += minibatch_cost
            minibatch_acc_sum += minibatch_acc

        # mean cost and accuracy
        test_cost = minibatch_cost_sum / num_iter
        test_accuracy = minibatch_acc_sum / num_iter

        print('Test Cost = %.2f Test accuracy = %.2f' % (test_cost, test_accuracy))

    @abstractmethod
    def network(self):
        raise NotImplementedError('subclasses must override network()!')

    @abstractmethod
    def metrics(self):
        raise NotImplementedError('subclasses must override metrics()!')