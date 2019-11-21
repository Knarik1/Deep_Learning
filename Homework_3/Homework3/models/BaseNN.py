import tensorflow as tf
from data_loader import *
import numpy as np
from abc import abstractmethod

class BaseNN:
    def __init__(self, train_images_dir, val_images_dir, test_images_dir, num_epochs, train_batch_size,
                 val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels, 
                 num_classes, learning_rate, base_dir, max_to_keep, model_name, flatten):

        self.data_loader = DataLoader(train_images_dir, val_images_dir, test_images_dir, train_batch_size,
                                      val_batch_size, test_batch_size, height_of_image, width_of_image, num_channels, num_classes, flatten)

        self.config = {
            "num_epochs": num_epochs,
            "train_batch_size": train_batch_size,
            "val_batch_size": val_batch_size,
            "test_batch_size": test_batch_size,
            "img_h": height_of_image,
            "img_w": width_of_image,
            "img_chls": num_channels,
            "num_cls": num_classes,
            "learning_rate": learning_rate,
            "base_dir": base_dir,
            "max_to_keep": max_to_keep,
            "model_name": model_name
        }

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
        # initialize FileWriter to save summaries
        self.train_writer = tf.summary.FileWriter(self.summaries_path + '/train', self.sess.graph)
        self.val_writer = tf.summary.FileWriter(self.summaries_path + '/val', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.summaries_path + '/test', self.sess.graph)

    def create_placeholders(self):
        if self.data_loader.flatten:
            self.X_tf = tf.placeholder(tf.float32, shape=[None, self.config["img_h"] * self.config["img_w"] * self.config["img_chls"]], name='images')
        else:
            self.X_tf = tf.placeholder(tf.float32, shape=[None, self.config["img_h"], self.config["img_w"], self.config["img_chls"]], name='images')

        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.config["num_cls"]], name='labels')
        self.training_flag = tf.placeholder_with_default(False, shape=[], name='training_flag')

    def initialize_network(self):
        # initializing all the variables globally
        init = tf.global_variables_initializer()

        save_path = os.path.join(self.config["base_dir"], self.config["model_name"])
        self.checkpoints_path = os.path.join(save_path, "checkpoints")
        self.summaries_path = os.path.join(save_path, "summaries")

        # check if checkpoints' directory exists, otherwise create it.
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)

        # check if summaries' directory exists, otherwise create it.
        if not os.path.exists(self.summaries_path):
            os.makedirs(self.summaries_path)

        # get file paths
        checkpoints = glob.glob(self.checkpoints_path+'/*', recursive=True)

        # opening Session
        self.sess = tf.Session()

        # initializing Saver
        self.init_saver()

        # initializing FileWriter
        self.init_fwriter()

        # init/restore weights
        if checkpoints:
            self.saver.restore(self.sess, self.checkpoints_path + "/checkpoint_final.ckpt")
            print("----Restore params----")
        else:
            self.sess.run(init)
            print("----Init params----")

    def compute_cost(self):
        loss = self.metrics()
        self.cost = tf.reduce_mean(loss)

    def compute_accuracy(self):
        # Calculate the correct predictions
        predict_op = tf.argmax(self.y_preds_tf, axis=1)
        correct_prediction = tf.equal(predict_op, tf.argmax(self.y_tf, axis=1))

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
        num_iter = np.ceil(self.data_loader.m_train / self.config["train_batch_size"])

        for epoch in range(self.config["num_epochs"]):
            minibatch_cost_sum = 0
            minibatch_acc_sum = 0

            # getting new sequence for every epoch to shuffle dataset
            perm = np.random.permutation(self.data_loader.m_train)

            for iter in range(int(num_iter)):
                X_batch, y_batch, y_cls_batch = self.data_loader.train_data_loader(iter, perm=perm)

                feed_dict_train = {
                    self.X_tf: X_batch,
                    self.y_tf: y_batch,
                    self.training_flag: True
                }

                _, minibatch_cost, minibatch_acc = self.sess.run([self.optimizer, self.cost, self.accuracy], feed_dict=feed_dict_train)

                minibatch_cost_sum += minibatch_cost
                minibatch_acc_sum += minibatch_acc

            # mean cost and accuracy for each epoch
            minibatch_cost_mean = minibatch_cost_sum/num_iter
            minibatch_acc_mean = minibatch_acc_sum/num_iter

            if epoch % display_step == 0:
                print('Epoch %d: Train Cost = %.2f Train accuracy = %.2f' % (epoch, minibatch_cost_mean, minibatch_acc_mean))

            if epoch % validation_step == 0 or epoch % summary_step == 0:
                # getting new sequence to shuffle val batch
                perm_val = np.random.permutation(self.config["val_batch_size"])

                X_val_batch, y_val_batch, y_val_cls_batch = self.data_loader.val_data_loader(0, perm=perm_val)

                feed_dict_val = {
                    self.X_tf: X_val_batch,
                    self.y_tf: y_val_batch,
                    self.training_flag: False
                }

                with tf.name_scope('summaries_val'):
                    tf.summary.scalar("val_accuracy", self.accuracy)
                    tf.summary.scalar("val_cost", self.cost)
                    val_summary_merged = tf.summary.merge_all()

                val_cost, val_acc, val_summary = self.sess.run([self.cost, self.accuracy, val_summary_merged], feed_dict=feed_dict_val)

                # display val cost and accuracy
                if epoch % validation_step == 0:
                    print('Epoch %d: Val Cost = %.2f Val accuracy = %.2f' % (epoch, val_cost, val_acc))

                # write summary
                if epoch % summary_step == 0:
                    # val summary
                    self.val_writer.add_summary(val_summary, epoch)

                    # train summary
                    with tf.name_scope('summaries_train'):
                        train_summary = tf.Summary()
                        train_summary.value.add(tag="train_accuracy", simple_value=minibatch_acc_mean)
                        train_summary.value.add(tag="train_cost", simple_value=minibatch_cost_mean)

                    self.train_writer.add_summary(train_summary, epoch)

            if epoch % checkpoint_step == 0:
                self.saver.save(self.sess, self.checkpoints_path + "/checkpoint_" + str(epoch)+".ckpt")

        self.saver.save(self.sess, self.checkpoints_path + "/checkpoint_final.ckpt")
        self.train_writer.close()
        self.val_writer.close()

    def test_model(self):
        X_test_batch, y_test_batch, y_test_cls_batch = self.data_loader.test_data_loader(0)

        feed_dict_test = {
            self.X_tf: X_test_batch,
            self.y_tf: y_test_batch,
            self.training_flag: False
        }

        with tf.name_scope('summaries_test'):
            tf.summary.scalar("test_accuracy", self.accuracy)
            tf.summary.scalar("test_cost", self.cost)
            test_summary_merged = tf.summary.merge_all()

        test_cost, test_accuracy, test_summary = self.sess.run([self.cost, self.accuracy, test_summary_merged], feed_dict=feed_dict_test)

        print('Test Cost = %.2f Test accuracy = %.2f' % (test_cost, test_accuracy))

        self.val_writer.add_summary(test_summary)
        self.test_writer.close()

    @abstractmethod
    def network(self):
        raise NotImplementedError('subclasses must override network()!')

    @abstractmethod
    def metrics(self):
        raise NotImplementedError('subclasses must override metrics()!')
