import tensorflow as tf
from models.DNN import *
from models.CNN import *
from models.VGG import *
from models.RNN import *

# Datasets
tf.app.flags.DEFINE_string('train_images_dir', '', 'Training images data directory.')
tf.app.flags.DEFINE_string('val_images_dir', '', 'Validation images data directory.')
tf.app.flags.DEFINE_string('test_images_dir', '', 'Testing images data directory.')

tf.app.flags.DEFINE_boolean('train', True, 'whether to train the network')
tf.app.flags.DEFINE_integer('num_epochs', 10000, 'epochs to train')
tf.app.flags.DEFINE_integer('train_batch_size', 100, 'number of elements in a training batch')
tf.app.flags.DEFINE_integer('val_batch_size', 100, 'number of elements in a validation batch')
tf.app.flags.DEFINE_integer('test_batch_size', 100, 'number of elements in a testing batch')

tf.app.flags.DEFINE_integer('height_of_image', 28, 'Height of the images.')
tf.app.flags.DEFINE_float('width_of_image', 28, 'Width of the images.')
tf.app.flags.DEFINE_float('num_channels', 1, 'Number of the channels of the images.')
tf.app.flags.DEFINE_float('num_classes', 10, 'Number of classes.')

tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate of the optimizer')

tf.app.flags.DEFINE_integer('display_step', 20, 'Number of steps we cycle through before displaying detailed progress.')
tf.app.flags.DEFINE_integer('validation_step', 60, 'Number of steps we cycle through before validating the model.')

tf.app.flags.DEFINE_string('base_dir', './results', 'Directory in which results will be stored.')
tf.app.flags.DEFINE_integer('checkpoint_step', 300, 'Number of steps we cycle through before saving checkpoint.')
tf.app.flags.DEFINE_integer('max_to_keep', 2, 'Number of checkpoint files to keep.')

tf.app.flags.DEFINE_integer('summary_step', 20, 'Number of steps we cycle through before saving summary.')

tf.app.flags.DEFINE_string('model_name', 'softmax_classifier', 'name of model')

tf.app.flags.DEFINE_string('model', 'CNN', 'DNN, CNN or RNN')

FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    model = CNN(
        train_images_dir='data/train/',
        val_images_dir='data/val/',
        test_images_dir='data/test/',
        num_epochs=40,
        train_batch_size=1000,
        val_batch_size=1000,
        test_batch_size=10000,
        height_of_image=28,
        width_of_image=28,
        num_channels=1,
        num_classes=10,
        learning_rate=0.001,
        base_dir='results',
        max_to_keep=2,
        model_name="CNN",
        model='CNN'
    )

    model.create_network()
    model.initialize_network()

    if True:
        model.train_model(1, 1, 1, 4)
    else:
        model.test_model()


if __name__ == "__main__":
    tf.app.run()