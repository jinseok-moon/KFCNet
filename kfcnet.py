from data_loader import Dataset
import network
import torch
import torch.nn as nn
import torch.optim as optim

from absl import app
from absl import flags
from absl import logging


flags.DEFINE_enum(
    'mode', 'test', ['train', 'eval', 'test'],
    help='Train model or eval model or test image')
flags.DEFINE_string('test_path', './test', help='The path for testing images')
flags.DEFINE_string('data_path', './dataset', help='The dataset directory.')
flags.DEFINE_integer('batch_size', int(32), help='The batch size.')
flags.DEFINE_integer('num_epochs', int(25), help='Number of steps of training.')
flags.DEFINE_float('learning_rate', 1e-4, help='Learning rate.')
flags.DEFINE_string('model_path', './model',
                    help='The path for saving checkpoints of the model.')
flags.DEFINE_string('output_path', './output', help='The path for saving outputs.')

FLAGS = flags.FLAGS


def main(_):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    dataset = Dataset(FLAGS.data_path, FLAGS.batch_size)
    model = network.KFCNet(dataset.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    if FLAGS.mode == 'train':
        network.train_model(model, dataset, criterion, optimizer, device, True, num_epochs=FLAGS.num_epochs)
    else:
        _, model = network.load_model(model, FLAGS.model_path)
        if FLAGS.mode == 'eval':
            network.test_model(model, dataset, device)
        else:
            network.test_case(model, FLAGS.test_path, dataset, device)


if __name__ == '__main__':
    app.run(main)

