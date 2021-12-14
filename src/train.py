from argparse import ArgumentParser

import tensorflow.keras as keras

from model.constants import WEIGHTS_FILE_NAME
from model.model import build_model
from model.train import train_model

if __name__ == "__main__":
    argument_parser = ArgumentParser("Model training suite")

    argument_parser.add_argument(
        "-l",
        "--load-weights",
        action="store_true",
        help="Load weights from storage",
    )

    argument_parser.add_argument(
        "-f",
        "--weights-file",
        default=WEIGHTS_FILE_NAME,
        type=str,
        help="What file to load weights from",
    )

    argument_parser.add_argument(
        "-s",
        "--store-weights",
        action="store_true",
        help="Store weights to storage",
    )

    argument_parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=1000,
        help="Number of epochs used for training",
    )

    argument_parser.add_argument(
        "-se",
        "--start-epoch",
        type=int,
        default=0,
        help="Which epoch to start counting from",
    )

    argument_parser.add_argument(
        "-g",
        "--show-graphics",
        action="store_true",
        help="Show graphics",
    )

    argument_parser.add_argument(
        "-p",
        "--print-model",
        action="store_true",
        help="Print model",
    )

    args = argument_parser.parse_args()

    model = build_model()

    if args.print_model:
        print("Printing model...")
        keras.utils.plot_model(model, show_shapes=True)

    if args.load_weights:
        print("Loading weights...")
        model.load_weights(args.weights_file)

    train_model(
        model, args.epochs, args.store_weights, args.show_graphics, args.start_epoch
    )
