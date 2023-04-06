import argparse, sys

import os
from datetime import datetime

from model import GPT

WEIGHTS_DIR = "weights"


def run(kwargs):
    save_weights = kwargs.pop("save_weights", False)
    from_weights = kwargs.pop("from_weights", None)

    save_weights_file = (
        create_weights_file(kwargs.get("input_file")) if save_weights else None
    )
    from_weights_file = get_weights_file(from_weights) if from_weights else None

    gpt = GPT(**kwargs)

    if not from_weights_file:
        gpt.train(save_weights_file=save_weights_file)

    gpt.generate(load_weights_file=from_weights_file)


def get_weights_file(from_weights_input):
    return os.path.join(WEIGHTS_DIR, from_weights_input + ".pth")


def create_weights_file(input_file):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    input_file_root, _ = os.path.splitext(os.path.basename(input_file))
    filename = (
        input_file_root
        + "_weights_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".pth"
    )
    filepath = os.path.join(WEIGHTS_DIR, filename)
    return filepath


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="GPT training arguments")
    arg_parser.add_argument(
        "--max_len",
        nargs="?",
        type=int,
        default=1000,
        required=False,
        help="length of output",
    )
    arg_parser.add_argument(
        "--max_iters",
        nargs="?",
        type=int,
        default=5000,
        required=False,
        help="training loop iterations",
    )
    arg_parser.add_argument(
        "--eval_iters",
        nargs="?",
        type=int,
        default=200,
        required=False,
        help="iterations per evaluation in training loop",
    )
    arg_parser.add_argument(
        "--eval_interval",
        nargs="?",
        type=int,
        default=100,
        required=False,
        help="interval at which to evaluate when training",
    )
    arg_parser.add_argument(
        "--manual_seed",
        nargs="?",
        type=int,
        default=42,
        required=False,
        help="manual seed for reproducibility",
    )
    arg_parser.add_argument(
        "--stream",
        nargs="?",
        type=bool,
        default=True,
        required=False,
        help="Whether or not to stream the generated output",
    )
    arg_parser.add_argument(
        "--input_file",
        nargs="?",
        type=str,
        default="data/nietzsche_aphorisms.txt",
        required=False,
        help="input file to train on",
    )
    arg_parser.add_argument(
        "--save_weights",
        nargs="?",
        type=bool,
        default=False,
        required=False,
        help="whether or not to save weights from training",
    )
    arg_parser.add_argument(
        "--from_weights",
        nargs="?",
        type=str,
        default=None,
        required=False,
        help="weights file to load from, skips training",
    )

    kwargs = vars(arg_parser.parse_args(sys.argv[1:]))
    run(kwargs)
