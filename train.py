import argparse, sys

from model import GPT


def run(kwargs):
    gpt = GPT(**kwargs)
    gpt.train()
    gpt.generate()


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
        help="evaluation iterations inside training loop",
    )
    arg_parser.add_argument(
        "--eval_interval",
        nargs="?",
        type=int,
        default=500,
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
        "--input_file",
        nargs="?",
        type=str,
        default="data/nietzsche_aphorisms.txt",
        required=False,
        help="input file to train on",
    )

    kwargs = vars(arg_parser.parse_args(sys.argv[1:]))
    run(kwargs)
