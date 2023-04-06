# Pico-GPT

Pico-GPT is a simple, lightweight generative pre-trained transformer (GPT) model used to generate text from a given input. 
The model is trained on a single input text file and produces text based on learned relationships between characters or words.

## Requirements

To install the required packages, you can use the provided Pipfile with pipenv:

```pipenv install```

Or you can manually install the required packages:

```pip install torch torchvision transformers datasets wandb```

Make sure you have Python 3.8.2 installed.

## Usage

To use Pico-GPT, first, train the model with your input text file using the following command:

```python run.py --input_file <your_input_text_file>```


You can customize the training process and output by modifying the arguments passed to the script:

- `--max_len`: Length of the output (default: 1000)
- `--max_iters`: Training loop iterations (default: 1000)
- `--eval_iters`: Iterations per evaluation in the training loop (default: 200)
- `--eval_interval`: Interval at which to evaluate when training (default: 100)
- `--manual_seed`: Manual seed for reproducibility (default: 42)
- `--stream`: Whether or not to stream the generated output (default: True)
- `--save_weights`: Whether or not to save weights from training (default: False)
- `--from_weights`: Weights file to load from, skips training (default: None)

For example, to train the model on a custom text file with a manual seed and save the weights:

```python run.py --input_file my_text_file.txt --manual_seed 123 --save_weights True```

To generate text using the trained model, use the following command:

```python run.py --from_weights <your_weights_file_without_extension>```

For example:

```python run.py --from_weights my_text_file_weights_20230405_125500```

