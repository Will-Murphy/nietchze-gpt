# Pico-GPT

Pico-GPT is a simple, lightweight generative pre-trained transformer (GPT) model used to generate text. It is trained on a single text file and produces text based on learned relationships between characters or words. It is based on https://github.com/karpathy/nanoGPT and designed for use on a laptop.

## Requirements

To install the required packages, you can use the provided Pipfile with pipenv:

```pipenv install```

Or you can manually install the required packages:

```pip install torch torchvision transformers datasets wandb```

Make sure you have Python 3.8.2 installed. If you used pipenv shell ensure you run 
the following before following the below usage directions:

```pipenv shell```

## Usage

To use Pico-GPT, train and generate from the model with the run command:

```python run.py```

You can customize the training process and output by modifying the arguments passed to the script:
- `--input_file`: path to input file (default: data/nietzsche_aphorisms.txt)
- `--max_len`: Length of the output (default: 1000)
- `--max_iters`: Training loop iterations (default: 1000)
- `--eval_iters`: Iterations per evaluation in the training loop (default: 200)
- `--eval_interval`: Interval at which to evaluate when training (default: 100)
- `--manual_seed`: Manual seed for reproducibility (default: 42)
- `--stream`: Whether or not to stream the generated output (default: False)
- `--save_weights`: Whether or not to save weights from training (default: False)
- `--from_weights`: Weights file to load from, skips training (default: None)

For example, to train the model for 10,000 iterations on a custom text file named nietzsche_aphorisms.txt in the 
data directory, and save the weights (so you don't have to run intensive training each time):

```python run.py --input_file data/nietzsche_aphorisms.txt --max_len 1000 --max_iters 10000 --save_weights True```

To generate text using a previously trained model, use the following command:

```python run.py --from_weights <your_weights_file_without_extension>```

For example, to generate random text from our first training example, but with a new random seed we do:

``` python run.py --from_weights nietzsche_aphorisms_weights_20230405_231919  --manual_seed 1997```

## Performance

The example training and generation outlined in usage above took about 60 minutes on a 2015 macbook pro.
The following is a snippet of its ouput:

```
864 The world. Sential These psychological races of her knowledge, from step, though, that
of humaning. The are live to ention divine or metaphod was all the logical locks, the simple, 
the suffering, and whatever, as a sive for example in its Herding to life! _Deed is in consciousness,
we about and asquieted as the only that how much, into arguants harm on halgs, beforent raelia-semilation 
of selfishness, it is by them avoiding its meantical man is now purposed human so that music in
imposime something in its as at a name, confiling of things flictive of which in those 
only there--against of life.
```

When compare to an example from the training set, we can see it mirrors the style and structure of the input fairly well despite 
being nonsensical and gramatically incorrect:

```
112. At the Contemplation of Certain Ancient Sacrificial Proceedings. --How
many sentiments are lost to us is manifest in the union of the farcical,
even of the obscene, with the religious feeling. The feeling that this
mixture is possible is becoming extinct. We realize the mixture only
historically, in the mysteries of Demeter and Dionysos and in the
Christian Easter festivals and religious mysteries. But we still
perceive the sublime in connection with the ridiculous, and the like,
the emotional with the absurd. Perhaps a later age will be unable to
understand even these combinations.
```

If we were to increase training iterations (i.e --max_iters 10000) & scale up the model hyperparameters in model.py,
we could achieve much better performance in matching the input text.

