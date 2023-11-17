# Vanilla Transformer in PyTorch

## Overview

Implementation of a self-made Encoder-Decoder Transformer in PyTorch (Multi-Head Attention is implemented too), inspired by "Attention is All You Need." Primarily designed for Neural Machine Translation (NMT), specifically for Chinese to Thai translation.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Training](#training)
- [Testing](#testing)
- [References](#References)

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

Before using the model, make sure to download the necessary datasets and preprocess them accordingly. You can find an example of how to structure your data in the data.py file.

To train the model, execute:

```bash
python train.py
```

For testing, you can use the example script:

```bash
python test_example.py
```

Make sure to customize the paths and configurations in the param.py and data.py file to match your setup as these codes were used for training Chinese to Thai translation model.

## File Structure
The project structure is organized as follows:

* models/: Contains the core components of the Transformer.
  * layers/: Various layers used in the model.
  * model/: Implementation of the Encoder, Decoder, and the complete Transformer architecture.
* train.py: Script for training the model.
* test_example.py: Script for testing the trained model on an example.

## Training

Adjust the hyperparameters and configurations in the param.py file before starting the training process. The trained model will be saved in the models/ directory.

## Testing

Test the model using the testing script test_example.py. Modify the input examples to observe the model's outputs.


## References

1. **Attention is All You Need Paper**
   - Authors: Ashish Vaswani et al.
   - Paper Link: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

2. **Transformer Implementation by Kevin Ko**
   - GitHub Repository: [github.com/hyunwoongko/transformer](https://github.com/hyunwoongko/transformer)

3. **The Illustrated Transformer by Jay Alammar**
   - Article Link: [jalammar.github.io/illustrated-transformer](http://jalammar.github.io/illustrated-transformer/)

4. **TensorFlow Transformer Tutorial**
   - TensorFlow Text Tutorials: [TensorFlow Transformer Tutorial](https://www.tensorflow.org/text/tutorials/transformer)



