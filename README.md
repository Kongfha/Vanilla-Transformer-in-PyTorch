# Vanilla Transformer in PyTorch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
- [License](#license)

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

Make sure to customize the paths and configurations in the param.py file to match your setup.

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

Evaluate the model using the testing script test_example.py. Modify the input examples to observe the model's performance.


## References
