# DARTS-CIFAR10-NAS

This project is an implementation and experimental study of **Differentiable Architecture Search (DARTS)** on the CIFAR-10 dataset using PyTorch.

The main goal of this project is to understand how DARTS can automatically search for a good neural network architecture instead of manually designing the architecture. The experiment was carried out using Google Colab with an NVIDIA Tesla T4 GPU.

## About the Project

DARTS treats the architecture search problem as a continuous optimization problem. Instead of choosing one operation directly, different candidate operations are assigned learnable architecture parameters. During training, these parameters are updated along with the network weights, and the learned architecture can later be converted into a discrete cell structure.

In this experiment, the search space contains 8 candidate operations and each cell contains 14 possible edges.

### Candidate Operations

- `none`
- `max_pool_3x3`
- `avg_pool_3x3`
- `skip_connect`
- `sep_conv_3x3`
- `sep_conv_5x5`
- `dil_conv_3x3`
- `dil_conv_5x5`

## Experiment Setup

- Dataset: CIFAR-10
- Framework: PyTorch
- Search method: First-order DARTS
- GPU: NVIDIA Tesla T4
- Model parameters: approximately 1.9M
- Initial channels: 16
- Number of layers: 8
- Batch size: 64
- Weight learning rate: 0.025
- Architecture learning rate: 0.0003
- Weight decay: 0.0003
- Drop path probability: 0.3

## Results

The experiment was planned for 50 epochs. However, the Google Colab runtime stopped during the experiment because of runtime/session limitations.

The search successfully progressed through Epoch 6. At this stage, the model achieved approximately:

- Training accuracy: **78.5%**
- Validation accuracy: **~75%**

The architecture probabilities also changed from their initial nearly uniform distribution during the search. Normal and reduction cell genotypes were extracted as the architecture search progressed.

These results represent the intermediate progress of the search and are not the final results of a completed 50-epoch experiment.

## Project Structure

```text
darts-cifar10-nas/
│
├── README.md
├── architect.py
├── model_search.py
├── operations.py
├── genotypes.py
├── train_search.py
│
└── results/
    └── training_log.txt
Files

architect.py
Contains the architecture optimization procedure used to update the architecture parameters.

model_search.py
Contains the DARTS search network, cells, and mixed operations used during the architecture search.

operations.py
Contains the candidate operations used in the architecture search space.

genotypes.py
Contains the operation definitions and genotype structure used to represent the searched architecture.

train_search.py
Main training script used to train the network and perform the architecture search.

results/
Contains experiment logs and other results.

Running the Project

The project can be run using Google Colab or another environment with a CUDA-enabled GPU.

Install the required packages if they are not already available:

pip install torch torchvision numpy

Then run:

python train_search.py

For Google Colab, a GPU runtime is recommended because neural architecture search is computationally expensive.

Acknowledgements

This project is based on the original DARTS (Differentiable Architecture Search) work. The existing implementation was used as a starting point and was adapted and modified for this experiment and for running in a modern PyTorch and Google Colab environment.

Original DARTS repository:

https://github.com/quark0/darts

Reference

Liu, H., Simonyan, K., & Yang, Y. (2019).

DARTS: Differentiable Architecture Search.

International Conference on Learning Representations (ICLR).

Note

This repository is mainly for learning and research purposes. The current results show the progress of the architecture search up to Epoch 6. A complete 50-epoch search was not finished because of the available Google Colab runtime limitations.
