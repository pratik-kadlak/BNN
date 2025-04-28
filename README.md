# BBN (Binarized Neural Networks)

This repository implements **Binarized Versions** of popular deep learning models including **AlexNet**, **VGG**, and **ResNet**.  
Binarized Neural Networks (BNNs) are designed to reduce memory usage and accelerate inference by constraining both the weights and activations to binary values (+1 or -1), making them highly efficient for deployment on edge devices and resource-constrained environments.

---

## Papers Referenced

- [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830)  
  *Authors: Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, Yoshua Bengio*

---

## Models Implemented

| Model  | Details                                  |
|--------|------------------------------------------|
| AlexNet | Binarized Weights & Activations |
| VGG     | Binarized Weights & Activations |
| ResNet  | Binarized Weights & Activations |

Each model is a binarized version of the standard architecture, carefully modified to maintain as much accuracy as possible while significantly reducing computational complexity.

---

## Features

- Full binarization of both weights and activations.
- Custom binarized convolution and linear layers.
- Training and evaluation scripts for binarized models.
- Easily extensible to other architectures.

---

## Repository Structure

```
BBN/
├── models/
│   ├── __init__.py
│   ├── alexnet_binary.py
│   ├── alexnet.py
│   ├── resnet_binary.py
│   ├── resnet.py
│   ├── vgg_cifar10_binary.py
│   ├── vgg_cifar10.py
│   ├── binarized_modules.py # custom binarized layers
├── data.py
├── main_binary_hinge.py
├── main_binary.py
├── main_mnist.py
├── preprocess.py
├── utils.py
└── README.md
```

---

## How to Use

**Training On Cifar10:**
```bash
python3 main_binary.py --model resnet_binary --dataset cifar10 --epochs 100
```
```bash
python3 main_binary.py --model alexnet_binary --dataset cifar10 --epochs 100
```
```bash
python3 main_binary.py --model vgg_cifar10_binary --dataset cifar10 --epochs 100
```

**Training On Cifar100:**
```bash
python3 main_binary.py --model resnet_binary --dataset cifar100 --epochs 100
```
```bash
python3 main_binary.py --model alexnet_binary --dataset cifar100 --epochs 100
```
```bash
python3 main_binary.py --model vgg_cifar100_binary --dataset cifar100 --epochs 100
```


All hyperparameters (learning rate, batch size, optimizer, etc.) can be adjusted via command-line arguments.

---

## Command Line Arguments

| Argument | Type | Default | Description |
|:---------|:-----|:--------|:------------|
| `--results_dir` | str | `./results` | Directory to save results. |
| `--save` | str | `''` | Name of folder to save models/checkpoints. |
| `--dataset` | str | `imagenet` | Name or path of dataset. |
| `--model`, `-a` | str | `alexnet` | Model architecture. Choices: alexnet, vgg, resnet. |
| `--input_size` | int | `None` | Input image size. |
| `--model_config` | str | `''` | Additional model configuration. |
| `--type` | str | `torch.cuda.FloatTensor` | Tensor type (e.g., `torch.cuda.HalfTensor`). |
| `--gpus` | str | `'0'` | GPUs to be used (e.g., `0,1,2`). |
| `-j`, `--workers` | int | `8` | Number of data loader workers. |
| `--epochs` | int | `2500` | Number of total epochs to train. |
| `--start-epoch` | int | `0` | Manual epoch number (for restarts). |
| `-b`, `--batch-size` | int | `256` | Mini-batch size. |
| `--optimizer` | str | `SGD` | Optimizer to use (e.g., `SGD`, `Adam`). |
| `--lr`, `--learning_rate` | float | `0.1` | Initial learning rate. |
| `--momentum` | float | `0.9` | Momentum for SGD optimizer. |
| `--weight-decay`, `--wd` | float | `1e-4` | Weight decay (L2 regularization). |
| `-p`, `--print-freq` | int | `10` | Frequency of printing training status. |
| `--resume` | str | `''` | Path to checkpoint to resume training from. |
| `-e`, `--evaluate` | str | `None` | Evaluate model from given checkpoint on validation set. |

## 🔥 Results

| Model           | Dataset | Accuracy (Binarized) | Notes |
|-----------------|---------|----------------------|-------|
| Binarized AlexNet | CIFAR-10 | ~xx%                  | baseline |
| Binarized VGG     | CIFAR-10 | ~xx%                  | deeper network |
| Binarized ResNet  | CIFAR-10 | ~xx%                  | with skip connections |

(*Note: Replace `xx%` with your experimental results.*)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- Inspiration from the original BNN and XNOR-Net papers.
- Based on standard PyTorch implementations of AlexNet, VGG, and ResNet.

---

Would you also like me to prepare a small **`requirements.txt`** and **sample train.py** snippet if you want to make the repo instantly runnable? 🚀  
Would look super clean!






Perfect — thanks for sharing the actual list of supported arguments.

Here’s the **full README** updated properly, including a **nice table of the command-line arguments** based on what you gave:

---
