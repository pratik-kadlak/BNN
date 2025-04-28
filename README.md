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

## 🛠️ Installation

```bash
git clone https://github.com/your_username/BBN.git
cd BBN
pip install -r requirements.txt
```

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

## 🧠 How to Use

**Training Example:**
```bash
python train.py --model binarized_resnet --dataset cifar10 --epochs 100
```

**Testing Example:**
```bash
python test.py --model binarized_vgg --dataset cifar10
```

All hyperparameters (learning rate, batch size, optimizer, etc.) can be adjusted via command-line arguments.

---

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
