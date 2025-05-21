## MiniTorch Overview
This repository contains the full implementation of **MiniTorch**, a simplified deep learning framework inspired by PyTorch. Built from the ground up, it includes support for tensors, automatic differentiation, neural network modules, optimized CPU/GPU backends, and training pipelines for real-world tasks like image classification and sentiment analysis. The project was developed as part of the Machine Learning Engineering course at Cornell Tech, taught by Professor Sasha Rush.

You can find the original course project at [minitorch.github.io](https://minitorch.github.io/).

<br>

## 🧩 Submodules
MiniTorch is organized into several modular sub-repositories, each focusing on a key component of the deep learning framework.:
| Module | Description |
|--------|-------------|
| [`minitorch-primitives`](https://github.com/kpan02/minitorch-primitives) | Core math operations, functional tools, and module scaffolding |
| [`minitorch-autodiff`](https://github.com/kpan02/minitorch-autodiff) | Scalar-based automatic differentiation and backpropagation |
| [`minitorch-tensors`](https://github.com/kpan02/minitorch-tensors) | Tensor data structure, broadcasting, and autograd over tensors |
| [`minitorch-fasttensor`](https://github.com/kpan02/minitorch-fasttensor) | Optimized tensor ops using Numba (CPU) and CUDA (GPU) |



<br><br><br>

# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py

---

### 4.5 - Training Logs
[sentiment.txt](sentiment.txt)

[mnist.txt](mnist.txt)
