# Nonlocal-Biharmonic
This repository contains the implementation of the paper [A Variational Nonlocal Biharmonic Model with Gamma-Convergence and Numerical Implementation](https://arxiv.org/abs/2504.16395), written in Python.

## Dependencies
NumPy and SciPy. The code is tested on the following version:
```
python 3.13
numpy 2.4.4
scipy 1.17.1
```

## Files
- [biharmonic_1d.py](biharmonic_1d.py) and [biharmonic_2d.py](biharmonic_2d.py) are independent scripts for testing 1D and 2D cases.
- [utils.py](utils.py) contains the core implementation, including:
  - quadrature rule construction
  - cubic basis polynomials
  - assembly of the linear system

The implementation avoids heavy for-loops and leverages sparsity for efficiency. However, some parts may require effort to fully understand. Please refer to the detailed docstrings in each function.

## Usage
Run the following commands:
```
python biharmonic_1d.py --n 20 --func poly
python biharmonic_2d.py --n 20 --func log
```
The 1D script runs in approximately 0.4 seconds for `n=20` on personal laptop.  
The runtime increases for the 2D case, larger delta (less sparsity), or larger `n`.
