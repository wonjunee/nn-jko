# Neural network implementation for solving JKO scheme with various internal energy functional.

**Note:** This repository is based on the following repository: [https://github.com/EmoryMLIP/OT-Flow](https://github.com/EmoryMLIP/OT-Flow).

The code is solving the following variational formulation

$$
    \rho^{n+1} = \text{arg}\min_{\rho} \mathcale{E}(\rho) + \frac{1}{2} W^2_2(\rho, \rho^n)
$$

where this formulation is a discrete-in-time variational formulation of the following PDE:

$$
    \partial_t \rho - \nabla \cdot (\rho \nabla \delta \mathcal{E}(\rho)) = 0.
$$

Currently, the choice of the internal energy functional in the code `train.py` is

$$
    \mathcal{E}(\rho) = \int \rho(x) \log \left( \frac{\rho(x)}{q(x)} \right) \, dx
$$

where $q$ is a reference probability distribution.