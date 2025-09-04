# Space and Time adaptivity

#### Numerical Methods for Partial Differential Equations – Project 2024/25 - Politecnico di Milano

This project explores **adaptive methods** in both **space and time** for solving partial differential equations. We focus on the heat equation with a localized, time-dependent forcing term characterized by sharp, frequent impulses. The goal is to capture steep solution features accurately while minimizing computational cost through dynamic mesh and time step refinement based on error estimates. We implemented an adaptive solver using the ```deal.II``` library and evaluated its effectiveness in terms of accuracy and efficiency.

# TODO add colab notebook and final results.csv file inside results folder

# TODO add cool image
![cool_image]()


**Students**
* [Leonardo Arnaboldi](https://github.com/leo-arnaboldi)
* [Luca Donato](https://github.com/lucacris72)
* [Tommaso Crippa](https://github.com/crippius)


**Professor**: *Alfio Maria Quarteroni*


**Assistant Professor**: *Michele Bucelli*

## Strong formulation
$$
\begin{cases} 
\begin{aligned}
\frac{\partial u}{\partial t} - \nabla \cdot (\mu \nabla u) &= f && \text{in } \Omega \times (0,T), \\
\mu \nabla u \cdot \mathbf{n} &= 0 && \text{on } \partial \Omega \times (0,T), \\
u &= 0 && \text{in } \Omega \times \{0\},
\end{aligned}
\end{cases} 
$$

$$
f(\mathbf{x},t) = g(t)h(\mathbf{x}), \quad
g(t) = \frac{\exp(-a \cos(2N \pi t))}{\exp(a)}, \quad
h(\mathbf{x}) = \exp\left(-\left(\frac{|\mathbf{x} - \mathbf{x}_0|}{\sigma}\right)^2\right),
$$

$$
\text{where } a > 0,\; N \in \mathbb{N},\; x_0 \in \Omega,\; \text{and } \sigma > 0.
$$


## Prerequisites

* **deal.II** ≥ 9.0.
* **C++** compiler (C++11 or later).
* **CMake** ≥ 3.12.

To utilize the automatic tester or visualize the solver's statistic on the notebook inside the results folder, you can set up the conda environment named `pdeEnv` to properly run the python scripts in two simple steps:

---

1. **Create the environment:**
```bash
conda env create -f pdeEnv.yml
```

2. **Activate the environment:**
```bash
conda activate pdeEnv
```

## Compiling
To build the executable, make sure you have loaded the needed modules with
```bash
$ module load gcc-glibc dealii
```
Then run the following commands:
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```

## Execution

The program can be executed from the `build` folder through

```bash
$ ./main path/to/parameter_file.prm
```
where ```parameter_file.prm``` is a file that contains all of the hyperparameters utilized by the solver like:
- Space/Time Adaptivity flags
- Discretization Details
- More advanced parameters for space and time adaptivity

The template parameter file at the root directory ```parameters_base.prm``` can be used for the execution by writing:

```bash
$ ./main ../../parameters_base.prm
```

To run the automatic tester that executes different configurations of the solver type:
```bash
$ python main_pipeline.py 
```
The program will save each simulation ```.vtu``` files inside a dedicated folder in the ```test_runs``` directory, together with the output log and their own parameter file. The summary of all the runs will be found inside ```results.csv```. 


## Project Structure

# TODO check if correct

```text
root/                                      
├── src/
│   ├── Heat.hpp   
│   ├── Heat.cpp    
│   └── main.cpp               
├── results/
│   ├── plot_results.ipynb        
│   └── results.csv
├── CMakeLists.txt 
└── report.pdf          
```
