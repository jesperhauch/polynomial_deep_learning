# Simulation Metamodeling with Neural Networks
This repository is dedicated to the work done in connection with my master thesis at Technical University of Denmark ([arXiv submission]( https://arxiv.org/abs/2307.10892)). The code is written in [Lightning](https://lightning.ai/docs/pytorch/latest/) and can be made compatible to run on GPU but this is not implemented currently.

The project concerns the use of neural networks in simulation metamodeling. More specifically, the use of custom neural networks that can fit higher-order polynomials and generalize to out-of-distribution inputs. Neural network architectures are Polynomial Activation Neural Network (PANN), Coupled decomposition (CCP) from the [Pi-Nets paper](https://arxiv.org/abs/2006.13026) [(Github)](https://github.com/grigorisg9gr/polynomial_nets), PDC from [Augmenting Deep Classifiers with Polynomial Neural Networks](https://arxiv.org/pdf/2104.07916.pdf) and PDCLow proposed based on PDC.

## Installing
To install this project and run it on your own machine, please do the following:
1. Install dependencies - Use one of the two below
    - *environment.yaml* (requires conda) - `conda env create -f environment.yml`
    - *requirements.txt* - `pip install -r requirements.txt` (please use Python 3.10)
2. Install `setup.py` to ensure proper imports.
    - `pip install -e .`

## Folder Structure
The folder structure for this project does not follow a specific convention but should still be intuitive. A overview of the folders is given below
```
├── .vscode
├── data
├── models
├── scripts
├── run_files
├── training
├── utils
├── .gitignore
├── environment.yml
├── README.md
├── requirements.txt
├── result_calculation.ipynb
└── setup.py
```

The folder names are pretty much self-explanatory but a `README.md` file is found inside each folder that explains further about its contents.

## Example runs
To execute runs, the files inside the [training/](https://github.com/jesperhauch/polynomial_deep_learning/tree/master/training) folder must be run from the root folder of the repository. A distinction is made between types of runs possible in this project:
1. Arbitrary polynomials defined by `lambda` functions
2. Constrained optimization functions 
3. SIR simulation

The first two types are both executed through the [`training_polynomials.py`](https://github.com/jesperhauch/polynomial_deep_learning/blob/master/training/training_polynomials.py) file, where a runtime argument for `--model` must always be specified. For the constrained optimization functions, one of the classes inside [simulation_functions.py](https://github.com/jesperhauch/polynomial_deep_learning/blob/master/data/simulation_functions.py) must be specified for the `--data_generator` argument. See example runs for each of the first two types below:
```
python training/training_polynomials.py --model CCP --polynomial "lambda x: x**2" --polynomial_name x2
python training/training_polynomials.py --model CCP --data_generator Currin
```

The third run type is executed through the [training_epidemiology.py] file, where the model is specified in the `--multiplication_net` argument. 
```
python training/training_epidemiology.py --multiplication_net CCP
```

To view which arguments are possible to specify, please refer to the corresponding training files.

## Logging
Logging is done with Tensorboard through Lightning. Lightning automatically creates a `tb_logs` folder the Tensorboard log files. To view finished and running runs, use the following command:
```
tensorboard --logdir=tb_logs/
```

The Jupyter Notebook [result_calculation.ipynb](https://github.com/jesperhauch/polynomial_deep_learning/blob/master/result_calculation.ipynb) also shows how to extract the values from the `.tfevents` files generated by the Tensorboard logger in Lightning. However, you need to have log files inside a tb_logs folder for the notebook to work properly.

## Extending this work
If you wish to extend this work by creating your own custom models, ensure that [BaseModel](https://github.com/jesperhauch/polynomial_deep_learning/blob/fe248839d22413f0ee97496e8bc7b576346bc398/models/base_model.py#L9) is used as the parent class. 
