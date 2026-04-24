# CLC
Code for unpublished paper [Still thinking about title lol]. Keep updating......

## System Requirements
The listed environment was the one we ran the code. Packages in other versions may also work.
### General
* Python 3.12.10
* Scipy 1.13.1
* Numpy 1.26.4
* Pandas 2.2.2
* Matplotlib 3.10.0
* tqdm 4.66.4

### FTDT simulations
* Meep 1.30.0

### Deep learning
* PyTorch 2.5.1
* Torchvision 0.20.1

## Instructions
### ./FTDT:
Contains the code to perform FTDT simulations and visualize the result. To run the simulation:

```python3 Run_FTDT.py [polarization-flow angle in rad] [tilted angle in degree]```

An example of parallel running of different angles on a HPC cluster was provided in submit.sh and config.txt. The result can be visualized via Plot_FDTD.ipynb.

### ./detectors: 
Contains the code to collect training data and reconstruct spectrums and polarization from raw sensor reading. Code of five experiments was in the separated folders. Prior to run the code, dataset should be reached and properly placed into the corresponding folder. To prepare dataset and train the model:

```
# skip if no this film in the folder
python3 prepare_dataset.py 
python3 train.py
```

plot.ipynb in each folder can be used to evaluate and visualize the model performance.

