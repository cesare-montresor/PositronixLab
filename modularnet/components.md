# ModularNet Components

## CORE

### ModularSpace

ModularSpace is the core component of this project. It's an holografic version on the neural network, where each computation unit (CU, aka, neuron) now is projected into an high dimensional space. Via contrastive learning the CUs are organized in space in computational groups (CGs, aka, functional modules).
This process happens during training, and it's independent of the architecture of the model ( currently supports FC, CNN ).
ModularSpace attaches to a model, scans for compatible layers and register a forward_hook. During the forward pass, the hook pass the activations of the CUs to the ModularSpace. 
The computational graph of ModularSpace is deteched from the main model.
Ideally the hook should be as light as possible, to avoid hindering the training process. 
At the moment, a separate thread on the main training process is used to transform activations into "scores" and then queued.
On a separate process ModularSpace do all the heavy lifting, computing the loss to organize the CUs in space.

#### ModularSpacePluginInterface

ModularSpace attach to a model

__Inputs__
- activations:
    - mandatory
    - from: model forward hook
    - shape: [batch_size, num_cu]
    - dtype: float

- forward mask:
    - optional (default: 1)
    - from: function set_mask()
    - shape: [batch_size, num_cu]
    - dtype: bool

- module labels:
    - optional (default: -1)
    - from: function set_labels()
    - shape: [num_cu]
    - dtype: int

- 



__Outputs__
- loss components:
    - total
    - layer
    - center asymmetry
    - alignment
    - batch

- scores
- idx_on
- idx_off
- weights_on
- weights_off
- center_on
- center_off


#### ModularSpacePlugin

It's an extension of 



## EXPERIMENTS

### ax.dev 
- Run NAS experiments under different conditions.
- For a MetaModelTemplate, it will first run N trials with "no variants" to initialize the BO prior. it's called"common".
- Then it will run M trials for each variant and optimize the search space parameters.
- The common phase can run as many trials in parallel as possible. They are all random picked.
- Each individual variant, instead, has to be run sequentially, as it's using Byesian HyperOPT.
- Multiple variants can be run in parallel.


### MetaModelTemplate 
- Define a NAS template for a given Experiment   
- It has fixed parameters, like archiecture and dataset
- It has a search space of parameters to be optimized
- Currently it has a flat structure, using the name prefix to define the groups
- It can be at least divided in 2 main groups
    - Fixed parameters
    - Search space parameters
- Experiments supports also "variants". Each variant fix some otherwise searchable parameters.
- Currently variants are simply named A, B, C, D. 


### MetaModelConfig 
- Define a specific configuration of a MetaModelTemplate, it's one trial of the experiment.
- It's a flat dictionary of parameters 


### MetaModel
- MetaModel, ideally, should be able to produce any sort of model (FC, CNN, RNN, Transformer, etc )
- It's a PyTorch model that is built using the MetaModelConfig.
- Ideally the MetaModel can eneable or disable various settings, dimensions and features.
