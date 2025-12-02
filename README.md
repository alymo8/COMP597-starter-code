# COMP597-starter-code
This repository contains starter code for COMP597: Responsible AI - Energy Efficiency analysis using CodeCarbon. 
TODO: add more description on for course description, project description and instructions on the project.
### Course Description
TODO: course description.

### Project Description
TODO: project description.

### Instructions
TODO: instructions for the project. eg:
1. Set up your environment using the provided instructions below under [Environment Setup](#environment-setup).
2. Familiarize yourself with the CodeCarbon library and its usage. Resources can be found in the [CodeCarbon Resources](#codecarbon-resources) section.
3. Implement your language/vision/other model and run experiements to collect data.
4. Document your process and findings in a report.

---

### Repository Structure
```
COMP597-starter-code
.
├── energy_efficiency
│   ├── src
│   │   ├── config                          # Configuration related files                     
│   │   ├── models
│   │   │   ├── gpt2
│   │   │   └── ...                 
│   │   ├── trainer                         
│   │   │   ├── stats                       # Stats collection for trainers
│   │   │   │  ├── base.py 
│   │   │   │  └── ...
│   │   │   ├── base.py                     # Trainer base class
│   │   └── ...   
│   ├── launch.py                           # Main script to launch training experiments                           
│   ├── requirements.txt                        
│   └── start-gpt2.sh
├── .gitignore
├── env_setup.sh                            # Script to setup the conda environment                               
└── README.md
``` 

##### On a high level, the execution flow follows this structure:
1. **Entry point**: `launch.py`: file acts as the main entry point for training. It parses command-line arguments (into a configuration object conf), prepares the dataset, and then initialises the trainer object that will be used to train the model via process_conf(): `model_trainer, model_kwards = process_conf(conf, dataset)`. Internally, `process_conf()` calls the set-up funciton o fth emodel specified in the configuration object. For example, if `conf.model == "gpt2"`, it calls `gpt2.init_from_conf(conf, dataset)`. The `launch.py` file then receives back a Trianer instance and any model-specific keyword arguments. Finally, it calls the `train()` method of the Trainer instance to start training.
2. **Model setup**: `src/models/<model_name>/<model_name>.py`: Each model has its own directory under `src/models/`. The setup function `<model_name>_init(...)` is defined for each model under `src/models/`. For example, ``gpt2_init(...)` is defined under `src/models/gpt2/gpt2.py`. This `init()` function starts by handling all the ocmmon initialization steps for the model, such as loading the tokenizer, model configuration, and data collator. It then calls a trainer-specific setup function (e.g., `simple_trainer(...)`) based on the CLI arguments passed to initialize the trainer object with the model, dataset, optimizer, and other components.
3. **Trainer construction**: Each of these setup `init()` functions conclude by returning an instance of a Trainer class (defined under `src/trainer/`) that encapsulates the training logic. The Trainer class is responsible for managing the training loop, including forward and backward passes, optimization steps, and stats collection. In other words, the functinos inside the model classes set up the model with all the parameters needed for training (the tokens, the model, the backend connections with hardware, etc), then use these parameters to create a new Trainer object of matching type. It is this Trainer object which is then called on to actually begin the training. 
4. **Trainer execution**: Back in `launch.py`, the returned Trainer object is used to start the training process by calling its `train()` method. This method encapsulates the full training loop: iterating over batches, running forward and backward passes, stepping the optimizer, and calling the appropriate statistics hooks (e.g. timers, profilers, or energy tracking via CodeCarbon).

##### A closer look at the Trainer class:
- The *Trainer* class is defined under `src/trainer/base.py` as an abstract base class. It defines the basic structure and methods that all trainer implementations must follow. It establishes the general structure of the training loop. It contains two core concrete methods:
    - `train(self, kwargs)`: iterates over every batch in the dataset. For every batch, triggers the training step logic (forward pass, backward pass, and optimiser step) by calling self.step().
    - `step(iteration_num, batch, model_kwargs)`: given a batch of data, performs one iteration step. This is a template method, and it delegates the actual forward pass, backward pass and optimiser step to abstract classes forward(), backward(), and optimizer_step(). The step() method combines the steps that are common to all trainers, such as loading the batch or calling the stats trackers.
    - The abstract classes forward(), backward(), and optimizer_step() are left to be implemented by each concrete subclass of Trainer. This separation allows the train() and step() methods to stay identical for all trainer types, while each concrete subclass defines its own way of running a batch through the system via its custom implementation of the the forward pass, backward pass and optimiser step. 
    - The Trainer subclasses currently implemented are:
        - `SimpleTrainer` (defined under `src/trainer/simple_trainer.py`): a basic trainer used for single-GPU that implements the abstract methods of Trainer with straightforward logic for forward pass, backward pass, and optimiser step.
        - Additional trainers can be added as needed by creating new subclasses of Trainer and implementing the required abstract methods.
- The *TrainerStats* class is defined under `src/trainer/stats/base.py` as an abstract base class for collecting and reporting statistics during training. It defines methods that are called at various points in the training loop to track metrics such as loss, accuracy, time taken, and energy consumption. It defines a set of hooks that the trainers can call to measure and log what's happening at different phases of the training. The base class contains the following abstract methods at increasing levels of granularity, where every trainer calls these hooks before and after each different phase of training, allowing the trackers to record metrics like time, energy consumption, traces, and more. This makes it easy to add a new tracker: just extend the base class TrainerStats, and meanwhile the integration with the trainers is already done:
    - start_train(), stop_train(): track the full training period
    - start_step(), stop_step(): track the full training step for one batch
    - start_forward(), stop_forward(): track the forward pass 
    - start_backward(), stop_backward(): track the backward pass 
    - start_optimiser_step(), stop_optimiser_step(): track the optimiser step 
    - log_step(), log_stats(): log the collected statistics at each step and at the end of training
- Avalible TrainerStats subclasses:
    - `NOOPTrainerStats` (defined under `src/trainer/stats/noop.py`): a no-operation stats tracker that does nothing. It leaves all methods blank. Dummy default that doesn't track anything.
    - `SimpleTrainerStats` (defined under `src/trainer/stats/simple.py`): a basic stats tracker that measures time taken for each phase and logs loss at each step. It measures how much time each phase of the training loop takes. It uses torch.cuda.synchronize() to ensure CUDA timings are accurate, and stores each measurement using a helper called RunningTimer from utils.py. At the end of each training step it prints how long each subphase took (forward, backward & optimiser step), and at the end of training, it prints a breakdown of averages and quantiles.
    - `CodeCarbonStats` (defined under `src/trainer/stats/codecarbon.py`): a stats tracker that integrates with the CodeCarbon library to measure energy consumption during training. It measure the energy consumption and CO2 emissions associated with different phases of training: the full training loop, individual training steps, and passes within training steps (forward, backward, optimiser). Outputs the data in a csv file. Implemented using CodeCarbon's OfflineEmissionsTracker class.
    - Additional stats trackers can be added as needed by creating new subclasses of TrainerStats and implementing the required abstract methods.

### External Resources
### CodeCarbon Resources
- [CodeCarbon Colab Tutorial](https://colab.research.google.com/drive/1eBLk-Fne8YCzuwVLiyLU8w0wNsrfh3xq)

---

### Environment setup

We will use a Conda envrionment to install the required dependencies. The steps below will walk you through the steps. A setup script `env_setup.sh` is also provided and will execute all the steps below given as input the path `SOME_PATH` as described in step one below.

1. **Setting up storage** <br> Your home directory on the McGill server is part of a network file system where users get limited amounts of storage. You can check your storage usage and how much you are allowed to use using the command `quota`. Python packages, pip's cache, Conda's cache and datasets can use quite a bit of storage, so we need to ensure they are stored outside your directory to avoid any issues with disk quotas. Say you have your own directory, stored in `SOME_PATH`, on a server that is not part of the network file system (hence not affected by disk quotas). Use `export BASE_STORAGE_PATH=SOME_PATH` where you replace `SOME_PATH` with the actual path. The steps to go around the disk quota are as follows:
    1. We can make a cache directory using `mkdir ${BASE_STORAGE_PATH}/cache`. 
    2. For pip's cache, we can redirect it to that directory using `export PIP_CACHE_DIR=${BASE_STORAGE_PATH}/cache/pip`. 
    3. For Hugging Face datasets, we can use `export HF_HOME=${BASE_STORAGE_PATH}/cache/huggingface`. While this variable is not strictly needed for the environment set up, it is needed when using the Hugging Face datasets module.
2. **Initializing Conda** <br> If you have never used Conda with this user, you need to initialize Conda with `conda init bash`. This modifies the `~/.bashrc` file. Unfortunately, the `~/.bashrc` file is not always executed at login, depending on the server configurations. For that reason, it is recommended to run `. ~/.bashrc` before running any Conda commands. 
3. **Creating the project environment** <br> First, let's make sure to create the directory to store the environment using `mkdir -p ${BASE_STORAGE_PATH}/conda/envs`. You can now simply run `conda create --prefix ${BASE_STORAGE_PATH}/conda/envs/COMP597-project python=3.14` to create the environment. 
4. **Activating the environment** <br> You can use your environment by activating it with `conda activate ${BASE_STORAGE_PATH}/conda/envs/COMP597-project`. 
5. **Installing dependencies** <br> The dependencies are provided as a requirements file. You can install them using `pip install -r energy_efficiency/requirements.txt`.
6. **Using the environment** <br> For any future use of the environment, you can create a script, let's name it `local_env.sh`, which will contain the configuration to set up the environment. You can then execute the script with `. local_env.sh` to set up activate your environment. The script would look like this (where you need to replace `SOME_PATH`):
    ```
    #!/bin/bash
    
    . ~/.bashrc
    conda activate SOME_PATH/conda/envs/COMP597-project
    export PIP_CACHE_DIR=SOME_PATH/cache/pip
    export HF_HOME=SOME_PATH/cache/huggingface
    ```
7. **Quitting** <br> If you want to quit the environment, or reset your sheel to before you activate the environment, simply run `conda deactivate`.

---

TODO: FINSIH section about how to use the codebase, with GPT2 as an example + instructions on how to add new models.
### GPT2 example
#### How to setup a new model (GPT2)
Files to edit/add:
- Add a new model under the [models](energy_efficiency/src/models/) directory; `energy_efficiency/src/models/gpt2/gpt2.py` : contains the GPT2 model definition, optimizer initialization, and trainer setup.
- Create a bash script to run the experiments (optional); `energy_efficiency/start-gpt2.sh` : script to launch experiments with GPT2 model.
- Edit the main [launch](energy_efficiency/launch.py) file to add the new model; `energy_efficiency/launch.py` : add the model choice in the argument parser.
- Edit the [configuration](energy_efficiency/src/config/config.py) file to add any model-specific configuration;, `energy_efficiency/src/config/config.py`.
- Edit the [requirements](energy_efficiency/requirements.txt) file if new dependencies are needed; `energy_efficiency/requirements.txt`.
- Add any additional files as needed for data processing, evaluation, etc.
- Add [trainer objects](energy_efficiency/src/trainer/) and/or [trainer stats](energy_efficiency/src/trainer/stats/) if needed under `energy_efficiency/src/trainer/`.

Setting up a model - GPT2 example:
1. Find and setup a tokenizer from Hugging Face transformers. Make adjustements as needed to make it compatible with your dataset.
2. Find and setup an optimizer. Make sure to set the learning rate from the configuration object.
3. Setup data processing using the tokenizer and dataset.
4. Setup the model using data collator, a config from the model and a model (Note: do not take the pretrained one).
5. Implement the trainer setup function. You can start with a simple trainer as shown in the example. You can implement more complex trainers as needed.
6. Initialize the model and add it to the [init file](energy_efficiency/src/models/__init__.py) in the model factory. Make sure to add the model choice in the launch file argument parser and add any needed arguments to the configuration file.

#### How to run experiments with GPT2
Example commands to run experiments with GPT2 can be found in the [start script](energy_efficiency/start-gpt2.sh).

To run the model with codecarbon tracking, make necessary modifications to the [codecarbon trainer stats](energy_efficiency/src/trainer/stats/codecarbon.py) and run the experiments as shown in the script.
Add any other trainer stats objects as needed and run experiments accordingly.

#### How to run the codebase 
1. Always activate the environment first using `source local_env.sh` or `. local_env.sh` or the commands provided in the [Environment Setup](#environment-setup) section if it is the first time.
2. To train a model, use the `launch.py` script with appropriate command-line arguments. For example, to train the GPT2 model with the simple trainer and simple stats, you can run:
   ```
   python energy_efficiency/launch.py --model gpt2 --trainer simple --batch_size 4 --learning_rate 1e-6 --dataset_split "train[:100]" --train_stats simple
   ```
    > List of command-line arguments can be found in the `get_args` function in `energy_efficiency/launch.py`.
    > - **Models (`--model`)**: the model to train. Currently supports "gpt2". Add the model you need to implement in the codebase as shown in the [How to setup a new model (GPT2)](#how-to-setup-a-new-model-gpt2) section.
    > - **Trainers (`--trainer`)**: the training method to use. Currently supports "simple". More trainers can be added as needed. 
    > - **Training Stats (`--train_stats`)**: the stats collection method to use during training. Currently supports "simple" and "codecarbon". More stats collection methods can be added as needed. TODO (ADD OR REMOVE TO BE DECIDED).
    > - **Dataset (`--dataset`)**: the dataset to use. Currently supports "allenai/c4". TODO (THIS SHOULD BE CHANGED AND ALSO NEED TO ADD A FUNCTION TO PROCESS DATASET DEPENDING ON MODEL TYPE).
    > - **Batch Size (`--batch_size`)**: the batch size for training.
    > - **Learning Rate (`--learning_rate`)**: the learning rate for training. Adjust it based on the model and training setup as needed.
    > - **Dataset Split (`--dataset_split`)**: the split of the dataset to use for training. For example, "train[:100]" uses the first 100 samples from the training set.

---
