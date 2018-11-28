<img src="data/logo.png" width=25% align="right" />

# POIS (NeurIPS 2018)

This repository contains the implementation of the [POIS algorithm](https://arxiv.org/abs/1809.06098).
Click [here](https://t3p.github.io/NIPS/) for more info about our NeurIPS 2018 paper.

The implementation is based on OpenAI [baselines](https://github.com/openai/baselines).
We are working on synchronising this repository to the current version of baselines.

## What's new
We provide 3 different flavours of the POIS algorithm:
- **pois1**: control-based POIS (cpu)
- **pois2**: control-based POIS (gpu optimized, used in complex environments or complex policies)
- **pbpois**: parameter-based POIS (cpu)

## Minimal install with Docker
To test POIS on classic control environments within minutes, you can build a Docker image. This solution does not support Mujoco environments.

First, you need to [install Docker](https://docs.docker.com/get-started/#prepare-your-docker-environment). 
Then, clone the repository and build the image:

```bash
git clone https://github.com/T3p/pois
cd pois
docker build -t pois .
```

You can run the image as:


```bash
docker run -it pois
```

This will create a docker container and give you access to an interactive shell.
To test your installation (within the container):


```bash
cd pois1
python run_rllab.py
```

This should run pois1 (action-based POIS for CPU) on the 'cartpole' environment (rllab version).

## Full install (tested on Ubuntu 16.04)
First, you will need **python3** (>=3.5) and your favourite **tensorflow** version (tested with python 3.5.2 and tensorflow 1.12.0).
To use pois2, you will need a version of [tensorflow with gpu support](https://www.tensorflow.org/install/gpu).

To install pois with all the necessary requirements:

```bash
git clone https://github.com/T3p/pois
apt-get update
apt-get install ffmpeg git wget python-dev python3-dev libopenmpi-dev python-pip zlib1g-dev cmake python-opencv swig
cd pois
pip install -e .
```

If you want to test pois on rllab environments, you also need to install rllab:

```bash
git clone https://github.com/rll/rllab
cd rllab
pip install -e .
```

To test on [MuJoCo](http://www.mujoco.org) environments, you need a MuJoCo license. Instructions on setting up MuJoCo can be found [here](https://github.com/openai/mujoco-py).

## Usage
Scripts for running the algorithms on gym and rllab environments can be found under the algorithms folders. The scripts accept a large number of optional command-line arguments. Typical usage is:

```bash
python run_[gym | rllab].py --env [environment name] --seed [random seed] --policy [nn | linear]
```
The results are saved in csv and tensorboard formats under the ./logs directory.

The repository also ships with some original baselines algorithms for comparison, namely TRPO and PPO.

## Citing
To cite the POIS paper:

    @incollection{NIPS2018_7789,
        title = {Policy Optimization via Importance Sampling},
        author = {Metelli, Alberto Maria and Papini, Matteo and Faccio, Francesco and Restelli, Marcello},
        booktitle = {Advances in Neural Information Processing Systems 31},
        editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
        pages = {5443--5455},
        year = {2018},
        publisher = {Curran Associates, Inc.},
        url = {http://papers.nips.cc/paper/7789-policy-optimization-via-importance-sampling.pdf}
    }
    
 ## Acknowledgements
 The gpu-optimized version of POIS was developed by [Nico Montali](https://github.com/nicomon24), who also contributed to the overall refactoring of the code.

## Trivia
"Pois", pronounced **/pwa/**, means "polka dots" in Italian. The word derives from the French expression "à pois", which means "dotted".
If you like the style, check out this [Polka Dot Pattern Generator](https://rdyar.gitlab.io/background-generator/background-generators/polka-dot-pattern-generator/) by Ron Dyar.
