
# POIS (NeurIPS 2018)

This repository contains the implementation of the [POIS algorithm](https://arxiv.org/abs/1809.06098).
It is based on the OpenAI [baselines](https://github.com/openai/baselines) implementation, and uses the same implementation backbone.
We are working on synchronising this repository to the current version of OpenAI baselines.

## What's new
We provide 3 different flavours of the POIS algorithm:
- **POIS1**: control-based POIS (cpu)
- **POIS2**: control-based POIS (gpu optimized, used in complex environments or complex policies)
- **PBPOIS**: parameter-based POIS (cpu)

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

This should run pois1 (action-based POIS for CPU) on the cartpole environment (rllab version)
