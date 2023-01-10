# Reinforcement Learning Final Report
> Guided by Prof. Xu Chen

This repository holds the code that is used to reproduce the paper "[Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)".

## Authors
* Zhe Tang (Email: tangzh_33@163.com)

## Prerequisites
* For training, an NVIDIA GPU is strongly recommended. CPU is also supported but significantly decreases training speed.

## Datasets
We use continuous tasks provided by **OpenAI Gym API**. You can find the list of tasks [here](https://gym.openai.com/envs/#classic_control). We mainly use Pendulum-v0, LunarLanderContinuous-v2, MountainCarContinuous-v0 and HopperBulletEnv-v0 for testing.

## Installing Environment

### Basic Environment Introduction 

The Basic Environment Introduction is listed in the chart as follows:

| Environment |             Details             |
| :---------: | :-----------------------------: |
|     CPU     | AMD EPYC 7742 64-Core Processor |
|     RAM     |              252G               |
|     GPU     |         NVIDIA A100 40G         |
|     OS      |            Debian 11            |
|  Compiler   |            GNU@7.5.0            |
|    CUDA     |           CUDA@11.7.1           |
|   Pytorch   |       Torch@1.13.0+cu117        |

### Software

We strongly recommend using Anaconda 3 to set the environment. To install the environment in Anaconda use the following command:
```bash
conda create -n rl_final python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia #Install according to your own machines.
conda install tensorboard gym pybullet
conda install swig # needed to build Box2D in the pip install
pip install box2d-py # a repackaged version of pybox2d
```
To then activate this environment use:
```console
conda activate rl_final
```

## Training models
- In this project, I reproduced the original DDPG algorithm based on OpenAI Gym and Pytorch,  you can refer to the `ddpg_baseline.py`

- According to the test results of a large number of experiments, I analyzed the sensitive items and weaknesses of the baseline version of the DDPG algorithm

- Aiming at the weakness, I propose improvements in three different directions for DDPG
  -   Dueling Network: You can refer to the `ddpg_dueling_network.py`
  -   Finetuing Hyper-parameters: You can refer to the report.
  -   Import auto-tuning methods: You can refer to the `ddpg_Simulated_Annealing_lr.py`
  -   For more details, you can refer to the reproducibility report.
- For reproduction, you can refer to the source code and use the hyper-parameters combination using running time variants.(e.g. if we want to run the baseline code with learning rate 5e-4 in Pendulum-v0, you can run `python --gym_id Pendulum-v0 --lr 5e-4 --train`)

## Results

One of the results runing the task Pendulum-v0:

![Pendulum-v0_results](./pic/Pendulum-v0_result.png)
