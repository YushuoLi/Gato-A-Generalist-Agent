# A Generalist Agent: Gato

## Overview

Unofficial code for [A Generalist Agent](https://arxiv.org/abs/2205.06175). The agent Gato works as a multi-modal, multi-task, multi-embodiment generalist policy for different tasks using a single, large, transformer sequence model with only one single set of weights. In this repository we achieve a rough implementation for atari tasks and mujoco control tasks in OpenAI gym.
![3488245043cb29bef82fe47cf2ca82b.jpg](https://s2.loli.net/2022/10/27/igeTKucDHWvRlMr.jpg)

## Architecture

![3ebd64c19c0f44b91cda199213bd0ad.jpg](https://s2.loli.net/2022/10/27/3ndYFMVJPERCZTI.jpg)

Following the ideas in the paper, we realized:
* Tokenizer
* Embedding and Encoding
* Sequence
* Transformer
* Prompt

Check the paper and the code for details. 
### Tokenizer


### Embedding
### Sequence
### Transformer
### Prompt
![5ee1df75182416bf5483b19f0ace73f.jpg](https://s2.loli.net/2022/10/27/FYliV1U7zLSagtR.jpg)



## Requirements
### Install Gym
```bash
pip install gym==0.23.1
```
### Install Mujoco and D4RL
Download the mujoco210.
```bash
mkdir ~/Downloads/
cd ~/Downloads/
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -zxvf mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
cp -r mujoco210 ~/.mujoco
gedit ~/.bashrc  
export MUJOCO_KEY_PATH=~/.mujoco${MUJOCO_KEY_PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/shenxi/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
source ~/.bashrc

cd ~/.mujoco/mujoco210/bin/
./simulate ../model/humanoid.xml
```
Install mujoco_py.
```bash
git clone https://github.com/openai/mujoco-py.git
cd mujoco-py
pip install -e .
pip install mujoco_py
```
```bash
sudo apt install libosmesa6-dev
sudo apt-get install libglew-dev glew-utils
sudo apt-get -y install patchelf
sudo apt install gcc
```
Install dm_control
```bash
pip install dm_control
```

Install D4RL
```bash
pip install absl-py
pip install matplotlib
git clone https://github.com/rail-berkeley/d4rl.git
```
Find the setup.py document
```bash
install_requires=['gym',
                      'numpy',
                      # 'mujoco_py',
                      'pybullet',
                      'h5py',
                      'termcolor',  # adept_envs dependency
                      'click',  # adept_envs dependency
                      # 'dm_control' if 'macOS' in platform() else
                      # 'dm_control @ git+git://github.com/deepmind/dm_control@master#egg=dm_control',
                      'mjrl @ git+git://github.com/aravindr93/mjrl@master#egg=mjrl'],
```
Install and test:
```bash
# installing
pip install -e .

import gym
import d4rl # Import required to register environments
```

### Install Atari
```bash
pip install atari_py
pip install gym[atari]
```



## Results

**Note:** these results are mean and variance of 3 random seeds obtained after 20k updates (due to timelimits on GPU resources on colab) while the official results are obtained after 100k updates. So these numbers are not directly comparable, but they can be used as rough reference points along with their corresponding plots to measure the learning progress of the model. The variance in returns and scores should decrease as training reaches saturation.


| Dataset | Environment | DT (this repo) 20k updates | DT (official) 100k updates|
| :---: | :---: | :---: | :---: |
| Medium | HalfCheetah | 42.18 ± 00.59 | 42.60 ± 00.10 |
| Medium | Hopper | 69.43 ± 27.34 | 67.60 ± 01.00 |
| Medium | Walker | 75.47 ± 31.08 | 74.00 ± 01.40 |





## References
- Decision Transformer: Reinforcement Learning via Sequence Modeling [code](https://github.com/kzl/decision-transformer) and [paper](https://arxiv.org/abs/2106.01345)
- Minimal Implementation of Decision Transformer [code](https://github.com/nikhilbarhate99/min-decision-transformer)
- Unofficial Gato: A Generalist Agent [code](https://github.com/OrigamiDream/gato)