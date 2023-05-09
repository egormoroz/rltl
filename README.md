# Learning an RL agent for traffic lights 

## Description

A simple MLP agent is trained in a gym wrapper around CityFlow using PPO.

The environment wrapper can be found in `env1x1.py`. 
An agent can be trained by running `python train.py` and then evaluated and compared
to random and cyclic policies by running `python evaluate.py`.

# Comparison of implemented policies

| Name        | Mean waiting time (lower is better) |
| ----------- | ----------------------------------- |
| MLP         | 57.0                                |
| Circular 1  | 154.9                               |
| Random      | 421.9                               |
| Circular 2  | 744.4                               |
| Circular 4  | 906.9                               |


