Attention!: This repo is not ready! Work in progress🚧
# FastFlow

An unofficial implementation of the architecture of FastFlow [(Jiawei Yu et al.)](https://arxiv.org/pdf/2111.07677v2.pdf).
Starting from [this](https://github.com/marco-rudolph/differnet) implementation of Differnet of Marco Rudolph, I'm trying to create an easy to use implementation of FastFlow.

Python version required >= 3.8

If you use neptune, create a file named `neptuneparams.py` and insert this code
```
project="insert_name_of_neptune_project_here"
api_token="inset_token_here
```
These parameters are generated when you create a project on neptune, and you can find them there.
If you are not interested in using neptune, you can comment the neptune code and the import statement in `train.py`.

This project assumes that you have the mvtec dataset in the following path structure:
```
- data:
    - mvtec:
        - hazelnut
        - toothbush
        - ...
```
