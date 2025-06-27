# FastFlow

An unofficial implementation of the FastFlow architecture by Jiawei Yu et al. This implementation builds upon the [DifferNet implementation by Marco Rudolph](https://github.com/marco-rudolph/differnet) to provide an easy-to-use FastFlow implementation.

## Requirements

- Python >= 3.8

## Setup

### Neptune Integration (Optional)

If you want to use Neptune for experiment tracking:

1. Create a file named `neptuneparams.py` in the root directory
2. Add the following content:

```python
project = "insert_name_of_neptune_project_here"
api_token = "insert_token_here"
```

> **Note:** These parameters are generated when you create a project on Neptune. You can find them in your Neptune project dashboard.

If you don't want to use Neptune, simply comment out the Neptune-related code and import statements in `train.py`.

### Dataset Structure

This project expects the MVTec dataset to be organized in the following directory structure:

```
data/
└── mvtec/
    ├── hazelnut/
    ├── toothbrush/
    └── [other_categories]/
```

## Usage

*Usage instructions will be added as development progresses.*

## Contributing

This project is currently under active development. Contributions and feedback are welcome!

## References

- Original FastFlow paper: [FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows](https://arxiv.org/abs/2111.07677) by Jiawei Yu et al.
- Based on DifferNet implementation by Marco Rudolph
