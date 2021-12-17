# Snake Reinforcement Learning

Teaching a reinforcement learning agent to play the classical game of snake.

## Description

This project contains 2 implementations of machine learning agents for playing snake, one created using Convolutional Neural Networks (CNN) and one using a densly connected network.

## Dependencies

For running this project you'll need a modern Python version (>=3.9) as well as the following packages:

- [TensorFlow](https://www.tensorflow.org/)
- [pygame](https://www.pygame.org/)
- [NumPy](https://numpy.org/)
- [tqdm](https://github.com/tqdm/tqdm)

These are specified in the `requirements.txt` file and can be easily installed using pip with the following command:

```bash
pip install -r requirements.txt
```

## Training

### Densly connected model

For training the densly connected model simply run:

```bash
python src/nn_model.py
```

This should start running the training for a new model with the default configuration (found in `src/nn_model.py`)

### CNN Model

For training the CNN model, you can run the following command:

```bash
python src/train_cnn.py
```

If you want to configure some variables, like if weights should be loaded before running, if graphics should be used, and how many epochs it should be trained on, run the following command to get a description on how to set these:

```bash
python src/train_cnn.py --help
```

## Playing

### Human

If you want to play the game yourself, simply run:

```bash
python src/play.py
```

### Densly connected network

To play the game using a trained densly connected network run:

```bash
python src/play.py -m nn -w <PATH TO WEIGHTS>
```

### CNN

To play the game using a trained CNN run:

```bash
python src/play.py -m cnn -w <PATH TO WEIGHTS>
```
