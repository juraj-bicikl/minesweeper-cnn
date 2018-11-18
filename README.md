# minesweeper-cnn
Convolutional neural network (CNN) based AI for Minesweeper game.


### Description

AI for arcade games is often developed using reinforcement learning as supervised learning methods are difficult 
to apply since there is usually no direct relationship between current action and final reward 
(i.e. winning or losing).

Minesweeper is somewhat special and it allows us to take a different approach:
1. We define our strategy (or 'policy') to be: at each step of the game choose the field with the smallest probability 
of containing a mine.
2. Probability that a field contains a mine given the current state of the board is provided by 
convolutional neural network trained using supervised learning.

Our strategy (choosing the field with smallest proability of containing a mine) is not necessarily optimal,
however there is good evidence that it is quite close to it and it performs very well in practice.

The GUI for the game is based on **pygame** package, while neural network is trained using **tensorflow**.

At the moment, the only available difficulty level is *intermediate* (16 x 16 field containing 40 mines).

### Instructions

Playing the game will require installing several packages like *pygame*, *tensorflow*, *numpy*.

The game with AI assistance can be started using the following command:
```
python3 game.py
```

For playing the usual Minesweeper (without the AI assistance) run:
```
python3 gui_game.py
```

For an example of a script used for training a model take a look at *training_example.py*.

### AI performance

The AI currently used to assist in playing the game wins around 62% of games. The underlying model is a 13 layers
deep conolutional network.

### Todo list

There is place for improvement at several places:
1. Improve the AI performance by experimenting with different network topologies, training with more data, ...
2. Add AI for *beginner* and *expert* difficulty levels.
3. Improve GUI.

