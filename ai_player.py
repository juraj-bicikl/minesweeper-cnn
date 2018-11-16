import numpy as np
from abc import ABC, abstractmethod
from tf_model_runner import TfModelRunner
from game_core import GameCore


class AIPlayer(ABC):

    @abstractmethod
    def next_move(self, visible_board):
        """ Returns coordinates of the next move.

            Args:
                visible_board (numpy.ndarray): board based on encoding described in GameCore.get_visible_board()
        """
        ...

    @abstractmethod
    def predicted_mine_probability(self, visible_board):
        """ Returns a 2d array of predicted probability of finding a mine at a particular field.

            Args:
                visible_board (numpy.ndarray): board based on encoding described in GameCore.get_visible_board()
        """
        ...

    @abstractmethod
    def play(self, game: GameCore):
        """ Plays the 'game'.

            Args:
                game (GameCore): game instance

            Returns:
                1   won
                -1  lost
        """
        ...

    @abstractmethod
    def turn_on(self):
        """ Turns on the player.

            Returns:
                self
        """
        ...

    @abstractmethod
    def turn_off(self):
        """ Turns off the player.

            Returns:
                self
        """
        ...


class TensorflowPlayer(AIPlayer):

    def __init__(self, model_runner: TfModelRunner):
        self.model_runner = model_runner

    def next_move(self, visible_board):
        mine_prob = self.predicted_mine_probability(visible_board)
        # setting all the values of explored fields to np.inf, so that we can find the unexplored field with minimum
        # mine probability:
        mine_prob[visible_board > -1] = np.inf
        i, j = np.unravel_index(np.argmin(mine_prob, axis=None), mine_prob.shape)
        assert mine_prob[i, j] != np.inf
        return i, j

    def predicted_mine_probability(self, visible_board):
        model_inputs = visible_board.reshape([1] + list(visible_board.shape))
        return self.model_runner.evaluate(model_inputs)[0, :, :, 0]

    def play(self, game: GameCore):
        # print("starting to play the game ...")
        while game.get_status() == 0:
            i, j = self.next_move(game.get_visible_board())
            game.press_button(i, j)
            # print("pressed (%d, %d)" % (i, j))

        return game.get_status()

    def turn_on(self):
        self.model_runner.start_session()
        self.model_runner.load()
        return self

    def turn_off(self):
        self.model_runner.end_session(save=False)
        return self
