import numpy as np
import time
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

    def play(self, game: GameCore):
        """ Plays the full 'game'.

            Args:
                game (GameCore): game instance

            Returns:
                1   won
                -1  lost
        """
        while game.get_status() == 0:
            self.play_step(game)

        return game.get_status()

    @abstractmethod
    def play_step(self, game: GameCore):
        """ Plays one 'step' of the 'game'. Here the 'step' is whatever one wants it to be.

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

    def play_step(self, game: GameCore):
        assert game.get_status() == 0
        i, j = self.next_move(game.get_visible_board())
        game.press_button(i, j)
        return game.get_status()

    def turn_on(self):
        self.model_runner.start_session()
        self.model_runner.load()
        return self

    def turn_off(self):
        self.model_runner.end_session(save=False)
        return self


class TensorflowPlayerFast(AIPlayer):

    def __init__(self, model_runner: TfModelRunner, fast_move_threshold=0.01):
        self.model_runner = model_runner
        self.fast_move_threshold = fast_move_threshold

    def next_move(self, visible_board):
        mine_prob = self.predicted_mine_probability(visible_board)
        # setting all the values of explored fields to np.inf, so that we can find the unexplored field with minimum
        # mine probability:
        mine_prob[visible_board > -1] = np.inf
        i, j = np.unravel_index(np.argmin(mine_prob, axis=None), mine_prob.shape)
        assert mine_prob[i, j] != np.inf
        return i, j

    def fast_moves(self, visible_board):
        mine_prob = self.predicted_mine_probability(visible_board)
        # setting all the values of explored fields to np.inf, so that we can find the unexplored field with minimum
        # mine probability:
        mine_prob[visible_board > -1] = np.inf

        moves = []
        for i in range(mine_prob.shape[0]):
            for j in range(mine_prob.shape[1]):
                if mine_prob[i, j] < self.fast_move_threshold:
                    moves.append((i, j))

        if not moves:
            i, j = np.unravel_index(np.argmin(mine_prob, axis=None), mine_prob.shape)
            moves.append((i, j))

        return moves

    def predicted_mine_probability(self, visible_board):
        model_inputs = visible_board.reshape([1] + list(visible_board.shape))
        return self.model_runner.evaluate(model_inputs)[0, :, :, 0]

    def play_step(self, game: GameCore):
        assert game.get_status() == 0
        moves = self.fast_moves(game.get_visible_board())
        for i, j in moves:
            if game.press_button(i, j) != 0:
                break
        return game.get_status()

    def turn_on(self):
        self.model_runner.start_session()
        self.model_runner.load()
        return self

    def turn_off(self):
        self.model_runner.end_session(save=False)
        return self
