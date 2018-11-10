import numpy as np
from abc import ABC, abstractmethod


class GameCore(ABC):

    @abstractmethod
    def get_status(self):
        """ Returns current game 'status' (int):
             0: running
             1: game over - won
            -1: game over - lost
        """
        ...

    @abstractmethod
    def get_board(self):
        """ Returns a 2D np.ndarray of np.int32 representing the current state of the game.

            Codes:
                explored fields: -1, 0, 1, ..., 8 (-1: mine, while others denote number of mines in neighbourhood)

                unexplored fields: 9, 10, ..., 18 (9: mine, while others denote number of mines in neighbourhood + 10)

            Users should NOT modify the array on their own. The array is (indirectly) modified only by calling
            'press_button'.
        """
        ...

    @abstractmethod
    def get_height(self):
        """ Returns height of the board (int). """
        ...

    @abstractmethod
    def get_width(self):
        """ Returns width of the board (int). """
        ...

    @abstractmethod
    def get_num_mines(self):
        """ Returns the total number of mines on the board (int). """
        ...

    @abstractmethod
    def press_button(self, i, j):
        """ Press button at position (i, j)

            Args:
                i: vertical position (from top down)  (int)
                j: horizontal position (from left to right)  (int)
        """
        ...


class GameCoreV1(GameCore):
    """
        Simple implementation of GameCore interface.

        To create a class instance with random mine field, call 'initialize_game' method.
        If you wish create valid board from a give mine field, use 'mine_field_to_board' method.
    """

    def __init__(self, initial_board):
        """
            Args:
                 initial_board - see description under GameCore.get_board()
        """
        assert type(initial_board) == np.ndarray
        assert len(initial_board.shape) == 2

        self.height = initial_board.shape[0]
        self.width = initial_board.shape[1]
        self.num_mines = np.sum((initial_board == 9))
        self.board = initial_board.copy()
        self.unexplored_safe_fields = self.__produce_set_of_unexplored_safe_fields()
        self.status = 0  # status == 0 if not finished, 1 if finished (won), -1 if finished (lost)

    def get_status(self):
        return self.status

    def get_board(self):
        return self.board

    def get_height(self):
        return self.height

    def get_width(self):
        return self.width

    def get_num_mines(self):
        return self.num_mines

    def press_button(self, i, j):
        if self.board[i, j] >= 9:
            # updating (i, j) field, and checking/updating status
            self.board[i, j] -= 10
            if self.board[i, j] == -1:
                self.status = -1
            else:
                self.unexplored_safe_fields.remove((i, j))
                if not self.unexplored_safe_fields:
                    self.status = 1

            # pressing neighbouring buttons if the current field has no surrounding mines
            if self.board[i, j] == 0:
                for kk in range(i-1, i+2):
                    for ll in range(j-1, j+2):
                        if (kk >= 0) and (kk < self.height) and (ll >= 0) and (ll < self.width):
                            self.press_button(kk, ll)

    def __produce_set_of_unexplored_safe_fields(self):
        s = set()
        for i in range(self.height):
            for j in range(self.width):
                if self.board[i, j] > 9:
                    s.add((i, j))
        return s

    @staticmethod
    def mine_field_to_board(mine_field):
        """ Creates board state from positions of all the mines.

            Args:
                mine_field: 2d array with mine positions (code: 1 if field has mine, 0 otherwise)

            Returns:
                board (np.ndarray)
        """
        height, width = mine_field.shape
        board = np.empty(shape=mine_field.shape, dtype=np.int32)
        for i in range(height):
            h_start = max(0, i-1)
            h_end = min(height, i+2)
            for j in range(width):
                w_start = max(0, j-1)
                w_end = min(width, j+2)
                if mine_field[i, j]:
                    board[i, j] = 9
                else:
                    board[i, j] = np.sum(mine_field[h_start:h_end, w_start:w_end]) + 10
        return board

    @staticmethod
    def initialize_game(height, width, mine_fraction, seed=123, fixed_num_mines=True):

        rand_gen = np.random.RandomState(seed)

        if fixed_num_mines:
            num_mines = int(mine_fraction * height * width)
            random_mine_field = np.zeros(shape=[height * width], dtype=np.int32)
            for i in range(num_mines):
                random_mine_field[i] = 1
            random_mine_field = rand_gen.permutation(random_mine_field).reshape([height, width])
        else:
            random_mine_field = (rand_gen.rand(height, width) < mine_fraction).astype(np.int32)

        board = GameCoreV1.mine_field_to_board(random_mine_field)
        return GameCoreV1(board)
