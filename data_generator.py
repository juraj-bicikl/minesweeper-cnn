import numpy as np
import time
from game_core import GameCore, GameCoreV1
from tf_data import TfData
from abc import ABC, abstractmethod


class SimpleMinesweeperData(TfData):

    def __init__(self, target, inputs, seed=123):
        assert target.shape[0] == inputs.shape[0]
        self.target = target.astype(np.float32)
        self.inputs = inputs.astype(np.float32)  # we apply 'manual' scaling
        self.rand_gen = np.random.RandomState(seed)
        self.__num_rows = self.target.shape[0]

    def random_batch(self, batch_size):
        rand_indices = self.rand_gen.randint(low=0, high=self.__num_rows, size=batch_size)
        return self.target[rand_indices, :, :], self.inputs[rand_indices, :, :]

    def all_rows(self):
        return self.target, self.inputs

    def num_rows(self):
        return self.__num_rows


class GameDataGenerator(ABC):

    @abstractmethod
    def generate(self, game: GameCore, rand_gen: np.random.RandomState) -> (np.ndarray, np.ndarray):
        """ Generates data for NN training

            Args:
                game: GameCore instance
                rand_gen: numpy random number generator

            Returns:
                (np.ndarray, np.ndarray) pair of target, input data (both arrays are 3D arrays)
        """
        ...


class SimpleGameDataGenerator(GameDataGenerator):

    def __init__(self, snapshot_step: int):
        self.snapshot_step = snapshot_step

    def generate(self, game, rand_gen):

        num_steps = (game.get_height() * game.get_width() - game.get_num_mines())
        max_num_snapshots = num_steps // self.snapshot_step
        mine_field = (game.get_board() == 9).astype(np.int8)

        # pre-allocating the arrays
        target = np.empty(shape=[max_num_snapshots, game.get_height(), game.get_width()], dtype=np.int8)
        inputs = np.empty(shape=[max_num_snapshots, game.get_height(), game.get_width()], dtype=np.int8)

        # filling 'target'
        for i in range(max_num_snapshots):
            target[i, :, :] = mine_field

        step = 0
        snapshot_idx = 0
        # playing the game and filling 'inputs'
        while game.get_status() == 0:

            # taking snapshot
            if step % self.snapshot_step == 0:
                inputs[snapshot_idx, :, :] = game.get_visible_board()
                snapshot_idx += 1

            # making next step
            unexplored_safe_fields = list(game.get_unexplored_safe_fields())
            i, j = unexplored_safe_fields[rand_gen.randint(low=0, high=len(unexplored_safe_fields))]
            game.press_button(i, j)
            step += 1

        assert game.get_status() == 1  # making sure the game finished successfully

        target = target[0:snapshot_idx, :, :]
        inputs = inputs[0:snapshot_idx, :, :]

        return target, inputs


def generate_multi_game_data(num_games, height, width, mine_fraction, single_game_generator, seed):

    rand_gen = np.random.RandomState(seed)
    target_list = []
    inputs_list = []

    for i in range(num_games):
        game = GameCoreV1.initialize_game(height, width, mine_fraction, seed=rand_gen.randint(1000000000))
        tgt, inp = single_game_generator.generate(game, rand_gen)
        target_list.append(tgt)
        inputs_list.append(inp)

    target = np.concatenate(target_list, axis=0)
    inputs = np.concatenate(inputs_list, axis=0)

    return target, inputs


def test_1():
    game = GameCoreV1.initialize_game(8, 8, mine_fraction=0.1, seed=5723)
    target, inputs = SimpleGameDataGenerator(snapshot_step=10).generate(game, np.random.RandomState(123))

    for tgt, inp in zip(target, inputs):
        print("target ...")
        print(tgt)
        print("inputs ...")
        print(inp)
        print()


def test_2():
    target, inputs = generate_multi_game_data(num_games=2, height=8, width=8, mine_fraction=0.1,
                                              single_game_generator=SimpleGameDataGenerator(snapshot_step=10), seed=123)

    for tgt, inp in zip(target, inputs):
        print("target ...")
        print(tgt)
        print("inputs ...")
        print(inp)
        print()


if __name__ == '__main__':
    test_2()
