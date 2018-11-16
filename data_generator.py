import numpy as np
import time
from game_core import GameCoreV1
from tf_data import TfData


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


def generate_single_game_data(game, snapshot_step, seed=456):
    """ Generates data for NN training

        Args:
            game: GameCore instance
            snapshot_step: game snapshot is taken every 'snapshot_step' moves and added to the data set
            seed: random number generator seed

        Returns:
            (np.ndarray, np.ndarray) pair of target, input data (both arrays are 3D arrays)
    """

    num_steps = (game.get_height() * game.get_width() - game.get_num_mines())
    max_num_snapshots = num_steps // snapshot_step
    rand_gen = np.random.RandomState(seed)

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
        if step % snapshot_step == 0:
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


def generate_multi_game_data(num_games, height, width, mine_fraction, snapshot_step, seed):

    rand_gen = np.random.RandomState(seed)
    target_list = []
    inputs_list = []

    for i in range(num_games):
        game = GameCoreV1.initialize_game(height, width, mine_fraction, seed=rand_gen.randint(1000000000))
        tgt, inp = generate_single_game_data(game, snapshot_step, seed=rand_gen.randint(1000000000))
        target_list.append(tgt)
        inputs_list.append(inp)

    target = np.concatenate(target_list, axis=0)
    inputs = np.concatenate(inputs_list, axis=0)

    return target, inputs


def generate_multi_game_data_with_timings(num_games, height, width, mine_fraction, snapshot_step, seed):

    rand_gen = np.random.RandomState(seed)
    target_list = []
    inputs_list = []

    t_init = 0.0
    t_single_game = 0.0

    for i in range(num_games):
        t0 = time.time()
        game = GameCoreV1.initialize_game(height, width, mine_fraction, seed=rand_gen.randint(1000000000))
        t1 = time.time()
        tgt, inp = generate_single_game_data(game, snapshot_step, seed=rand_gen.randint(1000000000))
        t2 = time.time()
        t_init += t1 - t0
        t_single_game += t2 - t1
        target_list.append(tgt)
        inputs_list.append(inp)

    t0 = time.time()
    target = np.concatenate(target_list, axis=0)
    inputs = np.concatenate(inputs_list, axis=0)
    t1 = time.time()
    t_concat = t1 - t0

    print("t_init: %.2f s" % t_init)
    print("t_single_game: %.2f s" % t_single_game)
    print("t_concat: %.2f s" % t_concat)

    return target, inputs


def test_1():
    game = GameCoreV1.initialize_game(8, 8, mine_fraction=0.1, seed=5723)
    target, inputs = generate_single_game_data(game, snapshot_step=10)

    for tgt, inp in zip(target, inputs):
        print("target ...")
        print(tgt)
        print("inputs ...")
        print(inp)
        print()


def test_2():
    target, inputs = generate_multi_game_data(num_games=2, height=8, width=8, mine_fraction=0.1, snapshot_step=5, seed=123)

    for tgt, inp in zip(target, inputs):
        print("target ...")
        print(tgt)
        print("inputs ...")
        print(inp)
        print()


def test_3():
    h, w = 16, 16
    mine_fraction = 0.17
    generate_multi_game_data_with_timings(1000, h, w, mine_fraction, snapshot_step=4, seed=np.random.randint(int(1e9)))


if __name__ == '__main__':
    test_3()
