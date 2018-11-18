import numpy as np
import tensorflow as tf
from data_generator import generate_multi_game_data, SimpleMinesweeperData, SimpleGameDataGenerator
from tf_model_runner import TfModelRunner
from tf_model_trainer import TfTrainer, TfTrainerWithRefreshingData
from simple_cnn import SimpleCnn
from ai_player import TensorflowPlayer, TensorflowPlayerFast
from game_core import GameCoreV1


def example_1():

    height = 16
    width = 16
    mine_fraction = 40.0 / (16 * 16)

    single_game_generator = SimpleGameDataGenerator(snapshot_step=4)

    def learn_data_generator():
        tgt, ins = generate_multi_game_data(10000, height, width, mine_fraction, single_game_generator, seed=np.random.randint(int(1e9)))
        return SimpleMinesweeperData(tgt, ins, 1)

    tgt_val, ins_val = generate_multi_game_data(400, height, width, mine_fraction, single_game_generator, seed=111)
    validation_data = SimpleMinesweeperData(tgt_val, ins_val, 2)

    graph_builder = SimpleCnn(height, width, num_hidden_layers=12, num_hidden_channels=16, kernel_width=5,
                              l2_penalty=1e-5, optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
                              include_layer_norm=True, seed=333)
    saved_model_path = "saved-models/test-01"
    model_runner = TfModelRunner(graph_builder.build(), saved_model_path)

    tensorboard_dir = "tf-logs/test-01"
    model_trainer = TfTrainerWithRefreshingData(model_runner, learn_data_generator, validation_data, batch_size=32, interval=200,
                                                max_num_intervals=250, tensorboard_dir=tensorboard_dir, learn_data_refresh_period=30)
    model_trainer.train(model_path=None)

    # playing games ...
    model_runner_cpu = TfModelRunner(graph_builder.build(), saved_model_path, session_config=tf.ConfigProto(device_count={'GPU': 0}))
    ai_player = TensorflowPlayerFast(model_runner_cpu).turn_on()
    rand_gen = np.random.RandomState(222)
    n_games = 1000
    n_won = 0
    for i in range(n_games):
        game = GameCoreV1.initialize_game(height, width, mine_fraction, rand_gen.randint(int(1e9)))
        if ai_player.play(game) == 1:
            n_won += 1
        # print("game %d done" % i)

    ai_player.turn_off()
    print("pct won: %.3f" % (n_won / n_games))


if __name__ == '__main__':
    example_1()
