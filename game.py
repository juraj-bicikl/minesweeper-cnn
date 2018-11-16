import numpy as np
import tensorflow as tf
from gui_game import GuiGame
from game_core import GameCoreV1
from simple_cnn import SimpleCnn
from tf_model_runner import TfModelRunner
from ai_player import TensorflowPlayer


def run_game():

    height = 16
    width = 16
    mine_fraction = 40.0 / (16 * 16)

    graph_builder = SimpleCnn(height, width, num_hidden_layers=12, num_hidden_channels=16, kernel_width=5,
                              l2_penalty=1e-5, optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
                              include_layer_norm=True, seed=333)
    saved_model_path = "example-models/run-02"
    model_runner = TfModelRunner(graph_builder.build(), saved_model_path)

    game_core = GameCoreV1.initialize_game(height, width, mine_fraction, seed=np.random.randint(int(1e9)))
    ai_player = TensorflowPlayer(model_runner)
    game = GuiGame(game_core, ai_assistant=ai_player, tile_size=28)
    game.run()


if __name__ == '__main__':
    run_game()
