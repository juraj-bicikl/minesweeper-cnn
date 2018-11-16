import numpy as np
import pygame
from game_core import GameCoreV1

# colors ...
RED = (255, 0, 0)
ORANGE = (255, 127, 0)
YELLOW = (255, 255, 0)
BLUE = (51, 51, 255)
WHITE = (245, 245, 245)
GREY = (187, 187, 187)
BLACK = (0, 0, 0)


# mouse clicks ...
LEFT_CLICK = 1
MIDDLE_CLICK = 2
RIGHT_CLICK = 3


class GuiGame:

    def __init__(self, game_core, ai_assistant=None, tile_size=32):
        """
            Args:
                game_core (GameCore): game instance
                ai_assistant (AIPlayer): AI player that will assist in playing the game
                tile_size: tile size
        """

        self.game_core = game_core
        self.ai_assistant = ai_assistant
        self.height = game_core.get_height()
        self.width = game_core.get_width()
        self.tile_size = tile_size
        self.panel_height = 50

        self.flag_grid = np.full(shape=(self.height, self.width), fill_value=False)
        self.screen_height = self.height * self.tile_size
        self.screen_width = self.width * self.tile_size
        self.screen = None
        self.font_mine_count = None
        self.font_prob_estimate = None
        self.clock = None
        self.mine_prob_estimate = None

    def act_on_mouse_click(self, mouse_x, mouse_y, click):
        j = mouse_x // self.tile_size
        i = (mouse_y - self.panel_height) // self.tile_size
        assert i >= 0 & i < self.height
        assert j >= 0 & j < self.width

        value = self.game_core.get_board()[i, j]
        tile_hidden = value >= 9
        tile_flagged = self.flag_grid[i, j]

        if click == LEFT_CLICK and (not tile_flagged):
            self.game_core.press_button(i, j)
        if click == RIGHT_CLICK and tile_hidden:
            self.flag_grid[i, j] = not self.flag_grid[i, j]
        if click == MIDDLE_CLICK and value > 0 and value < 9:
            h_start = max(0, i-1)
            h_end = min(self.height, i+2)
            w_start = max(0, j-1)
            w_end = min(self.width, j+2)
            neighbouring_flags = np.sum(self.flag_grid[h_start:h_end, w_start:w_end])
            if neighbouring_flags == value:
                for kk in range(h_start, h_end):
                    for ll in range(w_start, w_end):
                        if not self.flag_grid[kk, ll]:
                            self.game_core.press_button(kk, ll)

        # updating mine probability estimate
        if self.ai_assistant:
            self.mine_prob_estimate = self.ai_assistant.predicted_mine_probability(self.game_core.get_visible_board())

    def draw_board(self):
        self.screen.fill(GREY)
        self.__draw_panel()
        for i in range(self.height):
            for j in range(self.width):
                prob = None if (self.mine_prob_estimate is None) else self.mine_prob_estimate[i, j]
                self.__draw_tile(i, j, prob)

    def __draw_panel(self):
        pygame.draw.rect(self.screen, ORANGE, (0, 0, self.screen_width, self.panel_height))
        status = self.game_core.get_status()
        if status != 0:
            msg = "YOU WON" if status == 1 else "YOU LOST"
            text = self.font_mine_count.render(msg, True, BLACK)
            text_x = text.get_rect().width
            text_y = text.get_rect().height
            self.screen.blit(text, ((self.screen_width // 2) - (text_x // 2), (self.panel_height // 2) - (text_y // 2)))

    def __draw_tile(self, i, j, mine_prob_estimate=None):
        x, y = (j * self.tile_size, i * self.tile_size + self.panel_height)
        rect = (x, y, self.tile_size, self.tile_size)
        value = self.game_core.get_board()[i, j]
        has_flag = self.flag_grid[i, j]
        is_visible = value < 9
        assert not (has_flag and is_visible)

        if has_flag:
            pygame.draw.rect(self.screen, YELLOW, rect)
        elif is_visible:
            has_mine = value == -1
            if has_mine:
                pygame.draw.rect(self.screen, RED, rect)
            else:
                pygame.draw.rect(self.screen, WHITE, rect)
                if value > 0:
                    text = self.font_mine_count.render(str(value), True, BLACK)
                    text_x = text.get_rect().width
                    text_y = text.get_rect().height
                    self.screen.blit(text, (x + (self.tile_size // 2) - (text_x // 2), y + (self.tile_size // 2) - (text_y // 2)))

        if mine_prob_estimate and (not is_visible):
            text = self.font_prob_estimate.render(str(int(round(mine_prob_estimate * 100))), True, BLUE)
            text_x = text.get_rect().width
            text_y = text.get_rect().height
            self.screen.blit(text, (x + (self.tile_size // 2) - (text_x // 2), y + (self.tile_size // 2) - (text_y // 2)))

        pygame.draw.rect(self.screen, BLACK, rect, 2)

    def run(self):

        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height + self.panel_height))
        pygame.display.set_caption("Minesweeper")
        self.clock = pygame.time.Clock()
        self.font_mine_count = pygame.font.SysFont("notomono", 20)
        self.font_prob_estimate = pygame.font.SysFont("notomono", 12)

        if self.ai_assistant:
            self.ai_assistant.turn_on()
            self.mine_prob_estimate = self.ai_assistant.predicted_mine_probability(self.game_core.get_visible_board())

        self.draw_board()
        pygame.display.flip()

        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos
                    if self.game_core.get_status() == 0:
                        self.act_on_mouse_click(mouse_x, mouse_y, event.button)

            self.draw_board()
            pygame.display.flip()

            self.clock.tick(50)

        if self.ai_assistant:
            self.ai_assistant.turn_off()
        pygame.quit()


def simple_game():
    height = int(input("height: "))
    width = int(input("width: "))
    mine_fraction = float(input("mine fraction (for example: 0.15): "))
    game_core = GameCoreV1.initialize_game(height, width, mine_fraction, seed=int(np.random.randint(int(1e9))))
    game = GuiGame(game_core, tile_size=30)
    game.run()


if __name__ == '__main__':
    simple_game()
