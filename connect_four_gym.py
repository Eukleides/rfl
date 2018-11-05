import gym
import pygame
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

class FooEnv(gym.Env):
    metadata = {
                'render.modes': ['human'],
                'state_types': [0, 1, 2],
                'AI PLAYER': 1, # ai player
                'OPPONENT': 2  # opponent player
                }

    def __init__(self):
        self.reset()

    def checkDraw(self):
        return np.max(self.gridCounterHeights) == -1

    def checkWin(self, col, row):
        return (self.checkWinInRow(self.state[row, :]) or self.checkWinInRow(self.state[:, col])
              or self.checkWinInRow(np.diagonal(self.state, offset=col - row))
              or self.checkWinInRow(np.diagonal(np.fliplr(self.state), offset=self.COLUMNS - 1 - (col + row))))

    def checkWinInRow(self, line):
        count = 0
        for c in range(len(line)):
          if line[c] == self.turn:
              count += 1
              if count >= self.WINAMOUNT:
                  return True
          else:
              count = 0
        return False

    def actionIsValid(self, action):
        return self.gridCounterHeights[action] >= 0

    def getValidRandomAction(self):
        validAction = False
        action = None
        while not validAction:
            action = self.action_space.sample()
            validAction = self.actionIsValid(action)
        return action

    def isOpponentsTurn(self):
        return (self.turn == self.metadata['OPPONENT'])

    def step(self, action):
        assert(self.actionIsValid(action))

        self.state[self.gridCounterHeights[action], action] = self.turn

        reward = 0.

        done = self.checkWin(action, self.gridCounterHeights[action])
        self.gridCounterHeights[action] -= 1

        if done:
            if self.turn == 1:
                reward = 100.
            else:
                reward = -100.


        if (done is False) and self.checkDraw():
            return self.flatten_state(np.array(self.state)), 0.0, True, {}

        player = self.turn
        self.turn = self.turn % 2 + 1
        return self.flatten_state(np.array(self.state)), reward, done, player

    def flatten_state(self, s):
        nrows = s.shape[0]
        ncols = s.shape[1]
        nsqtypes = len(self.metadata['state_types'])
        f = np.zeros((1, s.shape[0], s.shape[1], nsqtypes), dtype=int)

        for i in range(nrows):
            for j in range(ncols):
                colour = s[i][j]
                f[0][i][j][colour] = 1

        return f

    def display_board(self):
        nrows = self.state.shape[0]
        ncols = self.state.shape[1]

        print('----------------------\n')
        for i in range(nrows):
            for j in range(ncols):
                colour = self.state[i][j]
                if colour == self.metadata['AI PLAYER']:
                    print('R')
                elif colour == self.metadata['OPPONENT']:
                    print('Y')
                else:
                    print('.')
            print('\n')
        print('----------------------\n')

    def reset(self):
        self.COLUMNS = 7
        self.ROWS = 6
        self.WINAMOUNT = 4
        self.WINDOWWIDTH = 650
        self.WINDOWHEIGHT = 530
        self.COUNTERDIAMETER = 50
        self.PLAYWIDTH = self.COLUMNS * (self.COUNTERDIAMETER + 25) + 25
        self.PLAYHEIGHT = self.ROWS * (self.COUNTERDIAMETER + 25) + 25
        self.PLAYX = 50
        self.PLAYY = 25
        self.COLGAP = (self.PLAYWIDTH - self.COLUMNS * self.COUNTERDIAMETER) / (self.COLUMNS + 1)
        self.ROWGAP = (self.PLAYHEIGHT - self.ROWS * self.COUNTERDIAMETER) / (self.ROWS + 1)
        self.pygame = False
        self.running = True
        self.playing = True
        self.state = np.zeros((self.ROWS, self.COLUMNS), dtype=int)
        self.gridCounterHeights = [self.ROWS - 1] * self.COLUMNS
        self.turn = 1

        self.action_space = spaces.Discrete(self.COLUMNS)
        self.observation_space = np.zeros((self.ROWS, self.COLUMNS, len(self.metadata['state_types'])), dtype=int)

        return self.flatten_state(self.state)

    def render(self, mode='human', close=False):
        if self.pygame is False:
            import os
            os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (20, 40)
            self.DISPLAYSURF = pygame.display.set_mode((self.WINDOWWIDTH, self.WINDOWHEIGHT))
            pygame.init()
            self.pygame = True

        self.DISPLAYSURF.fill((0, 0, 0))
        pygame.draw.rect(self.DISPLAYSURF, (0, 0, 255), (self.PLAYX, self.PLAYY, self.PLAYWIDTH, self.PLAYHEIGHT), 0)

        for row in range(self.ROWS):
            for col in range(self.COLUMNS):
                dcol = int(self.PLAYX + self.COUNTERDIAMETER * col + self.COLGAP *(col+1) + self.COUNTERDIAMETER/2)
                drow = int(self.PLAYY + self.COUNTERDIAMETER * row + self.ROWGAP * (row+1) + self.COUNTERDIAMETER/2)
                if self.state[row][col] == 0:
                    pygame.draw.circle(self.DISPLAYSURF, (255, 255, 255), (dcol,drow) ,int(self.COUNTERDIAMETER/2),0)
                elif self.state[row][col] == 1:
                    pygame.draw.circle(self.DISPLAYSURF, (255, 0, 0), (dcol,drow),int(self.COUNTERDIAMETER/2))
                else:
                    pygame.draw.circle(self.DISPLAYSURF, (255, 255, 0), (dcol,drow),int(self.COUNTERDIAMETER/2))

        self.DISPLAYSURF.blit(self.DISPLAYSURF, (0, 0))

        pygame.time.delay(1)
        pygame.display.update()

        # buttonPress=False
        # while not buttonPress:
        #     for event in pygame.event.get():
        #         if event.type == MOUSEBUTTONUP:
        #             buttonPress = True


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.pygame:
            pygame.quit()

# Notes for later
# while running:
#     mouseX, mouseY = pygame.mouse.get_pos()
#     for event in pygame.event.get():
#         if event.type == QUIT:
#             pygame.quit()
#             sys.exit()
#         if event.type == MOUSEBUTTONUP and playing:
#             for col in range(COLUMNS):
#                 refX = PLAYX + col*COUNTERDIAMETER + (col+1)*COLGAP
#                 if refX < mouseX < refX + COUNTERDIAMETER and PLAYY < mouseY < PLAYY + PLAYHEIGHT:
#                     if gridCounterHeights[col] >= 0:
#                         playGrid[gridCounterHeights[col],col] = turn
#                         winBool = checkWin(playGrid, col, gridCounterHeights[col], turn, WINAMOUNT)
#                         gridCounterHeights[col] -= 1
#                         turn = turn % 2 + 1
#                         if winBool:
#                             playing = False
#                             print("Player: " + str(turn) + " wins!")
