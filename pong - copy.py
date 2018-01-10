import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pygame
import os
import math
import time
import numpy as np
import random
import scratch
import tensorflow as tf
import pickle
import glob
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

class GameRecords():

    def __init__(self):
        self.game_record=GameRecord(observations=[], actions=[], ante_scores=[], post_scores=[])

    def append(self, gr):

        for i in range(len(gr.post_scores)):
            if gr.post_scores[i]>0:
                self.game_record.observations.append(gr.observations[i])
                self.game_record.actions.append(gr.actions[i])
                self.game_record.ante_scores.append(gr.ante_scores[i])
                self.game_record.post_scores.append(gr.post_scores[i])

    def get_observations(self):
        return self.game_record.observations

    def get_scores(self):
        return self.game_record.post_scores

    def get_actions(self):
        return self.game_record.actions

    def size(self):
        return len(self.game_records)

    def is_empty(self):
        return (self.size()==0)

    def clear(self):
        self.game_records = []

    def get_item(self, i):
        return self.game_records[i]

class GameRecord():

    def __init__(self, observations=[], actions=[], ante_scores=[], post_scores=[]):
        self.observations = observations
        self.actions = actions
        self.ante_scores = ante_scores
        self.post_scores = post_scores

    def add_move(self, observation, action, ante_score, post_score=None):
        self.observations.append(observation)
        self.actions.append(action)
        self.ante_scores.append(ante_score)
        self.post_scores.append(post_score)

    def size(self):
        return len(self.observations)

    def get_ante_scores(self):
        return self.ante_scores

    def get_actions(self):
        return self.actions

    def set_post_scores(self, ps):
        self.post_scores = ps

    def get_observations(self):
        return self.observations

class GameOfPong():

    def name(self):
        return 'Game Of Pong'

    def observation_size(self):
        return 5

    def num_one_hot_outputs(self):
        return len(self.move_to_one_hot(0))

    def initialise(self, visible):
        self.visible = visible

        if visible:
            pygame.init()
            info = pygame.display.Info()
            self.screen_width, self.screen_height = pygame.display.Info().current_w, pygame.display.Info().current_h
        else:
            self.screen_width = 3840
            self.screen_height = 2160

        self.game_width = 1000
        self.game_height = 800

        if visible:
            os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (self.screen_width,60)

            self.gameDisplay = pygame.display.set_mode((self.game_width, self.game_height))
            pygame.display.set_caption('Simple Pong')

            self.BACGROUND_COLOUR = (0, 0, 0) # black
            self.clock = pygame.time.Clock()

            self.brickImg = pygame.image.load(os.getcwd() + '\images\\brick.png')
            w, h = self.brickImg.get_size()
            self.brickImg = pygame.transform.scale(self.brickImg, (int(w * 0.1), int(h * 0.1)))

            self.brickvertImg = pygame.image.load(os.getcwd() + '\images\\brick_ver.png')
            w, h = self.brickvertImg.get_size()
            self.brickvertImg = pygame.transform.scale(self.brickvertImg, (int(w * 0.1), int(h * 0.1)))

            self.ballImg = pygame.image.load(os.getcwd() + '\images\\ball.png')
            w, h = self.ballImg.get_size()
            self.ballImg = pygame.transform.scale(self.ballImg, (int(w * 0.02), int(h * 0.02)))
            w, h = self.ballImg.get_size()

            self.ball_width = w
            self.ball_height = h

            pygame.key.set_repeat(80, 80)
        else:
            self.ball_width = 40
            self.ball_height = 40

        self.board_lext_x = 60
        self.board_right_x = 940
        self.board_top_y = 60
        self.board_bottom_y = 750

        self.bat_width = 100

    def ball(self, x, y):
        if self.visible:
            self.gameDisplay.blit(self.ballImg, (x, y))

    def brick(self, x, y, vert=False):
        if self.visible:

            if vert:
                self.gameDisplay.blit(self.brickvertImg, (x, y))
            else:
                self.gameDisplay.blit(self.brickImg, (x, y))

    def build_horizontal_wall(self, x1, x2, y):

        if self.visible:
            w, h = self.brickImg.get_size()

            for i in range(x1, x2, w):
                self.brick(i, y)

    def build_vertical_wall(self, y1, y2, x):

        if self.visible:

            w, h = self.brickvertImg.get_size()

            for i in range(y1, y2, h):
                self.brick(x, i, vert=True)

    def game_over(self):

        if self.visible:
            font = pygame.font.SysFont('Calibri', 100, True, False)
            text = font.render("GAME OVER!", True, [100, 100, 100])
            self.gameDisplay.blit(text, [220, self.board_bottom_y/2])
            pygame.display.flip()
            time.sleep(0.1)
            pygame.quit()

    def bat(self, x):

        if self.visible:
            y = self.game_height - 50
            pygame.draw.rect(self.gameDisplay, [255, 255, 255], (x, y, self.bat_width, 15))

    def build_walls(self):

        if self.visible:
            self.build_horizontal_wall(50, self.game_width - 60, 0)
            self.build_vertical_wall(30, self.game_height, 0)
            self.build_vertical_wall(30, self.game_height, self.game_width - 100)

    def initialise_board(self):
        self.ball_x = (self.board_lext_x + self.board_right_x)/2
        self.ball_y = (self.board_top_y + self.board_bottom_y)/2
        self.ball_x_change = 5
        self.ball_y_change = -5

        self.bat_x = random.randint(self.board_lext_x, self.board_right_x-self.bat_width)
        #self.bat_x = 223
        self.status_in_play = True
        self.ball_bounce_count = 0

    def update_for_boundaries(self):
        if self.bat_x < self.board_lext_x:
            self.bat_x = self.board_lext_x
        if self.bat_x > self.board_right_x-self.bat_width:
            self.bat_x = self.board_right_x-self.bat_width

        if self.ball_x < self.board_lext_x:
            self.ball_x = self.board_lext_x
            self.ball_x_change *= -1.0
        if self.ball_x > self.board_right_x-self.ball_width:
            self.ball_x = self.board_right_x-self.ball_width
            self.ball_x_change *= -1.0
        if self.ball_y < self.board_top_y:
            self.ball_y = self.board_top_y
            self.ball_y_change *= -1.0
        if self.ball_y > self.board_bottom_y-self.ball_height:
            if self.ball_x_change>0.0 and self.bat_x>self.ball_x and self.ball_x+0.5*self.ball_width>self.bat_x:
                self.ball_y = self.board_bottom_y-self.ball_height
                self.ball_y_change *= -1.0
                self.ball_x_change *= -1.0
                self.ball_bounce_count += 1
            elif self.ball_x_change<0.0 and self.bat_x+self.bat_width>self.ball_x and self.ball_x+0.6*self.ball_width>self.bat_x+self.bat_width:
                self.ball_y = self.board_bottom_y-self.ball_height
                self.ball_y_change *= -1.0
                self.ball_x_change *= -1.0
                self.ball_bounce_count += 1
            elif self.bat_x <= self.ball_x + 0.6 * self.ball_width and self.bat_x + self.bat_width - 20.0 >= self.ball_x:
                self.ball_y = self.board_bottom_y - self.ball_height
                self.ball_y_change *= -1.0
                self.ball_bounce_count += 1
            else:
                return False

        return True

    def state(self):

        # ball position
        ball_x = self.ball_x
        ball_y = self.ball_y

        # ball motion
        ball_x_change = self.ball_x_change
        ball_y_change = self.ball_y_change

        # bat position
        bat_x = self.bat_x

        # number of ball bounces so far
        ball_bounce_count = self.ball_bounce_count

        return [ball_x, ball_y, ball_x_change, ball_y_change, bat_x]

    def move_to_one_hot(self, m):

        if m>0:
            return [0, 0, 1]
        if m<0:
            return [1, 0, 0]

        return [0, 1, 0]

    def post_process_game_record(self, gr):
        # self.observations = []
        # self.actions = []
        # self.ante_scores = []
        # self.post_scores = []

        final_score = gr.get_ante_scores()[-1]
        if final_score == 0:
            return GameRecord(observations=[], actions=[], ante_scores=[], post_scores=[])

        observations = gr.get_observations()
        actions = gr.get_actions()
        ante_scores = gr.get_ante_scores()
        nobs = gr.size()

        proc_observations = []
        proc_actions = []
        proc_ante_scores = []
        proc_post_scores = []

        for i in range(1, nobs):
            if ante_scores[i]!=ante_scores[i-1]:
                if ante_scores[i-1] == final_score:
                    break
                else:
                    window_size = 10

                    if len(proc_observations)==0:
                        proc_observations = observations[i-window_size:i]
                        proc_actions = actions[i-window_size:i]
                        proc_ante_scores = ante_scores[i-window_size:i]
                        ws = len(ante_scores[i-window_size:i])
                        proc_post_scores = ws * [1]
                    else:
                        for j in range(i-window_size,i):
                            proc_observations.append(observations[j])
                            proc_actions.append(actions[j])
                            proc_ante_scores.append(ante_scores[j])
                            proc_post_scores.append(1)

        rec = GameRecord(proc_observations, proc_actions, proc_ante_scores, proc_post_scores)
        return rec

    def play(self, visible=False, record=False, player=None, play_from_recording=False, recording=None ):

        self.initialise(visible=visible)

        gr = GameRecord(observations=[], actions=[], ante_scores=[], post_scores=[])

        self.initialise_board()
        rec_index = 0

        while self.status_in_play:

            x_change = 0

            if play_from_recording is False:

                if player is not None:

                    x_change = player.getMove(self.state())

                elif self.visible and player is None:

                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.status_in_play = False

                        if event.type == pygame.KEYDOWN:

                            move_scale = 45.0
                            if event.key == pygame.K_LEFT:
                                x_change = -move_scale
                            if event.key == pygame.K_RIGHT:
                                x_change = move_scale
                else:
                    throw('Not expected this')

                self.bat_x = self.bat_x + x_change

                if record:
                    b = self.state()
                    moh = self.move_to_one_hot(x_change)
                    gr.add_move(observation=b, action=moh, ante_score=self.ball_bounce_count)

            else:
                recorded_observation = recording.get_observations()[rec_index]
                [self.ball_x, self.ball_y, self.ball_x_change, self.ball_y_change, self.bat_x] = recorded_observation
                rec_index += 1

            self.ball_x = self.ball_x + self.ball_x_change
            self.ball_y = self.ball_y + self.ball_y_change

            self.status_in_play = self.update_for_boundaries()

            if self.visible:
                self.gameDisplay.fill(self.BACGROUND_COLOUR)
                self.build_walls()

                self.ball(self.ball_x,self.ball_y)
                self.bat(self.bat_x)

            if self.status_in_play==False:
                self.game_over()
            elif self.visible:
                pygame.display.flip()
                self.clock.tick(60)

        if record is True:
            return self.post_process_game_record(gr), self.ball_bounce_count

        return None, self.ball_bounce_count

class RandomPlayer():

    def getMove(self, observation):
        return random.choice([-45, 45])

class TrainedPlayer():

    def __init__(self, name, game):

        self.name = name
        self.game_name = game.name

        tf.reset_default_graph()

        # The network
        self.nb_inputs = game.observation_size()
        self.nb_outputs = game.num_one_hot_outputs()

        self.model = Sequential()
        self.model.add(Dense(512, input_dim=self.nb_inputs))
        self.model.add(Activation('tanh'))
        self.model.add(Dense(512))
        self.model.add(Activation('tanh'))
        self.model.add(Dense(512))
        self.model.add(Activation('tanh'))
        self.model.add(Dense(self.nb_outputs))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam')

        self.trained = False

    def getMove(self, observation):
        if not self.trained:
            return random.choice([-45, 45])

        action_from_model = self.model.predict_proba(np.reshape(observation, (1,-1)))[0]
        pos = np.argmax(action_from_model)

        if pos==0:
            return -45
        if pos==2:
            return 45

        return 0

class Training():

    def __init__(self, game):
        self.game = game
        self.player = TrainedPlayer(name='Pavlos', game=self.game)
        self.trainingRecords = []

    def collectTrainingGames(self, max_score, numGames=100 ):

        self.trainingRecords = GameRecords()

        for i in range(numGames):
            score = 0

            while score < max_score:
                gr, score = self.game.play(visible=False, record=True, player=self.player)

            self.trainingRecords.append(gr)

    def getTrainingGames(self):
        return self.trainingRecords

    def train(self):

        for i in range(10):
            print('------------------------------------- Training: ' + str(i) )
            self.collectTrainingGames(max_score=1, numGames=100)

            # load the training data

            observations = self.trainingRecords.get_observations()
            target_actions = self.trainingRecords.get_actions()
            # pickle.dump(observations, open("observations.p", "wb"))
            # pickle.dump(target_actions, open("target_actions.p", "wb"))

            self.player.model.fit(observations, target_actions, batch_size=500, epochs=1000, verbose=2)

            self.player.trained = True
            self.game.play(visible=True, player=self.player)
            self.player.trained = False

    def get_player(self):
        return self.player

if __name__ == "__main__":

    random.seed(1)

    # g = GameOfPong()
    #
    # t = Training(game=g)
    # t.train()

    #g.play(visible=True, player=t.get_player())
    scratch.try_gym()

