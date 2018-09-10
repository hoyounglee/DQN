import gym
import logging
import numpy as np
from PIL import Image

class Environment():
    def __init__(self, name, window_size, input_shape, display=False):
        self.name = name
        self.display = display
        self.window_size = window_size
        self.input_shape = input_shape
        self.env = gym.make(name)
        self.num_action = self.env.action_space
        self.width = input_shape[0]
        self.height = input_shape[1]
        self.env.close()
        self.preprocessor = Preprocess(self.window_size, self.input_shape)
        self.reset()
        self.lives = self.env.unwrapped.ale.lives()

    def reset(self):
        self.env.reset()

        if self.display == True:
            self.env.render()

        self.env.step(0)
        state,_,_,_  = self.env.step(0)

        p_state=self.preprocessor.preprocess(state)

        return p_state


    def step(self, action, Training = True):

        state, reward, terminal, _ = self.env.step(action)

        p_state = self.preprocessor.preprocess(state)

        current_lives = self.env.unwrapped.ale.lives()

        if Training == True and self.lives > current_lives:
            terminal = True

        if self.display == True:
            self.env.render()



        return p_state, reward, terminal



class Preprocess:
    def __init__(self, window_size, input_shape):
        self.buff_size = window_size
        self.resize_w = input_shape[0]
        self.resize_h = input_shape[1]


    def update_state_buffer(self, state):
        pass

    def preprocess(self, state):
        image = Image.fromarray(state)
        image = image.convert('L')
        image = image.resize((self.resize_w, self.resize_h), Image.ANTIALIAS)
        #image.show()
        image = np.array(image, dtype=np.uint8)
        return image