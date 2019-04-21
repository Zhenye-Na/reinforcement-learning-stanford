import numpy as np

class ActionSpace(object):
    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.randint(0, self.n)


class ObservationSpace(object):
    def __init__(self, shape):
        self.shape = shape
        self.bad_state = np.random.randint(0, 50, shape, dtype=np.uint8)
        self.normal_state = np.random.randint(100, 150, shape, dtype=np.uint8)
        self.good_state = np.random.randint(200, 250, shape, dtype=np.uint8)
        self.states = [self.bad_state, self.normal_state, self.good_state]


class EnvTest(object):
    """
    Adapted from Igor Gitman, CMU / Karan Goel
    """
    def __init__(self, shape=(84, 84, 3)):
        #3 states
        self.rewards = [-0.1, 0, 0.1]
        self.cur_state = 0
        self.num_iters = 0
        self.was_in_second = False
        self.action_space = ActionSpace(4)
        self.observation_space = ObservationSpace(shape)
        

    def reset(self):
        self.cur_state = 0
        self.num_iters = 0
        self.was_in_second = False
        return self.observation_space.states[self.cur_state]
        

    def step(self, action):
        assert(0 <= action <= 3)
        self.num_iters += 1
        if action < 3:
            self.cur_state = action
        reward = self.rewards[self.cur_state]
        if self.was_in_second is True:
            reward *= -10
        if self.cur_state == 1:
            self.was_in_second = True
        else:
            self.was_in_second = False
        return self.observation_space.states[self.cur_state], reward, self.num_iters >= 5, {'ale.lives':0}


    def render(self):
        print(self.cur_state)