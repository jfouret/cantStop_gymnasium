
import numpy as np
from tensorflow import keras
from keras import Model, Sequential, Input
from keras.layers import Dense, Embedding, Reshape
from keras.optimizers import Adam

import random
from IPython.display import clear_output
from collections import deque
import progressbar

MAX_PLAYER = 4

FIRST_PAIR_ENCODING = np.array([
    [0,1],[0,2],[0,3],[1,2],[1,3],[2,3]
])

SECOND_PAIR_ENCODING = np.array([
    [2,3],[1,3],[1,2],[0,3],[0,2],[0,1]
])

ACTION_ENCODING = np.array([
  [0,0],[1,0],[2,0],[3,0],[4,0],[5,0],
  [0,1],[1,1],[2,1],[3,1],[4,1],[5,1]
])

class CantStopAgent():
  """ The agent for cant stop
  """
  def __init__(self, optimizer):
    # Initialize atributes
    self._state_size = (MAX_PLAYER + 1) * 11 + 4
    self._action_size = 12
    self._optimizer = optimizer
    
    self.expirience_replay = deque(maxlen=256)
    
    # Initialize discount and exploration rate
    self.gamma = 0.6
    self.epsilon = 0.1
    
    # Build networks
    self.q_network = self._build_compile_model()
    self.target_network = self._build_compile_model()
    self.alighn_target_model()
    
  def store(self, state, action, reward, next_state, terminated):
    self.expirience_replay.append((state, action, reward, next_state, terminated))
  
  def _build_compile_model(self):
    model = Sequential()
    model.add(Embedding(self._state_size, 10, input_length=1))
    model.add(Reshape((10,)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(self._action_size, activation='linear'))
    
    model.compile(loss='mse', optimizer=self._optimizer)
    return model
  
  def alighn_target_model(self):
    self.target_network.set_weights(self.q_network.get_weights())

  def encode_state(self, state):
    enc = np.zeros((self._state_size))
    for i in range(0, MAX_PLAYER):
      enc[11*i:11*(i+1)] = state["board"][i]
    enc[MAX_PLAYER*11:(MAX_PLAYER+1)*11] = state["current_player_progress"]
    enc[(MAX_PLAYER+1)*11:(MAX_PLAYER+1)*11 + 4] = state["dice_values"]
    return(enc)

  def is_legal_action(self, env, action):
    action = self.decode_action(action)
    first_value = np.sum(env._dice_values[FIRST_PAIR_ENCODING[action["pair_of_dice_chosen"]]])
    return(env._is_legal_value(first_value))

  def act(self, state, env, epsilon):
    if np.random.rand() <= epsilon:
      actions = np.arange(0, self._action_size)
      while True:
        action = np.random.choice(actions)
        if self.is_legal_action(env, action):
          return action
        actions = np.delete(actions, np.argwhere(actions == action))
    rank = 0
    q_values = self.q_network.predict(state, verbose = 0)
    while True:
      action = np.argsort(q_values[0])[::-1][rank]
      if self.is_legal_action(env, action):
          return action
      rank += 1
    
  def decode_action(self, action):
    enc = ACTION_ENCODING[action]
    return {
      "pair_of_dice_chosen": enc[0],
      "throw_dices": enc[1]
    }

  def retrain(self, batch_size):
    minibatch = random.sample(self.expirience_replay, batch_size)
        
    for state, action, reward, next_state, terminated in minibatch:
            
      target = self.q_network.predict(state, verbose = 0)
            
      if terminated:
        target[0][action] = reward
      else:
        t = self.target_network.predict(next_state, verbose = 0)
        target[0][action] = reward + self.gamma * np.amax(t)
            
    self.q_network.fit(state, target, epochs=1, verbose=0)

class CantStopEnv():
  """ 
  The CantStopEnv class represents the RL environment for the Can't stop game
  reward_type
  - negative_until_win
  - positive_at_win
  - positive_with_intermediate
  
  """    
  def __init__(self, num_players, reward_type):    
    """ Initialize a CantStopEnv instance

    Args:
        num_players (int): Number of players (Maximum 4)
        intermediate_reward (bool, optional): _description_. Defaults to False.
    """
    # number of players
    assert num_players in range(1, MAX_PLAYER + 1)
    self._num_players = num_players
    self.reward_type = reward_type
    
  def reset(self, seed = None):
    """ Reset the environment

    Args:
        seed (int, optional): random seed. Defaults to None.

    Returns:
        CantStopObservation: An observation
    """    
    # The board represents the paths advancment for all players.
    # The first dimension represent players.
    # The second dimension represent the paths.
    # Values are between zero and one.
    self._board = np.zeros((MAX_PLAYER,11))
    # id of the current player
    self._current_player_index = 0
    # current phase
    # 0: The player should decide whether to play or pass 
    # 1: dices have been thrown    
    self._current_player_progress = np.zeros(11)
    np.random.seed(seed)
    self._dice_values = np.random.randint(np.ones(4), 7 + np.zeros(4))
    observation = self._get_obs()
    self.illegal_values = []
    return observation

  def step(self, action):
    reward = 0
    player_index = self._current_player_index
    base_score = self._get_score(player_index)
    """
  - negative_until_win
  - positive_at_win
  - positive_with_intermediate
    """
    first_value = np.sum(self._dice_values[FIRST_PAIR_ENCODING[action["pair_of_dice_chosen"]]])
    if not self._is_legal_value(first_value):
      raise ValueError("first dice value is not legal")
    self._current_player_progress[first_value - 2] += 1/(13-2*abs(first_value-7))
    second_value = np.sum(self._dice_values[SECOND_PAIR_ENCODING[action["pair_of_dice_chosen"]]])
    
    if self._is_legal_value(second_value):
      self._current_player_progress[second_value - 2] += 1/(13-2*abs(second_value-7))
      
    if action["throw_dices"] == 0:
      self._board[self._current_player_index] += self._current_player_progress
      self._current_player_progress = np.zeros(11)
      self._current_player_index = (self._current_player_index + 1) % max((self._num_players - 1),1)
      if self.reward_type == "negative_until_win":
          reward = - 1
      self._dice_values = np.random.randint(np.ones(4), 7 + np.zeros(4))
    else:
      self._dice_values = np.random.randint(np.ones(4), 7 + np.zeros(4))
      if not self._has_legal_value():
        if self.reward_type == "negative_until_win":
          reward = - 1
        self._current_player_progress = np.zeros(11)
        self._current_player_index = (self._current_player_index + 1) % max((self._num_players - 1),1)
        
    while not self._has_legal_value():
      self._dice_values = np.random.randint(np.ones(4), 7 + np.zeros(4))
      self._current_player_index = (self._current_player_index + 1) % max((self._num_players - 1),1)
      if self._num_players == 1 and self.reward_type == "negative_until_win":
        reward -= 1
    
    new_score = self._get_score(player_index)
    is_terminated = new_score > 2.99
    if self.reward_type == "positive_with_intermediate":
      reward = new_score - base_score
    elif self.reward_type == "positive_at_win" and is_terminated:
      reward = 1

    observation = self._get_obs()
    return observation, reward, is_terminated

  def _get_score(self, player_index):
    """ Get score of the player. The score is the sum of the 3 best paths.

    Args:
        player_index (int): player index

    Returns:
        float: score
    """    
    player_progress = self._board[player_index]
    if player_index == self._current_player_index:
      player_progress = player_progress + self._current_player_progress
    return(np.sum(np.sort(player_progress)[-3:]))

  def _get_obs(self):
    """Observe the env

    Returns:
        CantStopObservation: An observation
    """    
    return {
      "board" : (
        self._board[self._current_player_index],
        self._board[(self._current_player_index + 1) % MAX_PLAYER],
        self._board[(self._current_player_index + 2) % MAX_PLAYER],
        self._board[(self._current_player_index + 3) % MAX_PLAYER]
      ),
      "current_player_index" : self._current_player_index,
      "current_player_progress" : self._current_player_progress,
      "dice_values" : self._dice_values
    }

  def _is_legal_value(self, value):
    """Check if a value is legal to play in the current situation

    Args:
        value (int): The value to check

    Returns:
        _type_: _description_
    """    
    is_legal_value = True
    used_values = np.where(self._current_player_progress > 0)[0] + 2
    if np.shape(used_values)[0] == 3:
      is_legal_value = value in used_values
    
    max_progress = max(
      (self._board[self._current_player_index] + self._current_player_progress)[value - 2],
      self._board[(self._current_player_index + 1) % MAX_PLAYER][value - 2],
      self._board[(self._current_player_index + 2) % MAX_PLAYER][value - 2],
      self._board[(self._current_player_index + 3) % MAX_PLAYER][value - 2]
    )
    
    return (is_legal_value and max_progress < 0.99)

  def _has_legal_value(self):
    has_legal_move = False
    for value in np.sum(self._dice_values[FIRST_PAIR_ENCODING], axis = 1):
      if self._is_legal_value(value):
        has_legal_move = True
        break
    return(has_legal_move)

