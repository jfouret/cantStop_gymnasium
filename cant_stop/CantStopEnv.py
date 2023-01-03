import gymnasium as gym
import numpy as np

MAX_PLAYER = 4
FIRST_PAIR_ENCODING = np.array([
    [0,1],[0,2],[0,3],[1,2],[1,3],[2,3]
  ])
SECOND_PAIR_ENCODING = np.array([
    [2,3],[1,3],[1,2],[0,3],[0,2],[0,1]
  ])

class CantStopEnv(gym.Env):
  metadata = {"render_modes": [], "render_fps": 1}
  
  def __init__(self, num_players, intermediate_reward = False):
    # number of players
    assert num_players in range(1, MAX_PLAYER + 1)
    self._num_players = num_players
    self._intermediate_reward = intermediate_reward
    
    self.observation_space = gym.spaces.Dict({
      "board": gym.spaces.Tuple((
        gym.spaces.Box(np.zeros(11), np.ones(11)),
        gym.spaces.Box(np.zeros(11), np.ones(11)),
        gym.spaces.Box(np.zeros(11), np.ones(11)),
        gym.spaces.Box(np.zeros(11), np.ones(11))
      )),
      "current_player_index": gym.spaces.Discrete(4),
      "current_player_progress": gym.spaces.Box(np.zeros(11), np.ones(11)),
      "dice_values": gym.spaces.Tuple((
        gym.spaces.Discrete(6, start = 1),
        gym.spaces.Discrete(6, start = 1),
        gym.spaces.Discrete(6, start = 1),
        gym.spaces.Discrete(6, start = 1)
      ))
    })

    self.action_space = gym.spaces.Dict({
      "pair_of_dice_chosen": gym.spaces.Discrete(6),
      "throw_dices": gym.spaces.Discrete(2)
    })
    
  def reset(self, seed = None):
    # The board represents the paths advancment for all players.
    # The first dimension represent players.
    # The second dimension represent the paths.
    # Values are between zero and one.
    self._board = (
      np.zeros(11),
      np.zeros(11),
      np.zeros(11),
      np.zeros(11)
    )
    # id of the current player
    self._current_player_index = 0
    # current phase
    # 0: The player should decide whether to play or pass 
    # 1: dices have been thrown    
    self._current_player_progress = np.zeros(11)
    np.random.seed(seed)
    self._dice_values = np.random.randint(np.ones(4), 7 + np.zeros(4))
    observation = self._get_obs()
    info = self._get_info()
    self.illegal_values = []
    return observation, info

  def _get_info(self):
    return {
      "score": (
        self._get_score(0),
        self._get_score(1),
        self._get_score(2),
        self._get_score(3)
      )
    }

  def _get_score(self, player_index):
    player_progress = self._board[player_index]
    if player_index == self._current_player_index:
      player_progress = player_progress + self._current_player_progress
    return(np.sum(np.sort(player_progress)[-3:]))

  def _get_obs(self):
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

  def step(self, action):
    player_index = self._current_player_index
    base_score = self._get_score(player_index)
    first_value = np.sum(self._dice_values[FIRST_PAIR_ENCODING[action["pair_of_dice_chosen"]]])
    if not self._is_legal_value(first_value):
      raise ValueError("first dice value is not legal")
    self._current_player_progress[first_value - 2] += 1/(12-2*abs(first_value-7))
    second_value = np.sum(self._dice_values[SECOND_PAIR_ENCODING[action["pair_of_dice_chosen"]]])
    if self._is_legal_value(second_value):
      self._current_player_progress[second_value - 2] += 1/(12-2*abs(second_value-7))
    if action["throw_dices"] == 0:
      self._board[self._current_player_index] += self._current_player_progress
      self._current_player_progress = np.zeros(11)
      self._current_player_index = (self._current_player_index + 1) % (self._num_players - 1)
    else:
      self._dice_values = np.random.randint(np.ones(4), 7 + np.zeros(4))
      if not self._has_legal_value():
        self._current_player_progress = np.zeros(11)
        self._current_player_index = (self._current_player_index + 1) % (self._num_players - 1)
    new_score = self._get_score(player_index)
    is_terminated = new_score > 2.99
    if self._intermediate_reward:
      reward = new_score - base_score
    elif is_terminated:
      reward = 1
    else:
      reward = 0
    observation = self._get_obs()
    info = self._get_info()
    return observation, reward, is_terminated, info