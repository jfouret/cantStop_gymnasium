
import numpy as np
import gymnasium as gym
from gymnasium.spaces.utils import flatten, unflatten, flatten_space
from enum import Enum
from typing import Type, Tuple, List
import matplotlib.pyplot as plt

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

B_LOW = np.zeros(11, dtype = np.uint8)

B_HIGH = np.array([3,5,7,9,11,13,11,9,7,5,3], dtype = np.uint8)

class RewardType(Enum):
  MINUS_ONE_PER_TURN = 1

class CantStopEnv(gym.Env):
  """ 
  The CantStopEnv class represents the RL environment for the Can't stop game
  reward_type
  - negative_until_win
  - positive_at_win
  - positive_with_intermediate
  """
  def __init__(self, num_players: int, 
    reward_type: Type[RewardType] = RewardType.MINUS_ONE_PER_TURN,
    generate_info: bool = True, board_one_hot : bool = True,
    dices_one_hot : bool = True) -> None:
    """ Initialize a CantStopEnv instance
    Args:
        num_players (int): Number of players.
        reward_type (Type[RewardType], optional): Type of reward.
        generate_info (bool, optioanl): Whether to generate info.
        board_one_hot: Whether to encode the board using one-hot.
        dices_one_hot: Whether to encode the dices using one-hot
    """

    if board_one_hot:
      board_space = gym.spaces.Tuple(
        [gym.spaces.MultiDiscrete(B_HIGH, dtype = np.uint8) for _ in range(num_players)]
      )
      current_player_progress_space = \
        gym.spaces.MultiDiscrete(B_HIGH, dtype = np.uint8)
    else: 
      board_space = gym.spaces.Tuple(
        [gym.spaces.Box(B_LOW, B_HIGH, dtype = np.uint8) for _ in range(num_players)]
      )
      current_player_progress_space = \
        gym.spaces.Box(B_LOW, B_HIGH, dtype = np.uint8)
    if dices_one_hot:
      dices_space = gym.spaces.MultiDiscrete(5*np.ones(4, dtype=np.uint8), dtype = np.uint8)
    else: 
      dices_space = gym.spaces.Box(np.zeros(4, dtype=np.uint8),
        5*np.ones(4, dtype=np.uint8), dtype = np.uint8)

    # number of players
    self._generate_info = generate_info
    self._num_players = num_players
    self._reward_type = reward_type
    self.observation_space = gym.spaces.Dict({
      "board": board_space,
      "current_player_progress": current_player_progress_space,
      "dice_values": dices_space
    })
    self.action_space = gym.spaces.Dict({
      "pair_of_dice_chosen": gym.spaces.Discrete(6),
      "throw_dices": gym.spaces.Discrete(2)
    })

  def reset(self, seed: int = None) -> np.ndarray[np.float32]:
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
    self._board = np.zeros((self._num_players,11), dtype=np.uint8)
    self._turn = 0
    # id of the current player
    self._current_player_index = 0
    # current phase
    # 0: The player should decide whether to play or pass 
    # 1: dices have been thrown    
    self._current_player_progress = np.zeros(11, dtype=np.uint8)
    self.seed(seed)
    self._fig = None
    self._ax = None
    self._throw_dices()
    self._bar_width = 0.9 / self._board.shape[0]
    self._cm = plt.get_cmap('Paired', self._board.shape[0])
    self._throw_dices()
    return self._get_obs()

  def seed(self, seed: np.uint32) -> None:
    rng = np.random.RandomState(seed)
    uint_max = np.iinfo(np.uint32).max
    s1, s2, s3 = \
      (rng.randint(uint_max, dtype = np.uint32) for _ in range(3))
    self._rng = np.random.RandomState(s1)
    self.observation_space.seed(int(s2))
    self.action_space.seed(int(s3))

  def step(self, action : np.ndarray[np.float32] ) -> Tuple[np.ndarray[np.float32], float, bool, dict] :

    state = self._get_obs()
    info = self._get_info()

    player_index = self._current_player_index
    base_score = self._get_score(player_index)

    action = self._unflatten(action)

    # Check action is legal
    first_value, second_value = self._get_values_from_dices(action["pair_of_dice_chosen"])
    if not self._is_legal_value(first_value):
      raise ValueError("first dice value is not legal")

    # Advance paths
    self._current_player_progress[first_value] += 1
    if self._is_legal_value(second_value):
      self._current_player_progress[second_value] += 1

    # resolve play or pass
    if action["throw_dices"] == 0: # pass
      self._next_player(keep_progression = True)

    # anyway dices are thrown
    self._throw_dices()

    # dices are thrown until a move is possible
    while not self._has_legal_value():
      self._throw_dices()
      self._next_player()

    is_terminated = self._get_score(player_index) > 2.99
    next_state = self._get_obs()
    next_info = self._get_info()

    reward = self._get_reward(state, info, next_state, next_info, is_terminated)

    observation = self._get_obs()
    return next_state, reward, is_terminated, next_info

  def render(self, mode : str = "human") -> None:
    if mode == "human":
      new_plot = False
      if self._fig is None or self._ax is None:
        self._fig, self._ax = plt.subplots()
        new_plot = True
        
      x = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
      x_pos = (2*x - (self._board.shape[0] - 1)  * (self._bar_width)) / 2
      self._ax.clear()
      for i in range(0, self._board.shape[0]):
          # Select the ith column as the y-values for the bar plot
          y = self._board[i, ]
          # Generate the bar plot with the appropriate color from the palette
          self._ax.bar(x_pos, y, width = self._bar_width, color=self._cm(i),
            label=f"player {i}")
          x_pos = x_pos + self._bar_width
      if new_plot:
        self._fig.show()
      else:
        self._fig.canvas.draw()

  def is_action_legal(self, action: np.ndarray[np.float32]) -> bool:
    first_value, second_value = self._get_values_from_dices(
      self._unflatten["pair_of_dice_chosen"])
    return self._is_legal_value(first_value)

  def _unflatten(self, action: np.ndarray[np.float32]) -> dict:
    action = unflatten(self.action_space, action)
    if not self.action_space.contains(action):
      raise ValueError("action is not defined")
    return action

  def _get_reward(self, state : np.ndarray[np.float32], info: dict, \
    next_state: np.ndarray[np.float32], next_info: dict, \
    is_terminated : bool) -> int:
    if self._reward_type == RewardType.MINUS_ONE_PER_TURN:
      if is_terminated and info["current_player_index"] == info["current_player_index"]:
        return(1)
      else:
        return(info["turn"] - next_info["turn"])

  def _get_values_from_dices(self, pair_of_dice_chosen: int) -> Tuple[int, int]:
    return (
      np.sum(self._dice_values[FIRST_PAIR_ENCODING[pair_of_dice_chosen]]),
      np.sum(self._dice_values[SECOND_PAIR_ENCODING[pair_of_dice_chosen]])
    )

  def _throw_dices(self) -> None:
    self._dice_values = \
      self._rng.randint(np.ones(4)*5, dtype=np.uint8)

  def _next_player(self, keep_progression: bool = False) -> None:
    self._current_player_index = \
      (self._current_player_index + 1) % max((self._num_players - 1),1)
    if self._current_player_index == 0: self._turn += 1
    if keep_progression:
      self._board[self._current_player_index] += self._current_player_progress
    self._current_player_progress = np.zeros(11, dtype=np.uint8)

  def _get_high_values(self) -> List[int]:
    high_values = []
    player_progress = self._board[self._current_player_index] + \
      self._current_player_progress
    player_progress = player_progress / B_HIGH
    for i in np.argsort(player_progress)[-3:]:
      if player_progress[i] != 0:
        high_values.append(i)
    return high_values

  def _get_possible_values(self) -> dict:
    possible_values = dict()
    for i in range(6):
      first, second = self._get_values_from_dices(i)
      if self._is_legal_value(first):
        if first in possible_values.keys():
          possible_values[first].append(i)
        else:
          possible_values[first] = [i]
    return(possible_values)

  def _get_score(self, player_index: int) -> np.ndarray[np.float32]:
    """ Get score of the player. The score is the sum of the 3 best paths.

    Args:
        player_index (int): player index

    Returns:
        float: score
    """
    player_progress = self._board[player_index]
    if player_index == self._current_player_index:
      player_progress = player_progress + self._current_player_progress
    player_progress = player_progress / B_HIGH
    return(np.sum(np.sort(player_progress)[-3:]))

  def _get_info(self) -> int:
    if self._generate_info:
      return {
        "turn": self._turn,
        "possible_values": self._get_possible_values(),
        "high_values" : self._get_high_values(),
        "current_player_index": self._current_player_index,
        "score": tuple(self._get_score(i) for i in range(self._num_players))
      }
    else:
      return {
        "turn": self._turn,
        "current_player_index": self._current_player_index,
        "score": tuple(self._get_score(i) for i in range(self._num_players))
      }

  def _get_obs(self) -> np.ndarray[np.float32]:
    """Observe the env

    Returns:
        np.ndarray[np.float32]: An observation
    """ 
    idx = self._current_player_index
    N = self._num_players
    return flatten(self.observation_space, {
      "board" : tuple(self._board[(idx + i) % N] for i in range(N)),
      "current_player_progress" : self._current_player_progress,
      "dice_values" : self._dice_values
    })

  def _is_legal_value(self, value : int) -> bool:
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
    
    idx = self._current_player_index
    N = self._num_players
    max_score = np.add(self._board[idx], self._current_player_progress)[value] / B_HIGH[value]
    if self._num_players > 1:
      max_score = max(max_score, np.max(
        np.array([self._board[(idx + i) % N][value] / B_HIGH[value] for i in range(N) \
          if i != idx])
      ))
    
    return is_legal_value and max_score < 0.99

  def _has_legal_value(self) -> bool:
    has_legal_move = False
    for value in np.sum(self._dice_values[FIRST_PAIR_ENCODING], axis = 1):
      if self._is_legal_value(value):
        has_legal_move = True
        break
    return(has_legal_move)

