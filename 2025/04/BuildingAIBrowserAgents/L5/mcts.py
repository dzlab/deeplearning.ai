from abc import ABC, abstractmethod
from typing import Generic, Protocol, Tuple, TypeVar, Union, runtime_checkable
import itertools
import math
from typing import Callable, Hashable, NamedTuple, Optional
import numpy as np
from tqdm import trange

#@title Abstract World Model
# Abstract Classes that define a model of how the environment is set up

State = TypeVar("State")
Action = TypeVar("Action")
Example = TypeVar("Example")
Trace = tuple[list[State], list[Action]]

class WorldModel(ABC, Generic[State, Action, Example]):
    def __init__(self) -> None:
        self.example = None
        self.prompt = None

    @abstractmethod
    def init_state(self) -> State: ...

    @abstractmethod
    def step(
        self, state: State, action: Action
    ) -> Union[State, Tuple[State, dict]]:
        """Returns the next state and optionally an auxiliary data dict

        :param state: The current state
        :param action: The action to take
        :return: The next state and optionally an auxiliary data dict
        """
        ...

    @abstractmethod
    def is_terminal(self, state: State) -> bool: ...

    def update_example(self, example: Example, prompt=None) -> None:
        if prompt is not None:
            self.prompt = prompt
        self.example = example


class SearchConfig(ABC, Generic[State, Action, Example]):
    def __init__(self) -> None:
        self.example = None
        self.prompt = None

    @abstractmethod
    def get_actions(self, state: State) -> list[Action]: ...

    def fast_reward(self, state: State, action: Action) -> tuple[float, dict]:
        return 0, {}

    @abstractmethod
    def reward(self, state, action, **kwargs) -> tuple[float, dict]: ...

    def update_example(self, example: Example, prompt=None) -> None:
        if prompt is not None:
            self.prompt = prompt
        self.example = example


@runtime_checkable
class AlgorithmOutput(Protocol[State]):
    terminal_state: State
    trace: Trace


class SearchAlgorithm(ABC):
    def __init__(self, **kwargs): ...

    @abstractmethod
    def __call__(
        self, world_model: WorldModel, search_config: SearchConfig, **kwargs
    ) -> AlgorithmOutput: ...

# Umbrella class that brings together the model of the world and the parameters of the search
class Reasoner(ABC, Generic[State, Action, Example]):
    def __init__(
        self,
        world_model: WorldModel[State, Action, Example],
        search_config: SearchConfig[State, Action, Example],
        search_algo: SearchAlgorithm,
    ) -> None:
        self.world_model = world_model
        self.search_config = search_config
        self.search_algo = search_algo

    def __call__(
        self, example: Example, prompt=None, **kwargs
    ) -> AlgorithmOutput[State]:
        self.world_model.update_example(example, prompt=prompt)
        self.search_config.update_example(example, prompt=prompt)
        return self.search_algo(self.world_model, self.search_config, **kwargs)

#@title MCTS Node Setup
# The information that the search tree stores at every node
class MCTSNode(Generic[State, Action, Example]):
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(
        self,
        state: Optional[State],
        action: Optional[Action],
        parent = None,
        fast_reward: float = 0.0,
        fast_reward_details=None,
        is_terminal: bool = False,
        calc_q: Callable[[list[float]], float] = np.mean,
    ):
        """
        A node in the MCTS search tree

        :param state: the current state
        :param action: the action of the last step, i.e., the action from parent node to current node
        :param parent: the parent node, None if root of the tree
        :param fast_reward: an estimation of the reward of the last step
        :param is_terminal: whether the current state is a terminal state
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        """
        self.id = next(MCTSNode.id_iter)
        if fast_reward_details is None:
            fast_reward_details = {}
        self.cum_rewards: list[float] = []
        self.fast_reward = self.reward = fast_reward
        self.fast_reward_details = fast_reward_details
        self.is_terminal = is_terminal
        self.action = action
        self.state = state
        self.parent = parent
        self.children = None
        self.calc_q = calc_q
        self.N = 0  # Visit count
        self._Q = 0  # Reward
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    def __str__(self):
        return f"MCTSNode(id={self.id}, state={self.state}, action={self.action}, reward={self.reward}, is_terminal={self.is_terminal})"

    @property
    def Q(self) -> float:
        if self.N == 0:
            return 0
        return self._Q  # Getter

    @Q.setter
    def Q(self, value: float):
        self._Q = value  # Setter


class MCTSResult(NamedTuple):
    terminal_state: State
    cum_reward: float
    trace: Trace
    trace_of_nodes: list[MCTSNode]
    tree_state: MCTSNode
    trace_in_each_iter: list[list[MCTSNode]] = None
    tree_state_after_each_iter: list[MCTSNode] = None
    aggregated_result: Optional[Hashable] = None


class MCTS(SearchAlgorithm, Generic[State, Action, Example]):
    def __init__(
        self,
        w_exp: float = 1.0,
        depth_limit: int = 5,
        n_iters: int = 10,
        cum_reward: Callable[[list[float]], float] = sum,
        calc_q: Callable[[list[float]], float] = np.mean,
        simulate_strategy: str | Callable[[list[float]], int] = "random",
        uct_with_fast_reward: bool = True,
        save_interval: int = 10,
    ):
        """
        MCTS algorithm
        :param w_exp: the weight of exploration in UCT
        :param cum_reward: the way to calculate the cumulative reward from each step. Defaults: sum
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        :param simulate_strategy: simulate strategy. Options: 'max', 'sample', 'random', or use a custom function
        :param uct_with_fast_reward: if True, use fast_reward instead of reward for unvisited children in UCT
                                     Otherwise, visit the *unvisited* children with maximum fast_reward first
        """
        super().__init__()
        self.world_model = None
        self.search_config = None
        self.w_exp = w_exp
        self.depth_limit = depth_limit
        self.n_iters = n_iters
        self.cum_reward = cum_reward
        self.calc_q = calc_q
        self.result_list = []
        default_simulate_strategies: dict[str, Callable[[list[float]], int]] = {
            "max": lambda x: np.argmax(x),
            "sample": lambda x: np.random.choice(len(x), p=x),
            "random": lambda x: np.random.choice(len(x)),
        }
        self.simulate_choice: Callable[[list[float]], int] = (
            default_simulate_strategies.get(simulate_strategy, simulate_strategy)
        )
        self.uct_with_fast_reward = uct_with_fast_reward
        self._output_iter: list[MCTSNode] = None
        self._output_cum_reward = -math.inf
        self.trace_in_each_iter: list[list[MCTSNode]] = None
        self.root: Optional[MCTSNode[State, Action, Example]] = None
        self.save_interval = save_interval

    def iterate(self, node: MCTSNode) -> list[MCTSNode]:
        # Choose a path to follow in the current MCTSTree until you reach
        # a leaf node in the tree
        path = self._select(node)

        if not self._is_terminal_with_depth_limit(path[-1]):
            self._expand(path[-1])
            self._simulate(path)
        cum_reward = self._back_propagate(path)
        return path

    def _is_terminal_with_depth_limit(self, node: MCTSNode):
        return node.is_terminal or node.depth >= self.depth_limit

    def _select(self, node: MCTSNode) -> list[MCTSNode]:
        path = []
        while True:
            path.append(node)
            if (
                node.children is None
                or len(node.children) == 0
                or self._is_terminal_with_depth_limit(node)
            ):
                return path
            node = self._uct_select(node)
            self.world_model.step(node.parent.state, node.action)

    def _uct(self, node: MCTSNode) -> float:
        return node.Q + self.w_exp * math.sqrt(math.log(node.parent.N) / (1 + node.N))

    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        # First, check for unvisited nodes
        for child in node.children:
            if child.N == 0:
                return child

        # If all nodes have been visited, use the UCB1 formula
        return max(node.children, key=self._uct)

    def _expand(self, node: MCTSNode):
        if node.state is None:
            node.state, aux = self.world_model.step(
                node.parent.state, node.action
            )
            # reward is calculated after the state is updated, so that the
            # information can be cached and passed from the world model
            # to the reward function with **aux without repetitive computation

            node.reward, node.reward_details = self.search_config.reward(
                node.parent.state, node.action, **node.fast_reward_details, **aux
            )

            node.is_terminal = self.world_model.is_terminal(node.state)

        if node.is_terminal:
            return

        children = []
        actions = self.search_config.get_actions(node.state)

        for action in actions:
            fast_reward, fast_reward_details = self.search_config.fast_reward(
                node.state, action
            )
            child = MCTSNode(
                state=None,
                action=action,
                parent=node,
                fast_reward=fast_reward,
                fast_reward_details=fast_reward_details,
                calc_q=self.calc_q,
            )
            children.append(child)

        node.children = children

    def _simulate(self, path: list[MCTSNode]):
        node = path[-1]
        while True:
            if node.state is None:
                self._expand(node)
            if self._is_terminal_with_depth_limit(node) or len(node.children) == 0:
                return
            fast_rewards = [child.fast_reward for child in node.children]
            node = node.children[self.simulate_choice(fast_rewards)]
            path.append(node)

    def _back_propagate(self, path: list[MCTSNode]):
        cum_reward = 0
        reward = path[-1].reward
        for node in reversed(path):
            cum_reward += node.reward
            node.Q = (node.Q * node.N + cum_reward) / (node.N + 1)
            node.N += 1
        return path[0].Q  # Return the root node's updated Q-value

    def _dfs_max_reward(self, path: list[MCTSNode]) -> tuple[float, list[MCTSNode]]:
        cur = path[-1]
        if cur.is_terminal:
            return self.cum_reward([node.reward for node in path[1:]]), path
        if cur.children is None:
            return -math.inf, path
        visited_children = [x for x in cur.children if x.state is not None]
        if len(visited_children) == 0:
            return -math.inf, path
        return max(
            (self._dfs_max_reward(path + [child]) for child in visited_children),
            key=lambda x: x[0],
        )

    def search(self):
        self._output_cum_reward = -math.inf
        self._output_iter = None
        self.root = MCTSNode(
            state=self.world_model.init_state(),
            action=None,
            parent=None,
            calc_q=self.calc_q,
        )

        for iter in trange(
            self.n_iters, disable=True, desc="MCTS iteration", leave=False
        ):
            path = self.iterate(self.root)
            if(iter % self.save_interval == 0):
                self.result_list.append(self.return_result())
                self._output_cum_reward, self._output_iter = self._dfs_max_reward(
                    [self.root]
                )
                if self._output_cum_reward == -math.inf:
                    self._output_iter = None

    def return_result(self):
      if self._output_iter is None:
          terminal_state = trace = None
      else:
          terminal_state = self._output_iter[-1].state
          trace = (
              [node.state for node in self._output_iter],
              [node.action for node in self._output_iter[1:]],
          )

      trace_in_each_iter = tree_state_after_each_iter = None
      result = MCTSResult(
          terminal_state=terminal_state,
          cum_reward=self._output_cum_reward,
          trace=trace,
          trace_of_nodes=self._output_iter,
          tree_state=self.root,
          trace_in_each_iter=trace_in_each_iter,
          tree_state_after_each_iter=tree_state_after_each_iter,
      )
      return result
        
    def __call__(
        self,
        world_model: WorldModel[State, Action, Example],
        search_config: SearchConfig[State, Action, Example],
        log_file: Optional[str] = None,
        **kwargs,
    ) -> list[MCTSResult]:
        MCTSNode.reset_id()
        self.world_model = world_model
        self.search_config = search_config
        self.search()
        self.result_list.append(self.return_result())
        return self.result_list
