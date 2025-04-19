from typing import List, NamedTuple, Tuple
from mcts import MCTS, MCTSResult, WorldModel, Reasoner, SearchConfig

class GridState(NamedTuple):
    position: Tuple[int, int]
    grid: List[List[int]]


class GridAction(NamedTuple):
    direction: str  # up, down, left, right

class GridWorldModel(WorldModel[GridState, GridAction, None]):
    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0])

    def init_state(self) -> GridState:
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i][j] == 2:
                    return GridState((i, j), self.grid)
        raise ValueError("No initial position (2) found in the grid")

    def step(
        self, state: GridState, action: GridAction
    ) -> Tuple[GridState, dict]:
        x, y = state.position
        if action.direction == "up":
            new_x, new_y = x - 1, y
        elif action.direction == "down":
            new_x, new_y = x + 1, y
        elif action.direction == "left":
            new_x, new_y = x, y - 1
        elif action.direction == "right":
            new_x, new_y = x, y + 1
        else:
            raise ValueError(f"Invalid action: {action}")

        # Check for valid position in the grid world
        if (
            0 <= new_x < self.height
            and 0 <= new_y < self.width
            and state.grid[new_x][new_y] != 1
        ):
            new_position = (new_x, new_y)
        else:
            new_position = state.position

        new_state = GridState(new_position, state.grid)
        return new_state, {}

    def is_terminal(self, state: GridState) -> bool:
        return is_terminal(state)


class GridSearchConfig(SearchConfig[GridState, GridAction, None]):
    def __init__(self):
        super().__init__()

    def get_actions(self, state: GridState) -> List[GridAction]:
        model = GridWorldModel(state.grid)
        actions = [
            GridAction("up"),
            GridAction("down"),
            GridAction("left"),
            GridAction("right"),
        ]

        res = []
        for action in actions:
            new_state, _ = model.step(state, action)
            if(new_state.position != state.position):
                res.append(action)
        return res

    def reward(
        self, state: GridState, action: GridAction, **kwargs
    ) -> Tuple[float, dict]:

        # Perform the computations again because our rewards are based purely on the resulting state we end up in
        # and not a function of the action that we take
        model = GridWorldModel(state.grid)
        res_state, _ = model.step(state, action)
        if is_terminal(res_state):
            return 1.0, {}  # Reached the end goal
        else:
            return -0.01, {}  # small penalty for each step to encourage shorter path


def is_terminal(state: GridState) -> bool:
    x, y = state.position
    return state.grid[x][y] == 3


class MCTSGridWrapper(Reasoner[GridState, GridAction, None]):
    def __init__(
        self,
        grid: List[List[int]],
        n_iterations: int = 1000,
        exploration_weight: float = 1.0,
        save_interval: int = 10,
    ) -> None:
        self.grid = grid
        world_model = GridWorldModel(grid)
        search_config = GridSearchConfig()
        search_algo = MCTS(
            n_iters=n_iterations,
            w_exp=exploration_weight,
            cum_reward=sum,
            simulate_strategy="random",
            depth_limit=len(grid) * len(grid[0]),
            save_interval=save_interval,
        )
        super().__init__(world_model, search_config, search_algo)

    def __call__(self) -> MCTSResult:
        return super().__call__(example=None)

    @staticmethod
    def print_path(result: MCTSResult):
        if result.trace is None or len(result.trace) == 0:
            print("No valid path found")
            return

        states, actions = result.trace
        print("Path found: ")
        for i, (state, action) in enumerate(zip(states, actions)):
            print(f"Step{i}: Position {state.position}, Action: {action.direction}")

        print(f"Final position: {states[-1].position}")
        print(f"Cumulative reward: {result.cum_reward}")


def dfs(current_tree_node, matrix):
    if current_tree_node.state:
      cur_position = current_tree_node.state.position
      if(current_tree_node.Q > matrix[cur_position[0]][cur_position[1]]):
        matrix[cur_position[0]][cur_position[1]] = current_tree_node.Q

    if(not current_tree_node.is_terminal and current_tree_node.children):
      for child in current_tree_node.children:
          dfs(child, matrix)