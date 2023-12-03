import numpy as np
import tensorflow as tf
import copy

# Define the Go board size
board_size = 9

# Neural Network Model
def create_policy_value_network():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=(board_size, board_size, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(board_size * board_size, activation='softmax'),  # Policy output
        tf.keras.layers.Dense(1, activation='tanh')  # Value output
    ])
    return model

# Game State Representation
class GameState:
    def __init__(self):
        self.board = np.zeros((board_size, board_size))
        self.current_player = 1  # 1 for black, -1 for white
        self.previous_state = None
        self.last_move = None

    def play_move(self, move):
        new_state = copy.deepcopy(self)
        new_state.board[move[0]][move[1]] = self.current_player
        new_state.current_player *= -1
        new_state.previous_state = self
        new_state.last_move = move
        return new_state

# Monte Carlo Tree Search
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def expand(self):
        legal_moves = [(i, j) for i in range(board_size) for j in range(board_size) if self.state.board[i][j] == 0]
        for move in legal_moves:
            new_state = self.state.play_move(move)
            new_node = MCTSNode(new_state, parent=self)
            self.children.append(new_node)

    def select(self, exploration_weight=1.0):
        return max(self.children, key=lambda node: node.value / (node.visits + 1e-5)
                   + exploration_weight * np.sqrt(np.log(self.visits + 1) / (node.visits + 1e-5)))

    def rollout(self):
        current_state = copy.deepcopy(self.state)
        while not self.is_terminal(current_state):
            legal_moves = [(i, j) for i in range(board_size) for j in range(board_size) if current_state.board[i][j] == 0]
            move = random.choice(legal_moves)
            current_state = current_state.play_move(move)
        return self.compute_value(current_state)

    def compute_value(self, state):
        # Simple heuristic: count the number of stones on the board
        return np.sum(state.board)

    def backpropagate(self, value):
        self.visits += 1
        self.value += value
        if self.parent:
            self.parent.backpropagate(value)

    def is_terminal(self, state):
        # Simple terminal condition: the game ends when there are no legal moves left
        return len([(i, j) for i in range(board_size) for j in range(board_size) if state.board[i][j] == 0]) == 0

# AlphaGo
class AlphaGo:
    def __init__(self):
        self.neural_network = create_policy_value_network()

    def get_action(self, state):
        root = MCTSNode(state)
        for _ in range(1000):  # Perform 1000 MCTS simulations
            node = root
            while not node.is_terminal(node.state):
                if len(node.children) == 0 or random.uniform(0, 1) < 0.2:
                    node.expand()
                    break
                node = node.select()
            value = node.rollout()
            node.backpropagate(value)
        best_child = max(root.children, key=lambda node: node.visits)
        return best_child.state.last_move

# Example Usage
if __name__ == "__main__":
    alpha_go = AlphaGo()
    game_state = GameState()

    for _ in range(50):  # Play 50 moves as an example
        action = alpha_go.get_action(game_state)
        game_state = game_state.play_move(action)
        print(f"Player {game_state.current_player} plays move {action}")
        print(game_state.board)
