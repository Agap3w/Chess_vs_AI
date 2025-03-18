import numpy as np 
import chess
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from reinf_encodeMove import move_to_index, index_to_move

class ChessGame:
    """ basic chess engine """
    
    def __init__(self):
        pass

    def get_initial_state(self):
        return chess.Board()

    def get_next_state(self, state, action):
        next_state = state.copy()
        next_state.push(action)
        return next_state
    
    def get_valid_moves(self, state):
        return list(state.legal_moves)
        
    def get_value_and_terminated(self, state):
        if not state.is_game_over():
            return 0, False
        if state.is_checkmate():
            return -1, True
        return 0, True

    def get_opponent_value(self, value):
        return -value

    @staticmethod
    def board_to_tensor(state):
        """
        Convert a chess.Board to a tensor representation with 14 planes:
        - 12 planes for piece positions (6 piece types × 2 colors)
        - 1 plane for side to move
        - 1 plane for castling rights for current player
        
        
        Returns:
            tensor: An 8×8×14 numpy array
        """
        # Initialize the planes
        planes = np.zeros((14, 8, 8), dtype=np.float32)
        
        # Piece planes (0-11): 6 piece types × 2 colors
        # Piece type order: pawn, knight, bishop, rook, queen, king
        # Color order: white, black
        piece_plane_map = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }
        
        # Fill piece planes
        for square in chess.SQUARES:
            piece = state.piece_at(square)
            if piece is not None:
                color_offset = 0 if piece.color == chess.WHITE else 6
                piece_plane = piece_plane_map[piece.piece_type] + color_offset
                
                # Convert chess square to tensor indices
                rank = 7 - chess.square_rank(square)  # Flip rank for standard orientation
                file = chess.square_file(square)
                
                planes[piece_plane, rank, file] = 1
        
        # Side to move plane (12)
        planes[12, :, :] = float(state.turn)

        # Current player's castling rights
        if state.turn == chess.WHITE:
            # White king's position
            planes[13, 7, 4] = 1
            # White kingside rook if kingside castling is available
            if state.has_kingside_castling_rights(chess.WHITE):
                planes[13, 7, 7] = 1
            # White queenside rook if queenside castling is available
            if state.has_queenside_castling_rights(chess.WHITE):
                planes[13, 7, 0] = 1
        else:  # Black's turn
            # Black king's position
            planes[13, 0, 4] = 1
            # Black kingside rook if kingside castling is available
            if state.has_kingside_castling_rights(chess.BLACK):
                planes[13, 0, 7] = 1
            # Black queenside rook if queenside castling is available
            if state.has_queenside_castling_rights(chess.BLACK):
                planes[13, 0, 0] = 1
        
        return planes

class ResNet(nn.Module):
    def __init__(self, device, channel=14, num_ResBlock=8, num_hidden=128):
        super().__init__()
        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(channel, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_ResBlock)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*8*8, 4672)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(channel*8*8, 1),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value

    @torch.no_grad()         
    def predict(self, state):
        """
        Predict policy and value for a given board state.
        
        Args:
            state: A chess.Board object
        
        Returns:
            tuple: (policy, value) where policy is a numpy array of shape (4096,)
                  and value is a float in the range [-1, 1]
        """
        # Convert board to planes
        planes = ChessGame.board_to_tensor(state)
        
        # Add batch dimension and convert to tensor
        x = torch.tensor(planes, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Forward pass
        policy_logits, value = self.forward(x)
        
        # Convert policy logits to probabilities
        policy = F.softmax(policy_logits, dim=1).detach().cpu().numpy()[0]
        
        # Get value as scalar
        value = value.item()
        
        return policy, value
    
    @torch.no_grad()         
    def batch_predict(self, states):
        """
        Make batch predictions for multiple states at once.
        
        Args:
            states: List of chess.Board objects
            
        Returns:
            tuple: (policies, values) as numpy arrays
        """
        if not states:  # Check if states list is empty
            return [], []
            
        # Convert boards to tensor representation
        batch_planes = np.stack([ChessGame.board_to_tensor(state) for state in states])
        
        # Convert to tensor and move to device
        batch_tensor = torch.tensor(batch_planes, dtype=torch.float32, device=self.device)
        
        # Forward pass through the model
        policy_logits, values = self.forward(batch_tensor)
        
        # Convert policy logits to probabilities
        policies = F.softmax(policy_logits, dim=1).detach().cpu().numpy()
        
        # Get values as numpy array
        values = values.detach().cpu().numpy().flatten()
        
        return policies, values
    
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
    
class Node:
    """ Node (for MCTS) """

    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []

        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child 
                best_ucb = ucb

        return best_child
    
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value =  child.value_sum / child.visit_count
        return q_value + self.args['C'] * (math.sqrt(self.visit_count)/(child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action_idx, prob in enumerate(policy):
            if prob > 0:
                action = index_to_move(action_idx)
                if action is not None and action in self.state.legal_moves:
                    child_state = self.game.get_next_state(self.state, action)
                    child = Node(self.game, self.args, child_state, self, action, prob)
                    self.children.append(child)
       
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)

class MCTS:
    """ MonteCarlo Tree Search implementation """

    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state, root_node):
        root_node.args = self.args
        for _ in range(self.args['num_searches']):
            node = root_node

            # SELECTION - find the most promising expandable node
            while node.is_fully_expanded():
                selected_node = node.select()
                if selected_node is None:  # If select returns None, break
                    break
                node = selected_node

            value, is_terminal = self.game.get_value_and_terminated(node.state)
            if not is_terminal:
                policy, value = self.model.predict(node.state)
                legal_policy = self._mask_policy_to_legal_moves(policy, node.state)

                # EXPAND - add a new child node
                node.expand(legal_policy)

            #BACKPROPAGATION - update statistics up the tree
            node.backpropagate(value)

    def _mask_policy_to_legal_moves(self, policy, state):
        valid_moves = self.game.get_valid_moves(state)
        
        # Create a mask for valid moves
        valid_moves_mask = np.zeros(4672)
        for move in valid_moves:
            idx = move_to_index(move)
            if idx is not None:
                valid_moves_mask[idx] = 1
        
        # Apply the mask to the policy
        masked_policy = policy * valid_moves_mask
        
        # Renormalize if sum is positive
        policy_sum = np.sum(masked_policy)
        if policy_sum > 0:
            return masked_policy / policy_sum
        else:
            print("Errore: policy_sum == 0, non ho valid legal moves?")
            # Create uniform distribution over valid moves
            uniform_policy = np.zeros_like(valid_moves_mask)
            if np.sum(valid_moves_mask) > 0:
                uniform_policy = valid_moves_mask / np.sum(valid_moves_mask)
            return uniform_policy

    def _get_action_probs_from_visits(self, root):
        action_probs = np.zeros(4672)
        
        # Use visit counts directly from children
        for child in root.children:
            if child.action_taken is not None:
                idx = move_to_index(child.action_taken)
                if idx is not None:
                    action_probs[idx] = child.visit_count
        
        # Normalize if the sum is positive
        visit_sum = np.sum(action_probs)
        if visit_sum > 0:
            action_probs /= visit_sum
        else:
            print("Errore: action probs/visit count == 0")
        
        return action_probs

class ParallelSPG:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.state = game.get_initial_state()
        self.root = Node(game, args, self.state)
        self.memory = []
        self.active = True
        self.move_count = 0  # Track moves to detect stalemates

class AlphaZero: 
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def selfPlay(self):
        """selfPlay method with parallel game handling."""
        return_memory = []

        # Initialize games
        games = [ParallelSPG(self.game, self.args) for _ in range(self.args['num_parallel_games'])]
        
        # Continue until all games are finished
        active_games = len(games)
        max_steps = 100  # Safety limit to prevent infinite games
        step = 0
        
        with trange(max_steps, desc="Simulating games") as progress:
            while active_games > 0 and step < max_steps:
                step += 1
                progress.update(1)
                progress.set_description(f"Active games: {active_games}/{len(games)}, Step: {step}")
                
                # Process each active game individually
                for game_idx, game in enumerate(games):
                    if not game.active:
                        continue
                        
                    # Perform MCTS search for this game
                    if len(game.root.children) == 0:  # Only search if not already expanded
                        self.mcts.search(game.state, game.root)
                    
                    # Get action probabilities from MCTS visit counts
                    action_probs = self.mcts._get_action_probs_from_visits(game.root)
                    
                    # Store the current state and action probabilities
                    game.memory.append((game.state.copy(), action_probs))
                    
                    # Sample an action based on the visit counts and temperature
                    temperature_action_prob = action_probs ** (1 / self.args['temperature'])
                    
                    # Ensure the distribution is valid
                    if np.sum(temperature_action_prob) > 0:
                        temperature_action_prob = temperature_action_prob / np.sum(temperature_action_prob)
                        
                        # Sample an action
                        action_idx = np.random.choice(len(temperature_action_prob), p=temperature_action_prob)
                        action = index_to_move(action_idx)
                        
                        if action in game.state.legal_moves:
                            # Update the game state
                            game.state = self.game.get_next_state(game.state, action)
                            game.move_count += 1
                            
                            # Create a new root node for the updated state
                            game.root = Node(self.game, self.args, game.state)
                            
                            # Check if the game is over
                            value, is_terminal = self.game.get_value_and_terminated(game.state)
                            
                            # Also terminate if too many moves (likely a draw)
                            if is_terminal or game.move_count >= 100:
                                if not is_terminal and game.move_count >= 100:
                                    value = 0  # Draw value for move limit
                                    
                                # Game is over, add results to memory
                                print(f"Game {game_idx} finished after {game.move_count} moves with value {value}")
                                for hist_state, hist_action_probs in game.memory:
                                    # For the winner's perspective
                                    player_perspective_value = value if hist_state.turn == game.state.turn else -value
                                    return_memory.append((
                                        ChessGame.board_to_tensor(hist_state),
                                        hist_action_probs,
                                        player_perspective_value
                                    ))
                                game.active = False
                                active_games -= 1
                        else:
                            # Invalid action, end this game
                            print(f"Warning: Invalid action selected in game {game_idx}")
                            game.active = False
                            active_games -= 1
                    else:
                        # No valid action probabilities, end this game
                        print(f"Warning: No valid actions in game {game_idx}")
                        game.active = False
                        active_games -= 1
                
                # Early termination if all games are inactive
                if active_games == 0:
                    break
        
        print(f"Self-play completed: {len(return_memory)} positions collected from {len(games)} games")
        return return_memory
 
    def train(self, memory):
        if not memory:
            print("Warning: Empty memory, nothing to train on")
            return
        
        random.shuffle(memory)
        total_loss = 0
        num_batches = 0

        for batch_idx in range(0, len(memory), self.args['batch_size']):
            sample=memory[batch_idx:min(len(memory), batch_idx+self.args['batch_size'])]
            state, policy_target, value_target = zip(*sample)

            state, policy_target, value_target = np.array(state), np.array(policy_target), np.array(value_target).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_target = torch.tensor(policy_target, dtype=torch.float32, device=self.model.device)
            value_target = torch.tensor(value_target, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            log_policy = F.log_softmax(out_policy, dim=1)
            policy_loss = F.kl_div(log_policy, policy_target, reduction='batchmean')
            value_loss = F.mse_loss(out_value, value_target)

            loss = policy_loss+value_loss
            total_loss += loss.item()
            num_batches += 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f"Training completed. Average loss: {avg_loss:.4f}")
            else:
                print("No batches processed during training")

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            print(f"\n--- Starting iteration {iteration+1}/{self.args['num_iterations']} ---")
            memory = []

            self.model.eval()
            for spg_iter in range(self.args['num_selfPlay_iterations']):
                print(f"Self-play iteration {spg_iter+1}/{self.args['num_selfPlay_iterations']}")
                iteration_memory = self.selfPlay()
                memory.extend(iteration_memory)
                print(f"Collected {len(iteration_memory)} examples, total memory: {len(memory)}")

            if not memory:
                print("Warning: No memory collected during self-play, skipping training")
                continue

            self.model.train()
            print(f"Training on {len(memory)} examples for {self.args['num_epochs']} epochs")
            for epoch in range(self.args['num_epochs']):
                print(f"Epoch {epoch+1}/{self.args['num_epochs']}")
                self.train(memory)

            # Save the model after each iteration
            model_path = f"model_{iteration}.pt"
            optim_path = f"optimizer_{iteration}.pt"
            print(f"Saving model to {model_path}")
            torch.save(self.model.state_dict(), model_path)
            torch.save(self.optimizer.state_dict(), optim_path)


def main():
    chessgame = ChessGame()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ResNet(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.0001)

    args = {
        'C':2,
        'num_searches': 50,
        'num_iterations': 3,
        'num_selfPlay_iterations': 2,
        'num_parallel_games': 2,
        'num_epochs': 2,
        'batch_size': 32,
        'temperature': 1.25
    }

    alphaZero=AlphaZero(model, optimizer, chessgame, args)
    alphaZero.learn()
    

if __name__ == "__main__":
    main()