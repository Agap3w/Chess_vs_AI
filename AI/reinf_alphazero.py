import numpy as np 
import chess
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import autocast 
from reinf_encodeMove import move_to_index, index_to_move

torch.set_float32_matmul_precision('high')

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
        # Most common case first: game is ongoing
        if not state.is_game_over():
            return 0, False  # No termination, value 0
        
        # Check for checkmate (current player loses)
        if state.is_checkmate():
            return -1, True
        
        # All other terminal states (draws)
        return -0.5, True

    def get_opponent_value(self, value):
        if value == -0.5:
            return value
        return -value

    @staticmethod
    def board_to_tensor(state):
        """
        Convert a chess.Board to a tensor representation with 16 planes:
        - 12 planes for piece positions (6 piece types × 2 colors)
        - 1 plane for side to move
        - 1 plane for castling rights for current player
        - 1 plane for repetition count
        - 1 plane for 50-move rule counter
        """
        # Initialize the planes and precompute turn to avoid multiple call
        planes = np.zeros((16, 8, 8), dtype=np.float32)
        turn = state.turn
        piece_map = state.piece_map()  
    
        # Piece planes (0-11) 
        if piece_map:  
            squares = np.array(list(piece_map.keys()))  
            pieces = np.array([piece_map[sq] for sq in squares])  
            piece_types = np.array([p.piece_type - 1 for p in pieces])  
            colors = np.array([int(not p.color) * 6 for p in pieces])  
            ranks = 7 - (squares // 8)  
            files = squares % 8  
            
            # Advanced indexing (no loops)  
            planes[piece_types + colors, ranks, files] = 1  

        # Side to mvoe plane (12)
        planes[12, :, :] = float(turn)

        # Castling rights plane (13)
        row = 7 * int(turn)  # White=7, Black=0
        planes[13, row, 7] = float(state.has_kingside_castling_rights(turn))
        planes[13, row, 0] = float(state.has_queenside_castling_rights(turn))
        planes[13, row, 4] = float(planes[13, row, 7] or planes[13, row, 0])

        # Repetition plan (14)
        planes[14, :, :] = state.is_repetition(count=2)
        
        # 50-move plan (15)
        planes[15, :, :] = state.halfmove_clock / 100.0

        return planes

class ResNet(nn.Module):
    def __init__(self, device, channel=16, num_ResBlock=12, num_hidden=256):
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
            # Expand features first
            nn.Conv2d(num_hidden, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Then focus down
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Output layer
            nn.Flatten(),
            nn.Linear(32*8*8, 4672)
        )

        self.valueHead = nn.Sequential(
            # Spatial Feature Compression
            nn.Conv2d(num_hidden, 64, kernel_size=3, padding=1),  # Input: 256 channels → Output: 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Channel Reduction
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 64 → 32 channels
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Global Context Pooling
            nn.AdaptiveAvgPool2d((1, 1)),  # Reduces 8x8→1x1, output: 32*1*1=32 features
            
            # Dense Layers for Evaluation
            nn.Flatten(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # Regularization
            nn.Linear(128, 1),
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
        
        with autocast(device_type='cuda'):
            # Forward pass through the model
            policy_logits, values = self.forward(batch_tensor)
            
            # Convert policy logits to probabilities
            policies = F.softmax(policy_logits, dim=1)
        
        #move to cpu at the end
        policies_np = policies.detach().cpu().numpy()
        values_np = values.detach().cpu().numpy().flatten()
        
        return policies_np, values_np
    
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

        self.visit_count = 1 if parent is None else 0  # Root starts at 1
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
        return q_value + self.args['C'] * child.prior * (math.sqrt(self.visit_count)/(1 + child.visit_count)) 
    
    def expand(self, policy):
        if self.parent is None:  # Root node
            legal_indices = np.where(policy > 0)[0]
            if len(legal_indices) > 0:
                dir_alpha = self.args['dir_alpha']
                legal_probs = policy[legal_indices]
                dir_noise = np.random.dirichlet([dir_alpha] * len(legal_indices))
                policy[legal_indices] = 0.8 * legal_probs + 0.2 * dir_noise

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
    def search(self, root_node):
        root_node.args = self.args
        leaf_buffer = []
        
        # SINGLE TREE FOR ALL SEARCHES
        for _ in range(self.args['num_searches']):
            node = root_node  # Always start from original root
            
            # SELECTION
            while node.is_fully_expanded():
                node = node.select()
                if node is None:
                    break
            
            # TERMINAL CHECK
            value, is_terminal = self.game.get_value_and_terminated(node.state)
            if is_terminal:
                node.backpropagate(value)
                continue
            
            # STORE LEAF FOR BATCH PROCESSING
            leaf_buffer.append(node)
            
            # BATCH EVALUATION
            if len(leaf_buffer) >= self.args['mcts_batch_size']:
                self._process_batch(leaf_buffer)
                leaf_buffer.clear()
        
        # PROCESS REMAINING LEAVES
        if leaf_buffer:
            self._process_batch(leaf_buffer)

    def _process_batch(self, leaf_buffer):
        states = [n.state for n in leaf_buffer]
        policies, values = self.model.batch_predict(states)
        
        for i, node in enumerate(leaf_buffer):
            # EXPAND ONLY ONCE PER NODE
            if not node.is_fully_expanded():
                policy = self._mask_policy_to_legal_moves(policies[i], node.state)
                node.expand(policy)
                node.backpropagate(values[i])

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
        self.move_count = 0

class AlphaZero: 
    def __init__(self, model, optimizer, game, args, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.scheduler = scheduler
        self.mcts = MCTS(game, args, model)
        self.memory_buffer = []
        self.max_buffer_size = self.args['max_buffer_size']

    def selfPlay(self):
        """selfPlay method with parallel game handling."""
        return_memory = []
        draw_count = 0

        # Initialize games
        games = [ParallelSPG(self.game, self.args) for _ in range(self.args['num_parallel_games'])]
        
        # Continue until all games are finished
        active_games = len(games)
        
        progress = tqdm(desc="Simulating games")
        while active_games > 0:
            progress.update(1)
            progress.set_description(f"Active games: {active_games}/{len(games)}")
            
            # Process each active game individually
            for game_idx, game in enumerate(games):
                if not game.active:
                    continue
                    
                # Perform MCTS search for this game
                if len(game.root.children) == 0:  # Only search if not already expanded
                    self.mcts.search(game.root)
                
                # Get action probabilities from MCTS visit counts
                action_probs = self.mcts._get_action_probs_from_visits(game.root)
                
                # Store the current state and action probabilities
                game.memory.append((game.state.copy(), action_probs))
                
                # Sample an action based on the visit counts and temperature
                temperature = (
                    8.0 if game.move_count < 8 else  # High exploration in opening
                    4.0 if game.move_count < 15 else  # High exploration in opening
                    2.0 if game.move_count < 30 else  # High exploration in opening
                    1.5 if game.move_count < 70 else  # Still diverse in early midgame
                    1.0 if game.move_count < 100 else  # Some randomness in late midgame
                    0.7
                )
                temperature_action_prob = action_probs ** (1 / temperature)
                
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
                        if is_terminal:
                            # Game is over, add results to memory
                            if value == -0.5:
                                draw_count += 1
                            for hist_state, hist_action_probs in game.memory:
                                # For the winner's perspective
                                player_perspective_value = value if hist_state.turn == game.state.turn else self.game.get_opponent_value(value)
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
        
        total_games = len(games)
        draw_percent = (draw_count / total_games) * 100 if total_games > 0 else 0
        tqdm.write(f"Self-play: {len(return_memory)} positions | Draw%: {draw_percent:.1f}%")
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

            loss = policy_loss+(0.8*value_loss)
            total_loss += loss.item()
            num_batches += 1

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"Training completed. Average loss: {avg_loss:.4f}")
        else:
            print("No batches processed during training")

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            print(f"\n--- Starting iteration {iteration+1}/{self.args['num_iterations']} ---")
            iteration_memory  = []

            self.model.eval()
            for spg_iter in range(self.args['num_selfPlay_iterations']):
                print(f"Self-play iteration {spg_iter+1}/{self.args['num_selfPlay_iterations']}")
                new_examples  = self.selfPlay()
                iteration_memory.extend(new_examples)
                tqdm.write(f"Collected {len(new_examples)} examples, total memory: {len(iteration_memory)}")

            if not iteration_memory:
                print("Warning: No memory collected during self-play, skipping training")
                continue

            # Process new examples to add priority
            prioritized_memory = []
            for state, policy, value in iteration_memory:
                # Higher priority for decisive outcomes (wins/losses)
                priority = 20.0 if abs(value) > 0.5 else 1.0  
                prioritized_memory.append((state, policy, value, priority))
            
            self.memory_buffer = [
                (state, policy, value, priority * self.args['buffer_decay'])
                for (state, policy, value, priority) in self.memory_buffer
            ]

            # Add new examples to the buffer
            self.memory_buffer.extend(prioritized_memory)
            
            # Trim buffer if it exceeds the maximum size
            if len(self.memory_buffer) > self.max_buffer_size:
                # Sort by priority (highest first) before trimming
                self.memory_buffer.sort(key=lambda x: x[3], reverse=True)
                self.memory_buffer = self.memory_buffer[:self.max_buffer_size]
            print(f"Buffer size after merging and trimming: {len(self.memory_buffer)}")
            
            # Create training dataset with prioritized sampling
            training_examples = self.sample_from_buffer(self.args['training_examples_per_iter'])
      
            self.model.train()
            print(f"Training on {len(training_examples)} examples for {self.args['num_epochs']} epochs")
            for epoch in range(self.args['num_epochs']):
                print(f"Epoch {epoch+1}/{self.args['num_epochs']}")
                self.train(training_examples)

            # Save the model after each iteration
            model_path = f"model_{iteration}.pt"
            optim_path = f"optimizer_{iteration}.pt"
            print(f"Saving model to {model_path}")
            torch.save(self.model.state_dict(), model_path)
            torch.save(self.optimizer.state_dict(), optim_path)
            if self.scheduler is not None:
                self.scheduler.step()

    def sample_from_buffer(self, sample_size):
        """Sample examples from the replay buffer with prioritization."""
        if not self.memory_buffer:
            return []
        
        # Calculate sampling probabilities based on priorities
        total_priority = sum(item[3] for item in self.memory_buffer)
        probabilities = [item[3]/total_priority for item in self.memory_buffer]
        
        # Sample indices based on priorities
        buffer_size = len(self.memory_buffer)
        sample_size = min(sample_size, buffer_size)
        indices = np.random.choice(buffer_size, size=sample_size, p=probabilities, replace=True)
        
        # Create the training sample without the priority value
        samples = [self.memory_buffer[i][:3] for i in indices]
        
        return samples

def main():
    chessgame = ChessGame()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ResNet(device)
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=0.01,          # Start high, reduce later
        momentum=0.9, 
        weight_decay=1e-4  # Match AlphaZero's regularization
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[13, 26, 39],  # Reduce LR at iterations specified
        gamma=0.1            # Reduce by factor of 10 each time
    )

    args = {
        'C': 4.0,
        'num_searches': 512,
        'num_iterations': 50,
        'num_selfPlay_iterations': 10,
        'num_parallel_games': 32,
        'num_epochs': 8,
        'batch_size': 512,
        'mcts_batch_size': 128,
        'dir_alpha': 0.4,
        'max_buffer_size': 350000,  # Maximum size of the experience replay buffer
        'training_examples_per_iter': 150000,  # Number of examples to sample for each training iteration
        'buffer_decay': 0.9
    }

    alphaZero=AlphaZero(model, optimizer, chessgame, args, scheduler)
    alphaZero.learn()

if __name__ == "__main__":
    main()