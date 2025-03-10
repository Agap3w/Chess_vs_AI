import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import chess
import numpy as np
import random
from collections import deque, defaultdict
import time
import os
import cProfile
import pstats  # For analyzing cProfile output
import io

# Set random seeds for reproducibility (tiene quasi fissi i valori random aiutando a paragonare/debugging diverse run)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class ChessEncoder:
    """
    Encodes chess boards into neural network inputs (15,8,8 matrix)
    A differenze di tf, in torch i channel vanno all'inizio e non alla fine. 
    TO DO: aggiungere castling rights and turn info?
    """
    
    def encode_board(self, board):
        """
        Encodes a chess board into a 15x8x8 tensor (6 piece types x 2 colors)
        """
        # Create numpy array for piece planes (6 piece types x 2 colors)
        state = np.zeros((15, 8, 8), dtype=np.float32)
        
        # Piece mapping: Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4, King=5 -- Color offset: White=0, Black=6
        piece_idx = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        # Fill piece planes
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                # Convert square (0-63) to rank and file
                rank, file = divmod(square, 8)
                piece_type = piece_idx[piece.piece_type]
                color_offset = 0 if piece.color == chess.WHITE else 6
                state[piece_type + color_offset][rank][file] = 1
        
        # Turn-aware castling rights
        if board.turn == chess.WHITE:
            if board.has_kingside_castling_rights(chess.WHITE):
                state[12].fill(1)
            if board.has_queenside_castling_rights(chess.WHITE):
                state[13].fill(1)
        else:
            if board.has_kingside_castling_rights(chess.BLACK):
                state[12].fill(1)
            if board.has_queenside_castling_rights(chess.BLACK):
                state[13].fill(1)

        # Turn information
        if board.turn == chess.WHITE:
            state[14].fill(1)

        return torch.FloatTensor(state)

class ChessNetwork(nn.Module):
    """
    Simple neural network for chess with policy and value heads
    architecture: 
    - shared conv: per imparare important features (es minacce, difese, caselle controllate)
    - architettura: CNN + residual blocks
    - a questo punto splitto in due 'teste'
    - value head (position score) -> scalar
    - policy head (best move) -> 4096 output, tipo supervised
    """
    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            residual = x
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            return F.relu(x + residual)
        
    def __init__(self, num_residual_blocks=6):
        super().__init__()
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(15, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            self.ResidualBlock(128) for _ in range(num_residual_blocks)
        ])
        
        # Policy head with more sophisticated architecture
        self.policy_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.policy_fc1 = nn.Linear(64 * 8 * 8, 1024)
        self.policy_fc2 = nn.Linear(1024, 4096)  # 64*64 possible moves

        # Value head with more layers
        self.value_conv = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 64)
        self.value_fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        # Shared layers
        x = self.initial_conv(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy = self.policy_conv(x)
        policy = policy.view(-1, 64 * 8 * 8)
        policy = F.relu(self.policy_fc1(policy))
        policy = self.policy_fc2(policy)
        
        # Value head
        value = self.value_conv(x)
        value = value.view(-1, 32 * 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = F.relu(self.value_fc2(value))
        value = torch.tanh(self.value_fc3(value))
        
        return policy, value

class ReplayBuffer: 
    """
    Store and sample experiences for training
    al momento ne salvo 50,000 ma posso giocare con questo numero (hyperparametro)
    al momento le salvo tutte indistintamente, ma posso creare un algoritmo per prioritizze quali salvare
    """
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)  # <-- Initialize buffer here
        self.priorities = deque(maxlen=capacity)  # Stores rarity scores
        self.epsilon = 1e-6  # Small constant to avoid zero probabilities

    def add(self, state, policy_target, value_target, rarity):
        self.buffer.append((state, policy_target, value_target))  # <-- Directly append to buffer
        self.priorities.append(rarity + self.epsilon)  # Track position rarity

    def compute_probs(self):
        # Convert priorities to probabilities using softmax
        priorities_array = np.array(self.priorities)
        probs = priorities_array / priorities_array.sum()
        return probs

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return torch.Tensor(), torch.Tensor(), torch.Tensor()

        # Compute sampling probabilities
        probs = self.compute_probs()
        
        # Sample indices based on rarity probabilities
        indices = np.random.choice(
            len(self.buffer), 
            size=min(batch_size, len(self.buffer)),
            p=probs,
            replace=False
        )
        
        # Get the sampled experiences
        batch = [self.buffer[i] for i in indices]
        states, policy_targets, value_targets = zip(*batch)
        
        # Convert to tensors (same as original)
        states = torch.cat(states, dim=0)
        policy_targets = torch.stack(policy_targets)
        value_targets = torch.tensor(value_targets, dtype=torch.float).view(-1, 1)
        
        return states, policy_targets, value_targets
    
    def __len__(self):
        return len(self.buffer)

class OptimizedMonteCarloTreeSearch:
    class Node:
        __slots__ = ['snapshot', 'parent', 'children', 'visits', 'value_sum', 'prior_prob']

        def __init__(self, snapshot=None, parent=None, prior_prob=1.0):
            self.snapshot = snapshot
            self.parent = parent
            self.children = {}
            self.visits = 0
            self.value_sum = 0
            self.prior_prob = prior_prob

        def q_value(self):
            """Average value of the node"""
            return self.value_sum / (self.visits + 1e-8)

        def u_value(self, exploration_constant=1.4):
            """Upper confidence bound value"""
            return exploration_constant * self.prior_prob * np.sqrt(self.parent.visits) / (1 + self.visits)

        def select_child(self):
            """Select child with highest Q + U value"""
            return max(self.children.items(),
                       key=lambda child: child[1].q_value() + child[1].u_value())

    class LightweightBoard:
        __slots__ = ['pieces', 'turn', 'castling', 'ep_square', 'halfmove_clock']
        
        def __init__(self, board):
            # Store minimal board state using bitboards or arrays instead of FEN strings
            self.pieces = np.array([board.piece_type_at(sq) * (1 if board.color_at(sq) else -1) 
                                if board.piece_type_at(sq) else 0 for sq in range(64)], dtype=np.int8)
            self.turn = board.turn
            self.castling = board.castling_rights
            self.ep_square = board.ep_square
            self.halfmove_clock = board.halfmove_clock
        
        def restore(self, original_board):
            original_board.clear()
            for sq in range(64):
                piece_type = abs(self.pieces[sq])
                if piece_type:
                    color = self.pieces[sq] > 0
                    original_board.set_piece_at(sq, chess.Piece(piece_type, color))
            
            original_board.turn = self.turn
            original_board.castling_rights = self.castling
            original_board.ep_square = self.ep_square
            original_board.halfmove_clock = self.halfmove_clock

    def __init__(self, agent, num_simulations=50, use_pruning=False):
        self.agent = agent
        self.num_simulations = num_simulations
        self.use_pruning = use_pruning
        self.max_workers = 4

    def run_search(self, board):
        """Run MCTS with batched evaluations for leaf nodes."""
        root = self.Node(self.LightweightBoard(board))

        # Batch collection variables
        max_batch_size = min(self.num_simulations, 1024)  # Adjust based on GPU
        leaf_states = torch.zeros(max_batch_size, 15, 8, 8, device=self.agent.device)
        leaf_nodes = []
        snapshots = []
        batch_idx = 0

        # Process all simulations with fewer callbacks to Python
        with torch.amp.autocast("cuda", enabled=True):
            for sim in range(self.num_simulations):
                node = root
                temp_board = chess.Board()
                root.snapshot.restore(temp_board)

                # Selection phase (implement in a way that minimizes FEN generation)
                while node.children and not temp_board.is_game_over():
                    if self.use_pruning:
                        node = self._prune_branches(node)
                    if not node.children:
                        break
                    move, node = node.select_child()
                    temp_board.push(move)

                # Skip if terminal state
                if temp_board.is_game_over():
                    value = self._get_game_result(temp_board)
                    self._backpropagate(node, value)
                    continue

                # Fill batch more efficiently
                leaf_states[batch_idx] = self.agent.encode_board_cached(temp_board)
                leaf_nodes.append(node)
                snapshots.append(self.LightweightBoard(temp_board))
                batch_idx += 1

                # Process batch when full
                if batch_idx == max_batch_size or sim == self.num_simulations - 1:
                    if batch_idx > 0:
                        # Process all leaves at once
                        policies, values = self.agent.network(leaf_states[:batch_idx])

                        # Expand all nodes in parallel where possible
                        for i in range(batch_idx):
                            current_node = leaf_nodes[i]
                            snapshot = snapshots[i]
                            policy_logits = policies[i]
                            value = values[i].item()

                            # Restore board state from snapshot
                            temp_board = chess.Board()
                            snapshot.restore(temp_board)

                            # Expand node with legal moves
                            legal_moves = list(temp_board.legal_moves)
                            if self.use_pruning:
                                legal_moves = self._prune_moves(temp_board, policy_logits.unsqueeze(0))

                            policy_probs = F.softmax(policy_logits.detach(), dim=0).cpu().numpy()
                            for move in legal_moves:
                                move_idx = self.agent.encode_move(move)
                                prior_prob = policy_probs[move_idx]

                                # Create child node
                                temp_board.push(move)
                                child_snapshot = self.LightweightBoard(temp_board)
                                child_node = self.Node(
                                    snapshot=child_snapshot,
                                    parent=current_node,
                                    prior_prob=prior_prob
                                )
                                current_node.children[move] = child_node
                                temp_board.pop()

                            # Backpropagate the value
                            self._backpropagate(current_node, value)

                        # Reset batch
                        batch_idx = 0
                        leaf_nodes = []
                        snapshots = []

        # Select best move
        if root.children:
            return max(root.children.items(), key=lambda x: x[1].visits)[0]
        else:
            legal_moves = list(board.legal_moves)
            if legal_moves:
                print("MCTS failed to create children. Selecting a random legal move.")
                return random.choice(legal_moves)
            else:
                raise ValueError("No legal moves available")
            
    def _single_simulation(self, root, snapshot):
        """Perform a single MCTS simulation"""
        node = root
        search_board = chess.Board()  # Create fresh board
        snapshot.restore(search_board)  # Restore state from snapshot


        # Selection phase with optional pruning
        while node.children and not search_board.is_game_over():
            if self.use_pruning:
                # Prune less promising branches
                node = self._prune_branches(node)

            move, node = node.select_child()
            search_board.push(move)

        # Expansion and evaluation phases (similar to previous implementation)
        if not search_board.is_game_over():
            state = self.agent.encode_board_cached(search_board).unsqueeze(0).to(self.agent.device)
            with torch.no_grad():
                policy_logits, value = self.agent.network(state)

            # Expand node with legal moves
            legal_moves = list(search_board.legal_moves)

            # Optional move pruning
            if self.use_pruning:
                legal_moves = self._prune_moves(search_board, policy_logits)

            for move in legal_moves:
                move_idx = self.agent.encode_move(move)
                prior_prob = F.softmax(policy_logits, dim=1)[0][move_idx].item()

                # Push move to the REUSABLE board
                search_board.push(move)
                
                # Create snapshot of CURRENT STATE
                child_snapshot = self.LightweightBoard(search_board)
                
                # Create node with SNAPSHOT
                child_node = self.Node(
                    snapshot=child_snapshot,  # Store snapshot instead of full board
                    parent=node,
                    prior_prob=prior_prob
                )
                node.children[move] = child_node
                
                # Undo move to reuse board
                search_board.pop()

            value = value.squeeze().item()
        else:
            value = self._get_game_result(search_board)

        # Backpropagation
        self._backpropagate(node, value)

        return value

    def _parallel_search(self, root, board):
        """Parallel MCTS simulations"""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            initial_snapshot = self.LightweightBoard(board)
            futures = [executor.submit(self._single_simulation, root, initial_snapshot) for _ in range(self.num_simulations)]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Simulation failed: {e}")

        # Select move with most visits
        return max(root.children.items(), key=lambda x: x[1].visits)[0]

    def _prune_branches(self, node, pruning_threshold=0.1):
        """Prune less promising branches with better safeguards"""
        if not node.children or node.visits < 10:  # Increased minimum visits
            return node

        # Calculate absolute visit threshold
        visit_threshold = max(2, int(node.visits * pruning_threshold))
        
        # Keep top 5 moves regardless of threshold
        kept_children = sorted(node.children.items(), 
                            key=lambda x: x[1].visits, 
                            reverse=True)[:5]
        
        # Convert to dictionary
        node.children = dict(kept_children)
        return node

    def _prune_moves(self, board, policy_logits, top_k=20):
        """Prune moves based on policy network confidence"""
        legal_moves = list(board.legal_moves)

        # Get move probabilities
        probs = F.softmax(policy_logits, dim=1).squeeze().cpu().numpy()

        # Sort moves by probability
        move_probs = [
            (move, probs[self.agent.encode_move(move)])
            for move in legal_moves
        ]
        move_probs.sort(key=lambda x: x[1], reverse=True)

        # Return top K moves
        return [move for move, _ in move_probs[:top_k]]

    def _backpropagate(self, node, value):
        """Efficient backpropagation without recursive calls"""
        while node:
            node.visits += 1
            node.value_sum += value
            value = -value  # Alternate value for alternating players
            node = node.parent

    def _get_game_result(self, board):
        value, _ = self.agent.calculate_game_result(board)
        return value or 0.0
    
class ChessRL:
    """Main chess reinforcement learning class"""
    
    def __init__(self, learning_rate=0.001, batch_size=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.encoder = ChessEncoder()
        self.network = ChessNetwork().to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=1e-4) #weight_decay = L2 regularization
        self.replay_buffer = ReplayBuffer(capacity=50000)
        self.batch_size = batch_size
        self.mcts = OptimizedMonteCarloTreeSearch(
            agent=self, 
            num_simulations=50,  # Increased from 10
            use_pruning=False      # Enable pruning
        )
        self.board_cache = {}  # FEN → encoded tensor
        self.num_workers = min(4, os.cpu_count())  # Use at most 4 CPU cores
    
        # Set PyTorch to use multiple threads for CPU operations
        if torch.cuda.is_available():
            torch.set_num_threads(self.num_workers)
            # Enable TF32 precision on Ampere GPUs (like 3080) for faster computation
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Set cuDNN to benchmark mode for potentially faster convolutions
            torch.backends.cudnn.benchmark = True
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        # Pre-computed move mappings
        self.move_to_index = {}
        self.index_to_move = {}

        # Initialize all possible moves
        for from_sq in range(64):
            for to_sq in range(64):
                move = chess.Move(from_sq, to_sq)
                idx = from_sq * 64 + to_sq
                self.move_to_index[move] = idx
                self.index_to_move[idx] = move

                # Add queen promotions
                if (from_sq // 8 == 1 and to_sq // 8 == 0) or (from_sq // 8 == 6 and to_sq // 8 == 7):
                    move_prom = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
                    self.move_to_index[move_prom] = idx

    def encode_board_cached(self, board):
        """Encode board with more efficient caching."""
        # Create a unique but efficient board hash that doesn't depend on FEN
        pieces = tuple((sq, board.piece_at(sq).piece_type if board.piece_at(sq) else 0, 
                    board.piece_at(sq).color if board.piece_at(sq) else 0) 
                    for sq in chess.SQUARES if board.piece_at(sq))
        board_hash = (pieces, board.turn, board.castling_rights, board.ep_square)
        
        if board_hash not in self.board_cache:
            self.board_cache[board_hash] = self.encoder.encode_board(board)
            
        # Clean cache periodically
        if len(self.board_cache) > 10000:
            # More efficient pruning
            self.board_cache = dict(list(self.board_cache.items())[-5000:])
            
        return self.board_cache[board_hash]
    
    def encode_move(self, move):
        return self.move_to_index[move]
    
    def decode_move(self, move_idx):
        return self.index_to_move[move_idx]
    
    def get_legal_moves_mask(self, board):
        """Generate mask with only queen promotions"""
        mask = torch.zeros(4096, device=self.device)
        
        for move in board.legal_moves:
            # Convert all promotions to queen promotions
            if move.promotion and move.promotion != chess.QUEEN:
                move = chess.Move(move.from_square, move.to_square, 
                                promotion=chess.QUEEN)
            
            mask[self.encode_move(move)] = 1
        
        return mask
    
    def select_move(self, board, temperature=1.0):
        """
        Modified move selection to first try MCTS, 
        then fall back to original policy network selection
        """
        # First try MCTS
        if hasattr(self, 'mcts'):  # Ensure MCTS is initialized
            try:
                # Use MCTS to select move
                mcts_move = self.mcts.run_search(board)
                
                return mcts_move
            except Exception as e:
                print(f"MCTS failed, falling back to policy network: {e}")
        
        # Fallback to original policy network selection
        self.network.eval()
        
        state = self.encode_board_cached(board).unsqueeze(0).to(self.device)
        
        # Get move probabilities from policy network
        with torch.no_grad():
            policy_logits, value = self.network(state)
        
        # Apply mask for legal moves
        mask = self.get_legal_moves_mask(board)
        masked_logits = policy_logits.squeeze(0)
        masked_logits[mask == 0] = float('-inf')  # Set illegal moves to -infinity
        
        # Apply temperature
        if temperature == 0:  # Deterministic selection
            move_idx = torch.argmax(masked_logits).item()
        else:
            # Apply softmax with temperature
            probs = F.softmax(masked_logits / temperature, dim=0)
            move_idx = torch.multinomial(probs, 1).item()
        
        # Convert to chess move
        move = self.decode_move(move_idx)
        
        # Handle special moves (castling, en passant, promotion)
        if move in board.legal_moves:
            return move
        else:
            # Check if it's a promotion move or select a random legal move
            for legal_move in board.legal_moves:
                if legal_move.from_square == move.from_square and legal_move.to_square == move.to_square:
                    if legal_move.promotion:
                        return chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
                    return legal_move
            
            # Fallback to a random legal move if the selected move is not legal
            return random.choice(list(board.legal_moves))
    
    def calculate_game_result(self, board):
        """Returns tuple: (value_target, result_string)"""
        if board.is_checkmate():
            value = -1.0 if board.turn else 1.0
            return (value, "1-0" if not board.turn else "0-1")
            
        if board.is_stalemate():
            value = -0.2 if board.turn else 0.2
            return (value, "1/2-1/2 (stalemate)")
            
        if board.is_insufficient_material():
            return (0.0, "1/2-1/2 (insufficient material)")
            
        if board.is_fifty_moves():
            return (0.0, "1/2-1/2 (50-move rule)")
            
        if board.is_repetition():
            value = -0.3 if board.turn else 0.3
            return (value, "1/2-1/2 (repetition)")
            
        return (None, None)  # Game ongoing
    
    def train_batch(self):
        """Train on a batch of data from the replay buffer"""

        if len(self.replay_buffer) < self.batch_size // 4: # mi assicuro di avere un po' di esempi prima di partire
            return 0.0, 0.0  # Not enough data
        
        self.network.train() # train (e non evaluate come prima)
        
        # Sample batch
        states, policy_targets, value_targets = self.replay_buffer.sample(min(self.batch_size, len(self.replay_buffer)))
        states = states.to(self.device)
        policy_targets = policy_targets.to(self.device)
        value_targets = value_targets.to(self.device)
        
        # Use automatic mixed precision
        with torch.amp.autocast("cuda"):
            policy_logits, value_preds = self.network(states)
            value_loss = F.mse_loss(value_preds, value_targets)
            policy_loss = -(policy_targets * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
            total_loss = policy_loss + value_loss
        
        # Optimize with gradient scaling
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return policy_loss.item(), value_loss.item()
    
    def save_model(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def play_game(agent, temperature_schedule=None, max_moves=150):
    """Play a complete self-play game"""
    board = chess.Board()
    game_history = []
    position_counter = defaultdict(int)  # Track frequency of board states
    repetition_count = defaultdict(int)  # For 3-fold repetition check
    fen_history = []

    # Temperature schedule (higher early for exploration, lower later for better play)
    if temperature_schedule is None:
        temperature_schedule = {
            0: 3.0,
            30: 2.0,
            60: 1.0,
            100: 0.5
        }
    
    move_count = 0
    
    # Play until game over or max moves reached
    while not board.is_game_over() and move_count < max_moves:
        # Store current board state for repetition detection
        current_fen = board.fen().split(' ')[0]
        fen_history.append(current_fen)
        repetition_count[current_fen] += 1  # Track repetitions
        position_counter[current_fen] += 1  # Track rarity

        
        # Determine temperature based on move count
        temperature = 0.5  # Default temperature
        for threshold in sorted(temperature_schedule.keys()):
            if move_count >= threshold:
                temperature = temperature_schedule[threshold]
        
        # Encode current state
        state = agent.encode_board_cached(board).unsqueeze(0)
        
        # Create policy target (will be filled later with actual policy)
        policy_target = torch.zeros(4096, device=agent.device)
        
        # Detect repetitions and force random move to avoid loops
        if repetition_count[current_fen] > 2:
            legal_moves = list(board.legal_moves)
            if legal_moves:
                move = random.choice(legal_moves)
                move_idx = agent.encode_move(move)
                policy_target[move_idx] = 1.0
            else:
                break
        else:
            # Select move using policy network
            move = agent.select_move(board, temperature=temperature)
            
            # Create one-hot policy target for the selected move
            try:
                move_idx = agent.encode_move(move)
                policy_target[move_idx] = 1.0
            except Exception as e:
                move_idx = 0
        
        # Store state and policy for training
        game_history.append({
            'state': state,
            'policy_target': policy_target,
            'fen': current_fen,
            'player_turn': board.turn
        })

        # Make move
        try:
            board.push(move)
            move_count += 1
        except Exception as e:
            break
    
    # Determine game result
    result_value, result = agent.calculate_game_result(board)
    
    # Handle move limit case
    if result_value is None:
        result_value = 0.0
        result = "1/2-1/2 (move limit)"   

    # Calculate rarity scores (inverse frequency)
    for idx, history_entry in enumerate(game_history):
        # Calculate value target based on game result and player turn
        is_same_player = (history_entry['player_turn'] == board.turn)
        value_target = result_value if is_same_player else -result_value
        
        # Calculate rarity using pre-stored FEN
        frequency = position_counter[history_entry['fen']]
        rarity = 1.0 / (frequency + 1e-6)  # Prevent division by zero

        # Add to replay buffer with all metadata
        agent.replay_buffer.add(
            state=history_entry['state'],
            policy_target=history_entry['policy_target'],
            value_target=value_target,
            rarity=rarity
        )

    return result, move_count

def train(num_games=10000, batch_size=512, save_interval=1000):
    """Train the agent through self-play"""
    agent = ChessRL(batch_size=batch_size)
    
    # Create model directory if it doesn't exist and initialize TensorBoard writer
    os.makedirs("models", exist_ok=True)

    # Training statistics
    stats = {
        "white_wins": 0,
        "black_wins": 0,
        "draws": 0,
        "game_lengths": [],
        "policy_losses": [],
        "value_losses": []
    }

    start_time = time.time()
    
    print(f"Starting training for {num_games} games with batch size {batch_size}")
    print(f"Saving models every {save_interval} games")

    for game_num in range(num_games):
        
        # Play a complete game
        result, moves = play_game(agent)
        
        # Update statistics
        stats["game_lengths"].append(moves)
        if result == "1-0":
            stats["white_wins"] += 1
        elif result == "0-1":
            stats["black_wins"] += 1
        else:
            stats["draws"] += 1

        # Train after each game
        policy_loss, value_loss = agent.train_batch()

        if policy_loss > 0:  # Only record if we actually trained
            stats["policy_losses"].append(policy_loss)
            stats["value_losses"].append(value_loss)
        
        # Log metrics (ex TensorBoard) every 100 games
        if (game_num + 1) % 100 == 0: # Changed frequency to 100
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elapsed = time.time() - start_time
            games_per_second = (game_num + 1) / elapsed
            avg_game_length = np.mean(stats["game_lengths"][-100:]) if stats["game_lengths"] else 0
            recent_policy_loss = np.mean(stats["policy_losses"][-100:]) if stats["policy_losses"] else 0
            recent_value_loss = np.mean(stats["value_losses"][-100:]) if stats["value_losses"] else 0

            white_win_rate = stats['white_wins'] / (game_num + 1)
            black_win_rate = stats['black_wins'] / (game_num + 1)
            draw_rate = stats['draws'] / (game_num + 1)

            log_output = (
                f"[{timestamp}] Game {game_num+1:6d} ({elapsed:.1f}s, {games_per_second:.2f} games/s)\n" # Added games/s
                f"  Win Rate: W={white_win_rate:.3%} B={black_win_rate:.3%} D={draw_rate:.3%}\n" # Win/Draw rates as percentages
                f"  Wins (Abs): W={stats['white_wins']:4d} B={stats['black_wins']:4d} D={stats['draws']:4d}\n" # Absolute wins/draws
                f"  Avg Moves (Recent 100): {avg_game_length:.1f}\n" # Avg moves (recent)
                f"  Loss (Recent 100): PL={recent_policy_loss:.4f} VL={recent_value_loss:.4f}" # Recent Losses
            )
            print(log_output)

        # Save periodically
        if (game_num + 1) % save_interval == 0:
            agent.save_model(f"models/chess_model_{game_num+1}.pt")
    
    # Save final model
    agent.save_model("models/chess_model_final.pt")
    print("Training completed!")
    
    # Print final statistics
    print("\nTraining Summary:")
    print(f"Total games: {num_games}")
    print(f"White wins: {stats['white_wins']} ({stats['white_wins']/num_games:.1%})")
    print(f"Black wins: {stats['black_wins']} ({stats['black_wins']/num_games:.1%})")
    print(f"Draws: {stats['draws']} ({stats['draws']/num_games:.1%})")
    print(f"Average game length: {np.mean(stats['game_lengths']):.1f} moves")
    
    return agent

def profile_train(num_games=10):
    """
    Profile the training process using cProfile and generate human-readable reports
    
    Args:
        num_games (int): Number of games to train
    """

    # Create a profiler
    profiler = cProfile.Profile()
    
    # Run the profiler
    profiler.enable()
    try:
        # Call the training function
        agent = train(num_games=num_games)
    finally:
        profiler.disable()
    
    # Create multiple output formats for analysis
    # 1. Text-based detailed stats
    print("\n--- Detailed Performance Statistics ---")
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    
    # Sort by total time and print top bottlenecks
    ps.sort_stats('tottime').print_stats(30)  # Top 30 time-consuming functions
    
    # Write detailed stats to a human-readable text file
    with open('performance_profile_detailed.txt', 'w') as f:
        ps.print_stats(20)  # Write top 20 to file
    
    # 2. Cumulative time view
    print("\n--- Cumulative Time View ---")
    ps.sort_stats('cumtime').print_stats(20)
    
    # 3. Calls per function view
    print("\n--- Function Call Frequency ---")
    ps.sort_stats('calls').print_stats(20)
    
    # 4. Generate additional text report with more context
    with open('performance_profile_summary.txt', 'w') as summary_file:
        summary_file.write("Performance Profiling Summary\n")
        summary_file.write("=" * 30 + "\n\n")
        
        # Safer way to get function stats
        total_time = 0
        top_functions = []
        
        # Iterate through stats directly
        for entry in ps.stats.values():
            total_time += entry[3]  # Total time
            top_functions.append((entry[4], entry[3], entry[0]))  # (function name, total time, call count)
        
        # Sort and take top 10
        top_functions.sort(key=lambda x: x[1], reverse=True)
        top_functions = top_functions[:10]
        
        summary_file.write(f"Total Profiling Time: {total_time:.4f} seconds\n\n")
        
        summary_file.write("Top 10 Time-Consuming Functions:\n")
        for func_name, total_time, call_count in top_functions:
            summary_file.write(f"- {func_name}: {total_time:.4f} sec ({call_count} calls)\n")
    
    print("\nProfiling complete. Check performance_profile_detailed.txt and performance_profile_summary.txt")
    
    return agent


if __name__ == "__main__":
    # Choose between regular training or profiling
    MODE = "train"  # Change to "train" for normal training
    
    if MODE == "train":
        train()
    elif MODE == "profile":
        profile_train()  # Reduced games for faster profiling