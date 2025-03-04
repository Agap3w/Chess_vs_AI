import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import chess
import numpy as np
import random
from collections import deque
import time
import os
from torch.utils.tensorboard import SummaryWriter


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
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, policy_target, value_target):
        self.buffer.append((state, policy_target, value_target))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policy_targets, value_targets = zip(*batch)
        
        # Convert to appropriate tensor formats
        states = torch.cat(states, dim=0)
        policy_targets = torch.stack(policy_targets)
        value_targets = torch.tensor(value_targets, dtype=torch.float).view(-1, 1)
        
        return states, policy_targets, value_targets
    
    def __len__(self):
        return len(self.buffer)

class MonteCarloTreeSearch:
    """
    Monte Carlo Tree Search implementation for chess
    """
    class Node:
        def __init__(self, board=None, parent=None, prior_prob=1.0):
            self.board = board
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

    def __init__(self, agent, num_simulations=100):
        self.agent = agent
        self.num_simulations = num_simulations
    
    def run_search(self, board):
        """
        Run MCTS to select the best move
        """
        root = self.Node(board.copy())
        
        for _ in range(self.num_simulations):
            node = root
            search_board = board.copy()
            
            # Selection phase
            while node.children and not search_board.is_game_over():
                move, node = node.select_child()
                search_board.push(move)
            
            # Expansion phase
            if not search_board.is_game_over():
                # Get policy probabilities from neural network
                state = self.agent.encoder.encode_board(search_board).unsqueeze(0).to(self.agent.device)
                with torch.no_grad():
                    policy_logits, value = self.agent.network(state)
                
                # Expand node with legal moves
                for move in search_board.legal_moves:
                    move_idx = self.agent.encode_move(move)
                    prior_prob = F.softmax(policy_logits, dim=1)[0][move_idx].item()
                    
                    child_board = search_board.copy()
                    child_board.push(move)
                    child_node = self.Node(child_board, parent=node, prior_prob=prior_prob)
                    node.children[move] = child_node
                
                # Evaluate position with network
                value = value.squeeze().item()
            else:
                # Game over: determine value
                value = self._get_game_result(search_board)
            
            # Backpropagation
            while node:
                node.visits += 1
                node.value_sum += value
                value = -value  # Alternate value for alternating players
                node = node.parent
        
        # Select move with most visits
        best_move = max(root.children.items(), key=lambda x: x[1].visits)[0]
        return best_move
    
    def _get_game_result(self, board):
        """Determine game result value"""
        if board.is_checkmate():
            return -1.0 if board.turn else 1.0
        return 0.0  # Draw or other termination

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
        self.mcts = MonteCarloTreeSearch(self, num_simulations=100)
        
        self.debug_mode = False # For debugging, setto true se voglio printare tutte le varie info di debugging
    
    def encode_move(self, move):
        """Convert chess move to index in policy output tensor (one-hot encoding)"""
        from_square = move.from_square
        to_square = move.to_square
        return from_square * 64 + to_square
    
    def decode_move(self, move_idx):
        """Convert policy index back to chess move"""
        from_square = move_idx // 64
        to_square = move_idx % 64
        return chess.Move(from_square, to_square)
    
    def get_legal_moves_mask(self, board):
        """Create a mask of legal moves (4096, hot encoded)"""
        mask = torch.zeros(4096, device=self.device)
        
        for move in board.legal_moves:
            # Handle promotion, for simplicity always promote to queen
            if move.promotion:
                move = chess.Move(move.from_square, move.to_square)
            
            try:
                move_idx = self.encode_move(move)
                mask[move_idx] = 1
            except Exception as e:
                if self.debug_mode:
                    print(f"Error encoding move {move}: {e}")
                continue
        
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
                
                # Debug info if needed
                if self.debug_mode:
                    print(f"MCTS selected move: {mcts_move.uci()}")
                
                return mcts_move
            except Exception as e:
                if self.debug_mode:
                    print(f"MCTS failed, falling back to policy network: {e}")
        
        # Fallback to original policy network selection
        self.network.eval()
        
        state = self.encoder.encode_board(board).unsqueeze(0).to(self.device)
        
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
            if self.debug_mode:
                print(f"Warning: Selected move {move} is not legal. Selecting random move.")
            return random.choice(list(board.legal_moves))
    
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
        
        # Forward pass
        policy_logits, value_preds = self.network(states)
        
        # Calculate loss
        value_loss = F.mse_loss(value_preds, value_targets)
        policy_loss = -(policy_targets * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
        
        # Weighted total loss 
        total_loss = policy_loss + value_loss
        
        # Backward pass and optimization
        self.optimizer.zero_grad() # resetto i gradienti di tutti gli optimized parameters prima di fare backpropagation
        total_loss.backward()
        
        # Gradient clipping to prevent exploding gradients (che renderebbero il training instabile  )
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        
        self.optimizer.step() #aggiusto i pesi per minimizzare la loss
        
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

def play_game(agent, temperature_schedule=None, max_moves=200):
    """Play a complete self-play game"""
    board = chess.Board()
    game_history = []
    
    # Temperature schedule (higher early for exploration, lower later for better play)
    if temperature_schedule is None:
        temperature_schedule = {
            0: 1.0,    # First 10 moves: high exploration
            10: 0.8,   # Moves 11-20: moderate exploration
            20: 0.5,   # Moves 20+: low exploration
        }
    
    move_count = 0
    repetition_count = {}  # Track position repetitions
    
    # Debug info
    if agent.debug_mode:
        print("Starting new game")
    
    # Play until game over or max moves reached
    while not board.is_game_over() and move_count < max_moves:
        # Store current board state for repetition detection
        board_key = board.fen().split(' ')[0]  # Just piece positions
        repetition_count[board_key] = repetition_count.get(board_key, 0) + 1
        
        # Determine temperature based on move count
        temperature = 0.5  # Default temperature
        for threshold in sorted(temperature_schedule.keys()):
            if move_count >= threshold:
                temperature = temperature_schedule[threshold]
        
        # Debug info
        if agent.debug_mode and move_count % 10 == 0:
            print(f"Move {move_count}: {board.fen()}")
            print(f"Legal moves: {len(list(board.legal_moves))}")
        
        # Encode current state
        state = agent.encoder.encode_board(board).unsqueeze(0)
        
        # Create policy target (will be filled later with actual policy)
        policy_target = torch.zeros(4096, device=agent.device)
        
        # Detect repetitions and force random move to avoid loops
        if repetition_count.get(board_key, 0) > 2:
            if agent.debug_mode:
                print(f"Position repeated, making random move")
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
                if agent.debug_mode:
                    print(f"Error creating policy target: {e}")
                move_idx = 0
        
        # Store state and policy for training
        game_history.append((state, policy_target, None))  # Value filled later
        
        # Make move
        if agent.debug_mode and move_count % 10 == 0:
            print(f"Selected move: {move.uci()}")
        
        try:
            board.push(move)
            move_count += 1
        except Exception as e:
            if agent.debug_mode:
                print(f"Error making move {move}: {e}")
            break
    
    # Debug info
    if agent.debug_mode:
        print(f"Game ended after {move_count} moves. Result: {board.result()}")
        print(f"Game over reason: checkmate={board.is_checkmate()}, stalemate={board.is_stalemate()}, "
              f"insufficient={board.is_insufficient_material()}, fifty={board.is_fifty_moves()}, repetition={board.is_repetition()}")
    
    # Determine game result
    if board.is_checkmate():
        result_value = -1.0  # Last player to move lost
        result = "1-0" if not board.turn else "0-1"
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition():
        result_value = 0.0  # Draw
        result = "1/2-1/2"
    else:
        # Game terminated by move limit
        result_value = 0.0  # Draw
        result = "1/2-1/2 (move limit)"
    
    # Fill in value targets based on game result
    for idx in range(len(game_history)):
        state, policy_target, _ = game_history[idx]
        # Flip value for alternating players
        if (len(game_history) - idx) % 2 == 0:
            value_target = result_value
        else:
            value_target = -result_value
        
        # Add to replay buffer
        agent.replay_buffer.add(state, policy_target, value_target)
    
    return result, move_count

def train(num_games=10000, batch_size=512, save_interval=1000, debug_interval=100):
    """Train the agent through self-play"""
    agent = ChessRL(batch_size=batch_size)
    
    # Create model directory if it doesn't exist and initialize TensorBoard writer
    writer = SummaryWriter("runs/chess_training")  # Create a log directory
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
    print(f"Debug output every {debug_interval} games")

    for game_num in range(num_games):
        agent.debug_mode = False
        #agent.debug_mode = (game_num % debug_interval == 0)
        
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

        # Log metrics to TensorBoard
        writer.add_scalar("White Wins", stats["white_wins"] / (game_num + 1), game_num)
        writer.add_scalar("Black Wins", stats["black_wins"] / (game_num + 1), game_num)
        writer.add_scalar("Draws", stats["draws"] / (game_num + 1), game_num)
        writer.add_scalar("Average Game Length", np.mean(stats["game_lengths"]), game_num)
        
        # Train after each game
        policy_loss, value_loss = agent.train_batch()
        if policy_loss > 0:  # Only record if we actually trained
            stats["policy_losses"].append(policy_loss)
            stats["value_losses"].append(value_loss)
        
            # Log losses to TensorBoard
            writer.add_scalar("Policy Loss", policy_loss, game_num)
            writer.add_scalar("Value Loss", value_loss, game_num)

        # Print progress
        if (game_num + 1) % debug_interval == 0:
            elapsed = time.time() - start_time
            games_per_second = (game_num + 1) / elapsed
            avg_game_length = np.mean(stats["game_lengths"][-100:]) if stats["game_lengths"] else 0
            
            # Calculate recent metrics
            recent_policy_loss = np.mean(stats["policy_losses"][-100:]) if stats["policy_losses"] else 0
            recent_value_loss = np.mean(stats["value_losses"][-100:]) if stats["value_losses"] else 0
            
            print(f"\nGame {game_num+1}/{num_games} ({elapsed:.1f}s, {games_per_second:.2f} games/s)")
            print(f"W:{stats['white_wins']} B:{stats['black_wins']} D:{stats['draws']} " 
                  f"Avg moves:{avg_game_length:.1f} PL:{recent_policy_loss:.4f} VL:{recent_value_loss:.4f}")
        
            # Log metrics to TensorBoard
            writer.add_scalar("White Wins", stats["white_wins"], game_num)
            writer.add_scalar("Black Wins", stats["black_wins"], game_num)
            writer.add_scalar("Draws", stats["draws"], game_num)
            writer.add_scalar("Average Game Length", avg_game_length, game_num)
            writer.add_scalar("Recent Policy Loss", recent_policy_loss, game_num)
            writer.add_scalar("Recent Value Loss", recent_value_loss, game_num)

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
    
    writer.close() # Close TensorBoard writer
    
    return agent


if __name__ == "__main__":
    train()
