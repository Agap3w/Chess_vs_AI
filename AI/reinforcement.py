# snellisco codice
# CPU bottleneck? GPU @13%
# MCST in C++ con un py wrapper? :Q__
# il modello non sta imparando? solo draw con value loss a 0 e policy loss altissima

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import chess
import numpy as np
import random
from collections import deque
import time
from tqdm import tqdm

class ChessEncoder:
    """Encodes chess boards into neural network inputs"""
    
    def encode_board(self, board):
        """
        Encodes a chess board into a 12x8x8 tensor (6 piece types x 2 colors)
        Plus 1 additional plane for turn (white/black to move)
        """
        # Create numpy array for 12 piece planes + 1 turn plane
        state = np.zeros((13, 8, 8), dtype=np.float32)
        
        # Piece mapping: Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4, King=5
        # Color offset: White=0, Black=6
        piece_idx = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        # Fill piece planes
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                rank, file = divmod(square, 8)
                piece_type = piece_idx[piece.piece_type]
                color_offset = 0 if piece.color == chess.WHITE else 6
                state[piece_type + color_offset][rank][file] = 1
        
        # Fill turn plane (plane 12): all 1s if white to move, all 0s if black
        if board.turn == chess.WHITE:
            state[12].fill(1)
        
        return torch.FloatTensor(state)

class ChessNetwork(nn.Module):
    """Neural network for chess with policy and value heads"""
    
    def __init__(self):
        super(ChessNetwork, self).__init__()
        # Input: 13 channels (12 piece planes + 1 turn plane)
        
        # Shared layers
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Policy head
        self.policy_conv = nn.Conv2d(128, 64, kernel_size=1)
        self.policy_fc = nn.Linear(64 * 8 * 8, 1968)  # 1968 = 64*64/2 possible moves (approx)
        
        # Value head
        self.value_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        # Shared layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Policy head
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(-1, 64 * 8 * 8)
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_conv(x))
        value = value.view(-1, 32 * 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class MCTSNode:
    """Monte Carlo Tree Search node"""
    
    def __init__(self, board=None, parent=None, move=None, prior=0.0):
        self.board = board          # Chess board at this node
        self.parent = parent        # Parent node
        self.move = move            # Move that led to this node
        self.children = []          # Child nodes
        
        # MCTS statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior          # Prior probability from policy network
        self.c_puct = 1.0           # Exploration constant
    
    def add_child(self, child):
        self.children.append(child)
    
    def update(self, value):
        self.visit_count += 1
        self.value_sum += value
    
    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def select_child(self):
        """Select child with highest UCB score"""
        if not self.children:
            return None
        
        # Ensure we explore unvisited nodes first
        unvisited = [child for child in self.children if child.visit_count == 0]
        if unvisited:
            return random.choice(unvisited)
        
        # Use PUCT formula for selection
        total_visits = sum(child.visit_count for child in self.children)
        best_score = -float('inf')
        best_child = None
        
        for child in self.children:
            # PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(sum(N(s,b))) / (1 + N(s,a))
            exploit = child.value
            explore = self.c_puct * child.prior * np.sqrt(total_visits) / (1 + child.visit_count)
            ucb_score = exploit + explore
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        
        return best_child

class ReplayBuffer:
    """Store and sample experiences for training"""
    
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

class ChessRL:
    """Main chess reinforcement learning class"""
    
    def __init__(self, learning_rate=0.001, batch_size=1024):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.encoder = ChessEncoder()
        self.network = ChessNetwork().to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = batch_size
    
    def move_to_index(self, move):
        """Convert chess move to index in our move space"""
        from_square = move.from_square
        to_square = move.to_square
        
        # Create a unique index for each possible move
        # We have at most 64*64=4096 possible moves (ignoring legality)
        # For simplicity, we'll map to a smaller space since most moves are impossible
        # Assuming each piece can move to about half the board on average
        return from_square * 64 + to_square
    
    def index_to_move(self, index):
        """Convert index back to chess move"""
        from_square = index // 64
        to_square = index % 64
        
        # Handle promotion (always promote to queen for simplicity)
        is_promotion = False
        
        # Check if this could be a pawn promotion move
        if (from_square // 8 == 1 and to_square // 8 == 0) or (from_square // 8 == 6 and to_square // 8 == 7):
            is_promotion = True
        
        if is_promotion:
            return chess.Move(from_square, to_square, promotion=chess.QUEEN)
        return chess.Move(from_square, to_square)
    
    def get_legal_move_mask(self, board):
        """Create a mask of legal moves"""
        mask = torch.zeros(1968, device=self.device)
        
        for move in board.legal_moves:
            try:
                move_idx = self.move_to_index(move)
                if 0 <= move_idx < 1968:  # Ensure index is within our move space
                    mask[move_idx] = 1
            except Exception:
                continue  # Skip moves that can't be encoded
        
        return mask
    
    def select_move_with_mcts(self, board, num_simulations=100, temperature=1.0):
        """Select a move using MCTS guided by the policy and value networks"""
        self.network.eval()
        
        # Create root node
        root = MCTSNode(board=board)
        
        # Run MCTS simulations
        for _ in range(num_simulations):
            node = root
            sim_board = board.copy()
            search_path = [node]
            
            # Selection: Find leaf node using UCB
            while node.children and not sim_board.is_game_over():
                node = node.select_child()
                if node.move:
                    sim_board.push(node.move)
                search_path.append(node)
            
            # Expansion: Use policy network to guide node expansion if game not over
            if not sim_board.is_game_over():
                # Get move probabilities from policy network
                state = self.encoder.encode_board(sim_board).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    policy_logits, value_pred = self.network(state)
                
                # Apply mask for legal moves
                mask = self.get_legal_move_mask(sim_board)
                masked_logits = policy_logits.squeeze(0)
                masked_logits[mask == 0] = float('-inf')
                probs = F.softmax(masked_logits, dim=0)
                
                # Add child nodes for all legal moves
                for move in sim_board.legal_moves:
                    move_idx = self.move_to_index(move)
                    if 0 <= move_idx < 1968:
                        prior = probs[move_idx].item()
                        child = MCTSNode(parent=node, move=move, prior=prior)
                        node.add_child(child)
                
                # If we expanded with legal moves, select a child and continue
                if node.children:
                    node = node.select_child()
                    sim_board.push(node.move)
                    search_path.append(node)
                
                # Use neural network value prediction
                value = value_pred.item()
                if not sim_board.turn:  # Adjust for black's perspective
                    value = -value
            else:
                # Game ended - use actual result
                if sim_board.is_checkmate():
                    value = -1.0  # Last player to move lost
                else:
                    value = 0.0   # Draw
            
            # Backpropagation: Update values up the tree
            for path_node in reversed(search_path):
                value = -value  # Flip for alternating players
                path_node.update(value)
        
        # Select move based on visit counts
        if temperature == 0 or len(root.children) == 0:
            # Select most visited move (deterministic)
            if not root.children:
                # No legal moves - should not happen in normal chess
                return None, None
            best_child = max(root.children, key=lambda c: c.visit_count)
            selected_move = best_child.move
        else:
            # Sample based on visit count distribution
            visits = np.array([child.visit_count for child in root.children])
            visits = visits ** (1.0 / temperature)  # Apply temperature
            visit_probs = visits / sum(visits)
            child_idx = np.random.choice(len(root.children), p=visit_probs)
            selected_move = root.children[child_idx].move
        
        # Create policy target for training
        policy_target = torch.zeros(1968, device=self.device)
        for child in root.children:
            move_idx = self.move_to_index(child.move)
            if 0 <= move_idx < 1968:
                policy_target[move_idx] = child.visit_count
        
        # Normalize policy target
        policy_sum = policy_target.sum()
        if policy_sum > 0:
            policy_target = policy_target / policy_sum
        
        return selected_move, policy_target
    
    def train_batch(self):
        """Train on a batch of data from the replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0  # Not enough data
        
        self.network.train()
        
        # Sample batch
        states, policy_targets, value_targets = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        policy_targets = policy_targets.to(self.device)
        value_targets = value_targets.to(self.device)
        
        # Forward pass
        policy_logits, value_preds = self.network(states)
        
        # Calculate loss
        policy_loss = F.cross_entropy(policy_logits, policy_targets)
        value_loss = F.mse_loss(value_preds, value_targets)
        total_loss = policy_loss + value_loss
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
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

def play_game(agent, temperature_schedule=None):
    """Play a complete self-play game"""
    board = chess.Board()
    game_history = []
    
    # Temperature schedule (higher early for exploration, lower later for better play)
    if temperature_schedule is None:
        temperature_schedule = {
            0: 1.0,    # First 10 moves: high temperature
            10: 0.5,   # Moves 11-20: medium temperature
            20: 0.25,  # Moves 21+: low temperature
        }
    
    move_count = 0
    temp_thresholds = sorted(temperature_schedule.keys())
    
    # Play until game over or max moves reached
    while not board.is_game_over() and move_count < 100:
        # Determine temperature based on move count
        temperature = 0.25  # Default low temperature
        for threshold in reversed(temp_thresholds):
            if move_count >= threshold:
                temperature = temperature_schedule[threshold]
                break
        
        # Select move using MCTS
        move, policy_target = agent.select_move_with_mcts(
            board, num_simulations=100, temperature=temperature
        )
        
        if move is None:
            break  # No legal moves
        
        # Store state and policy for training
        state = agent.encoder.encode_board(board).unsqueeze(0)
        game_history.append((state, policy_target, None))  # Value filled later
        
        # Make move and continue
        board.push(move)
        move_count += 1
    
    # Determine game result
    if board.is_checkmate():
        result_value = -1.0  # Last player to move lost
        result = "1-0" if not board.turn else "0-1"
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition():
        result_value = 0.0  # Draw
        result = "1/2-1/2"
    else:
        result_value = 0.0  # Game terminated by move limit
        result = "1/2-1/2"
    
    # Fill in value targets based on game result
    for idx in range(len(game_history)):
        state, policy_target, _ = game_history[idx]
        # Flip value for alternating players
        value_target = result_value if (len(game_history) - idx) % 2 == 1 else -result_value
        # Add to replay buffer
        agent.replay_buffer.add(state, policy_target, value_target)
    
    return result

def train(num_games=100000, batch_size=1024, save_interval=1000):
    """Train the agent through self-play"""
    agent = ChessRL(batch_size=batch_size)
    
    # Training statistics
    stats = {
        "white_wins": 0,
        "black_wins": 0,
        "draws": 0,
        "policy_losses": [],
        "value_losses": []
    }
    
    start_time = time.time()
    
    # Progress bar
    with tqdm(total=num_games) as pbar:
        for game_num in range(num_games):
            # Play a complete game
            result = play_game(agent)
            
            # Update statistics
            if result == "1-0":
                stats["white_wins"] += 1
            elif result == "0-1":
                stats["black_wins"] += 1
            else:
                stats["draws"] += 1
            
            # Train on a batch of data
            policy_loss, value_loss = agent.train_batch()
            stats["policy_losses"].append(policy_loss)
            stats["value_losses"].append(value_loss)
            
            # Save periodically
            if (game_num + 1) % save_interval == 0:
                agent.save_model(f"chess_model_{game_num+1}.pt")
                
                # Print statistics
                elapsed = time.time() - start_time
                win_rate = (stats["white_wins"] + stats["black_wins"]) / (game_num + 1)
                draw_rate = stats["draws"] / (game_num + 1)
                
                recent_policy_loss = np.mean(stats["policy_losses"][-save_interval:])
                recent_value_loss = np.mean(stats["value_losses"][-save_interval:])
                
                print(f"\nGames: {game_num+1}/{num_games} ({elapsed:.1f}s)")
                print(f"Win rate: {win_rate:.2%} | Draw rate: {draw_rate:.2%}")
                print(f"White wins: {stats['white_wins']} | Black wins: {stats['black_wins']} | Draws: {stats['draws']}")
                print(f"Policy loss: {recent_policy_loss:.4f} | Value loss: {recent_value_loss:.4f}")
            
            pbar.update(1)
    
    # Save final model
    agent.save_model("chess_model_final.pt")
    print("Training completed!")

def play_against_human(model_path):
    """Play against a human through console input"""
    agent = ChessRL()
    agent.load_model(model_path)
    
    board = chess.Board()
    print("Starting a game against the AI. You play as white.")
    print("Enter moves in UCI format (e.g., 'e2e4')")
    
    while not board.is_game_over():
        print("\n" + str(board))
        
        if board.turn == chess.WHITE:
            # Human's turn
            valid_move = False
            while not valid_move:
                try:
                    move_uci = input("Your move: ")
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        valid_move = True
                        board.push(move)
                    else:
                        print("Illegal move. Try again.")
                except ValueError:
                    print("Invalid input. Please use UCI format (e.g., 'e2e4')")
        else:
            # AI's turn
            print("AI is thinking...")
            move, _ = agent.select_move_with_mcts(board, num_simulations=100, temperature=0.1)
            print(f"AI plays: {move.uci()}")
            board.push(move)
    
    print("\nGame over!")
    print(str(board))
    print(f"Result: {board.result()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chess Reinforcement Learning")
    parser.add_argument('--mode', choices=['train', 'play'], default='train',
                        help='Training mode or play against AI')
    parser.add_argument('--games', type=int, default=100000,
                        help='Number of games for training')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint for playing')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print(f"Starting training for {args.games} games")
        train(num_games=args.games)
    else:
        if not args.model:
            print("Error: Please specify a model path with --model when using play mode")
        else:
            play_against_human(args.model)