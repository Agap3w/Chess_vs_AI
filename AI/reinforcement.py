import torch
import torch.nn as nn
import torch.optim as optim
import chess
import numpy as np
from collections import deque
import random
import time
from tqdm import tqdm

class ChessEncoder:
    """Encodes chess boards into neural network inputs"""
    
    def __init__(self):
        # Piece type mapping: None=0, Pawn=1, Knight=2, Bishop=3, Rook=4, Queen=5, King=6
        self.piece_to_int = {None: 0, chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
                            chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6}
    
    def encode_board(self, board):
        """
        Encodes a chess board into a 8x8x12 tensor without using in-place operations
        6 planes for white pieces, 6 for black (one plane per piece type)
        """
        # First create a numpy array (no gradients involved)
        state_array = np.zeros((12, 8, 8))
        
        # Fill the numpy array
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                rank = square // 8
                file = square % 8
                piece_type = self.piece_to_int[piece.piece_type] - 1
                channel = piece_type if piece.color else piece_type + 6
                state_array[channel][rank][file] = 1
        
        # Convert to tensor all at once
        return torch.FloatTensor(state_array)

class PolicyNetwork(nn.Module):
    """Neural network that predicts move probabilities"""
    
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        
        # Input: 12 channels (6 piece types × 2 colors)
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 4096)  # Output size: maximum possible moves
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Ensure input requires gradients
        x = x.requires_grad_()
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)
        
        return logits

class ChessRL:
    """Main reinforcement learning class"""
    
    def __init__(self, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.encoder = ChessEncoder()
        self.policy_net = PolicyNetwork().to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.move_memory = []
    
    def get_legal_move_mask(self, board):
        """Creates a mask of legal moves"""
        mask = torch.zeros(4096, device=self.device)
        for move in board.legal_moves:
            move_idx = self.move_to_index(move)
            mask[move_idx] = 1
        return mask
    
    def move_to_index(self, move):
        """Converts a chess move to an index (simplified version)"""
        return move.from_square * 64 + move.to_square
    
    def index_to_move(self, index):
        """Converts an index back to a chess move (simplified version)"""
        from_square = index // 64
        to_square = index % 64
        return chess.Move(from_square, to_square)
    
    def select_move(self, board):
        """Selects a move using the policy network"""
        self.policy_net.train()  # Ensure we're in training mode
        
        # Get board state and move to device
        state = self.encoder.encode_board(board).to(self.device)
        state = state.unsqueeze(0)  # Add batch dimension
        
        # Get move logits
        logits = self.policy_net(state)
        
        # Mask illegal moves
        mask = self.get_legal_move_mask(board)
        masked_logits = logits.squeeze(0)
        masked_logits[mask == 0] = float('-inf')
        
        # Get move probabilities
        probs = torch.softmax(masked_logits, dim=0)
        
        # Sample move from probability distribution
        move_idx = torch.multinomial(probs, 1).item()
        selected_move = self.index_to_move(move_idx)
        
        # Store for training
        self.move_memory.append((state, masked_logits, move_idx, None))
        
        return selected_move
    
    def update_reward(self, reward):
        """Updates the last move in memory with its reward"""
        if self.move_memory:
            state, logits, move_idx, _ = self.move_memory[-1]
            self.move_memory[-1] = (state, logits, move_idx, reward)
    
    def train_step(self):
        """Performs one training step using policy gradients"""
        if not self.move_memory:
            return
        
        total_loss = 0
        for state, logits, move_idx, reward in self.move_memory:
            if reward is not None:
                # Compute log probabilities
                log_probs = torch.log_softmax(logits, dim=0)
                # Get log probability of the selected move
                action_log_prob = log_probs[move_idx]
                # Compute loss for this move
                loss = -action_log_prob * reward
                total_loss += loss
        
        # Only proceed if we have some loss to optimize
        if total_loss != 0:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        self.move_memory = []

def play_game(agent):
    """Plays one game of chess using the agent"""
    board = chess.Board()
    
    while not board.is_game_over():
        move = agent.select_move(board)
        board.push(move)
        
        if board.is_game_over():
            if board.is_checkmate():
                reward = 1.0 if board.turn else -1.0
            else:
                reward = 0.0  # Draw
            agent.update_reward(reward)
    
    return board.result()

def main():
    # Initialize our chess agent
    agent = ChessRL(learning_rate=0.001)
    
    # Training parameters
    num_episodes = 1000
    evaluation_frequency = 50
    
    # Statistics tracking
    moving_average_length = 100
    results_history = deque(maxlen=moving_average_length)
    wins_white = 0
    wins_black = 0
    draws = 0
    
    print("Starting training...")
    start_time = time.time()
    
    # Main training loop
    for episode in tqdm(range(num_episodes)):
        # Play one game
        result = play_game(agent)
        agent.train_step()
        
        # Track statistics
        results_history.append(result)
        if result == "1-0":
            wins_white += 1
        elif result == "0-1":
            wins_black += 1
        else:
            draws += 1
            
        if (episode + 1) % evaluation_frequency == 0:
            win_rate = (wins_white + wins_black) / evaluation_frequency
            draw_rate = draws / evaluation_frequency
            
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
            print(f"Last {evaluation_frequency} games statistics:")
            print(f"Win rate: {win_rate:.2%}")
            print(f"Draw rate: {draw_rate:.2%}")
            print(f"White wins: {wins_white}, Black wins: {wins_black}, Draws: {draws}")
            
            wins_white = 0
            wins_black = 0
            draws = 0
        
        if (episode + 1) % 500 == 0:
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
            }, f'chess_model_episode_{episode+1}.pt')
    
    print("\nTraining completed!")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    
    torch.save({
        'episode': num_episodes,
        'model_state_dict': agent.policy_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
    }, 'chess_model_final.pt')

if __name__ == "__main__":
    main()