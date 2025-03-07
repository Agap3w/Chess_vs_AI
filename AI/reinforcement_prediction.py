import chess
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

model_path = r"C:\Users\Matte\main_matte_py\Chess_vs_AI\models\chess_model_5000.pt"

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

class ChessMovePredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.encoder = ChessEncoder()
        
        # Create move mappings
        self.move_to_index = {}
        for from_sq in range(64):
            for to_sq in range(64):
                move = chess.Move(from_sq, to_sq)
                idx = from_sq * 64 + to_sq
                self.move_to_index[move] = idx
                # Add queen promotions
                if (from_sq // 8 == 1 and to_sq // 8 == 0) or (from_sq // 8 == 6 and to_sq // 8 == 7):
                    move_prom = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
                    self.move_to_index[move_prom] = idx

    def _load_model(self, model_path):
        model = ChessNetwork()
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.eval().to(self.device)

    def predict_move(self, board, temperature=0.1):
        """Main prediction interface
        Args:
            board: chess.Board object
            temperature: float controlling randomness (0 = deterministic)
        Returns:
            chess.Move object ready to be pushed to the board
        """
        # Convert all promotions to queen promotions
        legal_moves = []
        for move in board.legal_moves:
            if move.promotion and move.promotion != chess.QUEEN:
                move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
            legal_moves.append(move)
        
        # Encode board
        state = self.encoder.encode_board(board).unsqueeze(0).to(self.device)
        
        # Get model output
        with torch.no_grad():
            policy_logits, _ = self.model(state)
        
        # Mask illegal moves
        legal_indices = [self.move_to_index[move] for move in legal_moves]
        legal_logits = policy_logits[0][legal_indices]
        
        # Apply temperature
        scaled_logits = legal_logits / temperature
        probs = torch.softmax(scaled_logits, dim=0)
        
        # Select move
        best_idx = torch.argmax(probs).item()
        return legal_moves[best_idx]