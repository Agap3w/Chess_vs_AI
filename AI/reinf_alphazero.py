import numpy as np 
import chess
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from reinf_encodeMove import move_to_index, index_to_move, MOVE_TO_ENCODING, ENCODING_TO_MOVE

class ChessGame:
    """ basic chess engine """
    
    def __init__(self):
        self.board = chess.Board()

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

def board_to_planes(board):
    """
    Convert a chess.Board to a tensor representation with 15 planes:
    - 12 planes for piece positions (6 piece types × 2 colors)
    - 1 plane for side to move
    - 1 plane for kingside castling rights for current player
    - 1 plane for queenside castling rights for current player
    
    Returns:
        tensor: An 8×8×15 numpy array
    """
    # Initialize the planes
    planes = np.zeros((15, 8, 8), dtype=np.float32)
    
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
        piece = board.piece_at(square)
        if piece is not None:
            color_offset = 0 if piece.color == chess.WHITE else 6
            piece_plane = piece_plane_map[piece.piece_type] + color_offset
            
            # Convert chess square to tensor indices
            rank = 7 - chess.square_rank(square)  # Flip rank for standard orientation
            file = chess.square_file(square)
            
            planes[piece_plane, rank, file] = 1
    
    # Side to move plane (12)
    if board.turn == chess.WHITE:
        planes[12, :, :] = 1
    
    # Current player's castling rights
    if board.turn == chess.WHITE:
        # Kingside castling rights for white
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[13, :, :] = 1
        # Queenside castling rights for white
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[14, :, :] = 1
    else:
        # Kingside castling rights for black
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[13, :, :] = 1
        # Queenside castling rights for black
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[14, :, :] = 1
    
    return planes

class ResNet(nn.Module):
    def __init__(self, game, channel=15, num_ResBlock=8, num_hidden=128):
        super().__init__()
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

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value

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

class ChessResNet(ResNet):
    def __init__(self, game):
        # Initialize with 15 input channels for our board representation
        super().__init__(game)

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
        planes = board_to_planes(state)
        
        # Add batch dimension and convert to tensor
        x = torch.FloatTensor(planes).unsqueeze(0)
        
        # Forward pass
        policy_logits, value = self.forward(x)
        
        # Convert policy logits to probabilities
        policy = F.softmax(policy_logits, dim=1).detach().numpy()[0]
        
        # Get value as scalar
        value = value.item()
        
        return policy, value
    
class Node:
    """ Node (for MCTS) """

    def __init__(self, game, args, state, parent=None, action_taken=None):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.expandable_moves = game.get_valid_moves(state)

        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.expandable_moves) == 0 and len(self.children) > 0

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
        q_value =  ((child.value_sum / child.visit_count) +1 ) / 2 #il +1 & /2 serve a trasformare un range -1/+1 in un range 0/+1
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count)/child.visit_count)
    
    def expand(self):
        action = np.random.choice(self.expandable_moves)
        self.expandable_moves.remove(action)
        
        child_state = self.state.copy()
        child_state = self.game.get_next_state(child_state, action)

        child = Node(self.game, self.args, child_state, self, action)
        self.children.append(child)
        return child

    def simulate(self):
        value, is_terminal = self.game.get_value_and_terminated(self.state)
        if is_terminal:
            return value
        
        rollout_state = self.state.copy()
        while True:
            valid_moves = self.game.get_valid_moves(rollout_state)
            action = np.random.choice(valid_moves)
            rollout_state = self.game.get_next_state(rollout_state, action)
            value, is_terminal = self.game.get_value_and_terminated(rollout_state)
            if is_terminal:
                return value
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)

class MCTS:
    """ MonteCarlo Tree Search implementation """

    def __init__(self, game, args):
        self.game = game
        self.args = args

    def search(self, state):
        
        root = Node(self.game, self.args, state)

        for search in range(self.args['num_searches']):
            node = root
            # SELECTION - find the most promising expandable node
            while node.is_fully_expanded():
                node = node.select()
            # Check if we've reached a terminal state
            value, is_terminal = self.game.get_value_and_terminated(node.state)
            if is_terminal:
                node.backpropagate(value)
                continue
        
            # EXPAND - add a new child node
            node = node.expand()

            # SIMULATION - play out to terminal state
            value = node.simulate()

            #BACKPROPAGATION - update statistics up the tree
            node.backpropagate(value)

        #return visit_counts
        valid_moves = self.game.get_valid_moves(root.state)
        action_probs = np.zeros(4672)

        # move masking:
        for move in valid_moves:
            idx = move_to_index(move)
            if idx is not None:
                action_probs[idx] = 1e-10
            
        for child in root.children:
            if child.action_taken is not None:
                idx = move_to_index(child.action_taken)
                if idx is not None:
                    action_probs[idx] = child.visit_count 
        
        action_probs /= np.sum(action_probs)
        return action_probs


def main():
    chessgame = ChessGame()

    args = {
        'C':1.41,
        'num_searches': 100
    }
    mcts = MCTS(chessgame, args)

    state = chessgame.get_initial_state()

    while True:
        valid_moves = chessgame.get_valid_moves(state)
        player = 1 if state.turn == chess.WHITE else -1
        if player == 1:
            print (state)
            print("Valid moves:", [move.uci() for move in valid_moves])

            action = chess.Move.from_uci(input(f"{player}:").strip())

            if action not in valid_moves:
                print(("Action non valid"))
                continue
        
        else:
            mcts_probs = mcts.search(state)
            action_index = np.argmax(mcts_probs)
            action = index_to_move(action_index)
            if action not in valid_moves:
                action = np.random.choice(valid_moves) 

        state = chessgame.get_next_state(state, action)

        value, is_terminal = chessgame.get_value_and_terminated(state)

        if is_terminal:
            print("state")
            if value == -1:
                print(player, "won")
            else:
                print("draw")
            break

if __name__ == "__main__":
    main()