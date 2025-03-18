import numpy as np 
import chess
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from reinf_encodeMove import move_to_index, index_to_move, MOVE_TO_ENCODING, ENCODING_TO_MOVE

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
            piece = state.piece_at(square)
            if piece is not None:
                color_offset = 0 if piece.color == chess.WHITE else 6
                piece_plane = piece_plane_map[piece.piece_type] + color_offset
                
                # Convert chess square to tensor indices
                rank = 7 - chess.square_rank(square)  # Flip rank for standard orientation
                file = chess.square_file(square)
                
                planes[piece_plane, rank, file] = 1
        
        # Side to move plane (12)
        if state.turn == chess.WHITE:
            planes[12, :, :] = 1
        
        # Current player's castling rights
        if state.turn == chess.WHITE:
            # Kingside castling rights for white
            if state.has_kingside_castling_rights(chess.WHITE):
                planes[13, :, :] = 1
            # Queenside castling rights for white
            if state.has_queenside_castling_rights(chess.WHITE):
                planes[14, :, :] = 1
        else:
            # Kingside castling rights for black
            if state.has_kingside_castling_rights(chess.BLACK):
                planes[13, :, :] = 1
            # Queenside castling rights for black
            if state.has_queenside_castling_rights(chess.BLACK):
                planes[14, :, :] = 1
        
        return planes

class ResNet(nn.Module):
    def __init__(self, channel=15, num_ResBlock=8, num_hidden=128):
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
        x = torch.FloatTensor(planes).unsqueeze(0)
        
        # Forward pass
        policy_logits, value = self.forward(x)
        
        # Convert policy logits to probabilities
        policy = F.softmax(policy_logits, dim=1).detach().numpy()[0]
        
        # Get value as scalar
        value = value.item()
        
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
                if action is not None:
                    child_state = self.game.get_next_state(self.state, action)
                    child = Node(self.game, self.args, child_state, self, action, prob)
                    self.children.append(child)
        return self.children[-1] if self.children else None
       
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
    def search(self, state):
        
        root = Node(self.game, self.args, state)

        for search in range(self.args['num_searches']):
            node = root
            # SELECTION - find the most promising expandable node
            while node.is_fully_expanded():
                node = node.select()
            # Check if we've reached a terminal state
            value, is_terminal = self.game.get_value_and_terminated(node.state)
            
            if not is_terminal:
                policy, value = self.model.predict(node.state)
                valid_moves = self.game.get_valid_moves(node.state)
                # Create a mask for valid moves
                valid_moves_mask = np.zeros(4672)
                for move in valid_moves:
                    idx = move_to_index(move)
                    if idx is not None:
                        valid_moves_mask[idx] = 1
                
                # Apply the mask and renormalize
                policy = policy * valid_moves_mask
                policy_sum = np.sum(policy)
                if policy_sum > 0:
                    policy /= policy_sum

                # EXPAND - add a new child node
                node.expand(policy)

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
        
        action_probs /= np.sum(action_probs) if np.sum(action_probs) > 0 else 1
        return action_probs

class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def selfPlay(self):
        memory = []
        state = self.game.get_initial_state()

        while True:
            action_probs = self.mcts.search(state)
            memory.append((state.copy(), action_probs))

            action_idx = np.random.choice(4672, p=action_probs)
            action = index_to_move(action_idx)

            state = self.game.get_next_state(state, action)

            value, is_terminal = self.game.get_value_and_terminated(state)
            if is_terminal:
                returnMemory = []
                for hist_state, hist_action_probs in memory:
                    hist_outcome = value
                    returnMemory.append((
                        ChessGame.board_to_tensor(hist_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory
 
    def train(self, memory):
        random.shuffle(memory)
        for batch_idx in range(0, len(memory), self.args['batch_size']):
            sample=memory[batch_idx:min(len(memory)-1, batch_idx+self.args['batch_size'])]
            state, policy_target, value_target = zip(*sample)

            state, policy_target, value_target = np.array(state), np.array(policy_target), np.array(value_target).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32)
            policy_target = torch.tensor(policy_target, dtype=torch.float32)
            value_target = torch.tensor(value_target, dtype=torch.float32)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_target)
            value_loss = F.mse_loss(out_value, value_target)

            loss = policy_loss+value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()

            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")


def main():
    chessgame = ChessGame()
    model = ResNet()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    args = {
        'C':2,
        'num_searches': 10,
        'num_iterations': 5,
        'num_selfPlay_iterations': 5,
        'num_epochs': 2,
        'batch_size': 128
    }

    alphaZero=AlphaZero(model, optimizer, chessgame, args)
    alphaZero.learn()
    

if __name__ == "__main__":
    main()