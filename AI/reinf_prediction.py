import torch
import chess
from AI.reinf_alphazero import ChessGame, ResNet, MCTS, Node


model_path = r"C:\Users\Matte\main_matte_py\Chess_vs_AI\AI\resources\model_reinf.pt"

def predict_best_move(board, num_searches=800, exploration_constant=0):
    """
    Predict the best move for a given chess position using the trained model.
    
    Args:
        board: A chess.Board object representing the current position
        num_searches: Number of MCTS searches to perform
        exploration_constant: MCTS exploration constant (default: 0 for pure exploitation)
    
    Returns:
        chess.Move: The best move according to the model
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the model
    model = ResNet(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model successfully loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
        
    model.eval()  # Set model to evaluation mode
    
    # Setup game and MCTS
    game = ChessGame()
    args = {
        'C': exploration_constant,  # Low for more exploitation
        'num_searches': num_searches,
        'mcts_batch_size': 8,  # Smaller batch size for prediction
        'dir_alpha': 1e-7  # Very small Dirichlet noise for more deterministic behavior
    }
    mcts = MCTS(game, args, model)
    
    # Make sure the state is a proper chess.Board object
    if not isinstance(board, chess.Board):
        board = chess.Board(board) if isinstance(board, str) else chess.Board()
    
    # Add debug prints
    print(f"Board FEN: {board.fen()}")
    print(f"Legal moves: {list(board.legal_moves)}")
    
    # Create root node with the state properly set
    # The key fix: We need to pass the board as the state attribute
    root = Node(None, None, None)  # Create empty node first
    root.state = board  # Set the state manually
    root.game = game  # Set the game reference
    root.args = args  # Set the args
    
    # Run MCTS to find best move
    print(f"Starting MCTS search with {num_searches} iterations...")
    mcts.search(root)
    
    # Debug: Check if children were created
    print(f"Number of children after search: {len(root.children)}")
    
    # If no children, try direct policy prediction as fallback
    if len(root.children) == 0:
        print("No children found. Attempting direct policy prediction...")
        # Convert board to tensor for direct prediction
        board_tensor = torch.tensor(
            ChessGame.board_to_tensor(board),
            dtype=torch.float32, 
            device=model.device
        ).unsqueeze(0)  # Add batch dimension
        
        policy_logits, _ = model(board_tensor)
        policy = torch.softmax(policy_logits, dim=1).squeeze().detach().cpu().numpy()
        
        # Apply legal moves mask
        from AI.reinf_encodeMove import move_to_index
        legal_moves = list(board.legal_moves)
        legal_move_indices = [move_to_index(move) for move in legal_moves if move_to_index(move) is not None]
        
        if not legal_move_indices:
            print("No valid move indices found")
            return None
            
        # Find best move from policy directly
        best_move_idx = None
        best_prob = -1
        
        for move, move_idx in zip(legal_moves, legal_move_indices):
            if move_idx is not None and policy[move_idx] > best_prob:
                best_prob = policy[move_idx]
                best_move_idx = move_idx
                best_move = move
        
        if best_move_idx is not None:
            print(f"Direct policy selected move: {best_move}")
            return best_move
            
        print("Direct policy selection failed")
        return None
    
    # Get the best move based on visit counts
    best_child = None
    best_visits = -1
    
    for child in root.children:
        if child.visit_count > best_visits:
            best_visits = child.visit_count 
            best_child = child
    
    if best_child is None:
        print("No best child found")
        return None
    
    print(f"Selected move: {best_child.action_taken} with {best_visits} visits")
    return best_child.action_taken
