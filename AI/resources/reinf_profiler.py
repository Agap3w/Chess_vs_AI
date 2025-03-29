#SPECIFIC PROFILE IMPORT (remove after)
import functools
import time
import cProfile
import pstats
import io
from collections import defaultdict, Counter
from contextlib import contextmanager

#(remove after profiling)
class ChessProfiler:
    """
    A profiler for AlphaZero chess implementation.
    Provides timing information for each function/process and game outcome statistics.
    """
    
    def __init__(self, alphaZero, num_games=20, detailed=True):
        """
        Initialize the profiler.
        
        Args:
            alphaZero: AlphaZero instance to profile
            num_games: Number of games to profile
            detailed: Whether to collect detailed function-level stats
        """
        self.alphaZero = alphaZero
        self.num_games = num_games
        self.detailed = detailed
        self.function_times = defaultdict(float)
        self.call_counts = defaultdict(int)
        self.game_outcomes = Counter()
        self.total_time = 0
        
    @contextmanager
    def _time_function(self, func_name):
        """Context manager to time function execution."""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.function_times[func_name] += elapsed
            self.call_counts[func_name] += 1
    
    def _get_game_outcome(self, board):
        """Get detailed game outcome from chess board."""
        if board.is_checkmate():
            winner = "Black" if board.turn == chess.WHITE else "White"
            return f"Checkmate ({winner} wins)"
        elif board.is_stalemate():
            return "Stalemate"
        elif board.is_insufficient_material():
            return "Insufficient material"
        elif board.is_fifty_moves():
            return "Fifty-move rule"
        elif board.is_repetition():
            return "Threefold repetition"
        elif board.is_game_over():
            return "Game over (other)"
        else:
            return "Unknown"
    
    def _profile_game(self):
        """Profile a single self-play game."""
        game = self.alphaZero.game
        state = game.get_initial_state()
        root = Node(game, self.alphaZero.args, state)
        
        move_count = 0
        while True:
            # Time MCTS search
            with self._time_function("MCTS Search"):
                self.alphaZero.mcts.search(root)
            
            # Time action selection
            with self._time_function("Action Selection"):
                action_probs = self.alphaZero.mcts._get_action_probs_from_visits(root)
                
                # Apply temperature
                temperature = 1.0 if move_count < 40 else 0.1
                temperature_action_prob = action_probs ** (1 / temperature)
                
                if sum(temperature_action_prob) > 0:
                    temperature_action_prob = temperature_action_prob / sum(temperature_action_prob)
                    action_idx = np.random.choice(len(temperature_action_prob), p=temperature_action_prob)
                    action = index_to_move(action_idx)
                else:
                    # No valid moves
                    break
            
            # Time state update
            with self._time_function("State Update"):
                state = game.get_next_state(state, action)
                root = Node(game, self.alphaZero.args, state)
                move_count += 1
            
            # Check for game termination
            value, is_terminal = game.get_value_and_terminated(state)
            if is_terminal:
                outcome = self._get_game_outcome(state)
                self.game_outcomes[outcome] += 1
                break
                
        return move_count
    
    def _detailed_profiling(self):
        """Run detailed profiling with cProfile."""
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Play one full game with the profiler
        self._profile_game()
        
        profiler.disable()
        
        # Get string buffer output
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(30)  # Print top 30 functions by cumulative time
        
        return s.getvalue()
    
    def run(self):
        """Run the profiling session."""
        print(f"Starting profiling session for {self.num_games} games...")
        start_time = time.time()
        
        # Run detailed profiling if requested
        if self.detailed:
            detailed_results = self._detailed_profiling()
        
        # Run the main profiling
        total_moves = 0
        for i in range(self.num_games):
            print(f"Profiling game {i+1}/{self.num_games}...")
            moves = self._profile_game()
            total_moves += moves
        
        self.total_time = time.time() - start_time
        
        # Display results
        self._display_results()
        
        # If detailed profiling was done, display those results too
        if self.detailed:
            print("\n=== DETAILED FUNCTION PROFILING ===")
            print(detailed_results)
    
    def _display_results(self):
        """Display profiling results."""
        print("\n=== PROFILING RESULTS ===")
        print(f"Total time: {self.total_time:.2f} seconds for {self.num_games} games")
        print(f"Average time per game: {self.total_time/self.num_games:.2f} seconds")
        
        # Function timing
        print("\n--- Function Timing ---")
        print(f"{'Function':<20} {'Time (s)':<10} {'Calls':<10} {'Time/Call (ms)':<15} {'%Total':<10}")
        print("-" * 65)
        
        for func_name, time_spent in sorted(self.function_times.items(), key=lambda x: x[1], reverse=True):
            calls = self.call_counts[func_name]
            time_per_call = (time_spent / calls) * 1000 if calls > 0 else 0
            percent = (time_spent / self.total_time) * 100 if self.total_time > 0 else 0
            
            print(f"{func_name:<20} {time_spent:<10.2f} {calls:<10} {time_per_call:<15.2f} {percent:<10.2f}")
        
        # Game outcomes
        print("\n--- Game Outcomes ---")
        for outcome, count in self.game_outcomes.items():
            print(f"{outcome}: {count} ({count/self.num_games*100:.1f}%)")
    
    def profile_specific_function(self, func, *args, **kwargs):
        """Profile a specific function call."""
        func_name = func.__name__
        
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        print(f"Function '{func_name}' executed in {elapsed:.4f} seconds")
        
        return result
    
    def profile_batch_operations(self, batch_sizes=[122, 128, 134, 140]):
        """Profile batch operations with different batch sizes."""
        print("\n=== BATCH SIZE PROFILING ===")
        print(f"{'Batch Size':<15} {'Time (s)':<10} {'Throughput (samples/s)':<25}")
        print("-" * 50)
        
        # Create a simple test case for batch prediction
        game = self.alphaZero.game
        test_boards = [game.get_initial_state() for _ in range(max(batch_sizes))]
        
        for batch_size in batch_sizes:
            # Test batch for this size
            test_batch = test_boards[:batch_size]
            
            # Time it
            start_time = time.time()
            _, _ = self.alphaZero.model.batch_predict(test_batch)
            elapsed = time.time() - start_time
            
            throughput = batch_size / elapsed if elapsed > 0 else float('inf')
            print(f"{batch_size:<15} {elapsed:<10.4f} {throughput:<25.2f}")

# Helper decorator for timing specific functions if needed (remove after profiling)
def timed_function(profiler, func_name=None):
    """Decorator to time a function."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func_name or func.__name__
            with profiler._time_function(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


#(remove after profiling)    
def mainProfile():
    chessgame = ChessGame()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ResNet(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.002, weight_decay=0.0001)

    args = {
        'C': 2.5,
        'num_searches': 600,
        'num_iterations': 20,
        'num_selfPlay_iterations': 20,
        'num_parallel_games': 20,
        'num_epochs': 10,
        'batch_size': 256,
        'mcts_batch_size': 128,
        'temperature': 1,
        'dir_alpha': 0.2
    }

    alphaZero=AlphaZero(model, optimizer, chessgame, args)

    profiler = ChessProfiler(alphaZero)
    profiler.run()
    profiler.profile_batch_operations()

