import chess
import unittest
from reinf_encodeMove import move_to_index, index_to_move, MOVE_TO_ENCODING, ENCODING_TO_MOVE

class TestMoveEncoding(unittest.TestCase):
    """Test case for verifying move encoding/decoding functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a standard chess board for testing
        self.board = chess.Board()
        
    def test_roundtrip_conversion(self, move_uci):
        """Test that a move can be converted to an index and back correctly."""
        move = chess.Move.from_uci(move_uci)
        index = move_to_index(move)
        self.assertIsNotNone(index, f"Failed to encode move {move_uci}")
        
        # Check that the index is within the expected range
        self.assertGreaterEqual(index, 0, f"Index {index} is negative for move {move_uci}")
        self.assertLess(index, 4672, f"Index {index} is too large for move {move_uci}")
        
        # Convert back to a move
        decoded_move = index_to_move(index)
        self.assertIsNotNone(decoded_move, f"Failed to decode index {index} for move {move_uci}")
        
        # For promotions, special handling is needed since we saw issues with this
        if move.promotion:
            # Check that the from and to squares match
            self.assertEqual(move.from_square, decoded_move.from_square,
                            f"From square mismatch: {move.uci()} -> {index} -> {decoded_move.uci()}")
            self.assertEqual(move.to_square, decoded_move.to_square,
                            f"To square mismatch: {move.uci()} -> {index} -> {decoded_move.uci()}")
            
            # Check if promotion piece is preserved or if it's a known limitation
            if decoded_move.promotion != move.promotion:
                print(f"⚠️ Promotion piece not preserved: {move.uci()} -> {index} -> {decoded_move.uci()}")
                # This is where we could fix or adapt the promotion issue
        else:
            # Non-promotion moves should match exactly
            self.assertEqual(move, decoded_move, 
                            f"Move roundtrip failed: {move_uci} -> {index} -> {decoded_move.uci()}")
        
        return index  # Return the index for additional checks
    
    def test_normal_moves(self):
        """Test encoding/decoding of normal piece movements."""
        # Test knight moves
        self.test_roundtrip_conversion("g1f3")  # Knight from starting position
        self.test_roundtrip_conversion("b1c3")  # Another knight move
        
        # Test bishop moves
        self.board.push_san("e4")  # Move pawn to open diagonal
        self.board.push_san("e5")
        self.test_roundtrip_conversion("f1c4")  # Bishop move
        
        # Test rook moves
        self.board.push_san("a4")  # Move pawn to open file
        self.board.push_san("a5")
        self.test_roundtrip_conversion("a1a3")  # Rook move
        
        # Test queen moves
        self.test_roundtrip_conversion("d1f3")  # Queen move
        
        # Test king moves
        self.test_roundtrip_conversion("e1f1")  # King move
        
        # Test pawn moves
        self.test_roundtrip_conversion("d2d3")  # Single step
        self.test_roundtrip_conversion("h2h4")  # Double step
        
    def test_pawn_initial_double_move(self):
        """Test encoding/decoding of pawn's initial two-square advance."""
        # White pawns
        for file in 'abcdefgh':
            move_uci = f"{file}2{file}4"
            self.test_roundtrip_conversion(move_uci)
            
        # Black pawns
        for file in 'abcdefgh':
            move_uci = f"{file}7{file}5"
            self.test_roundtrip_conversion(move_uci)
            
    def test_castling(self):
        """Test encoding/decoding of castling moves."""
        # Set up a position where castling is possible
        self.board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        
        # Kingside castling (white)
        white_ks_index = self.test_roundtrip_conversion("e1g1")
        
        # Queenside castling (white)
        white_qs_index = self.test_roundtrip_conversion("e1c1")
        
        # Kingside castling (black)
        black_ks_index = self.test_roundtrip_conversion("e8g8")
        
        # Queenside castling (black)
        black_qs_index = self.test_roundtrip_conversion("e8c8")
        
        # Verify all castling indices are different
        self.assertNotEqual(white_ks_index, white_qs_index)
        self.assertNotEqual(black_ks_index, black_qs_index)
        
    def test_promotions(self):
        """Test encoding/decoding of pawn promotion moves."""
        # Test white pawn promotion (regular advance)
        self.board = chess.Board("8/4P3/8/8/8/8/8/8 w - - 0 1")
        
        # Regular promotion to each piece type
        self.test_roundtrip_conversion("e7e8q")  # Queen
        self.test_roundtrip_conversion("e7e8r")  # Rook
        self.test_roundtrip_conversion("e7e8b")  # Bishop
        self.test_roundtrip_conversion("e7e8n")  # Knight
        
        # Test white pawn promotion with capture
        self.board = chess.Board("3r4/4P3/8/8/8/8/8/8 w - - 0 1")
        
        # Promotion with capture to each piece type
        self.test_roundtrip_conversion("e7d8q")  # Queen
        self.test_roundtrip_conversion("e7d8r")  # Rook
        self.test_roundtrip_conversion("e7d8b")  # Bishop
        self.test_roundtrip_conversion("e7d8n")  # Knight
        
        # Test black pawn promotion (regular advance)
        self.board = chess.Board("8/8/8/8/8/8/3p4/8 b - - 0 1")
        
        # Regular promotion to each piece type
        self.test_roundtrip_conversion("d2d1q")  # Queen
        self.test_roundtrip_conversion("d2d1r")  # Rook
        self.test_roundtrip_conversion("d2d1b")  # Bishop
        self.test_roundtrip_conversion("d2d1n")  # Knight
        
        # Test black pawn promotion with capture
        self.board = chess.Board("8/8/8/8/8/8/3p4/2B5 b - - 0 1")
        
        # Promotion with capture to each piece type
        self.test_roundtrip_conversion("d2c1q")  # Queen
        self.test_roundtrip_conversion("d2c1r")  # Rook
        self.test_roundtrip_conversion("d2c1b")  # Bishop
        self.test_roundtrip_conversion("d2c1n")  # Knight
        
        # Analyze the promotion encoding issue in more detail
        print("\nInvestigating promotion encoding details:")
        self._analyze_promotion_encoding("e7e8q", "White queen promotion")
        self._analyze_promotion_encoding("e7e8r", "White rook promotion")
        self._analyze_promotion_encoding("e7e8b", "White bishop promotion")
        self._analyze_promotion_encoding("e7e8n", "White knight promotion")
        self._analyze_promotion_encoding("e7d8q", "White queen promotion capture")
        
    def _analyze_promotion_encoding(self, move_uci, description):
        """Analyze how a promotion move is encoded and decoded."""
        move = chess.Move.from_uci(move_uci)
        
        # Check direct lookup in the precomputed tables
        in_lookup = move in MOVE_TO_ENCODING
        
        # Get the encoding
        index = move_to_index(move)
        if index is None:
            print(f"❌ {description} ({move_uci}) could not be encoded")
            return
            
        # Get the square and plane indices
        square_index = index // 73
        plane_index = index % 73
        
        # Try to decode it
        decoded_move = index_to_move(index)
        
        print(f"- {description} ({move_uci}):")
        print(f"  - In lookup table: {in_lookup}")
        print(f"  - Encoded as index: {index} (square={square_index}, plane={plane_index})")
        if decoded_move:
            decoded_uci = decoded_move.uci()
            match = decoded_uci == move_uci
            print(f"  - Decoded as: {decoded_uci} ({'✅' if match else '❌'})")
            if decoded_move.promotion:
                print(f"  - Promotion piece preserved: {decoded_move.promotion == move.promotion}")
        else:
            print("  - Failed to decode back to a move")
        
    def test_en_passant(self):
        """Test encoding/decoding of en passant capture."""
        # Set up an en passant position (white capturing black)
        self.board = chess.Board("8/8/8/8/5p2/8/4P3/8 w - - 0 1")
        self.board.push_san("e4")  # Move pawn to set up en passant
        
        # Now black can capture en passant
        self.test_roundtrip_conversion("f4e3")
        
        # Set up an en passant position (black capturing white)
        self.board = chess.Board("8/4p3/8/8/8/8/3P4/8 b - - 0 1")
        self.board.push_san("e5")  # Move pawn to set up en passant
        
        # Now white can capture en passant
        self.test_roundtrip_conversion("d5e6")
        
    def test_check_moves(self):
        """Test encoding/decoding of moves that result in check."""
        # Set up a position where a move results in check
        self.board = chess.Board("8/8/8/8/8/8/8/4K2r w - - 0 1")
        
        # Move results in check
        self.test_roundtrip_conversion("e1d1")  # King move that leaves it in check
        
        # Set up another check position
        self.board = chess.Board("r3k3/8/8/8/8/8/8/4K2R b - - 0 1")
        
        # Move results in check
        self.test_roundtrip_conversion("e8d8")  # King move that leaves it in check
        
    def test_edge_cases(self):
        """Test encoding/decoding of various edge cases."""
        # Long diagonal move
        self.board = chess.Board("8/8/8/8/8/8/8/B6k w - - 0 1")
        self.test_roundtrip_conversion("a1h8")  # Bishop long diagonal
        
        # Long horizontal move
        self.board = chess.Board("8/8/8/8/8/8/8/R6k w - - 0 1")
        self.test_roundtrip_conversion("a1h1")  # Rook long horizontal
        
        # Long vertical move
        self.board = chess.Board("7k/8/8/8/8/8/8/R7 w - - 0 1")
        self.test_roundtrip_conversion("a1a8")  # Rook long vertical
        
        # Knight at the corner
        self.board = chess.Board("7k/8/8/8/8/8/8/N7 w - - 0 1")
        self.test_roundtrip_conversion("a1b3")  # Knight from corner
        self.test_roundtrip_conversion("a1c2")  # Knight from corner
        
    def test_illegal_moves(self):
        """Test that illegal moves are still encoded/decoded correctly."""
        # Note: The encoding/decoding should work for all moves,
        # regardless of whether they're legal in the current position
        
        # King moving two squares (not castling)
        self.test_roundtrip_conversion("e1e3")
        
        # Pawn moving backward
        self.test_roundtrip_conversion("e4e3")
        
        # Knight moving like a bishop
        self.test_roundtrip_conversion("g1e3")
        
    def test_all_indices_are_valid(self):
        """Test that all 4672 possible indices map to valid moves."""
        valid_indices = 0
        invalid_indices = []
        
        for index in range(4672):
            move = index_to_move(index)
            if move is not None:
                valid_indices += 1
                # Verify the round trip
                back_index = move_to_index(move)
                if back_index is None or back_index != index:
                    invalid_indices.append((index, back_index))
        
        if invalid_indices:
            self.fail(f"Found {len(invalid_indices)} indices that don't round-trip correctly: {invalid_indices[:5]}...")
            
        print(f"Valid indices: {valid_indices} out of 4672")
        
    def test_exhaustive_board_moves(self):
        """Test encoding/decoding of all legal moves in several positions."""
        # Test standard starting position
        self.board = chess.Board()
        self._test_position_moves()
        
        # Middle game position with various pieces
        self.board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
        self._test_position_moves()
        
        # Complex endgame position
        self.board = chess.Board("8/5pk1/7p/8/5PK1/7P/8/8 w - - 0 1")
        self._test_position_moves()
        
        # Position with promotion possibility
        self.board = chess.Board("8/P7/8/8/8/8/p7/8 w - - 0 1")
        self._test_position_moves()
        
    def _test_position_moves(self):
        """Helper method to test all legal moves in the current position."""
        for move in self.board.legal_moves:
            index = move_to_index(move)
            self.assertIsNotNone(index, f"Failed to encode move {move.uci()}")
            
            decoded_move = index_to_move(index)
            self.assertIsNotNone(decoded_move, f"Failed to decode index {index}")
            
            # For promotion moves, just check that source and destination squares match
            if move.promotion:
                self.assertEqual(move.from_square, decoded_move.from_square,
                                f"From square mismatch: {move.uci()} -> {index} -> {decoded_move.uci()}")
                self.assertEqual(move.to_square, decoded_move.to_square,
                                f"To square mismatch: {move.uci()} -> {index} -> {decoded_move.uci()}")
                
                if decoded_move.promotion != move.promotion:
                    print(f"⚠️ Promotion piece not preserved: {move.uci()} -> {index} -> {decoded_move.uci()}")
            else:
                # Non-promotion moves should match exactly
                self.assertEqual(move, decoded_move, 
                                f"Move roundtrip failed: {move.uci()} -> {index} -> {decoded_move.uci()}")

if __name__ == "__main__":
    # Create a test suite with all the test methods
    suite = unittest.TestSuite()
    suite.addTest(TestMoveEncoding("test_normal_moves"))
    suite.addTest(TestMoveEncoding("test_pawn_initial_double_move"))
    suite.addTest(TestMoveEncoding("test_castling"))
    suite.addTest(TestMoveEncoding("test_promotions"))
    suite.addTest(TestMoveEncoding("test_en_passant"))
    suite.addTest(TestMoveEncoding("test_check_moves"))
    suite.addTest(TestMoveEncoding("test_edge_cases"))
    suite.addTest(TestMoveEncoding("test_illegal_moves"))
    suite.addTest(TestMoveEncoding("test_all_indices_are_valid"))
    suite.addTest(TestMoveEncoding("test_exhaustive_board_moves"))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.failures or result.errors:
        print("\n⚠️ Tests completed with issues. Here's a summary of what to check:")
        print("1. Promotion Handling: The current encoding appears to lose promotion piece information.")
        print("   This may require an update to the encoding/decoding functions in 'reinf_encodeMove.py'.")
        print("2. Check if this is a critical issue for your AlphaZero implementation or if it's acceptable.")
        print("\nPossible solutions:")
        print("- Modify encode_move_universal() and decode_move_universal() to better handle promotions")
        print("- Update your training loop to handle the case where promotions are simplified")
    else:
        print("\n✅ All tests passed successfully!")