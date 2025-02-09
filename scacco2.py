# TO DO:
# commento

# MINOR:
# faccio redraw solo sulle square in cui serve (evitando di appesantire le performance ridisegnando ogni volta tutto)
# differenziare outro win da outro lose
# sistemo grafica intro e outro

# VERY MINOR:
# scelgo bianco o nero
# aggiungere pulsante per arrendersi / chiedere la patta
# miglioro selected square (le metto entrambe?)
# suono mangio su en passant
# quando sono in promo menu posso spammare il suono mmmmm
# customizzo grafica pezzi
# Move Highlighting: Show legal moves for the selected piece NB attivo solo dopo mossa irregolare
# Piece Capturing Animation (Optional): Provide visual feedback for captures. gIdea= barra nera laterale in cui scorrono dal basso verso l'alto gli "spiriti" dei pezzi, con le ali che flappano e un breve bubble text tipo "was I a good {piece}" 

# DOUBT:
# sql
# documentation? (es spiegazione lunga funzioni)
# test?

import pygame
import chess
from constants import DIMENSION, SQUARE_SIZE, WIDTH, HEIGHT, LIGHT_COLOR, DARK_COLOR, UNICODE_PIECES, FPS, FONT

class ChessGame:
    def __init__(self):
        self.gui = GUI()
        self.sound_manager = SoundManager()
        self.game_logic = GameLogic()
        self.running = True
        self.gui.init_game()
        self.clock = pygame.time.Clock()  # Initialize the clock
        self.game_state = "intro"

    def main_game_loop(self):
        while self.running:
            # Handle events
            self._handle_events()
            
            # Update display if needed
            if self.gui.redraw_needed:
                self._update_display()
                self.gui.redraw_needed = False

            # Maintain frame rate
            self.clock.tick(FPS)

        self.sound_manager.stop_sound()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_click(event.pos)
    
    def _handle_mouse_click(self, pos):
        if self.game_state == "intro":
            if self.gui.get_submit_button_rect().collidepoint(pos):
                self.game_state = "playing"
                self.gui.redraw_needed = True
        
        elif self.game_state == "playing":
            move_info = self.game_logic.process_click(pos)
            
            # Update display if something changed
            if any([move_info["selected_square"], 
                   move_info["move"], 
                   move_info["awaiting_promotion"]]):
                self.gui.redraw_needed = True
            
            # Play sounds if a move was made
            if move_info["move"] or move_info["awaiting_promotion"]:
                self.sound_manager.play_sounds(
                    move_info["awaiting_promotion"],
                    move_info["captured_piece"],
                    move_info["is_check"]
                )
            
            # Check for game over
            if self.game_logic.is_game_over():
                self.game_state = "outro"
                self.gui.redraw_needed = True
        
        elif self.game_state == "outro":
            if self.gui.get_submit_button_rect().collidepoint(pos):
                self.game_state = "intro"
                self.game_logic = GameLogic()  # Reset game
                self.gui.redraw_needed = True

    def _update_display(self):
        if self.game_state == "intro":
            self.gui.draw_extra("Chess vs AI", "Play")
        
        elif self.game_state == "playing":
            self.gui.screen.fill((255, 255, 255))
            self._draw_game_state()
        
        elif self.game_state == "outro":
            self.gui.draw_extra("Game Over", "Retry")
        
        self.gui.refresh_display()

    def _draw_game_state(self):
        # Draw basic board and pieces
        self.gui.draw_board()
        self.gui.draw_pieces(self.game_logic.board)
        
        # Draw selected square if not in promotion state
        if not self.game_logic.awaiting_promotion:
            self.gui.draw_selected_square(self.game_logic.selected_square)
        
        # Draw promotion menu if needed
        if self.game_logic.awaiting_promotion:
            promotion_square = self.game_logic.get_promotion_square()
            if promotion_square is not None:
                self.gui.draw_promotion_menu(promotion_square, 
                                          self.game_logic.board.turn)

class GUI:
    def __init__(self):
        self.screen = None
        self.font = None
        self.redraw_needed = True
    
    def init_game(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess vs AI")
        
        # Try different fonts
        try:
            self.font = pygame.font.SysFont(FONT[0], 60)  # Windows
        except:
            try:
                self.font = pygame.font.SysFont(FONT[1], 60)  # Alternative
            except:
                self.font = pygame.font.Font(None, 60)  # Fallback

    def get_submit_button_rect(self):
        return pygame.Rect(WIDTH // 2 - 50, HEIGHT // 2 + 50, 100, 50)

    def draw_extra(self, title, CTA):
        self.screen.fill((0, 0, 0))
        text = self.font.render(title, True, (255, 255, 255))
        self.screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2))
        
        submit_button = self.get_submit_button_rect()
        pygame.draw.rect(self.screen, (0, 255, 0), submit_button)
        button_text = self.font.render(CTA, True, (255, 255, 255))
        self.screen.blit(button_text, (submit_button.x + (submit_button.width - button_text.get_width()) // 2, 
                                     submit_button.y + (submit_button.height - button_text.get_height()) // 2))

    def refresh_display(self):
        pygame.display.flip()
    
    def draw_board(self):
        for row in range(DIMENSION):
            for col in range(DIMENSION):
                color = LIGHT_COLOR if (row + col) % 2 == 0 else DARK_COLOR
                pygame.draw.rect(self.screen, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    
    def draw_selected_square(self, square):
        if square is not None:  # If a square is selected
            col = chess.square_file(square)
            row = chess.square_rank(square)
            pygame.draw.rect(self.screen, (255, 255, 0, 50), 
                             pygame.Rect(col * SQUARE_SIZE, (7 - row) * SQUARE_SIZE, 
                                         SQUARE_SIZE, SQUARE_SIZE), 3)

    def draw_pieces(self, board=None, promotion_square=None):
        if board is None:
            board = chess.Board()  # Default to starting position if no board provided

        for square in chess.SQUARES:
            piece = board.piece_at(square)  # Get the piece at the given square
            if piece:  # If the piece exists
                piece_symbol = UNICODE_PIECES[piece.symbol()]  # Get the correct symbol for the piece
                col = chess.square_file(square)
                row = chess.square_rank(square)
                
                # Set the piece's color and render it on the screen
                color = (255, 255, 255) if piece.color else (0, 0, 0)
                text = self.font.render(piece_symbol, True, color)
                text_rect = text.get_rect(center=(col * SQUARE_SIZE + SQUARE_SIZE // 2, 
                                                  (7 - row) * SQUARE_SIZE + SQUARE_SIZE // 2))
                self.screen.blit(text, text_rect)

    def draw_promotion_menu(self, square, is_white):
        """Draw the promotion piece selection menu."""
        col = chess.square_file(square)
        row = 7 if is_white else 0
        base_y = (7 - row) * SQUARE_SIZE if is_white else row * SQUARE_SIZE
        
        # Draw semi-transparent overlay
        self._draw_overlay()
        
        # Draw menu background and border
        menu_x = col * SQUARE_SIZE
        menu_rect = pygame.Rect(menu_x, base_y, SQUARE_SIZE, 4 * SQUARE_SIZE)
        pygame.draw.rect(self.screen, (100, 100, 100), menu_rect.inflate(4, 4))
        pygame.draw.rect(self.screen, (230, 230, 230), menu_rect)
        
        # Draw promotion pieces
        pieces = ['q', 'r', 'b', 'n'] if not is_white else ['Q', 'R', 'B', 'N']
        for i, piece in enumerate(pieces):
            self._draw_promotion_piece(piece, is_white, menu_x, base_y + i * SQUARE_SIZE)
            
            # Draw separator line
            if i < len(pieces) - 1:
                pygame.draw.line(self.screen, (100, 100, 100),
                            (menu_x, base_y + (i + 1) * SQUARE_SIZE),
                            (menu_x + SQUARE_SIZE, base_y + (i + 1) * SQUARE_SIZE))

    def _draw_overlay(self):
        """Draw semi-transparent overlay for promotion menu."""
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

    def _draw_promotion_piece(self, piece, is_white, x, y):
        """Draw a single promotion piece option."""
        # Draw piece background
        piece_rect = pygame.Rect(x, y, SQUARE_SIZE, SQUARE_SIZE)
        pygame.draw.rect(self.screen, (200, 200, 200) if y % (2 * SQUARE_SIZE) == 0 else (180, 180, 180), piece_rect)
        
        # Draw piece
        piece_symbol = UNICODE_PIECES[piece]
        color = (255, 255, 255) if is_white else (0, 0, 0)
        text = self.font.render(piece_symbol, True, color)
        text_rect = text.get_rect(center=(x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2))
        self.screen.blit(text, text_rect)

class SoundManager:
    def __init__(self):
        pygame.mixer.init()
        self.sounds = {}
        self.load_sounds()

    def load_sounds(self):
        sound_files = {
            "move": "static/move.wav",
            "gnam": "static/gnam.wav",
            "check": "static/check.wav",
            "promo": "static/promo.wav"
        }
        
        for sound_name, file_path in sound_files.items():
            try:
                self.sounds[sound_name] = pygame.mixer.Sound(file_path)
            except pygame.error as e:
                print(f"Warning: Could not load sound {sound_name} from {file_path}: {e}")
                self.sounds[sound_name] = None
                
        # Set volumes only for successfully loaded sounds
        for sound_name, volume in [("move", 0.4), ("gnam", 1.0), ("promo", 1.2), ("check", 1.2)]:
            if self.sounds[sound_name]:
                self.sounds[sound_name].set_volume(volume)

    def play_sounds(self, awaiting_promotion, captured_piece, is_check):
        self.sounds["move"].play()
        if awaiting_promotion:
            self.sounds["promo"].play()
        if captured_piece:
            self.sounds["gnam"].play()
        if is_check:
            self.sounds["check"].play()

    def stop_sound(self):
        for sound in self.sounds.values():
            if sound:
                sound.stop()
        pygame.mixer.quit()

class GameLogic:
    def __init__(self):
        self.board = chess.Board()
        self.selected_square = None  # To store the currently selected square
        self.awaiting_promotion = False  # New flag for promotion state
        self.promotion_move = None  # Store the potential promotion move

    #converte posizione del mouse in square (per localizzare click?)
    def get_square_under_mouse(self, pos):
        x, y = pos
        col = x // SQUARE_SIZE
        row = 7 - (y // SQUARE_SIZE)
        return chess.square(col, row)
        
    def get_promotion_square(self):
        """Returns the square where promotion is happening if awaiting promotion."""
        return self.promotion_move.to_square if self.awaiting_promotion else None
    
    #check per GameOver (incluso Stallo)
    def is_game_over(self):
        if self.board.is_checkmate():
            print("Checkmate!")
            return 1
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            print("Draw!")
            return 2
        return 0

    def process_click(self, pos):
        """Process a mouse click and return information about the resulting move."""
        clicked_square = self.get_square_under_mouse(pos)
        
        # Handle promotion selection if we're awaiting one
        if self.awaiting_promotion:
            promotion_piece = self._handle_promotion_selection(pos)
            if promotion_piece:
                move = chess.Move(
                    self.promotion_move.from_square,
                    self.promotion_move.to_square,
                    promotion=promotion_piece
                )
                self.board.push(move)
                self.awaiting_promotion = False
                self.promotion_move = None
                self.selected_square = None
                return self._create_move_info(move=move, is_check=self.board.is_check())
            return self._create_move_info()

        # Handle piece selection and movement
        if self.selected_square is None:
            return self._handle_first_selection(clicked_square)
        else:
            return self._handle_second_selection(clicked_square)

    def _handle_first_selection(self, clicked_square):
        """Handle the first click to select a piece."""
        piece = self.board.piece_at(clicked_square)
        if piece and piece.color == self.board.turn:
            self.selected_square = clicked_square
            return self._create_move_info(square=clicked_square)
        return self._create_move_info()

    def _handle_second_selection(self, clicked_square):
        """Handle the second click to move a piece or select a different piece."""
        piece = self.board.piece_at(clicked_square)
        
        # If clicking another friendly piece, switch selection
        if piece and piece.color == self.board.turn and clicked_square != self.selected_square:
            self.selected_square = clicked_square
            return self._create_move_info(square=clicked_square)
        
        # Try to make a move
        move = chess.Move(self.selected_square, clicked_square)
        
        # Check for promotion
        current_piece = self.board.piece_at(self.selected_square)
        if self._is_promotion_move(current_piece, clicked_square):
            self.awaiting_promotion = True
            self.promotion_move = move
            return self._create_move_info(square=self.selected_square, awaiting_promotion=True)
        
        # Make the move if legal
        if move in self.board.legal_moves:
            captured_piece = self.board.piece_at(clicked_square)
            self.board.push(move)
            self.selected_square = None
            return self._create_move_info(
                move=move,
                captured_piece=captured_piece,
                is_check=self.board.is_check()
            )
        
        return self._create_move_info(square=self.selected_square)

    def _is_promotion_move(self, piece, target_square):
        """Check if a move would result in a pawn promotion."""
        if not piece or piece.piece_type != chess.PAWN:
            return False
        return (piece.color == chess.WHITE and chess.square_rank(target_square) == 7) or \
            (piece.color == chess.BLACK and chess.square_rank(target_square) == 0)

    def _handle_promotion_selection(self, pos):
        """Convert click position to promotion piece selection."""
        x, y = pos
        col = self.promotion_move.to_square % 8
        row = 7 if self.board.turn else 0
        
        if col * SQUARE_SIZE <= x <= (col + 1) * SQUARE_SIZE:
            piece_y = (7 - row) * SQUARE_SIZE if self.board.turn else row * SQUARE_SIZE
            piece_positions = [
                (piece_y, chess.QUEEN),
                (piece_y + SQUARE_SIZE, chess.ROOK),
                (piece_y + 2 * SQUARE_SIZE, chess.BISHOP),
                (piece_y + 3 * SQUARE_SIZE, chess.KNIGHT)
            ]
            
            for start_y, piece in piece_positions:
                if start_y <= y <= start_y + SQUARE_SIZE:
                    return piece
        return None

    def _create_move_info(self, square=None, move=None, captured_piece=None, is_check=False, awaiting_promotion=False):
        """Create a standardized move information dictionary."""
        return {
            'selected_square': square,
            'move': move,
            'captured_piece': captured_piece,
            'is_check': is_check,
            'awaiting_promotion': awaiting_promotion or self.awaiting_promotion
        }

def main():
    game = ChessGame()  # Create the ChessGame instance
    game.main_game_loop()  # Run the main game loop
    pygame.quit()  # Properly quit Pygame after the game loop ends

if __name__ == "__main__":
    main()  