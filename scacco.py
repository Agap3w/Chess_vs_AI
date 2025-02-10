#SE SISTEMO LO SCORE E' FATTA

""" TO DO: """
# controllo modello score, sacrifica donna su pedone


""" MINOR: """
# non mangia col re
# non sa dare scacco matto
# sul retry AI muove bianco e si scazza tutto
# si sono scazzati i suono con l'AI
# faccio redraw solo sulle square in cui serve (evitando di appesantire le performance ridisegnando ogni volta tutto)
# differenziare outro win da outro lose
# creo menu selezione AI in intro

""" VERY MINOR: """
# colonna A1 non si seleziona
# se perdo non vedo come
# scelgo bianco o nero
# aggiungere pulsante per arrendersi / chiedere la patta
# suono mangio su en passant
# quando sono in promo menu posso spammare il suono mmmmm
# customizzo grafica pezzi
# Segnalo Mossa Irregolare: suono, highlight rosso, mostro possibili legal moves? 

""" DOUBT: """
# sql
# documentation? (es spiegazione lunga funzioni)
# test?
# AI ELO evaluation?

import pygame
import chess
from AI.heuristic import heuristic_best_move
from constants import DIMENSION, SQUARE_SIZE, WIDTH, HEIGHT, LIGHT_COLOR, DARK_COLOR, UNICODE_PIECES, FPS, FONT, FONT_SIZE, PIECE_VALUES, PIECE_POS_TABLE

class ChessGame:
    """main Class che unisce tutte le altre 4 subclass"""

    def __init__(self):
        # 4 subclass
        self.gui = GUI()
        self.sound_manager = SoundManager()
        self.game_logic = GameLogic()
        self.board_score = BoardScore(self.game_logic.board)

        # inizializzo alcuni parametri che userò nel main loop
        self.running = True
        self.game_state = "intro"
        self.ai_thinking = False

        # inizializzo game e clock
        self.gui.init_game()
        self.clock = pygame.time.Clock()

    def main_game_loop(self):
        
        # finché running, gestisco mosse (events) e risposte AI
        while self.running:
            self._handle_events()
            self._handle_AI_response()
            
            # Update display if needed
            if self.gui.redraw_needed:
                self._update_display()
                self.gui.redraw_needed = False

            self.clock.tick(FPS) # mantengo frame rate

        self.sound_manager.stop_sound() # alla fine del running, chiudo i suoni per evitare memory leak (il quit lo faccio in main)

    def _handle_events(self):

        # se chiudo finestra, quitta il gioco
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            # se invece clicco col pulsante sx del mouse, gestisce il click
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_click(event.pos)
    
    def _handle_AI_response(self):

        # se siamo in playing, tocca all'AI e non c'è una promo in ballo, cerco best move
        if self.game_state == "playing" and self.ai_thinking and not self.game_logic.awaiting_promotion:
            ai_move = heuristic_best_move(self.game_logic.board, self.board_score)
            
            # Se esiste una AI best move, eseguo mossa + suono
            if ai_move:
                captured_piece = self.game_logic.board.piece_at(ai_move.to_square)
                self.game_logic.board.push(ai_move)
                self.sound_manager.play_sounds(
                    ai_move.promotion,  # Pass True if it's a promotion, otherwise False
                    captured_piece,
                    self.game_logic.board.is_check()
                )
                self.gui.redraw_needed = True
                
                # Check for game over after AI move (posso toglierlo/spostarlo nel game loop?)
                if self.game_logic.is_game_over():
                    self.game_state = "outro"
            
            self.ai_thinking = False  # AI turn is complete
        
    def _handle_mouse_click(self, pos):

        # se siamo in intro, aspetto click su play
        if self.game_state == "intro":
            if self.gui.get_submit_button_rect().collidepoint(pos):
                self.game_state = "playing"
                self.gui.redraw_needed = True
        
        # se invece siamo in playing e non è il turno dell' AI (o non devo selezionare promo menu)
        elif self.game_state == "playing" and (self.game_logic.awaiting_promotion or not self.ai_thinking):
            move_info = self.game_logic.process_click(pos)
            
            # Aggiorno display se c'è un'azione del player'
            if any([move_info["selected_square"], 
                   move_info["move"], 
                   move_info["awaiting_promotion"]]):
                self.gui.redraw_needed = True
            
            # Eseguo suoni in base al tipo di mossa
            if move_info["move"] or move_info["awaiting_promotion"]:
                self.sound_manager.play_sounds(
                    move_info["awaiting_promotion"],
                    move_info["captured_piece"],
                    move_info["is_check"]
                )

                #ripasso la palla all' AI
                self.ai_thinking = True
            
            # Check for game over
            if self.game_logic.is_game_over():
                self.game_state = "outro"
                self.gui.redraw_needed = True
        
        # se sono in GameOver, aspetto il click su Retry
        elif self.game_state == "outro":
            if self.gui.get_submit_button_rect().collidepoint(pos):
                self.game_state = "intro"
                self.game_logic = GameLogic()  # Reset game
                self.board_score = BoardScore(self.game_logic.board)  # Reset BoardScore with new board
                self.gui.redraw_needed = True

    def _update_display(self):
        """ decide cosa mandare a schermo in base al game_state intro/playing/outro """
        if self.game_state == "intro":
            self.gui.draw_extra("Chess vs AI", "Play")
        
        elif self.game_state == "playing":
            self.gui.screen.fill((255, 255, 255))
            self._draw_game_state()
        
        elif self.game_state == "outro":
            self.gui.draw_extra("Game Over", "Retry")
        
        pygame.display.flip()

    def _draw_game_state(self):
        # disegno Scacchiera, Pezzi e selected_square
        self.gui.draw_board()
        self.gui.draw_pieces(self.game_logic.board)        
        if not self.game_logic.awaiting_promotion:
            self.gui.draw_selected_square(self.game_logic.selected_square)
        
        # Disegno promo menu se serve
        if self.game_logic.awaiting_promotion:
            promotion_square = self.game_logic.get_promotion_square()
            if promotion_square is not None:
                self.gui.draw_promotion_menu(promotion_square, 
                                          self.game_logic.board.turn)

class GameLogic:
    """Classe dedicata alle regole del gioco (come si muovono i pezzi, etc.) e meccaniche speciali (es. promozione del pedone)"""

    def __init__(self):
        self.board = chess.Board()
        self.selected_square = None  
        self.awaiting_promotion = False  
        self.promotion_move = None  

    def get_square_under_mouse(self, pos):
        # per localizzare i click: ottengo la square corrispondente alla posizione del mouse
        x, y = pos
        col = x // SQUARE_SIZE
        row = 7 - (y // SQUARE_SIZE)
        return chess.square(col, row)
        
    def get_promotion_square(self):
        # ritorno la square dove sta avvenendo la promozione (serve davvero?)
        return self.promotion_move.to_square if self.awaiting_promotion else None
    
    def is_game_over(self):
        # check per GameOver (incluso Stallo)
        if self.board.is_checkmate():
            print("Checkmate!")
            return 1
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            print("Draw!")
            return 2
        return 0

    def process_click(self, pos):
        """Processa il click e return un dict con info sulla mossa fatta"""

        clicked_square = self.get_square_under_mouse(pos)
        
        # Gestisce promo selection (if any)
        if self.awaiting_promotion:
            promotion_piece = self._handle_promotion_selection(pos)
            if promotion_piece:
                move = chess.Move(self.promotion_move.from_square, self.promotion_move.to_square, promotion=promotion_piece)
                self.board.push(move)
                self.awaiting_promotion = False
                self.promotion_move = None
                self.selected_square = None
                return self._create_move_info(move=move, is_check=self.board.is_check())
            return self._create_move_info()

        # Gestisce normale selezione e movimento pezzi
        if self.selected_square is None:
            return self._handle_first_selection(clicked_square)
        else:
            return self._handle_second_selection(clicked_square)

    def _handle_first_selection(self, clicked_square):
        """Con il primo click, seleziona un pezzo (aggiornando il dict move_info)"""
        piece = self.board.piece_at(clicked_square)
        if piece and piece.color == self.board.turn:
            self.selected_square = clicked_square
            return self._create_move_info(square=clicked_square)
        return self._create_move_info()

    def _handle_second_selection(self, clicked_square):
        """Con il secondo click, seleziona un pezzo diverso o esegue una mossa. (Un po' contorto questo?)"""
        piece = self.board.piece_at(clicked_square)
        
        # se clicco un altro mio pezzo, seleziono quello
        if piece and piece.color == self.board.turn and clicked_square != self.selected_square:
            self.selected_square = clicked_square
            return self._create_move_info(square=clicked_square)
        
        # altrimenti se ho cliccato una colonna libera, provo a simulare la mossa
        move = chess.Move(self.selected_square, clicked_square)
        
        # se è una mossa di promozione, aggiorno l'info nel dict
        current_piece = self.board.piece_at(self.selected_square)
        if self._is_promotion_move(current_piece, clicked_square):
            self.awaiting_promotion = True
            self.promotion_move = move
            return self._create_move_info(square=self.selected_square, awaiting_promotion=True)
        
        # se la mossa è legal, la ritorno
        if move in self.board.legal_moves:
            captured_piece = self.board.piece_at(clicked_square)
            self.board.push(move)
            self.selected_square = None
            return self._create_move_info(move=move, captured_piece=captured_piece, is_check=self.board.is_check())
        
        # se non è legal, non faccio nulla e torno al pezzo selezionato
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
        """Creo un move_info dict standardizzato"""

        return {
            'selected_square': square,
            'move': move,
            'captured_piece': captured_piece,
            'is_check': is_check,
            'awaiting_promotion': awaiting_promotion or self.awaiting_promotion
        }

class BoardScore:
    """Classe dedicata alla creazione degli score, utilizzato da Heuristic AI per best move"""

    def __init__(self, board):
        self.board = board

    def get_score(self):
        """Classe dedicata alla creazione degli score, utilizzato da Heuristic AI per best move. Score positivo = vantaggio bianco, Score negativo = vantaggio nero"""

        # se fine partita
        if self.board.is_checkmate():
            return -1000 if self.board.turn == chess.WHITE else 1000
        if self.board.is_stalemate():
            return 0
        
        # altrimenti inizializzo score e poi lo calcolo iterando per tutti i pezzi score + altri modifier
        score = 0
        for square, piece in self.board.piece_map().items():
            piece_value = self._get_piece_value(piece, square)
            
            # Add the piece value
            if piece.color == chess.WHITE:
                score += piece_value
            else:
                score -= piece_value

        score += self._get_check_status() # scacco
        score += self._get_mobility_advantage() # Reward for having more moves

        return score

    def _get_piece_value(self, piece, square=None):
        """Ritorno un valore del pezzo pesato per posizione e minacce subite"""

        # valore = pezzo + posiz. pezzo
        base_value = PIECE_VALUES.get(piece.piece_type, 0) 
        
        if piece.piece_type != chess.KING and square is not None:  # Avoid the king

            # riduco il base value in base al threat 
            base_value += (PIECE_POS_TABLE.get(piece.piece_type, [])[square] if piece.piece_type in PIECE_POS_TABLE else 0)
            color = piece.color
            if self.board.is_attacked_by(not color, square):
                threat_value = self._get_threat_value(square, base_value, color)
                return base_value * (1 - threat_value)

        return base_value

    def _get_threat_value(self, square, piece_value, color):
        """Calculate threat value considering piece type, attackers/defenders strength"""
        threat_value = 0

        if piece_value == PIECE_VALUES[chess.KING]:
            return threat_value

        # Get attackers and defenders with their values 
        attackers = [(self.board.piece_at(sq), self._get_piece_value(self.board.piece_at(sq))) for sq in self.board.attackers(not color, square)]
        defenders = [(self.board.piece_at(sq), self._get_piece_value(self.board.piece_at(sq)))      for sq in self.board.attackers(color, square)]

        if not attackers:
            return threat_value

        # Consider lowest value attacker
        min_attacker = min(attackers, key=lambda x: x[1])
        
        # If piece can be captured by lower value piece
        if min_attacker[1] < piece_value:
            threat_value = (piece_value - min_attacker[1]) / piece_value

        # If undefended or attackers > defenders, assess threat based on relative strengths
        if not defenders or len(attackers) > len(defenders):
            attacker_strength = sum(a[1] for a in attackers)
            defender_strength = sum(d[1] for d in defenders) if defenders else 0
            threat_value = max(threat_value, min(0.9, attacker_strength / (attacker_strength + defender_strength)))
        if color ==chess.WHITE:
            threat_value /=2
        return threat_value

    def _get_check_status(self):

        # aggiungo o tolgo un punto per scacco
        if self.board.is_check():
            return -1 if self.board.turn == chess.WHITE else 1
        return 0
    
    def _get_mobility_advantage(self):
        """Calculate mobility advantage for white and black."""
        white_mobility = self._get_possible_moves(chess.WHITE)
        black_mobility = self._get_possible_moves(chess.BLACK)
        return 0.05 * (white_mobility - black_mobility)

    def _get_possible_moves(self, color):
        """Conta le legal moves possibili per il colore in arg"""
        
        original_turn = self.board.turn # Save the current turn
        self.board.turn = color # Set the board's turn to the color we want to count moves for
        move_count = len(list(self.board.legal_moves)) # Count legal moves
        self.board.turn = original_turn # Restore the original turn

        return move_count

class GUI:
    """Classe dedicata alla Graphic User Intarface."""

    def __init__(self):
        self.screen = None
        self.font = None
        self.redraw_needed = True
    
    def init_game(self):
        """Inizializzo pygame, setto screen e carico i font."""
        
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess vs AI")
        
        # Try different fonts
        try:
            self.font = pygame.font.SysFont(FONT[0], FONT_SIZE)  # Windows
        except:
            try:
                self.font = pygame.font.SysFont(FONT[1], FONT_SIZE)  # Alternative
            except:
                self.font = pygame.font.Font(None, FONT_SIZE)  # Fallback

    def get_submit_button_rect(self):
        """Bottone per menu intro e outro"""

        return pygame.Rect(WIDTH // 2 - 90, HEIGHT // 2, 200, 100)

    def draw_extra(self, title, CTA):
        """Disegno schermate intro e outro, fornendo titolo e CTA bottone"""

        self.screen.fill((0, 0, 0))
        text = self.font.render(title, True, (255, 255, 255))
        self.screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 3))
        
        submit_button = self.get_submit_button_rect()
        pygame.draw.rect(self.screen, (0, 255, 0), submit_button)
        button_text = self.font.render(CTA, True, (255, 255, 255))
        self.screen.blit(button_text, (submit_button.x + (submit_button.width - button_text.get_width()) // 2, submit_button.y-5))
    
    def draw_board(self):
        """Disegno scacchiera"""

        for row in range(DIMENSION):
            for col in range(DIMENSION):
                color = LIGHT_COLOR if (row + col) % 2 == 0 else DARK_COLOR
                pygame.draw.rect(self.screen, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    
    def draw_selected_square(self, square):
        """Disegno selected square (quadrato giallo evidenziante)"""

        if square is not None:  # If a square is selected
            col = chess.square_file(square)
            row = chess.square_rank(square)
            pygame.draw.rect(self.screen, (255, 255, 0, 50), pygame.Rect(col * SQUARE_SIZE, (7 - row) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3)

    def draw_pieces(self, board=None):
        """Disegno pezzi"""

        if board is None:
            board = chess.Board()  # Default to starting position if no board provided

        for square in chess.SQUARES:
            piece = board.piece_at(square)  # Get the piece at the given square
            if piece:  # If the piece exists
                col = chess.square_file(square)
                row = chess.square_rank(square)
                
                # Set the piece's color and render it on the screen
                self._render_pieces(UNICODE_PIECES[piece.symbol()], piece.color, col * SQUARE_SIZE + SQUARE_SIZE // 2, (7 - row) * SQUARE_SIZE + SQUARE_SIZE // 2)

    def draw_promotion_menu(self, square, is_white):
        """Disegno menù promozione pedone"""

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
                pygame.draw.line(self.screen, (100, 100, 100), (menu_x, base_y + (i + 1) * SQUARE_SIZE), (menu_x + SQUARE_SIZE, base_y + (i + 1) * SQUARE_SIZE))

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
        self._render_pieces(UNICODE_PIECES[piece], is_white, x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2)

    def _render_pieces(self, piece, color_condition, x, y):
        """Disegno pezzo."""

        color = (255, 255, 255) if color_condition else (0, 0, 0)
        text = self.font.render(piece, True, color)
        text_rect = text.get_rect(center=(x,y))
        self.screen.blit(text, text_rect)

class SoundManager:
    """Classe dedicata a caricare e gestire i suoni"""

    def __init__(self):
        pygame.mixer.init()
        self.sounds = {}
        self.load_sounds()

    def load_sounds(self):
        """gestisce eccezioni, carica i suoni, regola i volumi"""

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
                
        for sound_name, volume in [("move", 0.4), ("gnam", 1.0), ("promo", 1.2), ("check", 1.2)]:
            if self.sounds[sound_name]:
                self.sounds[sound_name].set_volume(volume)

    def play_sounds(self, awaiting_promotion, captured_piece, is_check):

        # in base al move_info dict performa un certo suono        
        self.sounds["move"].play()
        if awaiting_promotion:
            self.sounds["promo"].play()
        if captured_piece:
            self.sounds["gnam"].play()
        if is_check:
            self.sounds["check"].play()

    def stop_sound(self):

        # chiude tutto
        for sound in self.sounds.values():
            if sound:
                sound.stop()
        pygame.mixer.quit()


def main():
    game = ChessGame()  
    game.main_game_loop()
    pygame.quit() 

if __name__ == "__main__":
    main()  