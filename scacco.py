""" TO DO: """
# mi informo su quale NN posso trainare i dati (parto da un pretrainato? pro & cons?)
# inizio a trainare i dati

""" MINOR: """
# evidenzio checkmate (creo draw_checkmate? highlight rosso su re e attacker?)
# colonna A1 non si seleziona
# sul retry AI muove bianco e si scazza tutto
# AI minimax non evita la patta per ripetizione se è in vantaggio
# faccio redraw solo sulle square in cui serve (evitando di appesantire le performance ridisegnando ogni volta tutto)

""" VERY MINOR: """
# suoni mangio AI non sempre a fuoco
# scelgo bianco o nero
# aggiungere pulsante per arrendersi / chiedere la patta (con AI che accetta se suo score >0)
# customizzo grafica pezzi
# Segnalo Mossa Irregolare: suono, highlight rosso, mostro possibili legal moves? 

""" NEXT TO DO: """
# come quarto importo Stockfish per showcase API management o mi lancio in un NN supervised learning?

""" DOUBT: """
# sql
# documentation? (es spiegazione lunga funzioni)
# test?
# AI ELO evaluation?

import textwrap
import pygame
import chess
import AI.basic_test
from AI.heuristic import heuristic_best_move
from AI.minimax import minimax_best_move
import AI.basic_test
from constants import DIMENSION, SQUARE_SIZE, WIDTH, HEIGHT, LIGHT_COLOR, DARK_COLOR, UNICODE_PIECES, FPS, FONT, FONT_SIZE, PIECE_VALUES, PIECE_POS_TABLE, INTRO, OUTRO


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
        self.opponent = 0

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
            if self.opponent == 1:
                ai_move = heuristic_best_move(self.game_logic.board, self.board_score)
            elif self.opponent == 2:
                ai_move = minimax_best_move(self.game_logic.board, self.board_score)
            elif self.opponent == 3:
                ai_move = AI.basic_test.test_model(self.game_logic.board)
            
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
                    self._update_display()
                    pygame.display.flip()
                    pygame.time.delay(4000)  # 2 second delay
                    self.game_state = "outro"
            
            self.ai_thinking = False  # AI turn is complete
        
    def _handle_mouse_click(self, pos):

        # se siamo in intro, aspetto click su una delle 4 AI (che verrà selezionata come opponent)
        if self.game_state == "intro":
            if self.gui.get_submit_button_rect()["Heuristic"].collidepoint(pos):
                self.opponent = 1
                self.game_state = "playing"
                self.gui.redraw_needed = True
        
            if self.gui.get_submit_button_rect()["Minimax"].collidepoint(pos):
                self.opponent = 2
                self.game_state = "playing"
                self.gui.redraw_needed = True

            if self.gui.get_submit_button_rect()["CNN"].collidepoint(pos):
                self.opponent = 3
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
            if self.gui.get_submit_button_rect()["GameOver"].collidepoint(pos):
                self.game_state = "intro"
                self.game_logic = GameLogic()  # Reset game
                self.board_score = BoardScore(self.game_logic.board)  # Reset BoardScore with new board
                self.gui.redraw_needed = True

    def _update_display(self):
        """ decide cosa mandare a schermo in base al game_state intro/playing/outro """
        
        if self.game_state == "intro":
            self.gui.draw_extra()        
        
        elif self.game_state == "playing":
            self.gui.screen.fill((255, 255, 255))
            self._draw_game_state()
        
        elif self.game_state == "outro":
            self.gui.draw_extra(self.game_logic.is_game_over())
        
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
                self.gui.draw_promotion_menu(promotion_square, self.game_logic.board.turn)

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
            if self.board.turn:
                return 1 # white lose = AI won
            return 2 # black lose = AI lost
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_repetition(3):
            print("Draw!")
            return 3
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
        if self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_repetition(3):
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
            threat_value *=0.2
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
        
        self.piece_font = None      
        self.title_font = None
        self.desc_font = None
        self.cta_font = None
        self.head_font = None
        
        self.screen = None
        self.redraw_needed = True
        self.intro_image = pygame.image.load('static/title_image.png')
 
    def init_game(self):
        """Inizializzo pygame, setto screen e carico i font."""
        
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess vs AI")
        
        try:
            self.piece_font = pygame.font.SysFont(FONT[0], FONT_SIZE[0])  # Piece font
            self.title_font = pygame.font.Font(FONT[2], FONT_SIZE[1])  # Title font
            self.desc_font = pygame.font.Font(FONT[3], FONT_SIZE[2])   # Description font
            self.cta_font = pygame.font.SysFont(FONT[0], FONT_SIZE[3])    # CTA font
            self.head_font = pygame.font.Font(FONT[4], FONT_SIZE[4])    # head font

        
        except Exception as e:
            print(f"Error loading fonts: {e}")
            
            try:
                self.piece_font = pygame.font.SysFont(FONT[0], FONT_SIZE[0])
                self.title_font = pygame.font.SysFont(FONT[0], FONT_SIZE[1])
                self.desc_font = pygame.font.SysFont(FONT[0], FONT_SIZE[2])
                self.cta_font = pygame.font.SysFont(FONT[0], FONT_SIZE[3])
                self.head_font = pygame.font.SysFont(FONT[0], FONT_SIZE[4])
            except:
                self.piece_font = pygame.font.Font(None, FONT_SIZE[0])
                self.title_font = pygame.font.Font(None, FONT_SIZE[1])
                self.desc_font = pygame.font.Font(None, FONT_SIZE[2])
                self.cta_font = pygame.font.Font(None, FONT_SIZE[3])
                self.head_font = pygame.font.SysFont(None, FONT_SIZE[4])

    def get_submit_button_rect(self):
        """Bottone per menu intro e outro"""
        
        # Use the same calculations as in draw_extra2
        rect_width = WIDTH // 4
        padding = 10
        button_width = 100
        button_height = 40
        
        # Calculate buttons x positions using the same rectangle positions
        button_positions = {
            "Heuristic": pygame.Rect(
                (0 * rect_width + padding) + (rect_width - 2 * padding - button_width) // 2,  # First rectangle
                HEIGHT // 4 + HEIGHT // 1.5 - button_height - 10,  # Same y position as in draw_extra2
                button_width,
                button_height+10
            ),
            "Minimax": pygame.Rect(
                (1 * rect_width + padding) + (rect_width - 2 * padding - button_width) // 2,  # Second rectangle
                HEIGHT // 4 + HEIGHT // 1.5 - button_height - 10,
                button_width,
                button_height+10
            ),
            "CNN": pygame.Rect(
                (2 * rect_width + padding) + (rect_width - 2 * padding - button_width) // 2,  # Third rectangle
                HEIGHT // 4 + HEIGHT // 1.5 - button_height - 10,
                button_width,
                button_height+10
            ),
            "Reinforce": pygame.Rect(
                (3 * rect_width + padding) + (rect_width - 2 * padding - button_width) // 2,  # Fourth rectangle
                HEIGHT // 4 + HEIGHT // 1.5 - button_height - 10,
                button_width,
                button_height+10
            ),
            "GameOver": pygame.Rect(  # Keep GameOver button centered
                WIDTH // 2 - 90,
                HEIGHT // 2,
                200,
                100
            )
        }
        
        return button_positions

    def draw_extra(self, id=0):
        """ disegna schermate extra game (intro e outro)"""

        self.screen.fill((0,0,0))

        # gestisco la schermata di gameover
        if id:
            text = self.title_font.render(OUTRO[id], True, (255, 255, 255))
            self.screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 3))
            
            submit_button = self.get_submit_button_rect()["GameOver"]
            pygame.draw.rect(self.screen, (0, 255, 0), submit_button)
            button_text = self.cta_font.render("Retry", True, (255, 255, 255))
            self.screen.blit(button_text, (submit_button.x + (submit_button.width - button_text.get_width()) // 2, submit_button.y-5))
            return

        # titolo nero su sfondo bianco
        title_rendered = self.head_font.render("Chess vs AI", True, (0,20,0))
        text_width, text_height = title_rendered.get_size()
        highlight_rect = pygame.Rect(WIDTH // 2.6 - 10, HEIGHT // 12 - 5, text_width + 20, text_height + 10)
        pygame.draw.rect(self.screen, (255,255,255), highlight_rect)

        self.screen.blit(title_rendered, (WIDTH  // 2.6, HEIGHT // 12 ))
        self.screen.blit(self.intro_image, (WIDTH  // 12, HEIGHT // 23))

        for i, content in enumerate(INTRO):
            
            # rettangoli
            rect = pygame.Rect((i * (WIDTH // 4)) + 10, HEIGHT // 4, (WIDTH // 4) - 2 * 10, (HEIGHT // 1.45)) 
            pygame.draw.rect(self.screen, ((255,255,255)), rect, 2)  
            
            # scritte (titolo+descrizione)
            self._wrapper(content['title'], 10, 225, self.title_font, rect.x) 
            self._wrapper(content['description'], 20, 90, self.desc_font, rect.x, rect.y) 

            # bottoni CTA
            button_rect = self.get_submit_button_rect()[INTRO[i]["title"]] 
            pygame.draw.rect(self.screen, ((0,60,0)), button_rect)

            # testo CTA
            button_text = self.cta_font.render("Play", True, ((255,255,255))) # crea testo bottone
            self.screen.blit(button_text, (button_rect.x + (button_rect.width - button_text.get_width()) // 2, button_rect.y + (button_rect.height - button_text.get_height()) // 2))

        pygame.display.flip()

    def _wrapper(self, text, width, y_offset, font, x_pos, y_pos=0):
        
        # textwrap chiude il testo in un box
        wrapped_text = textwrap.wrap(text, width)  
        for line in wrapped_text:
            text_rendered = font.render(line, True, (255, 255, 255))
            self.screen.blit(text_rendered, (x_pos + 10, y_pos + y_offset))
            y_offset += 30  # Aggiunge uno spazio dopo ogni riga

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
        text = self.piece_font.render(piece, True, color)
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