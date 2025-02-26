import chess

DIMENSION = 8
SQUARE_SIZE = 110
WIDTH = HEIGHT = DIMENSION * SQUARE_SIZE
FPS = 30

LIGHT_COLOR = (240, 217, 181)
DARK_COLOR = (181, 136, 99)

UNICODE_PIECES = {
    'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
    'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟'
}

FONT = ['segoe ui symbol', 'arial unicode ms', r'./static/FingerPaint.ttf', r'./static/Combo.ttf', r'./static/Barriecito.ttf']
FONT_SIZE = [int(SQUARE_SIZE*0.75), 38, 22, 32, 70]

PIECE_VALUES = {
            chess.PAWN: 10,
            chess.KNIGHT: 32,
            chess.BISHOP: 33,
            chess.ROOK: 50,
            chess.QUEEN: 90,
            chess.KING: 1000
        }


PIECE_POS_TABLE = {
    chess.PAWN : [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 5, 5, 5, 5, 5, 5, 5,
        1, 1, 2, 3, 3, 2, 1, 1,
        0.5, 0.5, 1, 2.5, 2.5, 1, 0.5, 0.5,
        0, 0, 0, 2, 2, 0, 0, 0,
        0.5, -0.5, -1, 0, 0, -1, -0.5, 0.5,
        0.5, 1, 1, -2, -2, 1, 1, 0.5,
        0, 0, 0, 0, 0, 0, 0, 0
    ],

    chess.KNIGHT : [
        -5, -4, -3, -3, -3, -3, -4, -5,
        -4, -2, 0, 0, 0, 0, -2, -4,
        -3, 0, 1, 1.5, 1.5, 1, 0, -3,
        -3, 0.5, 1.5, 2, 2, 1.5, 0.5, -3,
        -3, 0, 1.5, 2, 2, 1.5, 0, -3,
        -3, 0.5, 1, 1.5, 1.5, 1, 0.5, -3,
        -4, -2, 0, 0.5, 0.5, 0, -2, -4,
        -5, -4, -3, -3, -3, -3, -4, -5
    ],

    chess.BISHOP : [
        -2, -1, -1, -1, -1, -1, -1, -2,
        -1, 0, 0, 0, 0, 0, 0, -1,
        -1, 0, 0.5, 1, 1, 0.5, 0, -1,
        -1, 0.5, 0.5, 1, 1, 0.5, 0.5, -1,
        -1, 0, 1, 1, 1, 1, 0, -1,
        -1, 1, 1, 1, 1, 1, 1, -1,
        -1, 0.5, 0, 0, 0, 0, 0.5, -1,
        -2, -1, -1, -1, -1, -1, -1, -2
    ],

    chess.ROOK : [
        0, 0, 0, 0, 0, 0, 0, 0,
        0.5, 1, 1, 1, 1, 1, 1, 0.5,
        -0.5, 0, 0, 0, 0, 0, 0, -0.5,
        -0.5, 0, 0, 0, 0, 0, 0, -0.5,
        -0.5, 0, 0, 0, 0, 0, 0, -0.5,
        -0.5, 0, 0, 0, 0, 0, 0, -0.5,
        -0.5, 0, 0, 0, 0, 0, 0, -0.5,
        0, 0, 0, 0.5, 0.5, 0, 0, 0
    ],

    chess.QUEEN : [
        -2, -1, -1, -0.5, -0.5, -1, -1, -2,
        -1, 0, 0, 0, 0, 0, 0, -1,
        -1, 0, 0.5, 0.5, 0.5, 0.5, 0, -1,
        -0.5, 0, 0.5, 0.5, 0.5, 0.5, 0, -0.5,
        0, 0, 0.5, 0.5, 0.5, 0.5, 0, -0.5,
        -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0, -1,
        -1, 0, 0.5, 0, 0, 0, 0, -1,
        -2, -1, -1, -0.5, -0.5, -1, -1, -2
    ],

    chess.KING : [
        -3, -4, -4, -5, -5, -4, -4, -3,
        -3, -4, -4, -5, -5, -4, -4, -3,
        -3, -4, -4, -5, -5, -4, -4, -3,
        -3, -4, -4, -5, -5, -4, -4, -3,
        -2, -3, -3, -4, -4, -3, -3, -2,
        -1, -2, -2, -2, -2, -2, -2, -1,
        2, 2, 0, 0, 0, 0, 2, 2,
        2, 3, 1, 0, 0, 1, 3, 2
    ]
}


INTRO = [
    {"title": "Heuristic", "description": "Questo algoritmo rudimentale nasce negli anni '40. Pensa solo al proprio turno, valutando tutte le opzioni possibili, e scegliendo la mossa che gli porta il vantaggio immediato maggiore.\nE' Molto veloce a rispondere.", "cta_text": "Sfida Heuristic AI"},
    {"title": "Minimax", "description": "Avanziamo negli anni '50. Questo algoritmo analizza fino a tre turni futuri, anticipando le mosse dell'avversario. Leggero aumento del tempo di reazione, seppur mitigato dalla variante alpha-beta pruning", "cta_text": "Click 2"},
    {"title": "Super V", "description": "2020 circa, questo modello si basa su un supervised learning attraverso reti neruali CNN, migliorate da residual blocks e attention mechanism; è stato trainato su milioni di partite di veri scacchisti e ha 'imparato' a giocare come loro!", "cta_text": "Click 3"},
    {"title": "Reinforce", "description": "Una delle tecniche più interessanti del Machine Learning: il Reinforcement Learning, combinato con MCTS in self-play: questa AI ha imparato a giocare sfidando se stessa per milioni di partite! E ora vuole mettersi alla prova sfidando un avversario umano :)", "cta_text": "Click 4"},
]

OUTRO = [
    "Ops, something went wrong here", 
    "Uggh, you lost!",
    "Congratulation, you win!",
    "Oh, that's a draw!"
]