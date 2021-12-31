import Deck
import numpy as np

class HanabiSolver():
    def __init__(self):
        self.deck = Deck()
        self.fireworks = np.array([-1,-1,-1,-1,-1], dtype=np.int16) #no cards = -1, card 1 is in firework = 0 etc.
        self.blue_tokens = 0
        self.red_tokens = 0
        self.max_b = 8
        self.max_r = 3
    def FindMove():
        """Find the next move for main player given current informations"""
        pass
    
    def Enforce():
        """improve current informations while it's another player's turn"""
        pass

    def RecordMove(data, type, opt=None):
        """whenever a move has been confirmed, update the state"""
        pass
        
    