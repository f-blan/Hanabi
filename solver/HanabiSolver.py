from .Deck import Deck
import numpy as np

class HanabiSolver:
    def __init__(self):
        self.fireworks = np.array([-1,-1,-1,-1,-1], dtype=np.int16) #no cards = -1, card 1 is in firework = 0 etc.
        self.blue_tokens = 0
        self.red_tokens = 0
        self.max_b = 8
        self.max_r = 3
    def FindMove(self):
        """Find the next move for main player given current informations"""
        pass
    
    def Enforce(self):
        """improve current informations while it's another player's turn"""
        pass

    def RecordMove(self, data, type):
        """whenever a move has been confirmed, update the state"""
        pass
    def HintsToString(self, player_name):
        pass
        
    