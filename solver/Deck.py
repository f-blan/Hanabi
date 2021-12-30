import numpy as np

#this class represents the cards in the deck and in the player's unknown hand
class Deck():
    def __init__(self):
        self.cards = np.array([
                                [3,3,3,3,3],
                                [2,2,2,2,2],
                                [2,2,2,2,2],
                                [2,2,2,2,2],
                                [1,1,1,1,1]
                                ], dtype=np.int16) 