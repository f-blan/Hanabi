import numpy as np

#this class represents the cards in the deck and in the player's unknown hand
class Deck():
    def __init__(self):

        #name is deck, but this is meant to store cards both in the deck and in the hands of the agent
        
        self.deck = np.array([
                                [3,3,3,3,3],
                                [2,2,2,2,2],
                                [2,2,2,2,2],
                                [2,2,2,2,2],
                                [1,1,1,1,1]
                                ], dtype=np.int16) 

    def RemoveCards(self, cards):
        pass