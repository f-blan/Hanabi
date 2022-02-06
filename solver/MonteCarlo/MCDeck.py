#from turtle import fd
from solver.MonteCarlo.NodeDeck import NodeDeck
from solver.MonteCarlo.MCPlayer import MCPlayer
import numpy as np

class MCDeck():
    """
        Function thought as auxiliary to the Enforce method of MCSolver, but the idea was scrapped.
        In the current version, this class represents a deck up to date with the current server
        state. It contains and instance of NodeDeck which is the class used in MCTS nodes
    """

    def __init__(self, main_player:MCPlayer, n_players: int):
        #super().__init__(main_player, n_players)

        self.ndeck = NodeDeck(main_player, n_players)

        #unused values
        self.last_unknown_index = self.ndeck.n_cards_in_deck
        self.card_value_per_index = np.zeros(self.ndeck.n_cards_in_deck, dtype=np.int16)-1
        self.card_color_per_index = np.zeros(self.ndeck.n_cards_in_deck, dtype=np.int16)-1

    def RemoveCards(self, to_remove: np.ndarray):
        self.ndeck.RemoveCards(to_remove)
        #self.last_unknown_index -= to_remove.shape[1]
    
    def RemoveCardsFromGame(self, to_remove: np.ndarray):
        self.ndeck.RemoveCardsFromGame(to_remove)
    
    def update_expected_values(self, fireworks: np.ndarray, main_player):
        self.ndeck.update_expected_values(fireworks, main_player)
    
    def evaluate_known_cards(self, fireworks,to_evaluate,hard_unknowns):
        return self.ndeck.evaluate_known_cards(to_evaluate,fireworks,hard_unknowns, 0)
    
    def evaluate_unknown_cards(self,n_cards, fireworks, hints, hard_unknowns):
        return self.ndeck.evaluate_unknown_cards(n_cards,fireworks, hints, hard_unknowns)

    def DeckToString(self):
        return self.ndeck.DeckToString()
