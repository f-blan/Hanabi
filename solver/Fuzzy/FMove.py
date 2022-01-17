from solver.Fuzzy.FDeck import FDeck
from .. import Move
import numpy as np
from .. import utils

class FMove(Move):
    def __init__(self, type):
        super().__init__(type)
        self.score = 0

    """
        Criteria for move evaluation:
        -a move that grants a blue token has default +0.25. -0.25 if it uses a blue token
        -a play that can be done only by the current player has 1.2 multiplier
        -hints are evaluated on number of cards hinted, their playability and their discardability
        -a play without maximum playability is not penalised unless we only have
    """

    def define_hint(self, player_n, type, value):
        super().define_hint(player_n, type, value)
        if type == "color":
            self.type = 1
            self.value = utils.encode_color(value)
        else:
            self.type = 0
            self.value = utils.encode_value(value)


    def EvaluatePlay(self, playability: float, redTokens: int):
        if playability == 1:
            """
            multiplier = 1
            add = 0
            if deck.cards_in_game[playedCard[0], playedCard[1]] == 1:
                multiplier = 1.2
            
            if playedCard[0] == 4:
                add = 0.25
            self.score = (1+add)*multiplier
            """
            self.score = 1
        else:
            rt = redTokens*1.0
            self.score = playability - 0.5 * (rt/2)


        return self.score

    def EvaluateDiscard(self, discardability: float, blueTokens: int):
        #blue token contribution: the less blue token there are available, the more valuable is the discard
        bt = 0.35 * (blueTokens*1.0)/8
        
        #a play with max playability will generally have a score higher than a discard with max discardability:
        #playing makes other cards playable, while if you stole a play from someone they still gained a safe discard
        #hence why *0.75. The only case where a discard is better than a play is when there are no blue tokens

        self.score = (discardability + bt)*0.75 
        return 0

    def EvaluateHint(self, hinted_player, blueTokens):
        #cards with the same value (the other cards that will be targeted by the hint)
        same_value = hinted_player.cards[self.type, :] == self.value

        #filter wtr to hints already given
        filter_mask = hinted_player.hint_tracker[self.type] == False


        same_value = np.logical_and(same_value, filter_mask)

        #indices of cards benefitting from the hint
        same_value = same_value.nonzero()

        #for each card, compute how valuable the knowledge given by the hint is
        gain = (hinted_player.playabilities[same_value]+ hinted_player.discardabilities[same_value])*0.5
        tot_gain = np.sum(gain)

        #blue token contribution, the least tokens are available, the least score gets the hint
        bt = 0.35 * (blueTokens*1.0)/8
        self.score = gain - bt
        return self.score