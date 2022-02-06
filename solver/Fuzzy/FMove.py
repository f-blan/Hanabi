from solver.Fuzzy.FDeck import FDeck
from solver.Move import Move
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
    def define_play(self, card_n):
        return super().define_play(card_n)
    
    def define_discard(self, card_n):
        return super().define_discard(card_n)

    def define_hint(self, player, h_type, value):
        self.hd_player = player.order
        self.hd_type=h_type
        self.hd_value = value
        if h_type == 0:
            super().define_hint(player.name, "value", utils.decode_value(value))
        else:
            super().define_hint(player.name, "color", utils.decode_color(value))

    #TODO: add some minor endgame info (e.g. discards should be valued less than plays and hints)

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
            self.score = playability - 0.5 * (rt)


        return self.score

    def EvaluateDiscard(self, discardability: float, blueTokens: int):
        if discardability == 1:
            self.score = 0.95
            return
        #blue token contribution: the more blue tokens there are available, the less valuable is the discard
        bt = (blueTokens*1.0)/8
        
        #a play with max playability will generally have a score higher than a discard with max discardability:
        #playing makes other cards playable, while if you stole a play from someone they still gained a safe discard
        #hence why *0.75. The only case where a discard is better than a play is when there are no blue tokens

        self.score = max(discardability*bt, 0.90) 
        return

    def EvaluateHint(self, hinted_player, blueTokens):
        #cards with the same value (the other cards that will be targeted by the hint)
        same_value = hinted_player.cards[self.hd_type, :] == self.hd_value

        #filter wtr to hints already given
        filter_mask = hinted_player.hint_tracker[self.hd_type] == False


        same_value = np.logical_and(same_value, filter_mask)

        #indices of cards benefiting from the hint
        same_value = same_value.nonzero()

        #for each card, compute how valuable the knowledge given by the hint is
        gain = (hinted_player.playabilities[same_value]+ hinted_player.discardabilities[same_value])*0.5
        tot_gain = np.sum(gain)

        #blue token contribution, the least tokens are available, the least score gets the hint
        bt = 0#0.35 * (blueTokens*1.0)/8
        if blueTokens == 8:
            bt = 0.35
        self.score = max(tot_gain - bt, 0.99)
        return self.score

    def ToKey(self):
        string = ""
        if self.type == 0:
            string+=f"d{self.card_n}"
        elif self.type ==1:
            string+=f"p{self.card_n}"
        else:
            string+=f"h_p{self.hd_player}t{self.hd_type}v{self.hd_value}"
        return string
        
    def ToString(self):
        if self.type == 0:
            return f"Discard move of {self.card_n} with score {self.score}"
        if self.type == 1:
            return f"Play move of {self.card_n} with score {self.score}"
        else:
            if self.hd_type==1:
                return f"Hint move of color {self.h_value} to player {self.h_player} with score {self.score}" 
            else:
                return f"Hint move of value {self.h_value} to player {self.h_player} with score {self.score}"
