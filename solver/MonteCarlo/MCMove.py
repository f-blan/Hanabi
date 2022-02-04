from solver.Fuzzy.FMove import FMove
#from solver.MonteCarlo.MCPlayer import MCPlayer

import numpy as np

from solver.MonteCarlo.NodeDeck import NodeDeck

class MCMove(FMove):
    def __init__(self, type, playerOrder,known,cardHandIndex = -1, used_card = None, drawHappened= True, thunder = False):
        super().__init__(type)
        self.thunder = thunder
        #index of the card if play/discard
        self.cardHandIndex = cardHandIndex
        #index of player who moved
        self.playerOrder = playerOrder
        self.drawHappened = drawHappened

        self.used_card = used_card

        #if we know the details of the played/discarded/hinted card(s)
        self.known = known
        self.drawn_card = None
        self.known_draw = False
        

    def finalize_discard(self, drawn_card):
        self.drawn_card = drawn_card
        self.known_draw=True
    
    def finalize_play(self, drawn_card):
        #self.cardIndex = cardIndex
        self.drawn_card = drawn_card
        self.known_draw = True
    
    def finalize_hint(self,destination, positions=None):
        self.positions = positions
        self.destination = destination
        if self.known == False:
            #fix: we don't know what we're hinting at, some parameter of parent class needs to be initialized
            self.hd_type = 0
            self.hd_value = 0

    def EvaluateHint(self, hinted_player, blueTokens: int, deck: NodeDeck, known = 0):
        #This function is a variant of FMove that takes into account that players may have hard unknown cards in their hand

        #blue token contribution, the least tokens are available, the least score gets the hint
        bt = 0#0.35 * (blueTokens*1.0)/8
        if blueTokens == 8:
            bt = 0.35

        if known == 0:
            #we're evaluating a hint on a hard unknown
            #assert np.any(hinted_player.hard_unknowns)
            #ht = hinted_player.hint_tracker[:,hinted_player.hard_unknowns]
            #assert np.any(ht == False)
            
            #consider we hit only one card for this hint
            tot_gain = (deck.discardability_rc + deck.playability_rc)*0.5
        elif known == 1:
            #evaluate an hint on a card in the hands of the agent that is not an hard unknown
            #we hint for the most playable card that was not already hinted
            filter_mask = np.logical_and(hinted_player.hint_tracker[0], hinted_player.hint_tracker[1])==False
            max_index = np.argmax(hinted_player.playabilities)
            tot_gain= (hinted_player.playabilities[max_index]+ hinted_player.discardabilities[max_index])*0.5 
        else:
            #we called this function assuming we know the cards of the player (to a certain extent)
            #filter out hard unknowns for the computations
            known_cards = hinted_player.cards[:,hinted_player.hard_unknowns == False]
            known_hints = hinted_player.hint_tracker[:,hinted_player.hard_unknowns == False]
            #cards with the same value (the other cards that will be targeted by the hint)
            same_value = known_cards[self.hd_type, :] == self.hd_value

            #filter wtr to hints already given
            filter_mask = known_hints[self.hd_type] == False


            same_value = np.logical_and(same_value, filter_mask)

            #indices of cards benefiting from the hint
            same_value = same_value.nonzero()

            #for each card, compute how valuable the knowledge given by the hint is
            gain = (hinted_player.playabilities[same_value]+ hinted_player.discardabilities[same_value])*0.5
            tot_gain = np.sum(gain)

        self.score = tot_gain - bt
        return self.score
    
    def ToKey(self):
        if self.known == True:
            return super().ToKey()
        else:
            string = ""
            if self.type == 0:
                string+=f"d{self.card_n}u"
            elif self.type ==1:
                string+=f"p{self.card_n}u"
            else:
                string+=f"h{self.destination}u"
            return string
    
    def ToString(self):
        str = f"Player{self.playerOrder} performed "
        if self.known == True:
            if self.type == 0:
                return str+f"Discard move of {self.card_n} with score {self.score}"
            elif self.type == 1:
                return str+f"Play move of {self.card_n} with score {self.score}"
            else:
                if self.hd_type==1:
                    return str+f"Hint move of color {self.h_value} to player {self.h_player} with score {self.score}" 
                else:
                    return str+f"Hint move of value {self.h_value} to player {self.h_player} with score {self.score}"
        
        if self.type == 0:
            return str+f"Unknown Discard move of {self.card_n} with score {self.score}"
        if self.type == 1:
            return str+f"Unknown Play move of {self.card_n} with score {self.score}"
        else:            
            return str+f"Unknown Hint to player {self.h_player} with score {self.score}" 

        