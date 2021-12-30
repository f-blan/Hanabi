import utils
import numpy as np
import Move

class NPlayer:
    def __init__(self, name, main,cards=[]):
        self.name = name
        self.main = main
        self.cards = np.array([
                                [-1,-1,-1,-1,-1],
                                [-1,-1,-1,-1,-1],
                                ], dtype=np.int16)
        #0 no info given, -1 can't be target, 1 card is target 
        self.hint_color = np.array([
                                    [0,0,0,0,0],
                                    [0,0,0,0,0],
                                    [0,0,0,0,0],
                                    [0,0,0,0,0],
                                    [0,0,0,0,0],
                                    ])
        self.hint_value = np.array([
                                    [0,0,0,0,0],
                                    [0,0,0,0,0],
                                    [0,0,0,0,0],
                                    [0,0,0,0,0],
                                    [0,0,0,0,0],
                                    ])
        if main == False:
            i=0
            for c in cards:
                self.cards[0][i] = utils.encode_value(c.value)
                self.cards[1][i] = utils.encode_color(c.color)

    def GetSafePlay(fireworks):
        nums = fireworks +1 #required numbers
        colors = np.array([0,1,2,3,4], dtype=np.int16) #for given colors

        """
            take the columns of the hint infos that are relevant for our constraints
            Example: we need value 3 (encoded to 2) for color yellow (encoded to 1),
            we take col 2 from num hints and col 1 from color hints and sum them.
            If result is 2 (both constraints satisfied) for an index i, then card i
            is a yellow 3 and can be played 
        """
        
        #TODO: if all fireworks are at the same value, any card with such value+1 is safely playable

        # a column for each needed card, if a[i,j] == 1 then card i satisfies value/color constraint for needed card j
        a = self.hint_value[:, nums] 
        b = self.hint_color[:, colors]

        #1d array: taking the max for each card after summing a and b.Each position i tells how playable is card i
        card_playabilities = np.max(a + b, axis = 1) 

        best_card = np.argmax(card_playabilities) #most playable card (index)

        #if playability of best card is not 2, then our best playable card is an unsafe play, return nothing
        if card_playabilities[best_card] != 2: 
            return None
        
        #we have a safe play available, return it
        move = Move(1)
        move.define_play(best_card)
        return move
        
    
    def GetSafeDiscard(fireworks):
        f = np.array([0,1,2,3,4])

        #gather info from hints
        card_values = np.argmax(self.hint_num==1, axis = 1) #1d array with card values
        card_colors = np.argmax(self.hint_color==1, axis = 1) #1d array with card colors

        #gather assurances (i.e. if value_a[i] == True, then value found for card i can be trusted)
        value_a = self.hint_num[f, card_values] == 1
        color_a = self.hint_color[f, card_colors] == 1

        #any card whose value is less or equal than min(fireworks) can be discarded regardless of color information
        is_low = card_values <= min(fireworks)
        is_low = np.logical_and(is_low, value_a)
        if np.any(is_low):
            move = Move(0)
            move.define_discard(np.argmax(is_low))
            return move

        #likewise, any card of a color which was completed can be discarded regardless of value information
        completed_colors = fireworks == 4 
        a = self.hint_color[:, completed_colors] #hints related to completed colors
        is_useless_color = np.max(a, axis = 1) == 1 #b[i] == True means that card i is of a completed color
        if np.any(is_useless_color):
            move = Move(0)
            move.define_discard(np.argmax(is_useless_color))
            return move


        #decide based only on trusted information 
        lower_than_hinted_color = card_values <= fireworks[card_colors]
        lower_than_hinted_color = np.logical_and(lower_than_hinted_color,value_a)
        lower_than_hinted_color = np.logical_and(lower_than_hinted_color,color_a)

        if np.any(lower_than_hinted_color):
            move = Move(0)
            move.define_discard(np.argmax(lower_than_hinted_color))
            return move

        #no safe discard found
        return None

    def GetSafeHint(fireworks, players):
        return None
    
    def GetRandomHint(fireworks, players):
        return None
    
    def GetUnsafeDiscard(fireworks):
        return None

    def GetUnsafePlay():
        return None