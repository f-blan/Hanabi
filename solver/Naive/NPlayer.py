from numpy.core.fromnumeric import argmin
import utils
import numpy as np
import Move

class NPlayer:
    def __init__(self, name, order, main,cards=[]):
        self.name = name
        self.main = main
        self.order = order
        self.cards = np.array([
                                [-1,-1,-1,-1,-1],
                                [-1,-1,-1,-1,-1],
                                ], dtype=np.int16)
        self.cardIds = no.array([-1,-1,-1,-1,-1])
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
                self.cardIds[i] =c.id
                i+=1

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
        
        #if all fireworks are at the same value, any card with such value+1 is safely playable
        if np.all(fireworks == fireworks[0]):
            needed_val_hints = self.hint_value[:, fireworks[0]] == 1
            if np.any(needed_val_hints):
                move = Move(1)
                move.define_play(np.argmax(needed_val_hints))
                return move


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


        #decide based only on complete information 
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

        #process the cards needed to avoid a loop later (expand )
        #needed_cards = np.array([fireworks+1,[0,1,2,3,4]], dtype=np.int16)
        #print(np.transpose(needed_cards))
        nums = needed_cards[0, :]
        colors = needed_cards[1,:]

        """
            Give a hint on any card that can be played given current fireworks
        """

        #TODO: add info on playing order (i.e. prioritize hinting to players next to you)
        for p in players:
            for i in range(5):
                tmpv = p.cards == nums[i]
                tmpc = p.cards == colors[i]
                tmpv = np.logical_and(tmpv,tmpc) 
                
                #if player has a playable card
                if np.any(tmpv):
                    card_index = np.argmax(tmpv)
                    #check if color or value were already hinted, if not hint one of them
                    if p.hint_value[card_index, nums[i]] != 1:
                        move = Move(2)
                        move.define_hint(p.name, "value", utils.decode_value(nums[i]))
                        return move
                    elif p.hint_color[card_index, colors[i]] != 1:
                        move = Move(2)
                        move.define_hint(p.name, "color", utils.decode_color(colors[i]))
                        return move

        #no good hints found
        return None
    
    def GetRandomHint(fireworks, players):
        #we don't have any immediate decent play, just do something unharmful and pass the turn

        for p in players:
            #prioritize hinting high value value cards
            hot_card = np.argmax(p.cards[0])

            if p.hint_value[hot_card, p.cards[0,hot_card]] != 1:
                move = Move(2)
                move.define_hint(p.name, "value", utils.decode_value(p.cards[0, hot_card]))
                return move
            elif p.hint_color[hot_card, p.cards[1, hot_card]] !=1:
                move = Move(2)
                move.define_hint(p.name, "color", utils.decode_color(p.cards[1, hot_card]))
                return move

        #suboptimal hints not found, just do anything, even if it's just wasting a token (should be a rare occasion)
        next_player = (self.order + 1) % players.len
        next_cards = players[next_player].cards
        move = Move(2)
        move.define_hint(players[next_player].name, "value", utils.decode_value(next_cards[0,0]))
        return move
    
    def GetUnsafeDiscard(fireworks):
        #Naive: just discard one of your lowest cards
        min = np.argmin(self.cards[0])

        move = move(0)
        move.define_discard(min)

        return move

    def GetUnsafePlay():
        #Naive: discard any card. Note: we are never supposed to get here (we either 
        # choose a bad hint or unsafe discard before),but i coded this scenario either way
        move = move(1)
        move.define_play(0)
        return move