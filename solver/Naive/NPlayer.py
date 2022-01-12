from numpy.core.fromnumeric import argmin
import numpy as np
from .. import Move

from .. import utils

class NPlayer:
    def __init__(self, name, main, order, cards=None):
        self.name = name
        self.main = main
        self.order = order
        self.cardHandIndex = -1
        self.ncards = 5
        self.cards = np.array([
                                [-1,-1,-1,-1,-1],
                                [-1,-1,-1,-1,-1],
                                ], dtype=np.int16)
        self.cardIds = np.array([-1,-1,-1,-1,-1])
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

    def GetSafePlay(self,fireworks):
        nums = fireworks +1 #required numbers
        colors = np.array([0,1,2,3,4], dtype=np.int16) #for given colors
        
        #filter out completed fireworks
        filter = nums < 5
        nums = nums[filter]
        colors = colors[filter]

        """
            take the columns of the hint infos that are relevant for our constraints
            Example: we need value 3 (encoded to 2) for color yellow (encoded to 1),
            we take col 2 from num hints and col 1 from color hints and sum them.
            If result is 2 (both constraints satisfied) for an index i, then card i
            is a yellow 3 and can be played 
        """
        
        #if all fireworks are at the same value, any card with such value+1 is safely playable
        #print(f"Solving first case:{nums}")
        if np.all(nums == nums[0]):
            #print("got in if")
            needed_val_hints = self.hint_value[:, nums[0]] == 1
            if np.any(needed_val_hints):
                #print("safemove found")
                move = Move.Move(1)
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
        move = Move.Move(1)
        move.define_play(best_card)
        return move
        
    
    def GetSafeDiscard(self,fireworks):
        f = np.array([0,1,2,3,4])

        #gather info from hints
        card_values = np.argmax(self.hint_value==1, axis = 1) #1d array with card values
        card_colors = np.argmax(self.hint_color==1, axis = 1) #1d array with card colors

        #gather assurances (i.e. if value_a[i] == True, then value found for card i can be trusted)
        value_a = self.hint_value[f, card_values] == 1
        color_a = self.hint_color[f, card_colors] == 1

        #any card whose value is less or equal than min(fireworks) can be discarded regardless of color information
        is_low = card_values <= min(fireworks)
        is_low = np.logical_and(is_low, value_a)
        if np.any(is_low):
            move = Move.Move(0)
            move.define_discard(np.argmax(is_low))
            return move

        #likewise, any card of a color which was completed can be discarded regardless of value information
        completed_colors = fireworks == 4
        if np.any(completed_colors):
            a = self.hint_color[:, completed_colors] #hints related to completed colors
            is_useless_color = np.max(a, axis = 1) == 1 #b[i] == True means that card i is of a completed color
            if np.any(is_useless_color):
                move = Move.Move(0)
                move.define_discard(np.argmax(is_useless_color))
                return move


        #decide based only on complete information 
        lower_than_hinted_color = card_values <= fireworks[card_colors]
        lower_than_hinted_color = np.logical_and(lower_than_hinted_color,value_a)
        lower_than_hinted_color = np.logical_and(lower_than_hinted_color,color_a)

        if np.any(lower_than_hinted_color):
            move = Move.Move(0)
            move.define_discard(np.argmax(lower_than_hinted_color))
            return move

        #no safe discard found
        return None

    def GetSafeHint(self,fireworks, players):

        #process the cards needed to avoid a loop later (expand )
        needed_cards = np.array([fireworks+1,[0,1,2,3,4]], dtype=np.int16)
        #print(np.transpose(needed_cards))
        nums = fireworks +1
        colors = np.array([0,1,2,3,4])

        #filter out completed fireworks
        filter = nums < 5
        nums = nums[filter]
        colors = colors[filter]

        """
            Give a hint on any card that can be played given current fireworks
        """

        #TODO: add info on playing order (i.e. prioritize hinting to players next to you)
        for p in players:
            if p.order == self.order:
                #we can't give hints to ourselves
                continue
            for i in range(nums.shape[0]):
                tmpv = p.cards[0] == nums[i]
                tmpc = p.cards[1] == colors[i]
                tmpv = np.logical_and(tmpv,tmpc) 
                
                #if player has a playable card
                if np.any(tmpv):
                    card_index = np.argmax(tmpv)
                    #check if color or value were already hinted, if not hint one of them
                    if p.hint_value[card_index, nums[i]] != 1:
                        move = Move.Move(2)
                        move.define_hint(p.name, "value", utils.decode_value(nums[i]))
                        return move
                    elif p.hint_color[card_index, colors[i]] != 1:
                        move = Move.Move(2)
                        move.define_hint(p.name, "color", utils.decode_color(colors[i]))
                        return move

        #no good hints found
        return None
    
    def GetRandomHint(self,fireworks, players):
        #we don't have any immediate decent play, just do something unharmful and pass the turn

        for p in players:
            if p.order == self.order:
                #can't give hints to ourselves
                continue
            #prioritize hinting high value cards
            hot_card = np.argmax(p.cards[0, :])
            hot_value = p.cards[0, hot_card]
            hot_color = p.cards[1, hot_card]

            if p.hint_value[hot_card, hot_value] != 1:
                move = Move.Move(2)
                if hot_value == -1:
                    print(f"hinting {p.name} with cards {p.cards} card number {hot_card}")
                move.define_hint(p.name, "value", utils.decode_value(hot_value))
                return move
            elif p.hint_color[hot_card, hot_color] !=1:
                move = Move.Move(2)
                move.define_hint(p.name, "color", utils.decode_color(hot_color))
                return move

        #suboptimal hints not found, just do anything, even if it's just wasting a token (should be a rare occasion)
        next_player = (self.order + 1) % len(players)
        assert next_player != self.order
        next_cards = players[next_player].cards
        #if next_cards[0,0] == -1:
            #the ally has no cards: this shouldn't happen, maybe it's a bug. Anyway we just don't hint anything
            #return None
        move = Move.Move(2)
        print(f"random hint for value {next_cards[0,0]}")
        move.define_hint(players[next_player].name, "value", utils.decode_value(next_cards[0,0]))
        return move
    
    def GetUnsafeDiscard(self,fireworks):
        #Naive: just discard one of your lowest cards
        min = np.argmin(self.cards[0])

        move = Move.Move(0)
        move.define_discard(min)

        return move

    def HintsToString(self):
        out = f"value: \n{self.hint_value}\ncolor: \n{self.hint_color}" 
        return out

    def GetUnsafePlay(self):
        #Naive: discard any card. Note: we are never supposed to get here (we either 
        # choose a bad hint or unsafe discard before),but i coded this scenario either way
        move = Move.Move(1)
        move.define_play(0)
        return move
    
    def handle_draw(self, played_id, drawnCard = None):
        #cards below the played one are moved one index above: move them and their hints accordingly
        for i in range(played_id+1, 5):
            self.hint_color[i-1, :] = self.hint_color[i, :]
            self.hint_value[i-1, :] = self.hint_value[i, :]
            self.cards[:, i-1] = self.cards[:, i]
        
        #erase hints: player has no hints on the newly drawn card
        self.hint_value[4, :] = np.zeros(5)
        self.hint_color[4, :] = np.zeros(5)
        
        if drawnCard == None:
            #we don't know the new card or it simply was not drawn (deck has no cards)
            self.cards[0, 4] = -1
            self.cards[1, 4] = -1
        else:
            #drawnCard is not none
            self.cards[0, 4] = utils.encode_value(drawnCard.value)
            self.cards[1, 4] = utils.encode_color(drawnCard.color)
            #self.deck.remove_cards(drawn_card)
        return
