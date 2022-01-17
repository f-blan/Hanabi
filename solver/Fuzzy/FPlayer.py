from numpy.core.fromnumeric import argmin
import numpy as np
from numpy.random import choice

from solver.Fuzzy.FDeck import FDeck
from solver.Fuzzy.FMove import FMove
from . import Move

from .. import utils

from sortedcontainers import SortedDict

HINTS_COMPUTED = 10
HINTS_VERBOSE = FALSE

class FPlayer:
    def __init__(self, name, main, order, cardsInHand, cards=None):
        self.name = name
        self.main = main
        self.order = order
        self.cardHandIndex = -1
        self.cardsInHand = cardsInHand
        self.n_cards = cardsInHand
        self.cards = np.array([
                                [-1 for i in range(0,cardsInHand)],
                                [-1 for i in range(0,cardsInHand)]
                                ], dtype=np.int16)
        self.cardIds = np.array([-1 for i in range(0,cardsInHand)])
        #0 no info given, -1 can't be target, 1 card is target 

        self.hints = np.zeros((5,5,cardsInHand), dtype=np.int16)

        #keep track of what kind of hints were given for each card (color, value)
        self.hint_tracker = np.zeros((2, cardsInHand))!=0
        
        
        #playability and discardability of each card
        self.playabilities = np.zeros((cardsInHand, 5), dtype=np.float32)
        self.discardabilities = np.zeros((cardsInHand, 5), dtype=np.float32)
        
        if main == False:
            i=0
            for c in cards:
                self.cards[0][i] = utils.encode_value(c.value)
                self.cards[1][i] = utils.encode_color(c.color)
                self.cardIds[i] =c.id
                i+=1

    def handle_remove(self, cardHandIndex: int):
        self.cardHandIndex = cardHandIndex
        
        self.cards[0, cardHandIndex] = -1
        self.cards[1, cardHandIndex] = -1
        self.playabilities[cardHandIndex] = 0
        self.discardabilities[cardHandIndex] = 0

        #erase hints
        #self.hints[:,:, cardHandIndex] = np.zeros((5,5), dtype=np.int16)
        #self.hint_tracker[0, cardHandIndex] = False
        #self.hint_tracker[1, cardHandIndex] = False

    
    def handle_hint(self, type:int, value:int, positions:list()):
        p_mask = np.zeros((5,5),dtype=np.int16)-5
        n_mask = np.zeros((5,5),dtype=np.int16)
        if type == 0:
            p_mask[value,:] = 1
            n_mask[value,:] = -5
        else:
            p_mask[:,value] = 1
            n_mask[:,value] = -5
            

        for i in range(0, self.n_cards):
            if i in positions:
                self.hints[:,:,i] += p_mask
                self.hint_tracker[type, value] = True
            else:
                self.hints[:,:,i] += n_mask
    
    def handle_draw(self, playedId: int, drawnCard=None):
        #cards below the played one are moved one index above: move them and their hints accordingly
        for i in range(playedId+1, self.cardsInHand):
            self.hints[:,:i-1] = self.hint_color[:,:,i]
            self.cards[:, i-1] = self.cards[:, i]
            self.hint_tracker[:, i-1] = self.hint_tracker[:,i]
            self.playabilities[i-1] = self.playabilities[i]
            self.discardabilities[i-1] = self.discardabilities[i]
        
        #erase hints: player has no hints on the newly drawn card
        self.hints[:,:, self.cardsInHand] = np.zeros((5,5), dtype=np.int16)
        self.hint_tracker[0, self.cardsInHand-1] = False
        self.hint_tracker[1, self.cardsInHand-1] = False
        self.playabilities[self.cardsInHand-1] = 0
        self.discardabilities[self.cardsInHand-1] = 0
        
        if drawnCard == None:
            #we don't know the new card or it simply was not drawn (deck has no cards)
            self.cards[0, self.cardsInHand-1] = -1
            self.cards[1, self.cardsInHand-1] = -1
            
        else:
            #drawnCard is not none
            self.cards[0, self.cardsInHand-1] = utils.encode_value(drawnCard.value)
            self.cards[1, self.cardsInHand-1] = utils.encode_color(drawnCard.color)
            #self.deck.remove_cards(drawn_card)
        return

    def HintsToString(self):
        hintlist = list()

        for i in range(0, self.n_cards):
            hint = self.hints[:,:,i]
            string = ""
            if self.hint_tracker[0, i] == True and self.hint_tracker[1, i] == True:
                val, col = (self.hint == 2).nonzero()
                str = f"value: {utils.decode_value(val[0])}, color : {utils.decode_color(col[0])}"
                hintlist.append(str)
            elif self.hint_tracker[0, i] == True:
                val, col = (self.hint >0).nonzero()
                str = f"value: {utils.decode_value(val[0])}, color : unkown"
                hintlist.append(str)
            elif self.hint_tracker[1, i] == True:
                val, col = (self.hint >0).nonzero()
                str = f"value: unknown, color : {utils.decode_color(col[0])}"
                hintlist.append(str)
            else:
                str = f"value: unknown, color : unknown"
                hintlist.append(str)



        out = f"hints: {hintlist}" 
        return out

    def GetMoves(self, players: list(FPlayer), fireworks: np.ndarray, deck: FDeck, redTokens: int, blueTokens: int):
        moves = SortedDict()
        max_hints = HINTS_COMPUTED

        #add possible plays and discards (small amount, we put them all)
        for i in range(0, self.cardsInHand):
            curmove = FMove(1)
            curmove.define_play(i)

            score = curmove.EvaluatePlay(self.playabilities[i], redTokens)
            moves.__setitem__(score, curmove)
            if blueTokens > 0:
                #we can discard only if there are no blue tokens used
                curmove = FMove(0)
                curmove.define_discard(i)

                score = curmove.EvaluateDiscard(self.discardabilities[i], blueTokens)
                moves.__setitem__(score, curmove)
        
        if blueTokens == 8:
            return moves

        #hint selection: consider more hints to players that play right after you
        hints_per_player = np.arange(len(players))
        hints_per_player = np.roll(hints_per_player, self.order)
        hints_per_player = np.around((hints_per_player/np.sum(hints_per_player))*max_hints)

        #consider a certain number of hints for each player
        for i in range(0, len(players)):
            if hints_per_player[i] == 0:
                #should only happen when i is agent's position
                continue
            
            p = players[i]
            max_play_i = np.argsort((-p.playabilities))
            max_disc_i = np.argsort((-p.discardabilities))
            p_i = 0
            d_i = 0
            for j in range(0, hints_per_player[i]):
                
                if p_i >= len(max_play_i) and d_i >= len(max_disc_i):
                    #we considered all possibilities for this player
                    break
                elif p_i >= len(max_play_i):
                    probs = np.ndarray([0,1])
                elif d_i >= len(max_disc_i):
                    probs = np.ndarray([1,0])
                else:
                    probs = np.array([max_play_i[p_i], max_disc_i[d_i]])

                #choose between hinting a playable or a discardable card
                probs = probs/np.sum(probs)
                ch = choice(a=[True,False], p = probs)

                if ch == True:
                    #we consider hinting the playable if it makes sense (no full knowledge yet)
                    j, curmove = self.hint_define(max_play_i, p_i, j, p)
                    p_i += 1
                    
                else:
                    #we hint the discardable if it makes sense
                    j, curmove = self.hint_define(max_disc_i, d_i,j,p)
                    d_i += 1
                
                if curmove != None:
                    moves.__setitem__(score, curmove)
                


    def hint_define(self, card_i: np.ndarray, i: int,j: int, player):
        ret_j = j
        ret_move = None

        #we choose here if we want to hint the value or the color
        if player.hint_tracker[0,card_i[i]] == False:
            #if value was not hinted yet, we hint it
            curmove = FMove(2)
            curmove.define_hint(player.order, "value", utils.decode_value(player.cards[0,card_i[i]]))

            ret_j+=1
            
            curmove.EvaluateHint(player)

            ret_move = curmove
        elif player.hint_tracker[1,card_i[i]] == False:
            #value was hinted already but not color
            curmove = FMove(2)
            curmove.define_hint(player.order, "color", utils.decode_color(player.cards[1,card_i[i]]))

            ret_j+=1
            
            curmove.EvaluateHint(player)
            ret_move = curmove
        else:
            #player already has full knowledge, this iteration didn't count towards the number of hints
            ret_move = None
        return ret_j, ret_move







        