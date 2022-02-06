#from solver.MonteCarlo.MCNode import MCNode

#from solver.MonteCarlo.MCNode import PRINT_DEBUG
from solver.Naive.NPlayer import NPlayer
from solver.Move import Move
from .. import utils
import numpy as np

from numpy.random import randint
from numpy.random import shuffle
import random

DEBUG_PRINT = -1
class NSimulator:
    """
        This function is meant to simulate a playout following the policy of NPlayer 
        starting from a node of the MCTS algorithm
    """
    def __init__(self, node):
        self.players = list()
        self.node = node

        for p in node.players:
            nplayer = NPlayer(p.name, p.main,p.order,p.cardsInHand, initCards=False)
            self.players.append(nplayer)
        
        self.main_player_order = node.main_player_order
        self.curr_player_order = node.curr_player_order

        self.endgame = node.deck.endgame
        self.turns_until_end = node.deck.turns_until_end
        
        #a list of cards
        self.deck = list()

    
    def initialize_random_state(self, node):
        """
            From the node, set the game state related to the simulator
            as one of the possible state that can exist from the node.

            The idea is to initialize a coherent deck (in terms of ordered cards)
            and coherent hands for both the agents and the other players
        """
        deck_copy = np.copy(node.deck.deck)

        #get a coherent hand for the agent
        mc_mp = node.players[node.main_player_order]
        n_mp = self.players[self.main_player_order]

        self.initialize_player_cards(mc_mp, deck_copy, True,mc_mp.cardsInHand,n_mp )
        
        #get a coherent hand for other players:
        for p in node.players:
            if p.order != self.main_player_order:
                n_p = self.players[p.order]
                self.initialize_player_cards(p,deck_copy,False,p.cardsInHand, n_p) 
                #print(f"{n_p.cards} for player {n_p.order}")

        
        #set self.deck as a list of ordered cards
        m = deck_copy > 0
        x,y = m.nonzero()
        #arange = np.array([i for i in range(0, x.shape[0])])
        #shuffle(arange)

        for i in range(0, x.shape[0]):
            #index = arange[i]
            for j in range(0, deck_copy[x[i], y[i]]):
                self.deck.append(np.array([x[i], y[i]]))
        random.shuffle(self.deck)

        return

    def initialize_player_cards(self, mcp, deck_copy: np.ndarray, main:bool, cardsInHand: int, n_p: NPlayer):
        ret_c = np.zeros((2,cardsInHand),dtype=np.int16)-1
        ret_hv = np.zeros((cardsInHand,5),dtype=np.int16)
        ret_hc = np.zeros((cardsInHand,5),dtype=np.int16)
        #regular_cards = mcp.hard_unknowns == False
        regular_indexes = mcp.hard_unknowns == False
        arange = np.array([i for i in range(0, len(mcp.hard_unknowns))]) < mcp.n_cards
        regular_indexes = np.logical_and(arange,regular_indexes)
        
        if main == False:
            
            ret_c[:, regular_indexes] = mcp.cards[:, regular_indexes]
            if DEBUG_PRINT >=1:
                print(f"mcplayer hand:\n{mcp.cards}")
                print(f"nplayer hand:\n{ret_c}")
                print(f"reg indexes\n{regular_indexes}")
        else:
            #get a possible card for each hint
            randomMask = np.zeros((5,5))
            i=0
            for hint in mcp.hints[:,:,regular_indexes]:
                mask = deck_copy > 0
                #hint = mcp.hints[:,:, i]
                cont = True
                m = hint >= 2
                if np.any(m) and cont:
                    #we know the card
                    x,y = m.nonzero()
                    x=x[0]
                    y=y[0]
                    ret_c[0,i]=x
                    ret_c[1,i]=y
                    cont = False
                m= hint > 0 
                if np.any(m) and cont:
                    #we have partial knowledge
                    ret_c[0,i], ret_c[1,i]= self.handle_random_card_selection(m,deck_copy,mask,randomMask)
                    cont = False
                m = hint < 0
                if np.any(m) and cont:
                    #we have negative knowledge
                    ret_c[0,i], ret_c[1,i]= self.handle_random_card_selection(m,deck_copy,mask,randomMask)
                    cont = False
                m = np.copy(mask)
                if np.any(m) and cont:
                    #we have no knowledge
                    ret_c[0,i], ret_c[1,i]= self.handle_random_card_selection(m,deck_copy,mask,randomMask)
                    cont = False
                i+=1
        
        #remove hard unknowns
        first_hu = np.argmax(mcp.hard_unknowns)
        if first_hu != 0:
            for i in range(first_hu, mcp.n_cards):
                m = deck_copy > 0
                x,y = m.nonzero()
                index = randint(0, x.shape[0])
                ret_c[0,i]=x[index]
                ret_c[1,i]= y[index]
                deck_copy[x[index],y[index]]-=1

        #reconstruct hints
        for i in range(0, mcp.n_cards):
            if mcp.hint_tracker[0, i] == True:
                val = ret_c[0,i]
                ret_hv[i, val] = 1
            elif mcp.hint_tracker[0, i] == True:
                val = ret_c[1,i]
                ret_hc[i, val] = 1
        
        n_p.cards=ret_c
        #print(n_p.cards)
        n_p.hint_color = ret_hc
        n_p.hint_value = ret_hv

                    
                    



    def handle_random_card_selection(self, m: np.ndarray, deck: np.ndarray,mask:np.ndarray, randomMask: np.ndarray):
        #we have partial or no knowledge
        x,y = m.nonzero()
        mask_filter = mask[x, y]
        x = x[mask_filter]
        y = y[mask_filter]

        index = randint(0, x.shape[0])
        randomMask[x,y] = -1
        randomMask[x[index],y[index]] = 1 

        deck[x[index], y[index]]-=1      

        return x[index], y[index] 



    def perform_playouts(self, iterations):
        if DEBUG_PRINT >=0:
            print("------PERFORMING PLAYOUTS-------")
        tot_score = 0
        for i in range(0, iterations):
            self.initialize_random_state(self.node)
            if DEBUG_PRINT >=1:
                print("-------------STATE OF PLAYOUT-----------")
                print(f"main cards:\n{self.players[self.main_player_order].cards}\ndeck:\n{self.compact_deck()}\nfw: {self.node.fireworks}\nbt: {self.node.blue_tokens},rt:{self.node.red_tokens}")
                print(f"secondary player:\n {self.players[1].cards}")

            tot_score+=self.playout()
        
        if DEBUG_PRINT >=0:
            print(f"tot_score = {tot_score}")
            print("------END PLAYOUTS-------")

        return tot_score
    
    def playout(self):
        """
            we're starting from a random coherent state
        """
        curr_player_order = self.curr_player_order
        endgame = self.endgame
        turns_until_end = self.turns_until_end
        endgame, turns_until_end = self.update_endgame(endgame,turns_until_end)
        fireworks = np.copy(self.node.fireworks)
        blueTokens = self.node.blue_tokens
        redTokens = self.node.red_tokens
        if DEBUG_PRINT >= 2:
            print(f"endgame: {endgame}, turns until end: {turns_until_end}")
            
        
        while endgame==False or turns_until_end >=0:
            curr_player = self.players[curr_player_order]
            move = curr_player.GetMove(fireworks,blueTokens,redTokens,self.players)
            fireworks,  blueTokens, redTokens = self.apply_move(move,curr_player_order, fireworks, blueTokens, redTokens)
            endgame, turns_until_end = self.update_endgame(endgame, turns_until_end)
            

            if DEBUG_PRINT >=2:
                print(f"------PLAYOUT ITER-----")
                print(f"move:{move.ToKey()}, fw: {fireworks}, bt: {blueTokens}, rt: {redTokens}, cp: {curr_player_order}\nc_cards:")
                #print(f"hint color: {self.players[curr_player_order].hint_color}")
                #print(f"hint value: {self.players[curr_player_order].hint_value}")
                print(f"-----END ITER---")
            curr_player_order = self.get_next_player_order(curr_player_order)
            
        
        return np.sum(fireworks+1)

    def apply_move(self,move: Move, player_order: NPlayer, fireworks: np.ndarray, blueTokens:int, redTokens: int):
        ret_f = np.copy(fireworks)
        ret_bt = blueTokens
        ret_rt = redTokens
        p = self.players[player_order]
        if move.type == 0:
            #discard
            drawnCard = None
            drawHappened = False
            if len(self.deck)>0:
                drawnCard = self.deck.pop(0)
                drawHappened = True
            p.remove_card(move.card_n)
            p.handle_draw_v2(move.card_n, drawnCard=drawnCard, drawHappened = drawHappened)
           
            ret_bt -=1

        elif move.type == 1:
            #play
            drawnCard = None
            drawHappened = False
            if len(self.deck)>0:
                drawHappened = True
                drawnCard = self.deck.pop(0)
            playedCard = p.cards[:, move.card_n]
            
            p.remove_card(move.card_n)
            p.handle_draw_v2(move.card_n, drawnCard=drawnCard, drawHappened = drawHappened)
            if DEBUG_PRINT >=3:
                print(f"playing: {playedCard}")

            #print(ret_f)
            #print(playedCard)
            if ret_f[playedCard[1]]+1 == playedCard[0]:

                ret_f[playedCard[1]]+=1
                if ret_f[playedCard[1]]==4:
                    ret_bt-=1
            else:
                ret_rt+=1
        elif move.type == 2:
            #hint
            destination = self.get_player(move.h_player)
            if DEBUG_PRINT >=3:
                print(f"destination is {destination.name}, hint is {move.ToKey()}")
                print(f"hints:\n{destination.hint_color}\n{destination.hint_value}")
            if move.h_type == "color":
                destination.handle_hint(1,utils.encode_color(move.h_value))
            else:
                destination.handle_hint(0,utils.encode_value(move.h_value))
            if DEBUG_PRINT >=3:
                
                print(f"post hints:\n{destination.hint_color}\n{destination.hint_value}")
            ret_bt+=1
        return ret_f, ret_bt,ret_rt

    def get_player(self, name):
        for p in self.players:
            if p.name == name:
                return p

    def update_endgame(self, endgame, turns_until_end):
        ret_e = endgame
        ret_tue = turns_until_end
        if endgame == False and len(self.deck)>0:
            ret_e= False
            ret_tue = len(self.players)
        elif endgame == False and len(self.deck)<= 0:
            ret_e = True 
            ret_tue = len(self.players)
        elif endgame == True:
            ret_e = True
            ret_tue = turns_until_end-1
        
        return ret_e,ret_tue
    
    def get_next_player_order(self, curr_p_order):
        ret = (curr_p_order + 1)%len(self.players)
        assert ret != curr_p_order
        return ret

    def compact_deck(self):
        deck = np.zeros((5,5), dtype = np.int16)
        for c in self.deck:
            deck[c[0],c[1]]+=1
        return deck

            


