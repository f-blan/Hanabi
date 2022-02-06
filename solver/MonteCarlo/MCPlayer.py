from solver.Fuzzy.FPlayer import FPlayer
from solver.Fuzzy.FPlayer import HINTS_COMPUTED
from numpy.random import choice

import numpy as np
from sortedcontainers import SortedList

#from solver.MonteCarlo.NodeDeck import NodeDeck
from solver.MonteCarlo.MCMove import MCMove
from solver.MonteCarlo.NodeDeck import NodeDeck

#from solver.MonteCarlo.MCPlayer import MCPlayer

class MCPlayer(FPlayer):

    """
        Extension of the FPlayer class that keeps track of the hard unknown card on the player's hand
        (i.e. when we're deep in the MCTS and this player has drawn a card we didn't see in the server game state)

        Some functions are overridden to be compatible with the MCSolver and MCNode classes for managing the game
        state
    """
    def __init__(self, name, main, order, cardsInHand, cards=None):
        super().__init__(name, main, order,cardsInHand, cards)

        self.hard_unknowns = np.zeros(cardsInHand) != 0
        self.card_id_unknown = np.zeros(cardsInHand, dtype= np.int16) -1
        
        #this is used to evaluate a state and keep track of hints we don't know the outcome of
        self.base_knowledge = 0

    def handle_draw(self, playedId: int, draw_happened: bool,top_card_index:int, drawnCard:np.ndarray,endgame: bool,known_draw: bool,hardUnknown = False):
        """
            Overridden function for compatibility, See FPlayer
        """
        #TODO: differentiate better when card is just unknown or it wasn't drawn (DONE)
        if draw_happened == False:
            #we don't know the new card or it simply was not drawn (deck has no cards)
            self.n_cards-=1
            self.cards[0, self.cardsInHand-1] = -1
            self.cards[1, self.cardsInHand-1] = -1
            self.hard_unknowns[self.cardsInHand-1] = False 
            self.card_id_unknown[self.cardsInHand-1] = -1
        elif hardUnknown:
            self.hard_unknowns[self.cardsInHand-1] = True
            self.card_id_unknown[self.cardsInHand-1] = top_card_index
        elif known_draw==False:
            #draw happened (outside of MCTS) but we don't know the card
            self.cards[0, self.cardsInHand-1] = -1
            self.cards[1, self.cardsInHand-1] = -1
            self.hard_unknowns[self.cardsInHand-1] = False 
            self.card_id_unknown[self.cardsInHand-1] = top_card_index
        else:
            #drawnCard is not none
            #print(self.order)
            self.cards[0, self.cardsInHand-1] = drawnCard[0]
            self.cards[1, self.cardsInHand-1] = drawnCard[1]
            self.hard_unknowns[self.cardsInHand-1] =False 
            self.card_id_unknown[self.cardsInHand-1] = top_card_index
            #self.deck.remove_cards(drawn_card)
        return


    def handle_remove(self, cardHandIndex: int, known = True):
        """
            Overridden function for compatibility, See FPlayer.

            The MCPlayer stores a base knowledge scores that is meant to represent how much the player
            at this node knows about cards we (as the agent) don't know anything about. If an "unknown"
            card is removed from the hand, this base knowledge score is decreased
        """
        #removing a card means we are losing some knowledge about our cards
        if known == False:
            self.base_knowledge -= 2
            self.base_knowledge = max(0, self.base_knowledge)

        #cards below the played one are moved one index above: move them and their hints accordingly
        for i in range(cardHandIndex+1, self.cardsInHand):
            self.hints[:,:,i-1] = self.hints[:,:,i]
            self.cards[:, i-1] = self.cards[:, i]
            self.hint_tracker[:,i-1] = self.hint_tracker[:,i]
            self.playabilities[i-1] = self.playabilities[i]
            self.discardabilities[i-1] = self.discardabilities[i]
            self.hard_unknowns[i-1] = self.hard_unknowns[i]
            self.card_id_unknown[i-1] = self.card_id_unknown[i]
        
        #erase hints: player has no hints on the newly drawn card
        self.hints[:,:, self.cardsInHand-1] = np.zeros((5,5), dtype=np.int16)
        self.hint_tracker[0, self.cardsInHand-1] = False
        self.hint_tracker[1, self.cardsInHand-1] = False
        self.playabilities[self.cardsInHand-1] = 0
        self.discardabilities[self.cardsInHand-1] = 0

    def handle_hint(self, type: int, value: int, positions: list(), known = True):
        """
            Overridden function for compatibility, See FPlayer
        """
        if known == False:
            self.base_knowledge+=2
            if self.main:
                self.base_knowledge = min(self.base_knowledge, 0.5*np.sum(self.hint_tracker==False))
            else:
                self.base_knowledge = min(self.base_knowledge, np.sum(self.hard_unknowns))
        else:
            #print("----------------HINT RECEIVED--------------------")
            #print(self.hint_tracker)
            ret= super().handle_hint(type, value, positions)
            #print(self.hint_tracker)
            #print("-----------END HANDLE----------------")
            return ret 
    
    def set_expected_values(self, playabilities: np.ndarray, discardabilities: np.ndarray):
        """
            Sets playability and discardability values for the player's card
        """
        #cut the ranges in the case something went wrong with evaluation
        high_indexes = playabilities > 1
        playabilities[high_indexes] = 1
        low_indexes = playabilities < 0
        playabilities[low_indexes] = 0
        self.playabilities = playabilities

        high_indexes = discardabilities > 1
        discardabilities[high_indexes] = 1
        low_indexes = discardabilities<0
        discardabilities[low_indexes] = 0
        self.discardabilities = discardabilities

    def get_score(self, deck: NodeDeck):
        """
            Used in MCNode.MC_Simulate. This function returns a score representing how much this player
            knows about their cards
        """
        dhu=deck.discardability_rc
        phu=deck.playability_rc
        return self.base_knowledge*(phu+0.2*dhu) + 1.2*np.sum(self.hint_tracker[0,:]*(self.playabilities+0.2*self.discardabilities)) \
                    + np.sum(self.hint_tracker[1,:]*(self.playabilities+0.2*self.discardabilities))

    def GetMoves(self, players: list(), fireworks: np.ndarray, deck: NodeDeck, redTokens: int, blueTokens: int):
        """
            Overridden function for compatibility, See FPlayer.

            Compared to the FPlayer version, this function does not add the following moves to the list:
                -unsafe play performed by the agent
                -play of a hard unknown card
                -detailed hints (of given type and value) to the agent
            Additionally the following moves are considered:
                -discard of an hard unknown card
                -discard of a partially known card by the agent
                -hint to a partially known card owned by the agent 
                -hint to a hard unknown card 
        """
        
        #Very similar to GetMoves of FPlayer, but takes into account hard unknowns and hints on hard unknown cards
        moves = SortedList(key= lambda x: -x.score)
        max_hints = HINTS_COMPUTED

        #add possible plays and discards (small amount, we put them all)
        for i in range(0, self.cardsInHand):

            if self.hard_unknowns[i] == False:
                #consider a play only if the card is not hard unknown

                #differentiate if this is the agent: play only cards with perfect knowledge if you are the agent
                if self.main == False:
                    #this is not the agent, we know the card
                    used_card = np.array([[self.cards[0,i]], [self.cards[1,i]]])#we need this as a two dimensional array for later function compatibility
                    curmove = MCMove(1,self.order, True,cardHandIndex=i,used_card=used_card)
                    curmove.define_play(i)

                    curmove.EvaluatePlay(self.playabilities[i], redTokens)
                    moves.add(curmove)
                elif self.playabilities[i] == 1:
                    #you are the agent. Play this card only if playability is 1 (i.e. you can reconstruct what the card is), so that
                    #we can compute fireworks for the next state
                    curmove = MCMove(1,self.order, False,cardHandIndex=i)
                    curmove.define_play(i)

                    curmove.EvaluatePlay(self.playabilities[i], redTokens)
                    moves.add(curmove)

            if blueTokens > 0:
                #we can discard only if there are some blue tokens used
                if self.main == False:
                    used_card = np.array([[self.cards[0,i]], [self.cards[1,i]]])
                    curmove = MCMove(0,self.order, True,cardHandIndex=i, used_card=used_card)
                    curmove.define_discard(i)

                    curmove.EvaluateDiscard(self.discardabilities[i], blueTokens)
                    moves.add(curmove)
                else:
                    curmove = MCMove(0,self.order,False, cardHandIndex=i)
                    curmove.define_discard(i)

                    curmove.EvaluateDiscard(self.discardabilities[i], blueTokens)
                    moves.add(curmove)
        
        if blueTokens == 8:
            return moves

        #hint selection: consider more hints to players that play right after you
        hints_per_player = np.arange(len(players))
        hints_per_player = np.roll(hints_per_player, self.order)
        hints_per_player = np.array(np.around((hints_per_player/np.sum(hints_per_player))*max_hints),dtype=np.int16)


        #consider a certain number of hints for each player
        for i in range(0, len(players)):
            if hints_per_player[i] == 0 or players[i].order == self.order:
                #should only happen when i is hinting player's position
                continue
            
            p = players[i]

            if p.main == True:
                #handle the case where we're hinting to the agent

                #hint a card with possible hints
                curmove = MCMove(2,self.order, False)
                curmove.h_player = p.name
                curmove.EvaluateHint(p,blueTokens,deck,known=1)
                
                curmove.finalize_hint(p.order)
                moves.add(curmove)

                #hint possible hard unknowns
                if np.any(p.hard_unknowns):
                    curmove = MCMove(2,self.order, False)
                    curmove.EvaluateHint(p,blueTokens,deck,known=0)
                    curmove.h_player=p.name
                    curmove.finalize_hint(p.order)
                    moves.add(curmove)
                continue

            #filter out hard unknowns
            mask = p.hard_unknowns==False
            if p.n_cards != p.cardsInHand:
                mask[-1] = False
            playabilities = p.playabilities[mask]
            discardabilities = p.discardabilities[mask]

            #sorted indexes of known cards
            max_play_i = np.argsort((-playabilities[0:p.n_cards]))
            max_disc_i = np.argsort((-discardabilities[0:p.n_cards]))
            p_i = 0
            d_i = 0
            #print(f"play: {max_play_i} disc: {max_disc_i}")
            
            #dictionary to avoid giving two equivalent hints
            uq = {}
            #print(f"player:{p.order}{p.cards}")
            for j in range(0, hints_per_player[i]):
                
                if p_i >= len(max_play_i) and d_i >= len(max_disc_i):
                    #we considered all possibilities for this player
                    break
                elif p_i >= len(max_play_i):
                    probs = np.array([0,1])
                elif d_i >= len(max_disc_i):
                    probs = np.array([1,0])
                else:
                    
                    max_p = p.playabilities[max_play_i[p_i]] + 0.0001
                    max_d = p.discardabilities[max_disc_i[d_i]] + 0.0001
                    if p.playabilities[max_play_i[p_i]]<0 or p.playabilities[max_play_i[p_i]]>1:
                        print("---------PRE EXCEPTION---------")
                        print(f"deck:\n{deck.deck}\nhint:\n{p.hints[:,:,max_play_i[p_i]]}\nmask:\n{deck.mask}\nplayer h_tracker:\n{p.hint_tracker}\n")
                        print(f"playabilities:\n{p.playabilities}\nindex:{max_play_i[p_i]}")
                    #print(f"p_i: {p_i} max_p: {max_p} d_i: {d_i} max_d: {max_d}")
                    if p.discardabilities[max_disc_i[d_i]]<0 or p.discardabilities[max_disc_i[d_i]]>1:
                        print("---------PRE EXCEPTION---------")
                        print(f"deck:\n{deck.deck}\nhint:\n{p.hints[:,:,max_disc_i[d_i]]}\nmask:\n{deck.mask}\nplayer h_tracker:\n{p.hint_tracker}\n")
                        print(f"discardabilities:\n{p.discardabilities}\nindex:{max_disc_i[d_i]}")
                    probs = np.array([max_p, max_d], dtype=np.float16)
                    #probs = np.array([0.5, 0.5])
                #print(probs)
                #choose between hinting a playable or a discardable card
                probs = probs/np.sum(probs)
                #print(probs)
                ch = choice(a=[True,False], p = probs)

                if ch == True:
                    #we consider hinting the playable if it makes sense (no full knowledge yet)
                    
                    j, curmove = self.hint_define(max_play_i, p_i, j, p, blueTokens,0,uq,deck)
                    if curmove != None:
                        uq.__setitem__(curmove.ToKey(),0)
                        moves.add(curmove)
                        
                    j, curmove = self.hint_define(max_play_i, p_i, j, p, blueTokens,1,uq,deck)
                    if curmove != None:
                        uq.__setitem__(curmove.ToKey(),0)
                        moves.add(curmove)
                        
                    p_i += 1
                    
                else:
                    #we hint the discardable if it makes sense
                    
                    j, curmove = self.hint_define(max_disc_i, d_i,j,p, blueTokens,0,uq,deck)
                    if curmove != None:
                        uq.__setitem__(curmove.ToKey(),0)
                        moves.add(curmove)
                        
                    j, curmove = self.hint_define(max_disc_i, d_i,j,p, blueTokens,1,uq,deck)
                    if curmove != None:
                        uq.__setitem__(curmove.ToKey(),0)
                        moves.add(curmove)
                        
                    d_i += 1
            
            #consider also the hard unknown hint
            if np.any(p.hard_unknowns):
                curmove = MCMove(2,self.order, False,cardHandIndex=i)
                curmove.EvaluateHint(p,blueTokens,deck,known=0)
                curmove.h_player=p.name
                curmove.finalize_hint(p.order)
                moves.add(curmove)
        return moves

    def hint_define(self, card_i: np.ndarray, i: int,j: int, player, blueTokens: int,htype:int, uq, deck: NodeDeck):
        """
            Overridden function for compatibility, See FPlayer
        """
        ret_j = j
        ret_move = None
        
        #we choose here if we want to hint the value or the color
        if player.hint_tracker[htype,card_i[i]] == False and f"h_p{player.order}t{htype}v{player.cards[htype,card_i[i]]}" not in uq:
            
            #if value was not hinted yet, we hint it
            hvalue = player.cards[htype,card_i[i]]

            if hvalue < 0 or hvalue > 4:
                #ad hoc fix
                return ret_j+1,None
            curmove = MCMove(2,self.order,True)
            curmove.define_hint(player, htype, hvalue)
            positions = list((player.cards[htype, :]==hvalue).nonzero()[0])
            
            curmove.finalize_hint(player.order, positions = positions)

            ret_j+=1
            
            curmove.EvaluateHint(player, blueTokens,deck,known=2)

            ret_move = curmove
        else:
            #player already has full knowledge, this iteration didn't count towards the number of hints
            ret_move = None
        
        return ret_j, ret_move

    
