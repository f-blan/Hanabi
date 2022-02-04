from copy import deepcopy
import numpy as np
from solver.MonteCarlo.MCDeck import MCDeck
from solver.MonteCarlo.MCMove import MCMove
from solver.MonteCarlo.MCPlayer import MCPlayer
from solver.MonteCarlo.NodeDeck import NodeDeck
from sortedcontainers import SortedList
#from cpu_client import players

import math 

MAX_CHILDREN = 1
MAX_DEPTH = 10
MC_ITERATIONS = 1
D_FACTOR = 0.9 #discount factor
VERBOSE = False

class MCNode():
    def __init__(self, fireworks: np.ndarray, blue_tokens: int, red_tokens: int, players: list(), 
                deck: NodeDeck, curr_player: int, main_player: int, MainDeck: MCDeck, top_card_index: int,is_root,  move = None, opt = None, depth = 0, parent= None):
        
        #Game State: copy of normal game state of parent
        self.fireworks = np.copy(fireworks)
        self.blue_tokens = blue_tokens
        self.red_tokens = red_tokens
        self.players = deepcopy(players)
        self.deck = deepcopy(deck)
        self.curr_player_order = curr_player
        self.main_player_order = main_player
        self.MainDeck = MainDeck
        self.top_card_index = top_card_index #the index of the card at the top of the deck
        self.move = move

        self.is_root = is_root
        if move != None:
            #we initialize also the move if specified
            self.apply_move(move)
            self.deck.update_expected_values(self.fireworks,self.players[self.main_player_order])
            for p in self.players:
                if p.order == self.curr_player_order or p.order == self.main_player_order:
                    p.playabilities, p.discardabilities = self.deck.evaluate_unknown_cards(p.n_cards, self.fireworks,p.hints, p.hard_unknowns)
                else:
                    p.playabilities, p.discardabilities = self.deck.evaluate_known_cards(p.cards,self.fireworks,p.hard_unknowns)
        
        #MC parameters
        self.depth = depth
        self.n_simulations = 1
        self.score = self.MC_simulate()

        self.children_scores = self.score
        #self.MC_score = 
        self.expanded = False
        
        self.children = list()
        self.childrenKeys = {}
        
        self.parent = parent
        self.expanded = False
        self.terminal = False 

        #TODO: fix endgame check (DONE)
        if self.red_tokens >= 3 or (self.deck.endgame and self.deck.turns_until_end <= 0):
            self.terminal = True

        #self.max_score = 8+3+len(players)*2*players[0].cardsInHand+25*4
        print("--------------NODE GENERATED-----------------")
        print(f"node with key {self.ToKey()} has been generated, To String is:")
        print(self.ToString())
        print("--------------END GENERATION-----------------")

    def apply_move(self, move: MCMove):
        """
            Initialization function: apply one move to get to the actual game state for this node
        """
        next_player_order = (self.curr_player_order+1+len(self.players))%len(self.players)
        assert next_player_order!=self.curr_player_order
        curr_player = self.players[self.curr_player_order]
        if self.move != None:
            print(f"curr_player_order: {self.curr_player_order} real: {self.move.playerOrder} next: {next_player_order}")
            assert self.curr_player_order == self.move.playerOrder
        self.curr_player_order = next_player_order
        
        if move.type == 0:
            #discard
            known = True
            #update deck state 
            if move.known:
                #we know the card that was played
                self.deck.RemoveCardsFromGame(move.used_card)
                card = move.used_card
                if self.curr_player_order == self.main_player_order:
                    self.deck.RemoveCards(move.used_card)
            else:
                #we don't know the card, but we may use the hints to remove certain cards
                if curr_player.hard_unknowns[move.cardHandIndex] == False:
                    #self.deck.RemoveHintedCardFromGame(curr_player.hints[:,:, move.cardHandIndex])
                    self.deck.RemoveHintedCard(curr_player.hints[:,:, move.cardHandIndex])
                else:
                    known = False
                    self.deck.RemoveHardUnknownFromGame()
                    self.deck.RemoveHardUnkown()
            if move.known_draw and move.drawHappened and self.curr_player_order!=self.main_player_order:
                    self.deck.RemoveCards(move.drawn_card)
            #update player state
            curr_player.handle_remove(move.cardHandIndex, known)
            curr_player.handle_draw(move.cardHandIndex, move.drawHappened, self.top_card_index,
                                    move.drawn_card,self.deck.endgame, move.known_draw,hardUnknown = not(self.is_root))
            self.blue_tokens-=1
        elif move.type == 1:
            #play
            #Note: we never play an unknown card as it changes the game too drastically

            if  move.known:
                #print(move.ToString())
                #print(move.used_card)
                self.deck.RemoveCardsFromGame(move.used_card)
                card = move.used_card
                if self.curr_player_order == self.main_player_order:
                    self.deck.RemoveCards(move.used_card)

            else:
                #the agent preferably only plays cards with perfect knowledge in the MCTS.
                #In the case we play a card without perfect knowledge but perfect playability
                #(i.e. all ones are playable, we know we have a one but don't know the color),
                #the node is set as terminal since the game state is too uncertain from now on  
                hint = curr_player.hints[:,:, move.cardHandIndex]
                
                kn = hint >=2
                if np.any(kn):
                    #we take into account this play because we have perfect knowledge: understand what's the card
                    x,y = kn.nonzero()
                    card = np.array([[x[0]],[y[0]]])
                    self.deck.RemoveCardsFromGame(card)
                    self.deck.RemoveCards(card)
                else:
                    #we selected this move just because of playability but we don't know what it is exactly.
                    #pick the card as any of the possible ones, then set status as terminal (we won't expand this node)
                    x,y = np.unravel_index(np.argmax(hint), hint.shape)
                    card = np.array([[x], [y]])
                    self.deck.RemoveCardsFromGame(card)
                    self.deck.RemoveCards(card)
                    self.terminal = True
            
            if move.known_draw and move.drawHappened and self.curr_player_order!=self.main_player_order:
                self.deck.RemoveCards(move.drawn_card)


            curr_player.handle_remove(move.cardHandIndex, True)
            curr_player.handle_draw(move.cardHandIndex, move.drawHappened, 
                                    self.top_card_index,move.drawn_card,self.deck.endgame,move.known_draw, hardUnknown = not(self.is_root))
            if card[0] != self.fireworks[card[1]]+1 or move.thunder == True:
                self.red_tokens += 1
            else:
                
                self.fireworks[card[1]]+=1
                if card[0] == 4:
                    self.blue_tokens -= 1
                print(self.fireworks)
        elif move.type == 2:
            #hint
            #we accept hints on cards we don't know, a player will gain a base knowledge score when that happens
            dest_player = self.players[move.destination]
            dest_player.handle_hint(move.hd_type, move.hd_value, move.positions, known= move.known)
            self.blue_tokens += 1
        
                     

    def MC_simulate(self):
        #the basic unit of score is knowledge of 1 value (color or number) for 1 card and it's equal to 1

        #red token contribution: score is 0 with 3 red tokens, 
        rt_contrib = 0
        if self.red_tokens == 3:
            #we lost
            self.score = 0
            return 0
        else:
            rt_contrib += 3 - self.red_tokens #only light punishment if we got few thunderstrikes
        
        #blue token contribution: we assume that a token always gives only 0.95 score point
        #i.e. a played hint that gives one or more knowledge points is considered a better contribution
        bt_contrib = 0.95*(8-self.blue_tokens)
        #also we take into account if we're in end_game
        if self.deck.n_cards_in_filtered_deck<=0:
            bt_contrib = bt_contrib*0.25*(len(self.players)- self.curr_player_order)/len(self.players)

        
        kn_contrib = 0
        #add knowledge score of each player
        for p in self.players:
            #add only if we're not in end game or the player can still play
            if self.deck.endgame == False or self.deck.turns_until_end > 0:
                kn_contrib += p.get_score()

        #add firework contribution: we assume that a play is worth 2 knowledge point + 3 coincidence points
        #we also bias the evaluation to favor plays with lower value
        bias = np.array([1.0, 0.99, 0.98, 0.97, 0.96])
        fw_points = 100*(self.fireworks+1) * bias[self.fireworks]
        fw_contrib = np.sum(fw_points)

        self.score = rt_contrib+bt_contrib+kn_contrib+fw_contrib
        self.score = math.pow(D_FACTOR, self.depth)*self.score

        #TODO: take decision on score normalization
        #normalize
        #self.score /= self.max_score
        return self.score

    def MC_expand(self):
        mcplayer = self.players[self.curr_player_order]

        moves = mcplayer.GetMoves( self.players, self.fireworks, self.deck, self.red_tokens, self.blue_tokens)

        for i in range(0, min(MAX_CHILDREN, len(moves))):
            
            child = MCNode(self.fireworks,self.blue_tokens,self.red_tokens,self.players,self.deck,self.curr_player_order,
                            self.main_player_order,self.MainDeck,self.top_card_index,False, move=moves[i], depth=self.depth+1,parent=self)
            #print(f"creating node for move: {moves[i].ToString()}. Child has score: {child.score}")
            self.children.append(child)
            child.MC_backprop()
        
        self.expanded = True
        if self.depth >= MAX_DEPTH:
            self.terminal == True


    def MC_backprop(self):
        tmpnode = self.parent
        while tmpnode.is_root == False:
            tmpnode.children_scores += self.score
            tmpnode.n_simulations+=1
            tmpnode = tmpnode.parent
        tmpnode.children_scores += self.score
        tmpnode.n_simulations+=1
        

    def MC_select(self):
        tmpnode = self
        while tmpnode.expanded ==True:
            node = tmpnode.GetBestNodeUTC()
            if node == None:
                tmpnode.terminal = True 
                tmpnode = tmpnode.parent
            else:
                tmpnode = node
        return tmpnode

    def FindMove(self, iterations = MC_ITERATIONS):
        print("-----FINDING MOVE-----")
        print(f"root is {self.ToKey()}")
        assert self.curr_player_order == self.main_player_order
        for i in range(0, iterations):
            node = self.MC_select()
            #print(f"Selected: {node.ToString()}")
            node.MC_expand()
        

        bestNode=self.GetBestNodeAVGScore()
        print(f"Best move found is {bestNode.ToString()}")
        print("---------END FIND--------")
        return bestNode.move
        


    def Enforce(self):
        pass
    
    def GetBestNodeUTC(self):
        node = None 
        maxscore= -100
        for i in range(0, len(self.children)):
            child = self.children[i]
            if child.terminal:
                continue
            c_score = child.UTCscore()
            if c_score > maxscore:
                maxscore = c_score
                node = child
        
        return node 
    def GetBestNodeAVGScore(self):
        node = None 
        maxscore= -100
        for i in range(0, len(self.children)):
            child = self.children[i]
            print(f"{child.ToString()}")
            c_score = child.children_scores/child.n_simulations
            if c_score > maxscore:
                maxscore = c_score
                node = child
        
        return node 

    def UTCscore(self):
        #print(f"({self.children_scores}/{self.n_simulations})+ sqrt(2*log({self.parent.n_simulations}/{self.n_simulations}))")
        return (self.children_scores/self.n_simulations) + math.sqrt(10*math.log(self.parent.n_simulations/self.n_simulations))
    
    def has_child(self, move):
        return move.ToKey() in self.childrenKeys

    def ToString(self):
        str = f"NODE. player: {self.players[self.curr_player_order].name}, depth: {self.depth}, bt: {self.blue_tokens}, score: {self.score}, ch_score: {self.children_scores}, n_sim: {self.n_simulations}"
        if self.move != None:
            str += f"\nmove: {self.move.ToString()}-fireworks: {self.fireworks}"
        if VERBOSE:
            str += self.deck.DeckToString()
        return str
    
    def ToKey(self):
        str = ""
        if self.parent != None:
            str += self.parent.ToKey()

        if self.move != None:
            str+=f"|d{self.depth}-mp{self.main_player_order}-cp{self.curr_player_order}-{self.move.ToKey()}|"
        elif self.is_root:
            str+=f"|ROOT-mp{self.main_player_order}-cp{self.curr_player_order}|"
        return str 


