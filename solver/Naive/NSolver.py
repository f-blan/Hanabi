from solver.HanabiSolver import HanabiSolver
#from .. import HanabiSolver
from . import NPlayer
import numpy as np
from .. import utils

class NSolver(HanabiSolver):
    def __init__(self, data, players, player_name):
        super().__init__()
        self.players= list()
        self.main_play=-1 #for handling play or discard of a card
        self.last_play= -1 #for handling replacement after drawing
        i=0
        self.cardsInHand = 5
        if len(players)>3:
            self.cardsInHand = 4
        for p in players:
            if p != player_name:
                for po in data.players:
                    if po.name == p:
                        nplayer = NPlayer.NPlayer( po.name, False,i,self.cardsInHand, cards=po.hand)
                        self.players.append(nplayer)
            else:
                nplayer = NPlayer.NPlayer(p, True,i, self.cardsInHand)
                self.players.append(nplayer)
                self.main_player = self.players[-1]
            i+=1
        
        self.current_player=0
        

    def FindMove(self):
        """
        Simple priority: 
            safe play
            safe discard
            hint safe play
            hint random
            unsafe discard
            unsafe play
        """

        move = self.main_player.GetSafePlay(self.fireworks)
        if move != None:
            self.main_play = move.card_n
            return move
        
        if self.blue_tokens >0:
            move = self.main_player.GetSafeDiscard(self.fireworks)
            if move!=None:
                self.main_play = move.card_n
                return move
        
        if self.blue_tokens <self.max_b:
            move = self.main_player.GetSafeHint(self.fireworks, self.players)
            if move != None:
                return move

            move = self.main_player.GetRandomHint(self.fireworks, self.players)
            assert move != None
            return move
        
        move = self.main_player.GetUnsafeDiscard(self.fireworks)
        if move!= None:
            self.main_play = move.card_n
            return move

        move=self.main_player.GetUnsafePlay(self.fireworks)
        self.main_play = move.card_n
        assert move != none
        return move

    def Enforce(self):
        pass
     
            
    def RecordMove(self,data, mtype):
        #get the player that performed the action
        #nplayer = nplayer[0]
        #player_index = (nplayer.order + len(self.players) -1)%len(self.players)
        #nplayer = self.players[player_index]
    
        if mtype == "discard":
            nplayer = self.get_player(data.lastPlayer)
            self.remove_card(nplayer, data.cardHandIndex)
            self.blue_tokens -=1
        elif mtype == "play":
            nplayer = self.get_player(data.lastPlayer)
            self.remove_card(nplayer, data.cardHandIndex)
            #self.blue_tokens -=1
            self.fireworks[utils.encode_color(data.card.color)]+=1
        elif mtype == "hint":
            nplayer = self.get_player(data.destination)
            #nplayer = filter(lambda x: x.name == data.destination, self.players)
            hinted_val = -1
            if data.type == "color":
                hinted_val = utils.encode_color(data.value)
            else:
                hinted_val = utils.encode_value(data.value)
            vec = np.zeros(5)-1
            vec[hinted_val] = 1
            for p in data.positions:
                if data.type == "color":
                    nplayer.hint_color[p, :] = vec
                else:
                    nplayer.hint_value[p, :] = vec
            self.blue_tokens += 1
            self.last_play = -1
        elif mtype == "thunder":
            nplayer = self.get_player(data.lastPlayer)
            self.remove_card(nplayer, data.cardHandIndex)
            self.red_tokens += 1
        elif mtype == "draw":
            nplayer = self.get_player(data.currentPlayer, -1)
            if nplayer.cardHandIndex == -1:
                #case when last play was a hint, no draws happened
                return
            played_id = nplayer.cardHandIndex
            nplayer.cardHandIndex = -1
            
            if nplayer.main:
                #we can't know which card it was if we're the ones who drew it
                nplayer.handle_draw(played_id)
            else :
                #we can store the card since we can see it
                #drawn_card = data.players[nplayer.order].hand[played_id]
                handLength = -1
                drawingPlayer = None
                for p in data.players:
                    if p.name == nplayer.name:
                        #drawn_card = p.hand[4]
                        handLength = len(p.hand)
                        drawingPlayer = p
                if handLength ==self.cardsInHand:
                    drawn_card = drawingPlayer.hand[handLength-1]
                    nplayer.handle_draw(played_id, drawn_card)
                    self.deck.remove_cards(drawn_card)
                    #print(f"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAplayer {nplayer.name} has drawn {nplayer.cards[:, 4]}")
                else:
                    nplayer.handle_draw(played_id)
            

                

    def HintsToString(self, player_name):
        player_id = -1
        for p in self.players:
            if p.name == player_name:
                player_id = p.order
                break
        return self.players[player_id].HintsToString()

    def DeckToString(self):
        return "no deck info"

    def remove_card(self, nplayer, cardId):
        #erase the card
        """
        cardId = -1

        if nplayer.name == self.main_player.name:
            assert self.main_play >= 0
            cardId = self.main_play
            self.main_play = -1
        else:
            ids = nplayer.cardIds == card.id
            cardId = argmax(ids)
        """
        nplayer.cardHandIndex = cardId
        nplayer.cards[0, cardId] = -1
        nplayer.cards[1, cardId] = -1

        #erase hints
        nplayer.hint_value[cardId, :] = np.zeros(5)
        nplayer.hint_color[cardId, :] = np.zeros(5)

    def get_player(self, name, pos = 0):
        for p in self.players:
            if p.name == name:
                order = (p.order+pos+len(self.players))%len(self.players)
                return self.players[p.order + pos]
                



        

        
        