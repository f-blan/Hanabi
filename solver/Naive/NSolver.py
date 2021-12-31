import HanabiSolver
import NPlayer
import numpy as np
import utils

class NSolver(HanabiSolver):
    def __init__(self, data, players, player_name):
        super.__init__(self)
        self.players= []
        self.main_play=-1 #for handling play or discard of a card
        self.last_play= -1 #for handling replacement after drawing
        i=0
        for p in players:
            if p != player_name:
                pdata = filter(lambda x: x.name==p, data.players)
                self.players.append(NPlayer( pdata[0].name, False,i, cards=pdata[0].hand))
            else:
                self.players.append(NPlayer(p.name, True,i))
                self.main_player = self.players[-1]
            i+=1
        
        self.current_player=0

    def FindMove():
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
            return move
        
        move = self.main_player.GetUnsafeDiscard(self.fireworks)
        if move!= None:
            self.main_play = move.card_n
            return move

        move=self.main_player.GetUnsafePlay(self.fireworks)
        self.main_play = move.card_n
        return move

    def Enforce():
        pass

    def RecordMove(data, type,, opt=None):
        #get the player that performed the action
        nplayer = filter(lambda x: x.name == data.player, self.players)
        nplayer = nplayer[0]
        player_index = (nplayer.order + len(self.players) -1)%len(self.players)
        nplayer = self.players[player_index]
        match type:
            case "discard":
                remove_card(nplayer, data.card)
                self.blue_tokens -=1
            case "play":
                remove_card(nplayer, data.card)
                self.blue_tokens -=1
                self.fireworks[utils.encode_color(data.card.color)]+=1
            case "hint":
                nplayer = filter(lambda x: x.name == data.destination, self.players)
                hinted_val = -1
                if data.type == "color":
                    hinted_val = utils.encode_color(data.value)
                else:
                    hinted_val = utils.encode_value(data.value)
                vec = np.zeros(5)-1
                vec[hinted_val] = 1
                for p in data.positions:
                    if data.type == "color":
                        nplayer.hint_color[p, hinted_val] = vec
                    else:
                        nplayer.hint_value[p, hinted_val] = vec
                self.blue_tokens += 1
                self.last_play = -1
            case "thunder":
                remove_card(nplayer, data.card)
                self.red_tokens += 1
            case "draw":
                if self.last_play == -1:
                    #case when last play was a hint, no draws happened
                    return
                drawn_id = self.last_play
                self.last_play = -1
                drawn_card = data.players[player_index].hand[drawn_id]
                nplayer.cards[0, drawn_id] = utils.encode_value(drawn_card.value)
                nplayer.cards[1, drawn_id] = utils.encode_color(drawn_card.color)
                self.deck.remove_cards(drawn_card)



    def remove_card(nplayer, card):
        #erase the card
        cardId = -1

        if nplayer.name == self.main_player.name:
            assert self.main_play >= 0
            cardId = self.main_play
            self.main_play = -1
        else:
            ids = nplayer.cardIds == card.id
            cardId = argmax(ids)
        
        self.last_play = cardId
        nplayer.cards[0, cardId] = -1
        nplayer.cards[1, cardId] = -1

        #erase hints
        nplayer.hint_value[cardId, :] = np.zeros(5)
        nplayer.hint_color[cardId, :] = np.zeros(5)

                
                



        

        
        