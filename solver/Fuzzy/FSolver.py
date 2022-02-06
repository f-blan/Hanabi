from solver.HanabiSolver import HanabiSolver
#from .. import HanabiSolver
from solver.Fuzzy.FPlayer import FPlayer
from solver.Fuzzy.FDeck import FDeck
import numpy as np
from .. import utils



class FSolver(HanabiSolver):
    def __init__(self, data, players: list, player_name: str):

        """
            Initialize the internal game state:
            -Players
            -Deck
            -statistics on the deck and player cards
        """
        super().__init__()
        self.players= list()
        i=0
        self.cardsInHand = 5
        
        if len(players)>3:
            self.cardsInHand = 4
        for p in players:
            if p != player_name:
                for po in data.players:
                    if po.name == p:
                        nplayer = FPlayer( po.name, False,i,self.cardsInHand, cards=po.hand)
                        self.players.append(nplayer)
            else:
                fplayer = FPlayer(p, True,i, self.cardsInHand)
                self.players.append(fplayer)
                self.main_player = self.players[-1]
            i+=1
        
        self.current_player=0
        #initialize the deck
        self.deck = FDeck(self.main_player, len(self.players))

        #remove the known cards from the deck 
        for p in self.players:
            if p.name != self.main_player.name:
                self.deck.RemoveCards(p.cards)

        self.deck.update_expected_values(self.fireworks)
        #evaluate playabilities for each player:
        for p in self.players:
            if p.name != self.main_player.name:
                p.playabilities, p.discardabilities = self.deck.evaluate_known_cards(p.cards, self.fireworks)
        
        self.main_player.playabilities, self.main_player.discardabilities = self.deck.evaluate_unknown_cards(self.main_player.cardsInHand, self.fireworks, self.main_player.hints)
        
    def FindMove(self):
        """
            Basic idea: our agent selects a certain (limited) amount of plays available based on playabilities
            and discardabilities. To each of them is then assigned a score (as a move). The move with highest score
            is selected for being performed
        """
        moves = self.main_player.GetMoves(self.players, self.fireworks, self.deck, self.red_tokens, self.blue_tokens)
        #for m in moves:
            #print(f"{m.ToString()}")
        return moves[0]
    
    def Enforce(self):
        """
            scrapped idea
        """
        pass

    def RecordMove(self,data, mtype):
        """
            Function to update the internal game state after the server has sent us data.
            We also recompute statistics on the deck here
        """

        main_recompute = False
        others_recompute = False

        if mtype == "discard":
            fplayer = self.get_player(data.lastPlayer)
            fplayer.handle_remove(data.cardHandIndex)
            self.blue_tokens -=1
            playedCard = np.array([[utils.encode_value(data.card.value)], [utils.encode_color(data.card.color)]])
            self.deck.RemoveCardsFromGame(playedCard)
            self.drawHappened = data.handLength == self.main_player.cardsInHand
            if fplayer.main == True:
                self.deck.RemoveCards(playedCard)
        elif mtype == "play":
            fplayer = self.get_player(data.lastPlayer)
            if data.card.value == 5:
                self.blue_tokens-=1
            fplayer.handle_remove(data.cardHandIndex)
            #self.blue_tokens -=1
            self.fireworks[utils.encode_color(data.card.color)]+=1
            playedCard = np.array([[utils.encode_value(data.card.value)], [utils.encode_color(data.card.color)]])
            self.deck.RemoveCardsFromGame(playedCard)
            self.drawHappened = data.handLength == self.main_player.cardsInHand
            if fplayer.main == True:
                self.deck.RemoveCards(playedCard)
        elif mtype == "hint":
            fplayer = self.get_player(data.destination)
            
            hinted_val = -1
            type = -1
            if data.type == "color":
                hinted_val = utils.encode_color(data.value)
                type = 1
            else:
                hinted_val = utils.encode_value(data.value)
                type = 0
            fplayer.handle_hint(type, hinted_val, data.positions)
            self.blue_tokens += 1
            self.last_play = -1
            if fplayer.name == self.main_player.name:
                main_recompute = True
        elif mtype == "thunder":
            fplayer = self.get_player(data.lastPlayer)
            fplayer.handle_remove(data.cardHandIndex)
            self.red_tokens += 1
            playedCard = np.array([[utils.encode_value(data.card.value)], [utils.encode_color(data.card.color)]])
            self.deck.RemoveCardsFromGame(playedCard)
            self.drawHappened = data.handLength == self.main_player.cardsInHand
            if fplayer.main == True:
                self.deck.RemoveCards(playedCard)
        elif mtype == "draw":
            fplayer = self.get_player(data.currentPlayer, -1)
            if fplayer.cardHandIndex == -1:
                #case when last play was a hint, no draws happened
                return
            played_id = fplayer.cardHandIndex
            fplayer.cardHandIndex = -1
            
            if fplayer.main:
                #we can't know which card it was if we're the ones who drew it
                fplayer.handle_draw(played_id, self.drawHappened)
                main_recompute = True
            else :
                #we can store the card since we can see it
                #drawn_card = data.players[nplayer.order].hand[played_id]
                main_recompute = True 
                others_recompute = True
                handLength = -1
                drawingPlayer = None
                for p in data.players:
                    if p.name == fplayer.name:
                        #drawn_card = p.hand[4]
                        handLength = len(p.hand)
                        drawingPlayer = p
                if handLength ==self.cardsInHand:
                    #draw happened and we know the card
                    drawn_card = drawingPlayer.hand[handLength-1]
                    fplayer.handle_draw(played_id,True, drawn_card)
                    
                    playedCard = np.array([[utils.encode_value(drawn_card.value)], [utils.encode_color(drawn_card.color)]])
                    self.deck.RemoveCards(playedCard)
                else:
                    fplayer.handle_draw(played_id,False)
        
        
        if others_recompute:
            #a card has left the deck (we gained a lot of information). Update playabilities and discardabilities of players
            for p in self.players:
                if p.main == False:
                    p.playabilities, p.discardabilities = self.deck.evaluate_known_cards(p.cards, self.fireworks)
        if main_recompute:
            #from logic, others_compute implies main_recompute: we gained at least an hint to us of information. 
            #update the deck's expected values and the agent's playabilities/discardabilities
            self.deck.update_expected_values(self.fireworks)
            self.main_player.playabilities, self.main_player.discardabilities = self.deck.evaluate_unknown_cards(self.main_player.cardsInHand, self.fireworks, self.main_player.hints)

    def HintsToString(self, player_name):
        player_id = -1
        for p in self.players:
            if p.name == player_name:
                player_id = p.order
                break
        return self.players[player_id].HintsToString()
    
    def DeckToString(self):
        return self.deck.DeckToString()

    def get_player(self, name, pos = 0):
        """auxiliary function, gets the player as a function of the name"""
        for p in self.players:
            if p.name == name:
                order = (p.order+pos+len(self.players))%len(self.players)
                return self.players[p.order + pos]
