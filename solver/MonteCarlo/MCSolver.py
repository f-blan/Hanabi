from matplotlib.pyplot import draw
from solver.HanabiSolver import HanabiSolver
from solver.MonteCarlo.MCMove import MCMove
#from .. import HanabiSolver
from solver.MonteCarlo.MCPlayer import MCPlayer
from solver.MonteCarlo.MCDeck import MCDeck
from solver.MonteCarlo.MCNode import MCNode
import numpy as np
from .. import utils



class MCSolver(HanabiSolver):
    def __init__(self, data, players: list, player_name: str):
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
                        nplayer = MCPlayer( po.name, False,i,self.cardsInHand, cards=po.hand)
                        self.players.append(nplayer)
            else:
                fplayer = MCPlayer(p, True,i, self.cardsInHand)
                self.players.append(fplayer)
                self.main_player = fplayer
            i+=1
        
        self.current_player=0
        #initialize the deck
        self.deck = MCDeck(self.main_player, len(self.players))

        #remove the known cards from the deck 
        for p in self.players:
            if p.name != self.main_player.name:
                self.deck.RemoveCards(p.cards)

        self.deck.update_expected_values(self.fireworks,self.main_player)
        #evaluate playabilities for each player:
        for p in self.players:
            if p.name != self.main_player.name:
                p.playabilities, p.discardabilities = self.deck.evaluate_known_cards(self.fireworks,p.cards,p.hard_unknowns)
        
        self.main_player.playabilities, self.main_player.discardabilities = self.deck.evaluate_unknown_cards(self.main_player.cardsInHand, self.fireworks, self.main_player.hints, self.main_player.hard_unknowns)

        self.root = MCNode(self.fireworks,self.blue_tokens,self.red_tokens,self.players,self.deck.ndeck, self.current_player,
                            self.main_player.order,self.deck,self.deck.last_unknown_index,True)
        

    def FindMove(self):
        """
            Idea: we're giving the root a certain amount of iterations to find the best
            move. Move evaluation is left to the MCTS algorithm
        """
        best_move = self.root.FindMove()
        return best_move
    
    def Enforce(self):
        """
            Idea: it's not our turn, let's use this time to expand our Monte Carlo tree with 1 iteration
            note: not implemented
        """
        self.root.Enforce()
    
    def RecordMove(self, data, mtype):
        if mtype == "discard":
            mcplayer = self.get_player(data.lastPlayer)
            #self.lastMove = MCMove(0, mcplayer.order, True)

            #mcplayer.handle_remove(data.cardHandIndex)
            self.blue_tokens -=1
            playedCard = np.array([[utils.encode_value(data.card.value)], [utils.encode_color(data.card.color)]])
            self.deck.RemoveCardsFromGame(playedCard)
            self.drawHappened = data.handLength == self.main_player.cardsInHand
            if mcplayer.main == True:
                self.deck.RemoveCards(playedCard)
            self.lastMove = MCMove(0, mcplayer.order,True, cardHandIndex= data.cardHandIndex, used_card=playedCard,drawHappened=self.drawHappened)
            self.lastMove.define_discard(data.cardHandIndex)
        elif mtype == "play":
            mcplayer = self.get_player(data.lastPlayer)
            #self.lastMove = MCMove(1, mcplayer.order)

            #mcplayer.handle_remove(data.cardHandIndex)
            if data.card.value == 5 and self.blue_tokens > 0:
                self.blue_tokens -=1
            self.fireworks[utils.encode_color(data.card.color)]+=1
            playedCard = np.array([[utils.encode_value(data.card.value)], [utils.encode_color(data.card.color)]])
            self.deck.RemoveCardsFromGame(playedCard)
            self.drawHappened = data.handLength == self.main_player.cardsInHand
            if mcplayer.main == True:
                self.deck.RemoveCards(playedCard)
            self.lastMove = MCMove(1, mcplayer.order,True,cardHandIndex=data.cardHandIndex,used_card=playedCard,drawHappened= self.drawHappened)
            self.lastMove.define_play(data.cardHandIndex)
        elif mtype == "hint":
            mcplayer = self.get_player(data.source)
            destination = self.get_player(data.destination)
            self.drawHappened = False
            self.lastMove = MCMove(2, mcplayer.order,True,drawHappened=False)
            if data.type == "color":
                self.lastMove.define_hint(destination, 1, utils.encode_color(data.value))
            else:
                self.lastMove.define_hint(destination,0, utils.encode_value(data.value))
            
            self.lastMove.finalize_hint(destination.order, positions=data.positions)
        elif mtype == "thunder":
            mcplayer = self.get_player(data.lastPlayer)
            #self.lastMove = MCMove(1, mcplayer.order)

            #mcplayer.handle_remove(data.cardHandIndex)
            self.red_tokens += 1
            playedCard = np.array([[utils.encode_value(data.card.value)], [utils.encode_color(data.card.color)]])
            self.deck.RemoveCardsFromGame(playedCard)
            self.drawHappened = data.handLength == self.main_player.cardsInHand
            if mcplayer.main == True:
                self.deck.RemoveCards(playedCard)
            self.lastMove = MCMove(1, mcplayer.order, data.cardHandIndex, drawHappened= self.drawHappened, thunder=True,used_card=playedCard)
            self.lastMove.define_play(data.cardHandIndex)
        elif mtype == "draw":
            
            mcplayer = self.get_player(data.currentPlayer, -1)
            if self.lastMove.drawHappened and mcplayer.main == False:
                for p in data.players:
                    if p.name == mcplayer.name:
                        drawingPlayer = p
                        handLength = len(p.hand)
                    
                drawn_card= drawingPlayer.hand[handLength-1]
                drawn_card = np.array([[utils.encode_value(drawn_card.value)],[utils.encode_color(drawn_card.color)]])
                #self.deck.card_value_per_index[self.deck.last_unknown_index] = drawn_card[0]
                #self.deck.card_color_per_index[self.deck.last_unknown_index] = drawn_card[1]
                self.deck.RemoveCards(drawn_card)
                print(f"drawn_card: {drawn_card}")
                if self.lastMove.type == 0:
                    self.lastMove.finalize_discard(drawn_card)
                elif self.lastMove.type == 1:
                    self.lastMove.finalize_play(drawn_card)
            elif self.lastMove.drawHappened:
                self.deck.last_unknown_index-=1

            if self.root.has_child(self.lastMove):
                #To be implemented, for now i don't keep the previously computed tree among different moves
                return 
            print(f"last move was: {self.lastMove.ToString()}")
            if self.lastMove.drawHappened:
                print(f"drawn card was: {self.lastMove.drawn_card}")
            r = self.root
            newRoot = MCNode(r.fireworks,r.blue_tokens,r.red_tokens,r.players,self.deck.ndeck,r.curr_player_order,r.main_player_order,r.MainDeck,self.deck.last_unknown_index,True,move=self.lastMove)
            #newRoot = MCNode(r.fireworks,r.blue_tokens,r.red_tokens,r.players,r.deck,r.curr_player_order, r.main_player,self.deck,True,move=self.lastMove)
            self.root = newRoot

    def HintsToString(self, player_name):
        player_id = -1
        for p in self.root.players:
            if p.name == player_name:
                player_id = p.order
                break
        return self.root.players[player_id].HintsToString()
    
    def DeckToString(self):
        return self.deck.DeckToString()

    def get_player(self, name, pos = 0):
        for p in self.root.players:
            if p.name == name:
                order = (p.order+pos+len(self.players))%len(self.players)
                return self.root.players[p.order + pos]
    

    
