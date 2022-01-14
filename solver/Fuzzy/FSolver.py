from solver.HanabiSolver import HanabiSolver
#from .. import HanabiSolver
from . import FPlayer
import numpy as np
from .. import utils



class FSolver(HanabiSolver):
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