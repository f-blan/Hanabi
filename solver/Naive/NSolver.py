import HanabiSolver
import NPlayer
import numpy as np

class NSolver(HanabiSolver):
    def __init__(self, data, players, player_name):
        super.__init__(self)
        self.players= []

        for p in players:
            if p != player_name:
                pdata = filter(lambda x: x.name==p, data.players)
                self.players.append(NPlayer( pdata[0].name, False, cards=pdata[0].hand))
            else:
                self.players.append(NPlayer(p.name, True))
                self.main_player = self.players[-1]
        
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
            return move
        
        if self.blue_tokens >0:
            move = self.main_player.GetSafeDiscard(self.fireworks)
            if move!=None:
                return move
        
        if self.blue_tokens <self.max_b:
            move = self.main_player.GetSafeHint(self.fireworks, self.players)
            if move != None:
                return move

            move = self.main_player.GetRandomHint(self.fireworks, self.players)
            return move
        
        move = self.main_player.GetUnsafeDiscard(self.fireworks)
        if move!= None:
            return move

        return self.main_player.GetUnsafePlay(self.fireworks)
        

        
        