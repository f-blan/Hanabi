import HanabiSolver

import KPlayer

class KSolver(HanabiSolver):
    def __init__(self, data, player_name):
        super.__init__(self)
        self.players = []
        for p in data.players:
            if p.name != player_name:
                self.players.append(KPlayer(p))
                