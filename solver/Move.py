
class Move:
    def __init__(self, type):
        """type: 0 discard, 1 play, 2 hint"""
        self.type = type
    
    def define_discard(self,card_n):
        assert self.type == 0
        self.card_n = card_n

    def define_play(self,card_n):
        assert self.type == 1
        self.card_n = card_n
    
    def define_hint(self,player_n, type, value):
        self.h_type = type
        self.h_value = value
        self.h_player = player_n

    