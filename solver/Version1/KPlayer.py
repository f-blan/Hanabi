import numpy as np

class KPlayer():
    def __init__(self, hand, ):
        self.name = p.name
        self.cards = np.array([
                                [-1,-1,-1,-1,-1],
                                [-1,-1,-1,-1,-1],
                                ], dtype=np.int16)
        #y axis for card number (in hand), x axis for type
        self.color_knowledge = np.array([
                                [0.2,0.2,0.2,0.2,0.2],
                                [0.2,0.2,0.2,0.2,0.2],
                                [0.2,0.2,0.2,0.2,0.2],
                                [0.2,0.2,0.2,0.2,0.2],
                                [0.2,0.2,0.2,0.2,0.2],
                                ], dtype=np.float32)
        self.value_knowledge = np.array([
                                [0.2,0.2,0.2,0.2,0.2],
                                [0.2,0.2,0.2,0.2,0.2],
                                [0.2,0.2,0.2,0.2,0.2],
                                [0.2,0.2,0.2,0.2,0.2],
                                [0.2,0.2,0.2,0.2,0.2],
                                ], dtype=np.float32)
        #0 no info given, -1 can't be target, 1 card is target 
        self.hint_color = np.array([
                                    [0,0,0,0,0],
                                    [0,0,0,0,0],
                                    [0,0,0,0,0],
                                    [0,0,0,0,0],
                                    [0,0,0,0,0],
                                    ])
        self.hint_value = np.array([
                                    [0,0,0,0,0],
                                    [0,0,0,0,0],
                                    [0,0,0,0,0],
                                    [0,0,0,0,0],
                                    [0,0,0,0,0],
                                    ])
        for card in 
