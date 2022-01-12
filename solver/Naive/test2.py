
import NSolver

#from solver import Move
class one():
    def __init__(self):
        self.a = 1
        return
    def hi():
        print("gnegen")
        
class two(one):
    def __init__(self, b):
        super().__init__()
        print("hi")
        self.b = b
    def hi(msg):
        print(msg)

class three(one):
    def __init__(self, num):
        super().__init__()
        self.num = num
        print("hi")
    def RecordMove(self, data, type):
        print("perfomring")
        return super().RecordMove(data, type)


a =three(2)
n = NSolver(None, [], "algio")
n.RecordMove(None, "hint")
#print(a.a)
#a.hi("gnagna")