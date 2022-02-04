import numpy as np
from numpy.random import choice
from numpy.core.fromnumeric import argmax
from sortedcontainers import SortedList
#python3 client.py 

print(np.__version__)
numhint = np.array([
    [1,0,0,0,0],
    [0,0,1,0,0],
    [0,0,0,1,0],
    [0,1,0,0,0],
    [1,0,0,0,0],
    ])

colorhint = np.array([
    [0,0,0,0,0],
    [0,0,1,0,0],
    [0,0,0,1,0],
    [0,1,0,0,0],
    [0,0,0,0,0],
    ])
indexing = np.array([
                                [False,False,False,False,False],
                                [False,False,True,False,False],
                                [False,False,False,False,False],
                                [False,False,False,True,True],
                                [False,False,False,False,False],
                                ], dtype=np.int16) 

    
deck = np.array([
                                [3,3,3,3,3],
                                [2,2,2,2,2],
                                [2,2,2,2,2],
                                [2,2,2,2,2],
                                [1,1,1,1,1]
                                ], dtype=np.int16) 
                        

hint_for_a_card = np.array([
                            [-2,-2,-2,0,-2],
                            [-2,-2,-2,0,-2],
                            [0 ,0 ,0 ,2, 0],
                            [-2,-2,-2,0,-2],
                            [-2,-2,-2,0,-2],
                            ])
hint_for_a_card = np.array([
                            [-1,-1,-1,+1,-1],
                            [-1,-1,-1,+1,-1],
                            [-1,-1,-1,+1,-1],
                            [-1,-1,-1,+1,-1],
                            [-1,-1,-1,+1,-1],
                            ])
hint_for_a_card = np.array([
                            [ 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0],
                            ])

cards_in_game = np.array([
                                [0,0,0,0,0],
                                [0,2,2,2,0],
                                [2,0,2,2,2],
                                [2,0,2,0,0],
                                [1,1,1,1,1]
                                ], dtype=np.int16) 

fireworks = np.array([1,1,1,1,0])
colors = np.array([0,1,2,3,4])

death_line = np.array([5,5,5,5,5])
mask = cards_in_game == 0

testarray= np.array([True,False,True])
testarray= list(testarray.nonzero()[0])
x,y = np.unravel_index(np.argmax(cards_in_game), cards_in_game.shape)
print(f"{x}-{y}")

"""
for i in range(0,5):
    print(mask[fireworks[i]:, i])
    if np.any(mask[fireworks[i]:,i]):
        death_line[i] = np.argmax(mask[fireworks[i]:, i]) + fireworks[i]

"""
print(death_line)

"""
l = SortedList(key = lambda x: x[0])

l.add((1,"hi"))
l.add((0,"gnigni"))
l.add((-1, "fahfapsf"))

d = {}

d.__setitem__("h2c3", None)
d.__setitem__("h4c23", None)
a = True
print(a== True and "h2c3" in d)

if a == True and "h2c3" in d:
    print("hey") 

"""
"""
order = 2                            
a = np.array([0,1,2,3,4,5])
a = np.roll(a, 0)
b = np.array([0.5,0.25,0.75])
#s = choice([0,1,2,3,4,], p= [1,1,0,0])
x,y = indexing.nonzero()
print(x)
print(y)
"""
"""
fireworks = np.array([1,2,2,3,1])
needed_values = fireworks +1
needed_colors = np.array([i for i in range(0,4)])

discardabilities = np.zeros((25,25), dtype= np.float32)
a = np.zeros(5)
discardables =np.arange(5) < 3
x, y = indexing.nonzero()
print(f"{deck[x,y]}")
"""
"""
to_remove = np.array([
    [1,0],
    [0,4]
    ])

deck[to_remove[0,:],to_remove[1,:]]-=1

print(deck)
ar = np.ones([False, True, False])
print(ar)

ar = np.array([[-1 for i in range(0,4)],[-1 for i in range(0,4)]])
#print(ar)
ar2 = np.zeros((4,5), dtype=np.int16)
#print(ar2)
"""
"""
fireworks = np.array([4,2,4,1,1])
colors = np.array([0,1,2,3,4])
filter = fireworks < 4
print(fireworks[filter])
print(colors[filter])
print(f"shape = {fireworks.shape}")

needed_cards = np.array([fireworks+1,[0,1,2,3,4]], dtype=np.int16)

print(needed_cards)
"""
#print(needed_cards[needed_cards[0] < 5])
"""
needednums = np.array([4,3,4,3,3])
neededcolors = np.array([0,1,2,3,4])

string = f"hi + \n{needednums}"

print(string)

colorhint[2] = needednums
print(colorhint)
needednums[3] = 232323
print(colorhint)
i=0
"""

"""
needed_cards = np.array([needednums,[0,1,2,3,4]], dtype=np.int16)
#needed_cards= np.reshape(needed_cards, 10, order='F')
print(np.transpose(needed_cards))
cards = np.array([
                [4,2,4,1,3],
                [0,2,2,1,1]
                ])
rep = np.expand_dims(needed_cards,2)
print(rep)
rep=np.repeat(rep, 5, axis = 2)
print(rep[:,1,:])

print(cards == rep[:,1,:])
print(rep == cards)
#print(np.reshape(rep, [2,5,5]))
#print(cards)
print()
"""
"""
a = np.argmax(numhint==1, axis = 1) #array with card values
e = np.argmax(colorhint==1, axis = 1) #array with card colors
f = np.array([0,1,2,3,4])
b = numhint[f,a] == 1 #completeness assurance
g = colorhint[f,e] == 1 #completeness assurance
c = a< needednums[e]
d = np.logical_and(c,b)
d = np.logical_and(d, g)

completed_colors = needednums >= 4
a = colorhint[:,completed_colors]
b = np.max(a, axis=1)
c = b == 1
d = argmax(c)
print(completed_colors)
print(a)
print(b)
print(c)
print(d)
#print(np.any(d, axis = 1))
"""


"""
a =numhint[:,needednums]
b = colorhint[:,neededcolors]
c = np.max(a+b, axis =1)
print(a)
print(b)
print(a+b)
print(np.max(a+b, axis =1))
to_play = np.argmax(c) 


print(to_play)
"""