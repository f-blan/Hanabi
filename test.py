import numpy as np
from numpy.core.fromnumeric import argmax

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

ar = np.array([[-1 for i in range(0,4)],[-1 for i in range(0,4)]])
print(ar)
ar2 = np.zeros((4,5), dtype=np.int16)
print(ar2)

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