l = list(range(0,10))

pairs = list(zip(l,l[1:]+l[0:1],l[2:]+l[0:2],l[3:]+l[0:3]))

p = ((1,3),(3,1),(1,3))
s = set(map(tuple,map(sorted,p)))
print(s)
