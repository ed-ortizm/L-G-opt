import sys
from lm00 import *
n = int(sys.argv[1])
x = float(sys.argv[2])
y = float(sys.argv[3])
p0 = np.array([x,y])
f = Function(n=n)
p = lm(p0,f)
print("the optimal point is:", p)
