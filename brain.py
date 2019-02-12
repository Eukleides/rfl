
from nodes import Node_Input, Node_Sigmoid, Node_IF, Layer, InputLayer


l = InputLayer([0.1, -0.5])

c = Node_Sigmoid(l, [-0.4,0.9])
b = Node_Sigmoid(l, [0.3,0.4])

d = Node_Sigmoid([b, c], [0.3, -0.2])

e = Node_IF([b, c], [0.3, -0.2], leq=True, bias=1.0, threshold=0.8, default_out=0.75)
f = Node_IF([b, c], [0.3, -0.2], leq=False, bias=1.0, threshold=0.8, default_out=0.75)

print('{:.6f}'.format(d.out()))

print('{:.6f}'.format(e.out()))
print('{:.6f}'.format(f.out()))