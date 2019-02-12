import numpy as np

class Node:

    def __init__(self, prev=None):
        self.prev = prev

#################################################################################
# Description: a set of nodes sharing the same inputs and
# giving access to their combined output
class Layer():
    def __init__(self, nodes):
        self.nodes = nodes

    def out(self):
        return self.nodes

    def __getitem__(self, i):
        return self.nodes[i]

    def __len__(self):
        return len(self.nodes)

#################################################################################
# Description: a set of nodes sharing the same inputs and
# giving access to their combined output
class InputLayer(Layer):
    def __init__(self, inputs):
        nodes = []
        for in_val in inputs:
            nodes.append(Node_Input(in_val))

        super(InputLayer, self).__init__(nodes)


#################################################################################
# Description: single input value node
# Inputs: value
# Output: value
class Node_Input(Node):

    def __init__(self, value):
        self.value = value

    def out(self):
        return self.value

#################################################################################
# Description: sigmoid node
# Inputs: [prev node, weights, bias]
# Output: 1/(a+exp(x)), where x = (prev.out() .dot weights) + bias
class Node_Sigmoid(Node):

    def __init__(self, prev, weights, bias=1.0):
        self.prev = prev
        self.weights = weights
        self.bias = bias
        assert len(prev)==len(weights), "Incompatible weight size in sigmoid"

    def out(self):
        sum = self.bias
        n = len(self.weights)
        for i in range(n):
            sum += self.prev[i].out()*self.weights[i]

        res = 1.0/(1.0+np.exp(sum))
        return res

#################################################################################
# Description: IF node
# Inputs: [prev node, weights, bias, cond, threshold, default_out]
# Output:
# let x = (prev.out() .dot weights) + bias
# leq (<=)
# if  x <= threshold return default_out else 0
# not leq (>)
# if  x >= threshold return default_out else 0
class Node_IF(Node):

    def __init__(self, prev, weights, leq, bias=1.0, threshold=0., default_out=1.0):
        self.prev = prev
        self.weights = weights
        self.leq = leq
        self.bias = bias
        self.threshold = threshold
        self.default_out = default_out
        assert len(prev)==len(weights), "Incompatible weight size in sigmoid"

    def out(self):
        sum = self.bias
        n = len(self.weights)
        for i in range(n):
            sum += self.prev[i].out()*self.weights[i]

        res = 0.
        if self.leq: # <=
            if sum<=self.threshold:
                res = self.default_out
        else: # >=
            if sum>=self.threshold:
                res = self.default_out
        return res

#################################################################################
class Node_Mem(Node):

    def __init__(self, prev, weights, cond, bias, threshold, default_out=1.0):
        self.prev = prev
        self.weights = weights
        self.cond = cond
        self.bias = bias
        self.threshold = threshold
        self.default_out = default_out
        assert len(prev)==len(weights), "Incompatible weight size in sigmoid"

    def out(self):
        sum = self.bias
        for p, w in zip(self.prev, self.weights):
            sum += p.out()*w

        res = 0.
        if self.cond==0: # <=
            if sum<=self.threshold:
                res = self.default_out
        else: # >=
            if sum>=self.threshold:
                res = self.default_out
        return res
