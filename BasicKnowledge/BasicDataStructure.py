import torch

#arange,shape,numel
x = torch.arange(12)
print(x)
print(x.shape)  #shape
print(x.numel())  #number of element
X = x.reshape(3, 4)
print(X)

#zeros,ones,tensor
a = torch.zeros((2, 3, 4))
print(a)
a = torch.ones((2, 3, 4))
print(a)
a = torch.tensor([[[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]])
print(a.shape)

#calculate
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x**y)
print(torch.exp(x))

#cat
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))
print(X == Y)

#cat dim=2
X = torch.arange(12, dtype=torch.float32).reshape((2, 2, 3))
Y = torch.tensor([[[1.0, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]])
print(torch.cat((X, Y), dim=2))

print(X.sum())

#broadcasting mechanism
a = torch.arange(3).reshape((3, 1))  # broadcast to the whole roll
b = torch.arange(2).reshape((1, 2))  # broadcase to the whole column
print(a + b)

#visit
print(X[-1])  #the last element
print(X[1:3])  #the second and the third element
print(X[1, 1, 2])  #index 1,1,2
X[0:2, :] = 12  #area
print(X)

#id -->
before = id(Y)
Y = Y + X
# Y += X
print(id(Y) == before)  #false   id changed

#dont change id method
Z = torch.zeros_like(Y)
print("id(Z):", id(Z))
Z[:] = X + Y
print("id(Z):", id(Z))
#or like
X += Y
X[:] = X + Y  #dont change id

#numpy transform
A = X.numpy()
B = torch.tensor(A)
print(type(A))
print(type(B))

a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))  #tensor([3.5000]) 3.5 3.5 3
