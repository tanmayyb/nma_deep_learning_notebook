# nma_deep_learning_notebook

## W1D1

## Section 2: Basics of Pytorch
**flatten/reshape**
```
z = torch.arange(12).reshape(6, 2)
z.flatten()
```

**Squeezing Tensors**
To get rid of singleton dimensions in your data we use squeeze
```
x = torch.randn(1, 10)
x.squeeze(0)
```
**unsqueezing a tensor**
```
y = torch.randn(5, 5)
y = y.unsqueeze(1)
Shape of y: torch.Size([5, 5])
Shape of y: torch.Size([5, 1, 5])
```

**Permutation**
dim  [3×48×64] , but we want dim [48×64×3]
```
x = torch.rand(3, 48, 64)

# We want to permute our tensor to be [ image_height , image_width , color ]
x = x.permute(1, 2, 0)

torch.Size([48, 64, 3])
```
**Concat**
```
# Create two tensors of the same shape
x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# Concatenate along rows
cat_rows = torch.cat((x, y), dim=0)

# Concatenate along columns
cat_cols = torch.cat((x, y), dim=1)

Concatenated by rows: shape[6, 4] 
 tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [ 2.,  1.,  4.,  3.],
        [ 1.,  2.,  3.,  4.],
        [ 4.,  3.,  2.,  1.]])

 Concatenated by colums: shape[3, 8]  
 tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
        [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
        [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]])
```

