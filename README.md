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

## Section3: Neural Networks
### Section3.1 Load CSV

**Load CSV**
```
X_orig = data[["x0", "x1"]].to_numpy() # Create a numpy array from the x0 and x1 columns from the data imported from csv
y_orig = data["y"].to_numpy()

#tensors from numpy array
X = torch.tensor(X_orig, dtype=torch.float32)
y = torch.from_numpy(y_orig).type(torch.LongTensor)
```
**uploading tensors to GPU**
```
DEVICE = set_device() #checks if GPU is set up for runtime

# Upload the tensor to the device
X = X.to(DEVICE)
y = y.to(DEVICE)
```
### Section 3.2 Simple Neural Network

**Programing the Network**

PyTorch provides a base class for all neural network modules called [`nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). You need to inherit from `nn.Module` and implement some important methods:

* `__init__`
* `forward`
* `predict`
* `train`
<br>

### Section 3.3 Train Your Neural Network
