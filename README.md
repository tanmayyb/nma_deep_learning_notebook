# nma_deep_learning_notebook

W1D1 Notes


## Squeezing Tensors
To get rid of singleton dimensions in your data we use squeeze
```
x = torch.randn(1, 10)
x.squeeze(0)
```


## Permutation

torch.rand(3, 48, 64)
x.permute(1, 2, 0)
