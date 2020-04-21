import torch

# 测试简单的卷积
m = torch.nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input = torch.randn(20, 16, 50, 100)
output = m(input)
print(output.shape)   # torch.Size([20, 33, 26, 100])

# 测试简单的LSTM
# input_size – The number of expected features in the input x
# hidden_size – The number of features in the hidden state h
# num_layers - lstm layers
rnn = torch.nn.LSTM(10, 20, 2)
x = torch.randn(5, 3, 10)  # (batch_size,timestep,features)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
y, (hn, cn) = rnn(x, (h0, c0))
print(y.shape)  # (5,3,20)