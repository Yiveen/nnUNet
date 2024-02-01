from torch import nn

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        # 使用 1x1 卷积调整通道数
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1)

        self.psi = nn.Conv2d(F_int, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.W_g(g) #g is from seg branch
        x1 = self.W_x(x) #x is from key point branch
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))

        return x + x * psi #residual
