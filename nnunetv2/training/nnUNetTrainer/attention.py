from torch import nn

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        # 使用 1x1x1 卷积调整通道数，这里的 1x1x1 实际上是 kernel_size=(1, 1, 1)
        self.W_g = nn.Conv3d(F_g, F_int, kernel_size=1)
        self.W_x = nn.Conv3d(F_l, F_int, kernel_size=1)

        self.psi = nn.Conv3d(F_int, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.W_g(g) # g is from seg branch
        x1 = self.W_x(x) # x is from key point branch
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))

        # 注意，乘法是 element-wise 的，所以尺寸要匹配
        return x + x * psi  # 这里使用了 element-wise 乘法实现 'attention gating'
