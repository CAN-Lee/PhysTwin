现在，您的系统将物理参数从 原始可学习空间（Raw/Latent Space） 映射到了 对数-Sigmoid 空间（Log-Sigmoid Space）。
具体的数学映射公式如下：
设 $x_{\text{raw}}$ 为优化器(ADAMS)直接优化的原始参数（即 self.raw_E, self.raw_yield 等），$P$ 为最终用于模拟的物理参数（如杨氏模量 $E$，屈服应力 $\sigma_y$），$P_{\min}$ 和 $P_{\max}$ 为该参数在配置文件中设定的物理上下界。
映射过程分为两步：
Sigmoid 归一化：将无界的原始参数映射到 $[0, 1]$ 区间。
$$ s = \text{sigmoid}(x_{\text{raw}}) = \frac{1}{1 + e^{-x_{\text{raw}}}} $$
对数域插值与指数还原：在 $P_{\min}$ 和 $P_{\max}$ 的对数域之间进行线性插值，然后取指数还原回物理域。
$$ P = \exp\left( \ln(P_{\min}) + s \cdot (\ln(P_{\max}) - \ln(P_{\min})) \right) $$
或者等价地写成底数为10的形式（效果相同，只是系数不同）：
$$ P = 10^{\left( \log_{10}(P_{\min}) + s \cdot (\log_{10}(P_{\max}) - \log_{10}(P_{\min})) \right)} $$
直观理解：
如果 $x_{\text{raw}} \to -\infty$，则 $s \to 0$，物理参数 $P \to P_{\min}$。
如果 $x_{\text{raw}} \to +\infty$，则 $s \to 1$，物理参数 $P \to P_{\max}$。
如果 $x_{\text{raw}} = 0$，则 $s = 0.5$，物理参数 $P$ 等于 $P_{\min}$ 和 $P_{\max}$ 的几何平均值（即 $\sqrt{P_{\min} P_{\max}}$）。
为什么要这样做？
对于跨越多个数量级的物理参数（例如 $E \in [10^4, 10^7]$），普通的线性插值会导致：
$s=0.5$ 时，$P \approx 5 \times 10^6$（算术平均，偏向大值）。
在 $10^4 \sim 10^5$ 这个“软”区域，Sigmoid 需要输出极小的值（如 0.001），这处于 Sigmoid 的梯度饱和区，导致很难优化出“软”材料。
而在 Log-Sigmoid 映射下：
$s=0.5$ 时，$P \approx 10^{5.5} \approx 3 \times 10^5$（几何平均）。
参数在 $10^4, 10^5, 10^6, 10^7$ 这几个数量级上的分布是均匀的（在 $x_{\text{raw}}$ 空间上等距）。这使得优化器可以轻松地在不同数量级之间穿梭，不会被困在某个极端。