"""Define Mobile RRN architecture.

Mobile RRN is a lite version of Revisiting Temporal Modeling (RRN) which is a recurrent network for
video super-resolution to run on mobile.

Each Mobile RRN cell firstly concatenate input sequence LR frames and hidden state.
Then, forwarding it through several residual blocks to output prediction and update hidden state.

Reference paper https://arxiv.org/abs/2008.05765
Reference github https://github.com/junpan19/RRN/
"""

import tensorflow as tf


class MobileRRN(tf.keras.Model):
    """Implement Mobile RRN architecture.

    Mobile RRN 是 Revisiting Temporal Modeling (RRN) 的轻量版本，适用于移动设备的视频超分辨率任务。

    Attributes:
        scale: 一个 `int`，表示上采样率。
        base_channels: 一个 `int`，表示基础通道的数量。
    """

    def __init__(self,):
        """初始化 `MobileRRN` 模型。"""
        super().__init__()
        in_channels = 3  # 输入通道数（通常为 RGB 图像的 3 个通道）
        out_channels = 3  # 输出通道数（通常为 RGB 图像的 3 个通道）
        block_num = 5  # RNN 单元中的残差块数量

        self.base_channels = 16  # 基础通道数
        self.scale = 4  # 上采样倍数

        # 第一个卷积层，提取基础特征
        self.conv_first = tf.keras.layers.Conv2D(
            self.base_channels, kernel_size=3, strides=1, padding='SAME', activation='relu'
        )
        # 残差块序列，主要用于提取更深层次的特征
        self.recon_trunk = make_layer(
            ResidualBlock, block_num, base_channels=self.base_channels)

        # 最后的卷积层，用于生成高分辨率的输出
        self.conv_last = tf.keras.layers.Conv2D(
            self.scale * self.scale * out_channels, kernel_size=3, strides=1, padding='SAME'
        )
        # 隐藏状态的卷积层，用于更新隐藏状态
        self.conv_hidden = tf.keras.layers.Conv2D(
            self.base_channels, kernel_size=3, strides=1, padding='SAME', activation='relu'
        )

    def call(self, inputs, training=False):
        """前向传播。

        Args:
            inputs: 一个包含输入图像序列和隐藏状态的 `Tensor`。
            training: 一个布尔值，表示当前过程是训练还是测试。

        Returns:
            一个元组 `(out, hidden)`，其中 `out` 是输出 `Tensor`，`hidden` 是更新后的隐藏状态。
        """
        x, hidden = inputs  # 拆分输入和隐藏状态
        x1 = x[:, :, :, :3]  # 低分辨率输入图像
        x2 = x[:, :, :, 3:]  # 另一部分输入图像（可能是前一帧）
        _, h, w, _ = x1.shape.as_list()  # 获取图像的高度和宽度

        # 将低分辨率图像和隐藏状态拼接在一起
        x = tf.concat((x1, x2, hidden), axis=-1)
        out = self.conv_first(x)  # 第一个卷积层
        out = self.recon_trunk(out)  # 通过残差块序列
        hidden = self.conv_hidden(out)  # 更新隐藏状态
        out = self.conv_last(out)  # 最后的卷积层生成高分辨率输出

        # 上采样输出图像
        out = tf.nn.depth_to_space(out, self.scale)
        # 使用双线性插值将低分辨率图像调整到输出图像的尺寸
        bilinear = tf.image.resize(x2, size=(h * self.scale, w * self.scale))
        # 将高分辨率输出图像与双线性插值图像相加
        out = out + bilinear

        # 如果不是训练过程，限制输出的像素值在 [0, 255] 范围内
        if not training:
            out = tf.clip_by_value(out, 0, 255)

        return out, hidden  # 返回输出和更新后的隐藏状态


class ResidualBlock(tf.keras.Model):
    """残差块，用于 Mobile RRN 模型中。

    具有两个卷积层，采用残差连接。

    Attributes:
        base_channels: 一个 `int`，表示基础通道的数量。
    """

    def __init__(self, base_channels):
        """初始化 `ResidualBlock`。

        Args:
            base_channels: 一个 `int`，表示基础通道的数量。
        """
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            base_channels, kernel_size=3, strides=1, padding='SAME', activation='relu'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            base_channels, kernel_size=3, strides=1, padding='SAME')

    def call(self, x):
        """前向传播。

        Args:
            x: 输入 `Tensor`。

        Returns:
            一个输出 `Tensor`。
        """
        identity = x  # 保存输入作为残差
        out = self.conv1(x)  # 第一个卷积层
        out = self.conv2(out)  # 第二个卷积层
        return identity + out  # 残差连接


def make_layer(basic_block, block_num, **kwarg):
    """堆叠多个相同的块以形成网络层。

    Args:
        basic_block: 一个 `nn.module`，表示基本块。
        block_num: 一个 `int`，表示块的数量。

    Returns:
        一个 `nn.Sequential`，由堆叠的块组成。
    """
    model = tf.keras.Sequential()
    for _ in range(block_num):
        model.add(basic_block(**kwarg))
    return model
