"""Generate testing output."""

import argparse  # 用于解析命令行参数的模块
import pathlib  # 用于处理文件路径的模块

import imageio  # 用于读取和写入图像的库
import numpy as np  # 用于科学计算的库
import tensorflow as tf  # 用于构建和运行机器学习模型的库

from util import plugin  # 从本地 util 模块中导入 plugin 功能


def _parse_argument():
    """Return arguments for conversion.

    返回命令行参数，用于模型路径、检查点路径、数据目录等的设置。
    """
    parser = argparse.ArgumentParser(
        description='Testing.')  # 初始化参数解析器，描述为“Testing”
    parser.add_argument(
        '--model_path', help='Path of model file.', type=str, required=True)
    # 添加参数 --model_path，指定模型文件路径，字符串类型，必填

    parser.add_argument(
        '--model_name', help='Name of model class.', type=str, required=True)
    # 添加参数 --model_name，指定模型类名，字符串类型，必填

    parser.add_argument(
        '--ckpt_path', help='Path of checkpoint.', type=str, required=True)
    # 添加参数 --ckpt_path，指定检查点文件的路径，字符串类型，必填

    parser.add_argument(
        '--data_dir', help='Directory of testing frames in REDS dataset.', type=str, required=True
    )
    # 添加参数 --data_dir，指定 REDS 数据集的测试帧目录，字符串类型，必填

    parser.add_argument(
        '--output_dir', help='Directory for saving output images.', type=str, required=True
    )
    # 添加参数 --output_dir，指定保存输出图像的目录，字符串类型，必填

    args = parser.parse_args()
    # 解析传入的命令行参数并返回

    return args


def main(args):
    """Run main function for converting keras model to tflite.

    主函数，用于加载模型、检查点并进行测试帧推理和输出。

    Args:
        args: A `dict` contain augments.
        参数 args：包含命令行传入的参数。
    """
    # prepare dataset 准备数据集
    data_dir = pathlib.Path(args.data_dir)  # 将数据目录转换为 pathlib 的 Path 对象，便于路径操作

    # prepare model 准备模型
    model_builder = plugin.plugin_from_file(
        args.model_path, args.model_name, tf.keras.Model)
    # 动态加载指定的模型文件和类，通过 plugin_from_file 函数构建模型

    model = model_builder()  # 实例化模型

    # load checkpoint 加载检查点
    ckpt = tf.train.Checkpoint(model=model)  # 创建检查点对象，将模型传入
    ckpt.restore(args.ckpt_path).expect_partial()  # 从指定的检查点路径恢复模型权重

    save_path = pathlib.Path(args.output_dir)  # 将输出目录转换为 pathlib 的 Path 对象
    save_path.mkdir(exist_ok=True)  # 如果输出目录不存在，则创建

    # testing 测试过程
    for i in range(30):  # 循环处理 30 组视频
        for j in range(100):  # 每组视频中处理 100 帧图像
            if j == 0:  # 如果是第一个帧
                input_image = np.expand_dims(
                    imageio.imread(data_dir / str(i).zfill(3) / f'{str(j).zfill(8)}.png'), axis=0
                ).astype(np.float32)
                # 读取帧图像，将其扩展为四维张量，并转换为 float32 类型

                b, h, w, _ = input_image.shape  # 获取输入图像的形状（批次大小、高度、宽度）

                input_tensor = tf.concat([input_image, input_image], axis=-1)
                # 将图像自身拼接两次，形成用于模型输入的张量

                hidden_state = tf.zeros([b, h, w, model.base_channels])
                # 初始化隐藏状态为全零，大小为批次、高度、宽度和模型的基础通道数

                pred_tensor, hidden_state = model(
                    [input_tensor, hidden_state], training=False)
                # 调用模型进行前向推理，返回预测结果和更新后的隐藏状态
            else:
                input_image_1 = np.expand_dims(
                    imageio.imread(data_dir / str(i).zfill(3) / f'{str(j-1).zfill(8)}.png'), axis=0
                ).astype(np.float32)
                # 读取上一帧图像，扩展维度，转换为 float32 类型

                input_image_2 = np.expand_dims(
                    imageio.imread(data_dir / str(i).zfill(3) / f'{str(j).zfill(8)}.png'), axis=0
                ).astype(np.float32)
                # 读取当前帧图像，扩展维度，转换为 float32 类型

                input_tensor = tf.concat(
                    [input_image_1, input_image_2], axis=-1)
                # 将两帧图像拼接在一起作为输入张量

                pred_tensor, hidden_state = model(
                    [input_tensor, hidden_state], training=False)
                # 使用前后帧图像进行模型推理，并更新隐藏状态

                # 将预测结果转换为 0-255 范围的 uint8 数据类型
                pred_image = (pred_tensor[0].numpy() * 255).astype(np.uint8)

                # 保存图像
                imageio.imwrite(save_path / f'{str(i).zfill(3)}_{str(j).zfill(8)}.png', pred_image)



if __name__ == '__main__':
    arguments = _parse_argument()  # 获取命令行参数
    main(arguments)  # 调用主函数
