import argparse  # 用于解析命令行参数
import pathlib  # 用于处理文件路径
import imageio  # 用于读取和写入图像
import numpy as np  # 用于科学计算
import tensorflow as tf  # 用于TensorFlow Lite Interpreter

def _parse_argument():
    """返回推理所需的参数."""
    parser = argparse.ArgumentParser(description='TFLite Model Inference.')
    parser.add_argument('--tflite_model', help='Path of TFLite model file.', type=str, required=True)
    parser.add_argument('--data_dir', help='Directory of testing frames in REDS dataset.', type=str, required=True)
    parser.add_argument('--output_dir', help='Directory for saving output images.', type=str, required=True)

    args = parser.parse_args()
    return args

def main(args):
    """执行TFLite模型推理的主函数.

    Args:
        args: 包含参数的字典.
    """
    # 加载TFLite模型
    interpreter = tf.lite.Interpreter(model_path=args.tflite_model)
    interpreter.allocate_tensors()  # 准备模型进行执行

    # 获取输入和输出张量
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 准备数据集
    data_dir = pathlib.Path(args.data_dir)
    save_path = pathlib.Path(args.output_dir)
    save_path.mkdir(exist_ok=True)  # 创建输出目录（如果不存在）

    # 推理
    for i in range(30):  # 处理30个视频序列
        for j in range(100):  # 处理每个视频的100帧
            # 构建输入张量
            if j == 0:
                # 读取第一帧
                input_image = np.expand_dims(
                    imageio.imread(data_dir / str(i).zfill(3) / f'{str(j).zfill(8)}.png'), axis=0
                ).astype(np.float32)

                b, h, w, _ = input_image.shape
                print(input_image.shape)
                
                # 将第一帧与自身拼接
                # input_tensor = np.concatenate([input_image, input_image], axis=-1)
                input_tensor = tf.concat([input_image, input_image], axis=-1)
                print(f"Input tensor shape: {input_tensor.shape}")
                # 初始化隐藏状态为全零
                hidden_state = tf.zeros([b, h, w, 32])
                # 初始化隐藏状态为全零，大小为批次、高度、宽度和模型的基础通道数
                print(f"Hidden state shape: {hidden_state.shape}")
                #输入张量与隐藏状态
                interpreter.set_tensor(input_details[0]['index'], input_tensor)
                interpreter.set_tensor(input_details[1]['index'], hidden_state) 
                
                #执行推理
                interpreter.invoke()
                
                #获取隐藏状态与预测结果
                pred_tensor, hidden_state = interpreter.get_tensor(output_details[0]['index']), interpreter.get_tensor(output_details[1]['index'])
                
                # 保存预测的图像
                pred_image = np.clip(pred_tensor[0], 0, 255).astype(np.uint8)
                imageio.imwrite(save_path / f'{str(i).zfill(3)}_{str(j).zfill(8)}.png', pred_image)

            else:
                # 读取连续的两帧
                input_image_1 = np.expand_dims(
                    imageio.imread(data_dir / str(i).zfill(3) / f'{str(j-1).zfill(8)}.png'), axis=0
                ).astype(np.float32)

                input_image_2 = np.expand_dims(
                    imageio.imread(data_dir / str(i).zfill(3) / f'{str(j).zfill(8)}.png'), axis=0
                ).astype(np.float32)

                # 将两帧拼接
                input_tensor = tf.concat([input_image_1, input_image_2], axis=-1)
                # 将两帧图像拼接在一起作为输入张量

                # 使用上一次推理的隐藏状态
                interpreter.set_tensor(input_details[0]['index'], input_tensor)  # input_1
                interpreter.set_tensor(input_details[1]['index'], hidden_state)  # input_2

                # 执行推理
                interpreter.invoke()

                # 获取输出结果
                pred_tensor = interpreter.get_tensor(output_details[0]['index'])  # Identity
                hidden_state = interpreter.get_tensor(output_details[1]['index'])  # Identity_1

                imageio.imwrite(save_path / f'{str(i).zfill(3)}_{str(j).zfill(8)}.png', pred_tensor[0])
                

if __name__ == '__main__':
    arguments = _parse_argument()
    main(arguments)
