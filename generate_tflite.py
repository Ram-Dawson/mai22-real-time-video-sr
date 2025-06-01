import argparse  # 用于解析命令行参数
import pathlib  # 用于处理文件路径
import imageio.v2 as imageio # <--- 使用 imageio.v2 避免警告
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

    print("Model Input Details:")
    for detail in input_details:
        print(detail)
    print("\nModel Output Details:")
    for detail in output_details:
        print(detail)

    # 准备数据集
    data_dir = pathlib.Path(args.data_dir)
    save_path = pathlib.Path(args.output_dir)
    save_path.mkdir(exist_ok=True)  # 创建输出目录（如果不存在）

    # 推理 (为了演示，减少循环次数)
    num_videos_to_process = 1 # 您可以改回 30
    num_frames_per_video = 5  # 您可以改回 100

    for i in range(num_videos_to_process):
        # 初始化 hidden_state，其形状应与模型第二个输入的期望形状匹配
        hidden_state_shape_expected = input_details[1]['shape'] # (1, 180, 320, 32)
        hidden_state = tf.zeros(hidden_state_shape_expected, dtype=np.float32)
        print(f"Initial hidden_state shape for video {i}: {hidden_state.shape}")

        for j in range(num_frames_per_video):
            print(f"\nProcessing video {i}, frame {j}")
            current_frame_path = data_dir / str(i).zfill(3) / f'{str(j).zfill(8)}.png'
            
            if not current_frame_path.exists():
                print(f"Frame {current_frame_path} not found, skipping.")
                continue

            if j == 0:
                try:
                    input_image_orig = np.expand_dims(
                        imageio.imread(current_frame_path), axis=0
                    ).astype(np.float32)
                except FileNotFoundError:
                    print(f"Error: Image file not found: {current_frame_path}")
                    continue
                
                print(f"Original input image shape: {input_image_orig.shape}") # (1, 180, 320, 3)
                # 新模型期望 180x320 输入, 不再需要转置
                input_tensor = tf.concat([input_image_orig, input_image_orig], axis=-1) # (1, 180, 320, 6)
                print(f"Input tensor shape for model: {input_tensor.shape}")
                # hidden_state 已在视频序列开始时初始化
                print(f"Hidden state shape (before invoke): {hidden_state.shape}")
            else: # j > 0
                prev_frame_path = data_dir / str(i).zfill(3) / f'{str(j-1).zfill(8)}.png'
                if not prev_frame_path.exists():
                    print(f"Previous frame {prev_frame_path} not found, skipping frame {j}.")
                    continue
                try:
                    input_image_1_orig = np.expand_dims(
                        imageio.imread(prev_frame_path), axis=0
                    ).astype(np.float32)
                    input_image_2_orig = np.expand_dims(
                        imageio.imread(current_frame_path), axis=0
                    ).astype(np.float32)
                except FileNotFoundError:
                    print(f"Error: Image file not found during processing frame {j}")
                    continue

                # 新模型期望 180x320 输入, 不再需要转置
                input_tensor = tf.concat([input_image_1_orig, input_image_2_orig], axis=-1) # (1, 180, 320, 6)
                print(f"Input tensor shape for model (loop): {input_tensor.shape}")
                print(f"Hidden state shape (before invoke, loop): {hidden_state.shape}")

            # 设置输入张量
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.set_tensor(input_details[1]['index'], hidden_state)

            # 执行推理
            interpreter.invoke()

            # 获取输出结果
            pred_tensor = interpreter.get_tensor(output_details[0]['index']) # shape (1, 720, 1280, 3), dtype float32
            hidden_state = interpreter.get_tensor(output_details[1]['index']) # shape (1, 180, 320, 32), dtype float32

            print(f"Output pred_tensor shape: {pred_tensor.shape}, dtype: {pred_tensor.dtype}")
            print(f"Updated hidden_state shape: {hidden_state.shape}, dtype: {hidden_state.dtype}")

            # 从批处理中提取图像 (pred_tensor 已经是 numpy 数组)
            output_image_float32 = pred_tensor[0] # Shape: (720, 1280, 3), dtype float32

            # ***** 修改点：确保在保存前将 float32 转换为 uint8 *****
            try:
                # 将 float32 像素值裁剪到 0-255 范围并转换为 uint8
                image_to_save_uint8 = np.clip(output_image_float32, 0, 255).astype(np.uint8)
                
                output_filename = save_path / f'{str(i).zfill(3)}_{str(j).zfill(8)}.png'
                imageio.imwrite(output_filename, image_to_save_uint8)
                print(f"Successfully saved image for frame {j}: {output_filename}")
            except Exception as e:
                print(f"Error saving image for frame {j}: {e}")
                if hasattr(output_image_float32, 'shape'):
                    print(f"Shape of tensor intended for saving: {output_image_float32.shape}, dtype: {output_image_float32.dtype}")
                else:
                    print(f"output_image_to_save is not a shaped array. Type: {type(output_image_float32)}")
                

if __name__ == '__main__':
    arguments = _parse_argument()
    print(f"Running with arguments: {arguments}")
    main(arguments)