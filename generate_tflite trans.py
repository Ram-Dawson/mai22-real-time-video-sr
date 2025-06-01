import argparse
import pathlib
import imageio.v2 as imageio # 使用 v2 避免 DeprecationWarning
import numpy as np
import tensorflow as tf

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
    interpreter = tf.lite.Interpreter(model_path=args.tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Model Input Details:")
    for detail in input_details:
        print(detail)
    print("\nModel Output Details:") # 这个日志已经确认了输出形状
    for detail in output_details:
        print(detail) # pred_tensor 'Identity' index 58, shape (1, 1280, 720, 3)
                      # hidden_state 'Identity_1' index 54, shape (1, 320, 180, 32)

    data_dir = pathlib.Path(args.data_dir)
    save_path = pathlib.Path(args.output_dir)
    save_path.mkdir(exist_ok=True)

    num_videos_to_process = 1
    num_frames_per_video = 5 # 您可以改回实际需要的数量

    for i in range(num_videos_to_process):
        # 初始化 hidden_state
        # 根据 Model Output Details, hidden_state ('Identity_1') 的形状是 (1, 320, 180, 32)
        # 这个形状已经和之前的日志输出 "Initial hidden_state shape for video 0: (1, 320, 180, 32)" 一致
        hidden_state_shape_expected = output_details[1]['shape'] # index 54, 'Identity_1'
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
                input_image_transposed = tf.transpose(input_image_orig, perm=[0, 2, 1, 3]) # (1, 320, 180, 3)
                input_tensor = tf.concat([input_image_transposed, input_image_transposed], axis=-1) # (1, 320, 180, 6)
                print(f"Input tensor shape for model: {input_tensor.shape}")
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

                input_image_1_transposed = tf.transpose(input_image_1_orig, perm=[0, 2, 1, 3])
                input_image_2_transposed = tf.transpose(input_image_2_orig, perm=[0, 2, 1, 3])
                input_tensor = tf.concat([input_image_1_transposed, input_image_2_transposed], axis=-1)
                print(f"Input tensor shape for model (loop): {input_tensor.shape}")
                # hidden_state 使用上一轮的输出, 形状应保持 (1, 320, 180, 32)
                print(f"Hidden state shape (before invoke, loop): {hidden_state.shape}")

            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.set_tensor(input_details[1]['index'], hidden_state)

            interpreter.invoke()

            # output_details[0] is 'Identity' (pred_tensor)
            # output_details[1] is 'Identity_1' (hidden_state for next step)
            pred_tensor = interpreter.get_tensor(output_details[0]['index'])
            hidden_state = interpreter.get_tensor(output_details[1]['index'])

            print(f"Output pred_tensor shape: {pred_tensor.shape}") # (1, 1280, 720, 3)
            print(f"Updated hidden_state shape: {hidden_state.shape}") # (1, 320, 180, 32)

            # pred_tensor 已经是 NumPy 数组，形状为 (1, 1280, 720, 3)
            # 保存时取第一张图 (batch size 为 1)
            output_image_to_save = pred_tensor[0] # Shape: (1280, 720, 3)
            # print(f"Shape of tensor intended for saving (output_image_to_save): {output_image_to_save.shape}") # 这行日志已在错误信息中出现

            try:
                # output_image_to_save 已经是 NumPy 数组，所以不需要 .numpy()
                image_to_save_uint8 = np.clip(output_image_to_save, 0, 255).astype(np.uint8)
                imageio.imwrite(save_path / f'{str(i).zfill(3)}_{str(j).zfill(8)}.png', image_to_save_uint8)
                print(f"Successfully saved image for frame {j}: {save_path / f'{str(i).zfill(3)}_{str(j).zfill(8)}.png'}")
            except Exception as e:
                print(f"Error saving image for frame {j}: {e}")
                if hasattr(output_image_to_save, 'shape'):
                    print(f"Shape of tensor intended for saving: {output_image_to_save.shape}")
                else:
                    print(f"output_image_to_save is not a shaped array. Type: {type(output_image_to_save)}")

if __name__ == '__main__':
    arguments = _parse_argument()
    print(f"Running with arguments: {arguments}")
    main(arguments)