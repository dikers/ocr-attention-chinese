import argparse
import os
import sys
sys.path.append('../')
rootPath = os.path.dirname(sys.path[0])
import errno
import numpy as np
import glob
import tensorflow as tf
import cv2
import time
import sys
import os
import PIL.Image as Image
from create_data import show_process


def parse_arguments():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(
        description="输入的文件"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        nargs="?",
        help="The output directory",
        required=True
    )
    parser.add_argument(
        "-i",
        "--input_data_path",
        type=str,
        nargs="?",
        help="Data file path",
        required=True
    )
    parser.add_argument(
        "-d",
        "--dict_file",
        type=str,
        nargs="?",
        help="输入dict",
        required=True
    )
    parser.add_argument(
        "-t",
        "--tfrecord_type",
        type=str,
        nargs="?",
        default='train',
        help="输入tfrecord 类型  train valid test"
    )
    return parser.parse_args()




def encode_utf8_string(text, length, dic, null_char_id=5462):
    char_ids_padded = [null_char_id]*length
    char_ids_unpadded = [null_char_id]*len(text)
    for i in range(len(text)):
        hash_id = dic[text[i]]
        char_ids_padded[i] = hash_id
        char_ids_unpadded[i] = hash_id
    return char_ids_padded, char_ids_unpadded

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))



def _generate_tfrecord(dict_file, input_data_path, output_dir, tfrecord_type):
    dict={}

    null_char_id = -1
    with open(dict_file, encoding="utf") as dict_file:
        for line in dict_file:
            (key, value) = line.strip().split('\t')
            dict[value] = int(key)
            if value == '<>':
                null_char_id = key
    dict_length = len(dict)
    print('[Message]   dict length: {}   null_char_id: {}'.format(dict_length, null_char_id))

    image_path = os.path.join(output_dir, 'data/'+tfrecord_type+'/*.jpg')
    addrs_image = glob.glob(image_path)

    label_path = os.path.join(output_dir, 'data/'+tfrecord_type+'/*.txt')
    addrs_label = glob.glob(label_path)

    print(len(addrs_image))
    print(len(addrs_label))

    output_dir = os.path.join(output_dir, 'tfrecords')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    tfrecord_file = os.path.join(output_dir, tfrecord_type+"_tfrecord")
    tfrecord_writer = tf.io.TFRecordWriter(tfrecord_file)

    process_bar = show_process.ShowProcess(len(addrs_image))
    print('写入Tfrecord 记录 共计{}条 路径 {} '.format(len(addrs_image), tfrecord_file))
    for j in range(0, int(len(addrs_image))):

        # 这是写入操作可视化处理
        img = Image.open(addrs_image[j])

        img = img.resize((600, 150), Image.ANTIALIAS)
        np_data = np.array(img)
        image_data = img.tobytes()
        for text in open(addrs_label[j], encoding="utf"):
            char_ids_padded, char_ids_unpadded = encode_utf8_string(
                text=text,
                dic=dict,
                length=dict_length,
                null_char_id=null_char_id)

            process_bar.show_process()
            time.sleep(0.05)

        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image/encoded': _bytes_feature(image_data),
                'image/format': _bytes_feature(b"raw"),
                'image/width': _int64_feature([np_data.shape[1]]),
                'image/orig_width': _int64_feature([np_data.shape[1]]),
                'image/class': _int64_feature(char_ids_padded),
                'image/unpadded_class': _int64_feature(char_ids_unpadded),
                'image/text': _bytes_feature(bytes(text, 'utf-8')),
                # 'height': _int64_feature([crop_data.shape[0]]),
            }
        ))
        tfrecord_writer.write(example.SerializeToString())
    tfrecord_writer.close()
    sys.stdout.flush()
    process_bar.close()

def main():
    time_start = time.time()
    # Argument parsing
    args = parse_arguments()
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    _generate_tfrecord(args.dict_file, args.input_data_path,
                       args.output_dir, args.tfrecord_type)

    time_elapsed = time.time() - time_start
    print('The code run {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":
    main()





