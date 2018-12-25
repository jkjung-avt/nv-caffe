"""caffemodel_nv_to_bvlc.py

1. compile the nv-caffe's caffe.proto to caffe_pb2.py
2. use caffemodel_nv_to_bvlc.py to convert the nv-caffe caffemodel file
   into bvlc-caffe compatible format.

Example usage:
$ cd ${HOME}/project/nv-caffe
$ protoc src/caffe/proto/caffe.proto --python_out=.
$ python3 ./caffemodel_nv_to_bvlc.py <ini.caffemodel> <out.caffemodel>

Original code could be found at:
https://www.bountysource.com/issues/47592215-caffemodel-file-backward-compatibility-with-caffe-bvlc
"""


import array
import argparse

import src.caffe.proto.caffe_pb2 as pb


def caffe_nv_to_bvlc(nvidia_model_file, bvlc_model_file):

    param = pb.NetParameter()
    with open(nvidia_model_file, 'rb') as f:
        param.ParseFromString(f.read())

    for layer in param.layer:
        for blob in layer.blobs:
            if len(blob.raw_data) > 0 and blob.raw_data_type == pb.FLOAT:
                float_array = array.array('f')
                float_array.frombytes(blob.raw_data)
                blob.data.extend(float_array)
                blob.raw_data = bytes()

            if len(blob.raw_data) > 0 and blob.raw_data_type == pb.DOUBLE:
                double_array = array.array('d')
                double_array.frombytes(blob.raw_data)
                blob.double_data.extend(double_array)
                blob.raw_data = bytes()

            if len(blob.raw_diff) > 0 and blob.raw_diff_type == pb.FLOAT:
                float_array = array.array('f')
                float_array.frombytes(blob.raw_diff)
                blob.diff.extend(float_array)
                blob.raw_diff = bytes()

            if len(blob.raw_diff) > 0 and blob.raw_diff_type == pb.DOUBLE:
                double_array = array.array('d')
                double_array.frombytes(blob.raw_diff)
                blob.double_diff.extend(double_array)
                blob.raw_diff = bytes()

    with open(bvlc_model_file, 'wb') as f:
        f.write(param.SerializeToString())


def main():
    parser = argparse.ArgumentParser(
        description='Convert caffemodel from nv-caffe to bvlc format.')
    parser.add_argument(
        'nv_model',
        type=str,
        help='input file name (nv-caffe model)')
    parser.add_argument(
        'bvlc_model',
        type=str,
        help='output file name (bvlc-caffe model)')
    args = parser.parse_args()

    caffe_nv_to_bvlc(args.nv_model, args.bvlc_model)


if __name__ == '__main__':
    main()
