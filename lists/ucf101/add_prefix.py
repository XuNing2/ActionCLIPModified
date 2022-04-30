import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument(
        '--src_name',
        type=str,
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='/home/cike/projects/mmaction/data/ucf101/rawframes/')
    # parser.add_argument('--out_name', type=str, default=parser.src)
    # parser.add_argument('--out_path', type=str, default='./out/')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    f = open(args.src_name + 'yes', "w+")

    with open(args.src_name) as raw:
        line = raw.readline()
        while line:
            print(line, end='')  # 在 Python 3中使用
            f.write("/home/cike/projects/mmaction/data/ucf101/rawframes/" +
                    line)
            line = raw.readline()

    f.close()
