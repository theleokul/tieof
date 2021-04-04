import argparse

from dineof.model import Dineof
# from models import DINEOF3


def parse_args():
    parser = argparse.ArgumentParser(description='Dineof main entry.')
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    d = Dineof(args.config)
    d.fit()  # By default it keeps data only for summer
    d.predict()


if __name__ == '__main__':
    main()
