import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from models import DINEOF, DINEOF3


def parse_args():
    parser = argparse.ArgumentParser(description='DINEOF3 main entry.')
    parser.add_argument('tensor', type=str, help='Path to numpy representation of a tensor to reconstruct')
    parser.add_argument('-O', '--out', type=str, help='Save path for the reconstruction', required=True)
    parser.add_argument('-m', '--mask', type=str, help='Path to numpy representation of a mask', default='/mss3/baikal/kulikov/modis_aqua/2003/Input/static_grid/mask.npy')
    parser.add_argument('-R', '--rank', type=int, help='Rank to use in the decomposition algorithm', default=5)
    parser.add_argument('-L', '--length', type=int, help='Validation length', default=1)
    parser.add_argument('--tensor-shape', nargs=3, type=int, help='Tensor shape', default=[482, 406, 93])
    parser.add_argument('--decomposition-method', type=str, help='truncSVD, truncHOSVD, HOOI or PARAFAC', default='PARAFAC')
    parser.add_argument('--nitemax', type=int, default=100)
    parser.add_argument('--refit', action='store_true')
    parser.add_argument('--lat-lon-sep-centering', action='store_true')
    parser.add_argument('--random-seed', type=int, default=2434311)
    parser.add_argument('--val-size', type=float, default=0.045)
    args = parser.parse_args()
    return args


def get_model(args, R):
    if args.decomposition_method.lower() == 'truncsvd':
        d = DINEOF(
            R=R
            , tensor_shape=args.tensor_shape
            , nitemax=args.nitemax
            , mask=args.mask
        )
    else:
        d = DINEOF3(
            R=R
            , tensor_shape=args.tensor_shape
            , decomp_type=args.decomposition_method
            , nitemax=args.nitemax
            , lat_lon_sep_centering=args.lat_lon_sep_centering
            , mask=args.mask
        )
    
    return d


def main():
    args = parse_args()

    mask = np.load(args.mask).astype(bool)
    tensor = np.load(args.tensor)
    tensor[~mask] = np.nan
    X = np.asarray(np.nonzero(~np.isnan(tensor))).T
    y = tensor[tuple(X.T)]

    np.random.seed(args.random_seed)
    biner = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='kmeans')
    stratify_y = biner.fit_transform(y[:, None]).flatten().astype(int)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_size, random_state=args.random_seed, stratify=stratify_y)
    print(f'Missing ratio: {(mask.sum() * args.tensor_shape[-1] - y.shape[0]) / (mask.sum() * args.tensor_shape[-1])}')
    print(f'Train points: {y_train.shape[0]}, val points: {y_val.shape[0]}')

    val_errors = []
    Rs = list(range(args.rank, args.rank + args.length))
    for R in Rs:
        d = get_model(args, R)
        d.fit(X_train, y_train)
        val_errors.append(-d.score(X_val, y_val) * y_val.std())
        print(f'Validation error: {val_errors[-1]}')

    best_R = Rs[np.argmin(val_errors)]
    print(f'Best rank: {best_R}')

    if args.refit:
        d = get_model(args, best_R)
        d._fit(np.load(args.tensor))

    np.save(args.out, d.reconstructed_tensor)
    

if __name__ == '__main__':
    main()
