import numpy as np
import tensorly as tl


def tensor2supervised(tensor):
    """Form traditional X (lats, lons, times) and Y (tensor values) datasets from the tensor with shape (lats, lons, times)"""
    shape = tensor.shape
    lats_range = np.arange(shape[0])
    lons_range = np.arange(shape[1])
    ts_range = np.arange(shape[2])

    rectified_lats = np.array([lat for _ in ts_range for lat in lats_range for _ in lons_range])
    rectified_lons = np.array([lon for _ in ts_range for _ in lats_range for lon in lons_range])
    rectified_ts = np.array([t for t in ts_range for _ in lats_range for _ in lons_range])
    tensor_vals = np.moveaxis(tensor, -1, 0).flatten()

    X = np.array([rectified_lats, rectified_lons, rectified_ts], dtype=tensor.dtype).T
    Y = np.array([tensor_vals], dtype=tensor.dtype).T

    return X, Y


def supervised2tensor(X, y, shape):
    tensor = np.full(shape, np.nan)

    for i, d in enumerate(X):
        lat, lon, t = d.astype(np.int)
        tensor[lat, lon, t] = y[i]

    return tensor


def center_mat(mat):
    nan_mask = np.isnan(mat)
    temp_mat = mat.copy()
    temp_mat[nan_mask] = 0

    m0 = temp_mat.mean(axis=0)
    for i in range(temp_mat.shape[0]):
        temp_mat[i, :] -= m0

    m1 = temp_mat.mean(axis=1)
    for i in range(temp_mat.shape[1]):
        temp_mat[:, i] -= m1

    temp_mat[nan_mask] = np.nan
    return temp_mat, m0, m1


def decenter_mat(mat, m0, m1):
    temp_mat = mat.copy()

    for i in range(temp_mat.shape[0]):
        temp_mat[i, :] += m0

    for i in range(temp_mat.shape[1]):
        temp_mat[:, i] += m1

    return temp_mat


def center_3d_tensor(tensor, lat_lon_separately=True):
    temp_tensor = tensor.copy()

    # Spatial centering
    spatial_means = []
    if lat_lon_separately:
        m0 = np.nanmean(temp_tensor, axis=0)
        m0[np.isnan(m0)] = 0
        for i in range(temp_tensor.shape[0]):
            temp_tensor[i, :, :] -= m0
        spatial_means.append(m0)

        m1 = np.nanmean(temp_tensor, axis=1)
        m1[np.isnan(m1)] = 0
        for i in range(temp_tensor.shape[1]):
            temp_tensor[:, i, :] -= m1
        spatial_means.append(m1)
    else:
        # For each day unified spatial mean (in flattened spatial array)
        for t in range(temp_tensor.shape[-1]):
            mat_mean = np.nanmean(temp_tensor[:, :, t])
            if np.isnan(mat_mean):
                mat_mean = 0
            temp_tensor[:, :, t] -= mat_mean
            spatial_means.append(mat_mean)

    # Time centering
    m2 = np.nanmean(temp_tensor, axis=2)
    m2[np.isnan(m2)] = 0
    for i in range(temp_tensor.shape[2]):
        temp_tensor[:, :, i] -= m2

    return temp_tensor, spatial_means, m2


def decenter_3d_tensor(tensor, spatial_means, m2, lat_lon_separately=True):
    temp_tensor = tensor.copy()

    # Spatial decentering
    if lat_lon_separately:
        m0, m1 = spatial_means
        for i in range(temp_tensor.shape[0]):
            temp_tensor[i, :, :] += m0
        for i in range(temp_tensor.shape[1]):
            temp_tensor[:, i, :] += m1
    else:
        for t in range(temp_tensor.shape[-1]):
            temp_tensor[:, :, t] += spatial_means[t]

    # Time decentering
    for i in range(temp_tensor.shape[2]):
        temp_tensor[:, :, i] += m2

    return temp_tensor


def rectify_3d_tensor(tensor):
    rect_mat = []
    for t in range(tensor.shape[-1]):
        rect_mat.append(tensor[:, :, t].flatten())
    rect_mat = np.array(rect_mat)
    rect_mat = np.moveaxis(rect_mat, 0, -1)
    return rect_mat


def unrectify_mat(mat, spatial_shape):
    tensor = []

    for t in range(mat.shape[-1]):
        col = mat[:, t]
        unrectified_col = col.reshape(spatial_shape)
        tensor.append(unrectified_col)

    tensor = np.array(tensor)
    tensor = np.moveaxis(tensor, 0, -1)

    return tensor


def calculate_mat_energy(mat, s):
    sample_count_0 = mat.shape[1]
    sample_coef_0 = 1 / (sample_count_0 - 1)
    total_energy_0 = np.array([sample_coef_0 * np.trace(mat @ mat.T) for _ in range(len(s))])
    expl_energy_0 = -np.sort(-sample_coef_0 * s * s)
    expl_energy_ratio_0 = expl_energy_0 / total_energy_0

    sample_count_1 = mat.shape[0]
    sample_coef_1 = 1 / (sample_count_1 - 1)
    total_energy_1 = np.array([sample_coef_1 * np.trace(mat.T @ mat) for _ in range(len(s))])
    expl_energy_1 = -np.sort(-sample_coef_1 * s * s)
    expl_energy_ratio_1 = expl_energy_1 / total_energy_1

    return np.array([[total_energy_0, expl_energy_0, expl_energy_ratio_0],
                        [total_energy_1, expl_energy_1, expl_energy_ratio_1]])


def calculate_3d_tucker_energy(tensor, A):
    energy_stack = []
    for i in range(tensor.ndim):
        sample_count_i = np.prod(tensor.shape) / tensor.shape[i]
        sample_coef_i = 1 / (sample_count_i - 1)
        unfold_i = tl.unfold(tensor, i)
        tensor_proj_i = tl.tenalg.mode_dot(tensor, A[i].T, i)
        tensor_proj_unfold_i = tl.unfold(tensor_proj_i, i)

        full_cov_i = sample_coef_i * unfold_i @ unfold_i.T
        tensor_proj_cov_i = sample_coef_i * tensor_proj_unfold_i @ tensor_proj_unfold_i.T

        total_energy_i = np.trace(full_cov_i)
        expl_energy_i_per_component = -np.sort(-tensor_proj_cov_i.diagonal())
        expl_energy_ratio_i_per_component = expl_energy_i_per_component / total_energy_i
        total_energy_i_per_component = [total_energy_i for _ in range(len(expl_energy_i_per_component))]

        energy_stack.append([total_energy_i_per_component,
                                expl_energy_i_per_component,
                                expl_energy_ratio_i_per_component])
    energy_stack = np.array(energy_stack)

    return energy_stack
