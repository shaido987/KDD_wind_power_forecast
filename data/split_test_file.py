import os
import pandas as pd
import numpy as np

step_size = 10
save_path = './new_tests'
df_x = pd.read_csv("./test_x/0001in.csv")
df_y = pd.read_csv("./test_y/0001out.csv")


def strided_axis0(a, L):
    # Length of 3D output array along its axis=0
    nd0 = a.shape[0] - L + 1

    # Store shape and strides info
    m, n = a.shape
    s0, s1 = a.strides

    # Finally use strides to get the 3D array view
    return np.lib.stride_tricks.as_strided(a, shape=(nd0, L, n), strides=(s0, s0, s1))


if __name__ == '__main__':
    results = []
    for name, group in df_x.groupby('TurbID'):
        x = group.values
        x = strided_axis0(x, L=288)

        inputs = x[::step_size]
        outputs = x[288::step_size]
        n = outputs.shape[0]
        inputs = inputs[:n]

        results.append([inputs, outputs])

    results = np.transpose(np.array(results), axes=[2, 1, 0, 3, 4])
    results = results.reshape(n, 2, -1, 13)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, 'test_x'))
        os.mkdir(os.path.join(save_path, 'test_y'))

    for i in range(n):
        print('Saving', i)
        test_x = results[i][0]
        test_y = results[i][1]

        test_x = pd.DataFrame(test_x, columns=df_x.columns)
        test_y = pd.DataFrame(test_y, columns=df_y.columns)

        test_x.to_csv(os.path.join(save_path, 'test_x', f"{i:04}in.csv"), index=False)
        test_y.to_csv(os.path.join(save_path, 'test_y', f"{i:04}out.csv"), index=False)

    print('done')
