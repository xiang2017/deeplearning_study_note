import os
from scipy import io as spio
from PIL import Image
import numpy as np


f = __file__
cat_fold = f[:f.rfind('/')] + '/cat_or_not'


def load_dataset():
    cats = []
    not_cats = []
    for name in os.listdir(cat_fold):
        if not name.endswith('g'):
            continue
        im = Image.open(os.path.join(cat_fold, name))
        arr = np.array(im)
        if name.startswith('c'):
            cats.append(arr)
        else:
            not_cats.append(arr)


    # 1/2 的数据作为测试集
    t_cats_count = int(len(cats) / 5 * 4)
    t_not_cats_count = int(len(not_cats) / 5 * 4)

    classes = ['not cat', 'cat']

    t_x_cats = cats[:t_cats_count]
    t_x_not_cats = not_cats[:t_not_cats_count]
    t_x = np.concatenate((t_x_cats, t_x_not_cats), axis=0)
    y1 = np.repeat(1, len(t_x_cats))
    y2 = np.repeat(0, len(t_x_not_cats))
    t_y = np.concatenate((y1, y2), axis=0)

    train_set_x_orig, train_set_y = shuffle_union(t_x, t_y)
    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))

    c_x_cats = cats[t_cats_count:]
    c_x_not_cats = not_cats[t_not_cats_count:]
    c_x = np.concatenate((c_x_cats, c_x_not_cats), axis=0)
    c_y1 = np.repeat(1, len(c_x_cats))
    c_y2 = np.repeat(0, len(c_x_not_cats))
    c_y = np.concatenate((c_y1, c_y2), axis=0)

    test_set_x_orig, test_set_y = shuffle_union(c_x, c_y)
    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))

    return train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes


def shuffle_union(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

if __name__ == '__main__':
    load_dataset()