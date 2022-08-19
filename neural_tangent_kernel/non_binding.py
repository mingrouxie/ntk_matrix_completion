from enum import Enum

class NonBinding(Enum):
    ROW_MEAN = 1
    SMALL_POS = 2
    LARGE_POS = 3
    MAX_PLUS = 4

def fill_non_bind(mat, nb_type):
    if nb_type == NonBinding.ROW_MEAN:
        return mat.apply(lambda row: row.fillna(row.mean()), axis=1)
    elif nb_type == NonBinding.SMALL_POS:
        return mat.apply(lambda row: row.fillna(1e-5), axis=1)
    elif nb_type == NonBinding.LARGE_POS:
        # return mat.apply(lambda row: row.fillna(5), axis=1)
        return mat.apply(lambda row: row.fillna(10), axis=1)
        # return ground_tmatruth.apply(lambda row: row.fillna(30), axis=1)
    elif nb_type == NonBinding.MAX_PLUS:
        return mat.apply(lambda row: row.fillna(row.max() * 1.01), axis=1)
    else:
        raise Exception("Non-binding treatment unrecognised:", nb_type)