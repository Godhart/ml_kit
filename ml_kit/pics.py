import numpy as np

def rgb_to_sparse_classes(
    image_list,
    classes_map,
):

    result = []

    # Для всех картинок в списке:
    for d in image_list:
        sample = np.array(d)
        # Создание пустой 1-канальной картики
        y = np.zeros((sample.shape[0], sample.shape[1], 1), dtype='uint8')

        # По всем классам:
        for k, v in classes_map.items():
            # Нахождение 3-х канальных пикселей классов и занесение метки класса
            if not isinstance(v, list):
                v = [v]
            for vv in v:
                y[np.where(np.all(sample == vv, axis=-1))] = k

        result.append(y)

    return result
