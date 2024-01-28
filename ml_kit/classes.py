import copy
import numpy as np
import re

import sys
from pathlib import Path

ml_kit_path = str((Path(__file__).absolute() / ".." / "..").resolve())
if ml_kit_path not in sys.path:
    sys.path.insert(0, ml_kit_path)

try:
    from ml_kit.standalone import STANDALONE
except ImportError:
    STANDALONE = False

if STANDALONE:
    from ml_kit.texts import purify


class classes_list:

    def __init__(self, classes_labels: list, safe_properties:bool=True):
        if not isinstance(classes_labels, (list, tuple)):
            raise ValueError("'values' should be a list!")
        if len(classes_labels) == 0:
            raise ValueError("No classes specified!")
        self.safe_properties = safe_properties
        self._classes_labels = [*classes_labels]
        self._classes = []
        for i in range(len(self._classes_labels)):
            self._classes.append((i, self._classes_labels[i]))

    @property
    def classes(self):
        if self.safe_properties:
            return copy.deepcopy(self._classes)
        else:
            return self._classes

    @property
    def classes_labels(self):
        if self.safe_properties:
            return copy.deepcopy(self._classes_labels)
        else:
            return self._classes_labels

    def __len__(self):
        return len(self._classes)


def idx_to_ohe(idx, total, weight=1.):
    result = np.zeros(total)
    result[idx] = weight
    return result


class distinct_classes:

    def __init__(
        self,
        classes_labels          : list,
        oov                     = None,
        fuzzy                   : bool = False,
        fuzzy_toward_infinity   : bool = False,
        normalize_weight        : bool = False,
        purify_data             : bool = True
    ):
        """
        Args:
            classes (dict | None): dict where key is class and values - all it's possible values
            values_map (dict | None): dict where key is one of class values, and value is a class
            rest : class for values that are not in map
            fuzzy : true to do fuzzy matching
            normalize_weight (bool, optional): when true then sum of mhe vector is always 1.0
            normalize_weight_per_value (bool, optional): when true then value for each class in mhe vector depends on amount of values for that class in input data
        """

        self._classes = []
        for k in sorted(classes_labels):
            if k in self._classes:
                raise ValueError(f"Class '{k}' is mentioned multiple times!")
            self._classes.append(k)
        if oov is not None:
            if oov not in self._classes:
                raise ValueError("'oov' should be one of classes!")
        self._oov = oov

        self.fuzzy = fuzzy
        self.fuzzy_toward_infinity=fuzzy_toward_infinity
        self.normalize_weight = normalize_weight
        if purify_data:
            self._purify = purify
        else:
            def same(value):
                return value
            self._purify = same

        self._ohe = {}
        self._idx = {}
        idx = 0
        for cl in self._classes:
            self._idx[cl] = idx
            self._ohe[cl] = idx_to_ohe(idx, len(self._classes))
            idx += 1

    def __len__(self):
        return len(self._classes)

    @property
    def classes_labels(self):
        return self._classes

    def _v(self, value):
        if value not in self._ohe:
            if self._oov is not None:
                value = self._oov
            else:
                raise ValueError(f"Value '{value}' doesn't belongs to classes!")
        return value

    def ohe(self, value):
        if not self.fuzzy:
            value = self._purify(value)
            return self._ohe[self._v(value)]
        else:
            return self.fuzzy_ohe(value)

    def mpe(self, values, normalize_weight=None):
        normalize_weight = normalize_weight or self.normalize_weight
        if not self.fuzzy:
            if isinstance(values, (list, tuple)):
                values = [self._purify(v) for v in values]
            else:
                values = self._purify(values)
            result = np.zeros(len(self._classes))
            if normalize_weight:
                weight = 1. / len(values)
            else:
                weight = 1.
            for value in values:
                value = self._v(value)
                result[self._idx[value]] = weight
            return result
        else:
            return self.fuzzy_mpe(values, normalize_weight=normalize_weight)

    def fuzzy_ohe(self, value : int|float, toward_infinity=None):
        value = self._purify(value)
        toward_infinity = toward_infinity or self.fuzzy_toward_infinity

        idx = 0
        while idx+1 < len(self._classes) and value >= self._classes[idx+1]:
            idx += 1
        if toward_infinity and value > self._classes[idx]:
            idx += 1
        if idx >= len(self._classes):
            idx = len(self._classes) - 1
        return idx_to_ohe(idx, len(self._classes))

    def fuzzy_mpe(self, values : list[int|float], toward_infinity=None, normalize_weight=None):
        toward_infinity = toward_infinity or self.fuzzy_toward_infinity
        normalize_weight = normalize_weight or self.normalize_weight

        if not isinstance(values, (list, tuple)):
            values = [values]
        values = [self._purify(v) for v in values]

        hits = []
        for value in values:
            idx = 0
            while idx < len(self._classes) and value >= self._classes(idx):
                idx += 1
            if toward_infinity and value > self._classes[idx]:
                idx += 1
            if idx >= len(self._classes):
                idx = len(self._classes) - 1
            if idx not in hits:
                hits.append(idx)
        if normalize_weight:
            weight = 1. / len(hits)
        else:
            weight = 1.
        result = np.zeros(len(self._classes))
        for idx in hits:
            result[idx] = weight
        return result


class multivalue_classes:

    def __init__(
        self,
        classes                     : dict|None = None,
        values_map                  : dict|None = None,
        rest                        = None,
        fuzzy                       : bool = False,
        normalize_weight            : bool = False,
        normalize_weight_per_value  : bool = False,
        purify_data                 : bool = True,
        ):
        """
        Args:
            Mandatory:
              classes (dict | None): dict where key is class and values - all it's possible values
              values_map (dict | None): dict where key is one of class values, and value is a class
            NOTE: specify one and only one - classes or values_map

            Optional:
              rest : class for values that are not in map
              fuzzy : set to True to do fuzzy matching. Otherwise when rest is not set then any out of map value will rise exception
              normalize_weight (bool, optional): when true then sum of mhe vector is always 1.0
              normalize_weight_per_value (bool, optional): when true then value for each class in mhe vector depends on amount of values for that class in input data
        """
        if classes is None and values_map is None:
            raise ValueError("Specify 'classes' OR 'values_map'!")
        if classes is not None and values_map is not None:
            raise ValueError("Specify ONLY 'classes' OR ONLY 'values_map'!")

        self.fuzzy = fuzzy
        self.normalize_weight = normalize_weight
        self.normalize_weight_per_value = normalize_weight_per_value
        if purify_data:
            self._purify = purify
        else:
            def same(value):
                return value
            self._purify = same

        self._class_to_clsid = {}
        self._value_to_clsid = {}
        self._classes = {}
        self._values_map = {}

        idx = 0
        if classes is not None:
            for cl, values in classes.items():
                self._classes[cl] = []
                self._class_to_clsid[cl] = idx
                for v in values:
                    v = self._purify(v)
                    if v in self._values_map:
                        raise ValueError(f"Value '{v}' is used twice or more (first occurrence is in class '{self._values_map[v]}', second - in '{cl}')!")
                    self._value_to_clsid[v] = idx
                    self._values_map[v] = cl
                    self._classes[cl].append(v)
                idx += 1
        else:
            for v, cl in values_map.items():
                v = self._purify(v)
                if cl not in self._class_to_clsid:
                    self._class_to_clsid[cl] = idx
                    self._classes[cl] = []
                    cl_id = idx
                    idx += 1
                else:
                    cl_id = self._class_to_clsid[cl]
                self._value_to_clsid[v] = cl_id
                self._values_map[v] = cl
                self._classes[cl].append(v)

        if rest is not None:
            if rest not in self._values_map:
                raise ValueError("'rest' should be one of values!")
        self._rest = rest

        self._class_ohe = {
            label : idx_to_ohe(v, len(self._class_to_clsid)) for label, v in self._class_to_clsid.items()
        }
        self._value_ohe = {
            value : self._class_ohe[cl] for value, cl in self._values_map.items()
        }

    def __len__(self):
        return len(self._classes)

    def __contains__(self, item):
        return item in self._values_map

    def _v(self, value):
        if value not in self._value_ohe:
            if self._rest is not None:
                value = self._rest
            else:
                raise ValueError(f"Value '{value}' doesn't belongs to permitted values set!")
        return value

    @property
    def classes(self):
        return copy.deepcopy(self._classes)

    @property
    def classes_labels(self):
        return self._classes.keys()

    @property
    def values(self):
        return self._values_map.keys()

    @property
    def values_map(self):
        return copy.deepcopy(self._values_map)

    def _seek(self, label, value):
        if isinstance(label, str):
            return re.search(r"\b"+label+r"\b", value) is not None
        else:
            return label in value

    def ohe(self, value):
        value = self._purify(value)
        if not self.fuzzy:
            return self._value_ohe[self._v(value)]
        else:
            for label in self._value_to_clsid.keys():
                if self._seek(label, value):
                    return self._value_ohe[label]
            if self._rest is not None:
                return self._value_ohe[self._rest]
            else:
                raise ValueError(f"Nothing in value '{value}' matches any value")

    def mpe(self, values, normalize_weight=None, normalize_weight_per_value=None):
        normalize_weight = normalize_weight or self.normalize_weight
        normalize_weight_per_value = normalize_weight_per_value or self.normalize_weight_per_value

        if not isinstance(values, (list, tuple)):
            values = [values]
        values = [self._purify(v) for v in values]

        result = np.zeros(len(self._classes))
        if not normalize_weight:
            weight = 1.
            if not self.fuzzy:
                for value in values:
                    value = self._v(value)
                    result[self._value_to_clsid[value]] = weight
            else:
                any_found = False
                for label, idx in self._value_to_clsid.items():
                    for value in values:
                        if self._seek(label, value):
                            result[idx] = weight
                            any_found = True
                if not any_found and self._rest is not None:
                    result[self._value_to_clsid[self._rest]] = 1.
        elif normalize_weight_per_value:
            if not self.fuzzy:
                weight = 1. / len(values)
                for value in values:
                    value = self._v(value)
                    result[self._value_to_clsid[value]] += weight
            else:
                ids = []
                any_found = False
                for label, idx in self._value_to_clsid.items():
                    for value in values:
                        if self._seek(label, value):
                            ids.append(idx)
                if not any_found and self._rest is not None:
                    result[self._value_to_clsid[self._rest]] = 1.
                else:
                    weight = 1. / len(ids)
                    for idx in ids:
                        result[idx] += weight
        else:
            hits = {}
            any_found = False
            if not self.fuzzy:
                any_found = True
                for value in values:
                    value = self._v(value)
                    hits[self._value_to_clsid] = 1
            else:
                for label, idx in self._value_to_clsid.items():
                    for value in values:
                        if self._seek(label, value):
                            hits[idx] = 1
                            any_found = True
            if not any_found and self._rest is not None:
                result[self._value_to_clsid[self._rest]] = 1.
            else:
                weight = 1. / len(hits)
                for idx in hits.values():
                    result[idx] = weight
        return result


# TODO: unit tests
