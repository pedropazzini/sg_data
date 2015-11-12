from enum import Enum, unique

@unique
class Normalization(Enum):
    none = 1
    by_max = 2
    by_max_min = 3
    by_Z_normalization = 4
