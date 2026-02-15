from enum import Enum, auto

class Op(Enum):
    CREATE = auto(); POW = auto(); MEAN = auto(); SUM = auto();  EXP = auto()
    ADD = auto(); SUB = auto(); MUL = auto(); DOT = auto(); RESHAPE = auto()
    DIV = auto(); LOG = auto()
