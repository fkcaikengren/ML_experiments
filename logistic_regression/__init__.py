"""
    __init__.py运行时间：当package被import时。
    此外，该文件表明这个文件夹是一个package
"""

from .logistic import LogisticRegression


__all__ = ["LogisticRegression"]