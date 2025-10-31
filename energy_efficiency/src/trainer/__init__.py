"""Trainer for various training techniques of machine learning models.

This module provides various objects that are designed to train machine 
learning models. Please refer to each repesctive classes to know how to use 
them. 

Notes
-----
    Both the `Trainer` and `DistributedTrainer` classes are abstract classes 
    defining the required methods. Both already implement a basic training loop 
    such that they can easily be extended by creating a child class that only 
    needs to implement the `step` function, which should implement one 
    iteration of training.

"""
from src.trainer.base import Trainer
from src.trainer.simple import SimpleTrainer
from typing import Self
import enum

@enum.unique
class Trainers(enum.Enum):
    SIMPLE_TRAINER = enum.auto()

    @classmethod
    def from_str(cls, name : str) -> Self:
        if name == "simple":
            return cls.SIMPLE_TRAINER
        raise Exception(f"Unknown trainer {name}")
