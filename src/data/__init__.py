"""Thin wrapper around the datasets module.

This module is a very thing wrapper around the Hugging Face datasets module. It 
makes the translation between the config object and the datasets module, and it 
will make it easy to extend in the future if needs be.

"""
from src.data.data import load_data
import datasets
