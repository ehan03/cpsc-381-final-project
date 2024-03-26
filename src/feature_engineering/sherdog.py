# standard library imports
from typing import Tuple

# third party imports
import pandas as pd

# local imports
from .base import BaseFeatureGenerator


class SherdogFeatureGenerator(BaseFeatureGenerator):
    """
    Class for generating features from Sherdog data
    """
