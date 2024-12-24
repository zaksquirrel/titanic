
import pandas as pd  
import numpy as np

import matplotlib.pyplot as plt  
import seaborn as sns  

from sklearn.model_selection import train_test_split  # For splitting datasets
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For encoding and scaling
from sklearn.impute import SimpleImputer  # For handling missing data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # For evaluation

import tensorflow as tf  # Core TensorFlow
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense, Dropout  
from tensorflow.keras.optimizers import Adam  
from tensorflow.keras.callbacks import EarlyStopping  

import os  