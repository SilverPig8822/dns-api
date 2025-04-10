from flask import Flask, request, jsonify
import json
import numpy as np
import dns.resolver
import dns.rdatatype
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)

# Load the CNN phishing detection model.
# Make sure that 'new_st2_model.h5' is in the same directory or update the path accordingly.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "new_st2_model.h5")
model = load_model(MODEL_PATH)
model.summary()  # Optional: print the model summary

model.save('model.keras')