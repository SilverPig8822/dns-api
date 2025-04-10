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
MODEL_PATH = os.path.join(BASE_DIR, "model.keras")
model = load_model(MODEL_PATH)
model.summary()  # Optional: print the model summary

# Load the SentenceTransformer model.
st_model = SentenceTransformer("all-MiniLM-L6-v2")

# Set up DNS resolver using Cloudflare's DNS.
dns_resolver = dns.resolver.Resolver()
dns_resolver.nameservers = ["1.1.1.1"]

# Define the DNS record types to query.
FEATURES = {
    "PARTIAL": ["A", "MX", "CNAME", "NS", "TXT"],
    "FULL": dns.rdatatype.RdataType,
}

def dns_message_to_dict(domain):
    """
    Performs DNS queries for the given domain across various record types
    and returns a concatenated string of the results.
    """
    collected_data = ""
    # For each DNS record type defined in FEATURES["PARTIAL"]
    for qtype in FEATURES["PARTIAL"]:
        try:
            answer = dns_resolver.resolve(domain, qtype, lifetime=600)
            message = answer.response

            if qtype == "MX":
                message_dict = {
                    qtype: [
                        {
                            "name": str(rrset.name),
                            "ttl": rrset.ttl,
                            "email": [str(item.exchange) for item in rrset.items],
                        }
                        for rrset in message.answer
                    ]
                }
            elif qtype == "TXT":
                message_dict = {
                    qtype: [
                        {
                            "name": str(rrset.name),
                            "ttl": rrset.ttl,
                            "txt": " ".join([str(item.to_text().strip('"')) for item in rrset.items]),
                        }
                        for rrset in message.answer
                    ]
                }
            else:
                message_dict = {
                    qtype: [
                        {
                            "name": str(rrset.name),
                            "ttl": rrset.ttl,
                            "ip": (
                                [item.address for item in rrset.items]
                                if rrset.rdtype in [dns.rdatatype.A, dns.rdatatype.AAAA]
                                else []
                            ),
                        }
                        for rrset in message.answer
                    ]
                }
            # Convert the dictionary to a string and remove quotes as in the original code.
            record_str = json.dumps(message_dict, indent=None).replace('"', "").replace("'", "")
            collected_data += record_str
        except Exception as e:
            # In case of an error, return an “empty” record for that type.
            empty_message_dict = {
                qtype: [
                    {
                        "name": domain,
                        "ttl": 0
                    }
                ]
            }
            record_str = json.dumps(empty_message_dict, indent=None).replace('"', "").replace("'", "")
            collected_data += record_str
    return collected_data

@app.route("/")
def index():
    return "Flask API is up and running!"

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects a JSON POST request containing:
      { "domain": "example.com" }
    Returns the phishing probability based on DNS records of the input domain.
    """
    data = request.get_json()
    if not data or "domain" not in data:
        return jsonify({"error": "Please provide a 'domain' field in JSON"}), 400

    domain = data["domain"]
    
    # Get a string representation of the domain's DNS records.
    dns_data_str = dns_message_to_dict(domain)
    
    # Encode the DNS record string using the sentence transformer.
    encoded_vector = st_model.encode([dns_data_str])
    # Reshape to add the required channel dimension (e.g., shape becomes [batch_size, feature_dim, 1])
    encoded_vector = np.expand_dims(encoded_vector, axis=-1)
    
    # Use the CNN model to predict the probability.
    # The model outputs two probabilities: [non-phishing, phishing]
    prob = model.predict(encoded_vector).squeeze()
    phishing_probability = float(prob[1] * 100)

    # Return the result in JSON format.
    return jsonify({
        "domain": domain,
        "phishing_probability": phishing_probability
    })

if __name__ == "__main__":
    app.run(debug=True)
