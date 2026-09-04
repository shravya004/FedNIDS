"""
===========================================================
Attack Configuration
===========================================================

Configuration for Member 2 model-poisoning experiments.
"""

# Client IDs that should behave maliciously.
# Example: client 1 and client 2 are malicious.
MALICIOUS_CLIENTS = {1}

# Enable / disable model poisoning.
MODEL_POISONING_ENABLED = True

# Strength of the model poisoning attack.
POISONING_SCALE_FACTOR = -3.0