# lastmodel
only convo trained
from transformers import AutoTokenizer, AutoModel

# Choose a pre-trained model and tokenizer from Hugging Face's model hub
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Example input text
input_text = "Hello, how are you?"

# Tokenize input text
tokens = tokenizer(input_text, return_tensors="pt")

# Forward pass through the model
outputs = model(**tokens)

# Access the model's output (e.g., hidden states)
hidden_states = outputs.last_hidden_state
