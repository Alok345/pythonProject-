import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Sample sentence
sentence = "I live in New York and I visited California last summer."

# Process the sentence using spaCy
doc = nlp(sentence)

# Extract state names (GPE - Geopolitical Entity) from the sentence
state_names = []
for ent in doc.ents:
    if (
        ent.label_ == "GPE"
    ):  # Check if the entity is a geopolitical entity (e.g., state)
        state_names.append(ent.text)

# Print the recognized state names
print("Recognized state names:", state_names)
