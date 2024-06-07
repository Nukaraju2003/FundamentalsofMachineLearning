import nltk
from nltk.tokenize import word_tokenize 
from nltk import pos_tag, ne_chunk  
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# sample text  
sample_text = "Barack Obama was born in Mount Everest" 

# Tokenize the text 
tokens = nltk.word_tokenize(sample_text)

#part-of-speech tagging
postags = nltk.pos_tag(tokens)

#Named Entity Recognition (NER)
ner_tags = ne_chunk(postags)

print("Tokens:", tokens) 
print("Pos tags:", postags[0:6])
print("NER Tags:", ner_tags)