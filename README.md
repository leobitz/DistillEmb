# What?
A method to distill or compress word embeddings into a four-layer CNN 

# Why?
Most languages, especially African ones are under-resourced - either low dataset or low compute. 
Hence word embeddings are still relevant. However, word embeddings have many limitations:

- Linear memory requirement with respect to the number of tokens
- Out-of-vocabulary issues
- Hardly transferrable.
- These problems become even more challenging in morphologically complex languages

# Solution?

Compressing the entire embedding list into a fixed neural network takes little memory. Furthermore, as neural networks are good approximators, the CNN network can generate embeddings for unseen tokens. Finally, we can use the network with a similar language and benefit from the out-of-the-box cross-lingual transferability.

# How?

Use contrastive learning as a distillation loss. The anchor is what the CNN generates. The positive embedding is our target token. The negative one was extracted by picking the most similar embedding to the positive embedding out of a randomly sliced 32-length sequence.

The CNN network is fed the characters and from the embeddings of these characters, CNN learns word features.

# Evaluation?

We tested the method on a variety of tasks and languages. Amharic, Tigrigna, 10 African languages and a cross-linguality test were made. Overall, it achieved a 7% accuracy over the previous method and beat a 97M parameter model despite being less than 3M. Please read the [full report](https://leobitz.github.io/files/distill-emb.pdf)


