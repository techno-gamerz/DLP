# Practical 5 — Continuous Bag of Words (CBOW)

## Aim
To implement the Continuous Bag of Words (CBOW) word-embedding approach using a neural network and a small text corpus.

## 1. Theory
Natural Language Processing (NLP) involves computational methods for working with human language. Neural language models often represent words as numerical vectors called **word embeddings**. Embeddings can capture useful relationships between words when trained on sufficiently large and varied corpora.

### Word tokenization
A tokenizer maps words to integer IDs. For example, the sentence “deep learning helps” may become a sequence of integer tokens. The model operates on these integer IDs rather than directly on text strings.

### Word2Vec and CBOW
Word2Vec is a family of methods for learning word embeddings from context. One common architecture is **Continuous Bag of Words (CBOW)**. CBOW predicts a central target word using surrounding context words. The context is called a “bag” because a simple CBOW model combines the context embeddings without preserving their order.

For a sentence such as:
“deep learning makes computers intelligent”

and a context window of two words on each side, the model can use surrounding words to predict a central word when enough left and right context exists. The context contains four words: two before the target and two after it.

### Embedding layer
An embedding layer stores a trainable matrix with one vector per token ID. If the vocabulary size is \(V\) and the embedding dimension is \(d\), the matrix has approximately \(V \times d\) values. The embedding layer converts each token ID into its learned vector.

### Context aggregation and output
In this implementation, the embeddings for the four context words are averaged. The resulting vector is passed to a dense softmax output layer that predicts a probability distribution over the vocabulary. Training minimizes categorical cross-entropy between the predicted distribution and the correct target word.

### Limitations
The demonstration corpus is deliberately small, so its learned embeddings should not be treated as meaningful general-purpose language representations. Larger corpora, appropriate training settings and evaluation are needed for useful embeddings.

## 2. Code explanation
1. **Imports:** NumPy handles arrays, TensorFlow/Keras builds and trains the network, and `Tokenizer` maps words to integer IDs.
2. **Corpus:** Six short example sentences are defined.
3. **Tokenization:** `fit_on_texts` builds the vocabulary, and `texts_to_sequences` converts each sentence to token IDs. `word_index` maps words to IDs.
4. **Context-target generation:** For each valid target position, the code takes two token IDs to the left and two to the right, excluding the target itself. The target is the central token ID.
5. **One-hot target encoding:** `to_categorical` converts target IDs into one-hot vectors. This matches the categorical cross-entropy loss.
6. **Embedding layer:** The model learns a ten-dimensional vector for each vocabulary token.
7. **Mean aggregation:** `tf.reduce_mean(x, axis=1)` averages the context word vectors into one context representation.
8. **Softmax output:** A dense layer predicts the target token over the vocabulary.
9. **Training:** The model is trained on the context-target pairs.
10. **Prediction:** `argmax` selects the token with the largest predicted probability. The reverse dictionary converts the token ID back into a readable word.
11. **Embedding inspection:** The embedding matrix is extracted and the vector for the word “learning” is printed.

## 3. Algorithm
1. Import the required libraries.
2. Define a small text corpus.
3. Tokenize the corpus and construct a word-to-index dictionary.
4. Select a context-window size.
5. Generate context-target training pairs by using surrounding words as input and the central word as the target.
6. Convert target IDs to one-hot vectors.
7. Build a CBOW model using an embedding layer, context-vector aggregation and a softmax output layer.
8. Compile the model with Adam and categorical cross-entropy.
9. Train the model on the generated pairs.
10. Predict a target word from a sample context.
11. Inspect a learned word embedding and plot the training loss.

## 4. Result
A CBOW model is constructed and trained on a sample corpus. Context-target pairs are generated, a target word is predicted from surrounding words, and the learned embedding for a selected word is displayed.

## 5. Viva questions
1. **What is NLP?** The field of computing concerned with processing and understanding human language.
2. **What is a word embedding?** A learned numerical vector representation of a word.
3. **What is CBOW?** A Word2Vec approach that predicts a target word from surrounding context words.
4. **What is the context window?** The number of neighbouring tokens used to predict the target.
5. **Why tokenize text?** Neural networks operate on numerical inputs, so words are mapped to token IDs.
6. **What is a vocabulary?** The set of distinct tokens recognized by the tokenizer.
7. **What does an Embedding layer do?** It maps token IDs to trainable dense vectors.
8. **Why average the context embeddings?** To create a single fixed-size context representation from multiple word vectors.
9. **Why use softmax?** It produces a probability distribution over possible target words.
10. **What is a limitation of this notebook?** The tiny corpus is only for demonstration and cannot produce robust general-purpose embeddings.
