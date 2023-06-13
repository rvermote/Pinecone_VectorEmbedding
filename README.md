This is the general backend setup for the educational query website.

## SENTENCE TRANSFORMERS

A sentence transformer (here: MPNet) is a machine learning model that is trained to embed a snippet of text into vector space as keys optimized for querying. It does this using a transformer model with a pre-trained encoder and a few additional tricks to obtain normalized vectors.

## PINECONE

Pinecone is a vector database specialized in high-performance vector search applications. The embedded queries and original text are upserted into the database and APIs are sent to the database whenever a client makes a query. The database then returns the most similar keys (original texts) in the database.
