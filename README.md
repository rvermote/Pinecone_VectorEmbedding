This is the general backend setup for the educational query website.

## OBTAINING VIDEO FILES

Video files are scraped from Youtube using yt-dlp. This allows for easy scraping of large batches of videos and playlists.

## TEXT-TO-SPEECH: GETTING VIDEO TRANSCRIPTS

Video transcripts (Text-To-Speech) is done usin OpenAI Whisper, an open-source general-purpose speech recognition model (https://github.com/openai/whisper).

## TEXT EMBEDDING: SENTENCE TRANSFORMERS

A sentence transformer (here: MPNet) is a machine learning model that is trained to embed a snippet (128 symbols) of text into (768 dimensional) vector space. This vector represents a key optimized for later querying. It does this using a transformer model with a pre-trained encoder and a few additional tricks to obtain normalized vectors.

## PINECONE VECTOR DB

Pinecone is a vector database specialized in high-performance vector search applications. The embedded queries and original text are upserted into the database. When a query is made, the database then returns the most similar keys (original texts) in the database.

## CONNECTING TO FRONTEND

A query could be made from the backend (in the python file itself), but a frontend application is additionally build for easy user experience which uses an API to connect to pinecone and communicate the query and returing best matches (project QuickPhysics)
