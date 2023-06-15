#This is the main processing file. For a more explanatory but less efficient version based on one video, see processVideosExplained.ipynb

import whisper
import os
import numpy as np
import itertools
import json
from pathlib import Path
from dotenv import load_dotenv
import pinecone
from collections import defaultdict
from sentence_transformers import SentenceTransformer

load_dotenv()

#Setting constants, definitions and loading in models and data
videoFolder = Path(os.path.dirname(os.path.abspath('__file__'))).parent.parent / "PhysicsVideos"
model = whisper.load_model("small")

retriever = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base')
embed_dim = retriever.get_sentence_embedding_dimension()

pinecone.init(api_key=os.getenv("KEY"),
                environment=os.getenv("ENV"))
index = pinecone.Index("youtube-search")

video_meta=defaultdict(dict)
with open("video_meta.txt","r") as f:
    for line in f:
        data = line.strip().split("\\")
        video_meta[data[0]] = {"title":data[1],"url":data[2]}

BATCH_SIZE = 64
SENTENCE_TRANSFORMER_INPUT_TOKEN_LENGTH=128
NUM_OF_VIDEOS=136

#main loop for all videos

for i in range(1,NUM_OF_VIDEOS+1):

    vidIndex = str(i).zfill(3)
    print("Started processing video "+vidIndex)

    result = model.transcribe(whisper.load_audio(videoFolder / (vidIndex+".webm")))

    print("Finished Speech-To-Text video "+str(i))

    filename = "whisper_output.json"
    with open(filename, 'w') as f:
        json.dump(result, f, indent=4)

    segments = result["segments"]
    segInfo = [{"start": int(segment['start']),"end": int(segment['end']), "text": segment['text'].split()} for segment in segments]
    segInfo = segInfo[2:-2]

    #interpolate start-end time per word
    for idx,dict in enumerate(segInfo):
        segInfo[idx]["timeStamp"] = list(np.linspace(dict["start"],dict["end"],len(dict["text"]),dtype=int))

    #convert to single list of words and timestamps
    segInfo = list(map(lambda x:[x["text"],x["timeStamp"]],segInfo))
    segInfo = list(zip(*segInfo))
    segInfo = list(map(lambda x:list(itertools.chain.from_iterable(x)),segInfo))

    def endIdx(startIdx:int) -> int:
        return min(startIdx + SENTENCE_TRANSFORMER_INPUT_TOKEN_LENGTH,len(segInfo[0]))

    def wordsToToken(idx:int) -> str:
        return " ".join(segInfo[0][idx:endIdx(idx)])

    tokenInput = [{"text": wordsToToken(i), "start": segInfo[1][i], "end": segInfo[1][endIdx(i)-1]} for i in range(0, len(segInfo[0]), SENTENCE_TRANSFORMER_INPUT_TOKEN_LENGTH)]

    Total_package,filename = [],"pinecone_upserts.json"

    #process single list of words and timestamps into batches of token inputs
    for idx in range(0,len(tokenInput),BATCH_SIZE):
        batch = tokenInput[idx:min(idx+BATCH_SIZE,len(tokenInput))]

        ids = list(map(lambda x:str(vidIndex)+"-"+str(idx)+"-"+str(x["start"]), batch))
        #get embeddings using the sentence transformer
        embeddings = retriever.encode(list(map(lambda x:x["text"],batch))).tolist()
        metadata = list(map(lambda x:{"text":x["text"],"title":video_meta[vidIndex]["title"],"url":video_meta[vidIndex]["url"],"start":int(x["start"]),"end":int(x["end"])},batch))

        package = list(zip(ids,embeddings,metadata))
        package = list(map(lambda x:{'id':x[0],'values':x[1],'metadata':x[2]},package))

        #insert embeddings together with metadata into pinecone index
        index.upsert(vectors=package)
        Total_package.append(package)
    
    print("Succesfully upserted video "+vidIndex+" into pinecone index")

    with open(filename, 'w') as f:
        json.dump(Total_package, f, indent=4)

    index.describe_index_stats()