import csv
import os
import faiss
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.preprocessing import normalize


class MiimansaUtility:
    
    def __init__(self, index, metadata, direct_hit_model):
        self.index=index
        self.metadata=metadata
        self.model=direct_hit_model
        
    
    def is_direct_hit(self,query):
        """
        Checks z normalized cosine similarity between the query and questions in our qid2keyword

        Args:
            query (str): input query
            metadata (dict): Metadata containing embeddings
            direct_hit_model: SentenceTransformer model used to encode the query
            direct_hit_threshold (float): Threshold to determine if the query is a direct hit

        Returns:
            tuple: A tuple containing a boolean indicating if there's a direct hit and the QID of the direct hit if any
        """
        
        embedding_query = self.model.encode(query)
        mean_vector = self.metadata["mean_vector"]
        variance_vector = self.metadata["variance_vector"]
        #qid2emb = self.metadata["qid2emb"]
    
        #qids = list(qid2emb.keys())
        #embeddings = np.array([qid2emb[qid].squeeze() for qid in qids])
        # Normalize the query embedding
        embedding_query_normalized = (embedding_query - mean_vector) / np.sqrt(variance_vector)
        embedding_query_normalized = normalize(embedding_query_normalized.reshape(1, -1), axis=1)
        # Compute cosine similarities in batches with tqdm for progress tracking
        top_10_cosine_similarities={}
        k=10
        while(len(top_10_cosine_similarities)!=10):
            D, I = self.index.search(embedding_query_normalized, k) 
            #print(D,I)
            for t in range(len(I[0])):
                id = self.metadata["qid2id"][str(I[0][t])] 
                if id not in top_10_cosine_similarities.keys():
                    top_10_cosine_similarities[id]=D[0][t]
            k=k+1        
        
        return top_10_cosine_similarities
        #return False, None

    