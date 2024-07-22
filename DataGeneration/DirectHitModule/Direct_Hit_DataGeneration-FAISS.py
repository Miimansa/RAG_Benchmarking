import joblib
#from retriever import MiimansaClinicalTextRetriever
from sentence_transformers import SentenceTransformer
import numpy as np
from ragatouille import RAGPretrainedModel
import torch
from utilities_Faiss import MiimansaUtility
import joblib
from tqdm import tqdm
import json
from sklearn.preprocessing import normalize
import faiss

DB_PATH = "question_bank.csv"
METADATA_PATH = "./output/metadata.pkl"
VECTOR_DB_PATH = "./output"
#DIRECT_HIT_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
DIRECT_HIT_MODEL="./models/fine_tune_direct_hit_questions/checkpoint-250"

metadata = joblib.load(METADATA_PATH)
print('MetaData loaded')
direct_hit_model = SentenceTransformer(DIRECT_HIT_MODEL)



def load_data_fiass():
    mean_vector = metadata["mean_vector"]
    variance_vector = metadata["variance_vector"]
    qid2emb = metadata["qid2emb"]
    qid2id = metadata["qid2id"]

    # Prepare and normalize the embeddings
    qids = list(qid2emb.keys())
    embeddings = np.array([qid2emb[qid].squeeze() for qid in qids])
    embeddings_normalized = (embeddings - mean_vector) / np.sqrt(variance_vector)
    index_f = faiss.IndexFlatIP(1024)
    res = faiss.StandardGpuResources()  # Initialize GPU resources
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_f)
    embed=normalize(embeddings_normalized,axis=1)
    gpu_index.add(embed)
    return gpu_index

gpu_index=load_data_fiass()

print('Data Indexed on FIASS')

utility=MiimansaUtility(gpu_index,metadata,direct_hit_model)

print('Utility Loaded')
# RAG = MiimansaClinicalTextRetriever.from_index(
#     "output/colbert/indexes/Colbert-Experimental"
# )
# retriever = RAG.as_langchain_retriever(
#     metadata=metadata,
#     direct_hit_model=direct_hit_model,
#     direct_hit_threshold=0.91,
#     log_direct_hit=False,
#     log_dir="./logs",
#     k=5,
# )


def load_queries(path):
 with open (path,'r') as f:
    data=json.load(f)
 queries=[]
 for temp in data:
     for qa in temp['qas']:
         for key,value in qa.items():
             if(key=='question'):
               queries.append(value)
 return queries   

def load_queries_rephrased(path):
 with open (path,'r') as f:
    data=json.load(f)
 queries=[]
 #grouped_queries=[]
 for temp in data:
    questions=temp['rephrased']
    
    for ques in questions:
      queries.append(ques)
 return queries            

def get_relevant_documents(query):
    return utility.is_direct_hit(query)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def main():
    # DataGeneration for the llama qac triples # DataGeneration for the llama qac triples
    #path='output_all_qac_cleaned.json' 
    #queries=load_queries(path)
    #joblib.dump(queries, './all_queries.pkl')


    path='rephrased_questions_cleaned_with_context.json'
    queries=load_queries_rephrased(path)
    #joblib.dump(queries, './all_queries_rephrased.pkl')
    print(f'{len(queries)} Queries loaded')
    #print(queries[0:10])
    #load_data_fiass()
    
    results = {}
    query_i=0
    for query in tqdm(queries,total=len(queries),desc="Processing queries"):
        try:
            result = get_relevant_documents(query)
            #results.append((query, result))
            results[query_i]=result
            query_i=query_i+1
        except Exception as e:
            print(str(e))
            break
            #results.append((query, str(e)))
    print(results)
    with open('direct_hit_rephrased_results_finetuned.json', 'w') as file:
       json.dump(results, file,cls=CustomJSONEncoder)
    # Print the results
    # for query, result in results:
    #     print(f"Query: {query}")
    
    #     print(f"Result: {result}")
    #     print("\n")

if __name__ == "__main__":
    main()        
