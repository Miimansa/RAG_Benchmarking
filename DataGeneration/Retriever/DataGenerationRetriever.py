import joblib
from sentence_transformers import SentenceTransformer
import numpy as np
from ragatouille import RAGPretrainedModel
import torch
import json
from tqdm import tqdm

DB_PATH = "question_bank.csv"
METADATA_PATH = "./output/metadata.pkl"
VECTOR_DB_PATH = "./output"
DIRECT_HIT_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

retriever = RAGPretrainedModel.from_index("output/colbert/indexes/Colbert-Experimental")
print('Retriever Loaded')
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

def get_relevant_documents(query,n):
    temp=retriever.search(query,k=n)
    results={}
    multiple_instances={}
    for doc in temp:
        id=doc['document_id']
        if id in results.keys():
            if id in multiple_instances.keys():
                val=multiple_instances[id]
                multiple_instances[id]=val+1
                id=id+'('+str(val+1)+')'
            else:
                multiple_instances[id]=1
                id=id+'(1)'
            
        results[id]=doc['score']
    return results  

  

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def main():
    # DataGeneration for the llama qac triples
    #path='output_all_qac_cleaned.json' 
    #queries=load_queries(path)
    #joblib.dump(queries, './all_queries.pkl')

    # DataGeneration for rephrased questions
    path='rephrased_questions_cleaned_with_context.json'
    queries=load_queries_rephrased(path)
    joblib.dump(queries, './all_queries_rephrased.pkl')
    print(f'{len(queries)} Queries loaded')
    #print(queries[0:10])
    results = {}
    query_i=0
    for query in tqdm(queries,total=len(queries),desc="Processing queries"):
        try:
            result = get_relevant_documents(query,10)
            #results.append((query, result))
            results[query_i]=result
            query_i=query_i+1
        except Exception as e:
            print(str(e))
            break
            #results.append((query, str(e)))
    print(results)
    with open('Retriever_results_rephrased.json', 'w') as file:
       json.dump(results, file,cls=CustomJSONEncoder)
    # Print the results
    # for query, result in results:
    #     print(f"Query: {query}")
    
    #     print(f"Result: {result}")
    #     print("\n")

if __name__ == "__main__":
    main()              