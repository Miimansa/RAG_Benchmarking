import json

def load_json(path):
    with open(path,'r') as f:
        out= json.load(f)  
    return out  

def filter_docs_by_threshold(data, threshold):
    filtered_data = {}
    for query, docs in data.items():
        filtered_docs = {doc: score for doc, score in docs.items() if score >= threshold}
        if filtered_docs:
            filtered_data[query] = filtered_docs
    return filtered_data


def keep_the_top_match(data):
    filtered_data = {}
    for query, docs in data.items():
        filtered_docs = {k: v for k, v in [next(iter(docs.items()))]}
        if filtered_docs:
            filtered_data[query] = filtered_docs
    return filtered_data     


def recall_calculation_direct_hit(true_labels,results,threshold):
     total_queries=len(true_labels)
     if(threshold==0):
          
          sum=0
          min=float('inf')

          for query,doc_score in results.items():
               label=str(true_labels[query])
               docs=doc_score.keys()
               if label in docs:
                   sum=sum+1
                   score=doc_score[label]
                   if(score<min):
                        min=score
          print(f"The Threshold for this recall should be:{min}")              

          return (sum/total_queries)
     else:
          results_filtered=filter_docs_by_threshold(results, threshold)
          sum=0
          for query,doc_score in results_filtered.items():
               label=str(true_labels[query])
               docs=doc_score.keys()
               if label in docs:
                   sum=sum+1
                   
          return (sum/total_queries)
     

def recall_calculation_colbert(true_labels,results,k):
     total_queries=len(true_labels)
     if(k==10):
          
          sum=0
          

          for query,doc_score in results.items():
               label=str(true_labels[query])
               docs=doc_score.keys()
               if label in docs:
                   sum=sum+1
                          

          return (sum/total_queries)
     else:
          
          sum=0
          for query,doc_score in results.items():
               label=str(true_labels[query])
               docs=list(doc_score.keys())
               if label in docs[:k]:
                   sum=sum+1
                   
          return (sum/total_queries)  

def recall_calculation_both(true_labels,results_dh,results_cb,k,threshold):
     total_queries=len(true_labels)
     #results_dh_top=keep_the_top_match(results_dh)
     if(k==10):
          
          sum=0
          min=float('inf')
          for query,doc_score in results_cb.items():
               label=str(true_labels[query])
               dh_out=results_dh[query]
               doc,score=next(iter(dh_out.items()))
               if(doc==label and score>=threshold):
                    sum=sum+1
                    if(min>score):
                         min=score
                    continue     
               
               docs=doc_score.keys()
               if label in docs:
                   sum=sum+1
                          
          print(f'The ideal minimum threshold shoukd be:{min}')
          return (sum/total_queries)
     else:
          
          sum=0
          for query,doc_score in results_cb.items():
               label=str(true_labels[query])
               dh_out=results_dh[query]
               doc,score=next(iter(dh_out.items()))
               if(doc==label and score>=threshold):
                    sum=sum+1
                    if(min>score):
                         min=score
                    continue     
               docs=list(doc_score.keys())
               if label in docs[:k]:
                   sum=sum+1
          print(f'The ideal minimum threshold shoukd be:{min}')        
          return (sum/total_queries)             
                    

def combine_labels(path1,path2):
     label1=load_json(path1)
     label2=load_json(path2)
     total=len(label1)
     for key,value in label2.items():
          k=str(int(key)+total)
          label1[k]=value
     return label1     
 

def compute_recall_direct_hit(dataset,threshold=0):
       
       if(dataset=='llama'):
            true_labels=load_json('qrels_llama.json')
            result=load_json('direct_hit_results.json')
            return recall_calculation_direct_hit(true_labels,result,threshold)

       if(dataset=='rephrased'):
            true_labels=load_json('qrels_rephrased.json')
            result=load_json('direct_hit_rephrased_results.json')
            return recall_calculation_direct_hit(true_labels,result,threshold)

       if(dataset=='both'):
            true_labels=combine_labels('qrels_llama.json','qrels_rephrased.json')
            result=combine_labels('direct_hit_results.json','direct_hit_rephrased_results.json')
            return recall_calculation_direct_hit(true_labels,result,threshold)

def compute_recall_colbert(dataset,k=10):
       
       if(dataset=='llama'):
            true_labels=load_json('qrels_llama.json')
            result=load_json('Retriever_results.json')
            return recall_calculation_colbert(true_labels,result,k)

       if(dataset=='rephrased'):
            true_labels=load_json('qrels_rephrased.json')
            result=load_json('Retriever_results_rephrased.json')
            return recall_calculation_colbert(true_labels,result,k)

       if(dataset=='both'):
            true_labels=combine_labels('qrels_llama.json','qrels_rephrased.json')
            result=combine_labels('Retriever_results.json','Retriever_results_rephrased.json')
            return recall_calculation_colbert(true_labels,result,k)   

def compute_recall_both(dataset,k=10,threshold=0):
       
       if(dataset=='llama'):
            true_labels=load_json('qrels_llama.json')
            result_cb=load_json('Retriever_results.json')
            results_dh=load_json('direct_hit_results.json')
            return recall_calculation_both(true_labels,results_dh,result_cb,k,threshold)

       if(dataset=='rephrased'):
            true_labels=load_json('qrels_rephrased.json')
            result_cb=load_json('Retriever_results_rephrased.json')
            results_dh=load_json('direct_hit_rephrased_results.json')
            return recall_calculation_both(true_labels,results_dh,result_cb,k,threshold)

       if(dataset=='both'):
            true_labels=combine_labels('qrels_llama.json','qrels_rephrased.json')
            result_cb=combine_labels('Retriever_results.json','Retriever_results_rephrased.json')
            result_dh=combine_labels('direct_hit_results.json','direct_hit_rephrased_results.json')
            return recall_calculation_both(true_labels,results_dh,result_cb,k,threshold)         


                  
            
            

               


def main():
    module=input('Which module do you wish to test:\na) Direct Hit b) Colbert c)Both\n ')
    dataset=input('Give Dataset\n a) Llama Generated QAC b) Rephrased questions c) Both\n')
    print(module,dataset)
    match module:
        case 'a':
              thresh=float(input("Give Threshold(0 if you don't want to specify):\n"))
              if(dataset=='a'): print(compute_recall_direct_hit('llama',thresh))
              elif(dataset=='b'):print(compute_recall_direct_hit('rephrased',thresh))
              else:print(compute_recall_direct_hit('both',thresh))
        case 'b':
              k=int(input("Give k(10 if you don't want to specify):\n"))
              if(dataset=='a'): print(compute_recall_colbert('llama',k))
              elif(dataset=='b'):print(compute_recall_colbert('rephrased',k))
              else:print(compute_recall_colbert('both',k))
        case 'c':
              thresh=float(input("Give Threshold(0 if you don't want to specify):\n"))
              k=int(input("Give k(10 if you don't want to specify):\n"))
              if(dataset=='a'): print(compute_recall_both('llama',k,thresh))
              elif(dataset=='b'):print(compute_recall_both('rephrased',k,thresh))
              else:print(compute_recall_both('both',k,thresh))                

                   
                   
            

if __name__=='__main__':
    main()    
