from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
    SimilarityFunction
)
from datasets import Dataset
from sentence_transformers.losses import TripletDistanceMetric
import pickle
import json
import random
from sentence_transformers.evaluation import SentenceEvaluator
from CustomTripletEvaluator import CustomTripletEvaluator
from sentence_transformers.training_args import BatchSamplers



def load_data(path):
    with open(path,'rb') as f:
        return pickle.load(f)


def create_traing_data_cosine(anchor,positives,negatives):
    sent1=[]
    sent2=[]
    label=[]
    for i in range(len(anchor)):
        sent1.append(anchor[i])
        sent1.append(anchor[i])
        sent2.append(positives[i])
        sent2.append(negatives[i])
        label.append(1)
        label.append(0)

    train_dict={
        "sentence1": sent1,
        "sentence2": sent2,
        "score": label,
    }
    train_dataset = Dataset.from_dict(train_dict)
    return train_dataset


def create_test_data_cosine(path):
    with open(path, 'r') as f:
        test_data = json.load(f)

    selected_questions = random.sample(range(len(test_data)), 1000)
    different_questions = set()

    while len(different_questions) < 1000:
        num = random.randint(0, len(test_data) - 1)
        if num not in selected_questions:
            different_questions.add(num)

    different_questions = list(different_questions)
    
    sent1 = []
    sent2 = []
    label = []

    for i in range(1000):
        sent1.append(test_data[selected_questions[i]]['original'])
        sent1.append(test_data[selected_questions[i]]['original'])

        temp = random.randint(0, 1)
        sent2.append(test_data[selected_questions[i]]['rephrased'][temp])
        sent2.append(test_data[different_questions[i]]['original'])

        label.append(1)
        label.append(0)

    test_data_dict = {
        "sentence1": sent1,
        "sentence2": sent2,
        "score": label,
    }
    test_dataset = Dataset.from_dict(test_data_dict)
    return test_dataset 


def create_train_test_data_cosine_questions(path):
    with open(path, 'r') as f:
        test_data = json.load(f)

    selected_questions = random.sample(range(len(test_data)), 1250)
    different_questions = set()

    while len(different_questions) < 1250:
        num = random.randint(0, len(test_data) - 1)
        if num not in selected_questions:
            different_questions.add(num)

    different_questions = list(different_questions)
    
    sent1 = []
    sent2 = []
    label = []

    for i in range(1000):
        sent1.append(test_data[selected_questions[i]]['original'])
        sent1.append(test_data[selected_questions[i]]['original'])

        temp = random.randint(0, 1)
        sent2.append(test_data[selected_questions[i]]['rephrased'][temp])
        sent2.append(test_data[different_questions[i]]['original'])

        label.append(1)
        label.append(0)

    train_data_dict = {
        "sentence1": sent1,
        "sentence2": sent2,
        "score": label,
    }
    train_dataset = Dataset.from_dict(train_data_dict)
    
    sent1_test=[]
    sent2_test=[]
    label_test=[]
    
    for i in range(1000,1250):
        sent1_test.append(test_data[selected_questions[i]]['original'])
        sent1_test.append(test_data[selected_questions[i]]['original'])
        temp=random.randint(0,1)
        if(temp==0):
         sent2_test.append(test_data[selected_questions[i]]['rephrased'][0])
        else:
         sent2_test.append(test_data[selected_questions[i]]['rephrased'][1])
        sent2_test.append(test_data[different_questions[i]]['original'])
        label_test.append(1)
        label_test.append(0)

    test_data_dict={
        "sentence1": sent1_test,
        "sentence2": sent2_test,
        "score": label_test,
    }

    test_dataset = Dataset.from_dict(test_data_dict)
    return train_dataset,test_dataset 

def load_args():
    return  SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/fine_tune_direct_hit_questions",
    # Optional training parameters:
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
   
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=10,
    save_strategy="steps",
    save_steps=50,
    logging_steps=10,
    save_total_limit=1,
    load_best_model_at_end=True
    
)

def main():
    
    anchor = load_data('anchor.pkl')
    positives = load_data('positives.pkl')
    negatives = load_data('negatives.pkl')  

    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1") 
    print('Initial Model loaded')
    
    #train_dataset=create_traing_data_cosine(anchor,positives,negatives)
    #test_dataset=create_test_data_cosine('rephrased_questions_cleaned_with_context.json')
    train_dataset,test_dataset=create_train_test_data_cosine_questions('rephrased_questions_cleaned_with_context.json')
    print('Dataset_loaded')
    loss=losses.CosineSimilarityLoss(model)
    
    args=load_args()
    
    custom_evaluator = CustomTripletEvaluator(
        sentence1=test_dataset["sentence1"],
        sentence2=test_dataset["sentence2"],
        score=test_dataset["score"],
        name="rephrased_questions",
    )

    # Run the evaluation
    accuracy = custom_evaluator(model)
    print(f"Initial evaluation accuracy: {accuracy}")

    trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss=loss,
    evaluator=custom_evaluator,
)
    print('Trainer loaded')
    trainer.train()
     

if __name__ == "__main__" :
    main()
