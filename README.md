# Benchmarking Retrieval-Augmented Generation (RAG) Model

## Project Overview

The goal of this project is to create a benchmark dataset to test the effectiveness of our Retrieval-Augmented Generation (RAG) model. Our existing Claude QAC database contains Question-Answer-Corpus (QAC) triples that are matched against user queries. Including similar queries in the test dataset could lead to direct hits every time, thus not providing an actual test of the model’s capabilities. Therefore, to avoid this, we need to generate new QAC triples using other Large Language Models (LLMs).

## Model Selection

We chose the Meta-Llama-3-8B-Instruct model for generating new QAC triples. This selection was based on its promising performance observed during manual inspections on one CART-Protocol and one CART-Paper, where it was able to extract, on average, more than 15 QAC triples from every section of the document.

## Data Generation

Using the Meta-Llama-3-8B-Instruct model, a total of 174,554 new test question-answer pairs were generated. These pairs were derived from the contexts present in the Claude QAC database, which consists of 15,618 unique contexts. Each context is associated with up to 15 QAC triples in this test dataset.

### Data Cleaning and Processing

1. **Removed Identical Questions**: Ensured the dataset's uniqueness by removing completely identical questions to avoid cases where one question could be mapped to multiple contexts.
2. **Context Cleaning**: Removed contexts from which no questions were generated.


## Question Rephrasing
1. **Random Selection**: Selected 10 questions randomly from each of the 1114 documents available in the Claude QAC database to get a diverse set of questions from the document pool.
2. **Removed Near-Duplicate Questions**: Removed similar questions with a rouge score greater than 90 to maintain the dataset's diversity.
3. **Rephrased Questions**: Rephrased the remaining questions using the Meta-Llama-3-8B-Instruct model.
4. **Multiple Variations**: Generated two rephrased versions for each question to create multiple variations and test the performance of the direct hit approach.

## Benchmark Dataset

The initial and rephrased questions together formed a comprehensive test benchmark dataset, ready for further evaluation of retrieval strategies.

## Retrieval and Scoring

1. **Direct Hit Approach**: For every question in the test dataset, we found the top k similar questions (where k is a high value like 10) in the Claude QAC database. Computed and stored the scores for these top k questions using the same method used for evaluating direct hits to assess the relevance of this approach.
2. **ColBERT Model**: Applied the same retrieval approach using the ColBERT model. Retrieved the top k documents and computed their scores using the ColBERT model's evaluation criteria. Stored these scores for further analysis.

## Evaluation

Once we have all these scores, we can test different aspects of the pipeline such as:

1. Assess how effectively the direct hit approach retrieves relevant questions.
2. Investigate if relevant documents can be retrieved solely using the direct hit approach with an appropriate scoring threshold.
3. Determine the ideal threshold score for a direct hit to optimize the retrieval process.
4. Measure the recall rate when using only the ColBERT model.
5. Evaluate if the retrieval pipeline performs better when both the direct hit and ColBERT models are used together.
6. Determine the ideal k value that gives the best performance in retrieval using the ColBERT model.

## Files

- **Rag_Benchmarking_ClaudeContext.ipynb**: Python notebook to generate QAC triples using Llama and Claude Context.
- **Question_rephrasing.ipynb**: Python notebook to generate question rephrasing using Llama.
- **Fine_tune_sentence_transformer.ipynb**: Python notebook to generate the dataset for fine-tuning sentence transformers.
- **FineTuneSentenceTransformer.py**: Code to fine-tune sentence transformer.
- **DataGeneration/DirectHitModule**: Codes for generating direct hit scores.
- **DataGeneration/Retriever**: Codes for generating retriever scores.
- **DataGeneration/recall_calculation.py**: Code to generate recall.
- **Output/DataGeneration**: Output scores for the direct hit and the retriever module.
- **Output/llama_output**: Questions obtained from Llama and their count (testqa_count_cleaned.csv).
- **Output/FineTuneSentenceTransformer**: Data generated for fine-tuning sentence transformer.
