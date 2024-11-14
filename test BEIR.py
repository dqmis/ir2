# How to do BEIR
import pathlib, os
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
import sklearn.metrics
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import torch
import vec2text
from alive_progress import alive_bar
import sklearn
import numpy as np
import math
def get_gtr_embeddings(text_list,
                       encoder: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer) -> torch.Tensor:

    inputs = tokenizer(text_list,
                       return_tensors="pt",
                       max_length=128,
                       truncation=True,
                       padding="max_length",).to("cuda")

    with torch.no_grad():
        model_output = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        hidden_state = model_output.last_hidden_state
        embeddings = vec2text.models.model_utils.mean_pool(hidden_state, inputs['attention_mask'])

    return embeddings
# load dataset 
def load_dataset(dataset = "msmarco"):
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    #### Provide the data_path where scifact has been downloaded and unzipped
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels

def calc_cosine_score(query_embeddings, corpus_embeddings):
    dot = query_embeddings @ corpus_embeddings.T
    query_norm = torch.norm(query_embeddings, dim=1)
    corpus_norm = torch.norm(corpus_embeddings, dim = 1)
    
    cosine_sim = dot / torch.outer(query_norm, corpus_norm)
    
    return cosine_sim

def calc_NDCG(score_tensor, corpus_ids, query_ids, qrels):
    ranking = torch.argsort(score_tensor)
    normelizer = np.arange(2, 12)
    normelizer = np.log2(normelizer)
    NDCG = 0
    for query_id in query_ids:
        print(qrels[query_ids[0]]) 
        relevant_docs = qrels[query_id]
        ideal_score = np.array(sorted(list(relevant_docs.values()))[::-1][:10])
        query_ranking = ranking[query_id][:10]
        pred_score = np.array([relevant_docs[corpus_ids[doc_id]] for doc_id in query_ranking])
        DCG = np.sum(pred_score / normelizer)
        IDCG = np.sum(ideal_score / normelizer)
        NDCG += DCG / IDCG
    NDCG /= len(query_ids)
    return NDCG
          
def BEIR_loop(dataset = "scifact", corpus_batch_size = 64):
    corpus, queries, qrels = load_dataset(dataset)
    encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    query_embeddings = get_gtr_embeddings(list(queries.values()), encoder, tokenizer)
    total_batches = len(corpus_ids) // corpus_batch_size
    score_tensor = torch.zeros((len(query_ids), len(corpus_ids)))
    
    with alive_bar(total_batches) as bar:
        for i in range(total_batches):
            batch = corpus_ids[i*corpus_batch_size : (i+1) * corpus_batch_size]
            batch_text = [corpus[id]["text"] for id in batch]
            corpus_embeddings = get_gtr_embeddings(batch_text, encoder, tokenizer)
            cosine_sim = calc_cosine_score(query_embeddings, corpus_embeddings)
            score_tensor[:, i*corpus_batch_size : (i+1) * corpus_batch_size] = cosine_sim
            bar()
    # query_embeddings = embed_data()
    NDCG = calc_NDCG(score_tensor, corpus_ids, query_ids, qrels)
    

# corpus, queries, qrels = load_dataset()
# print(len(queries))

BEIR_loop("msmarco")