# How to do BEIR
import pathlib, os
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import torch
import vec2text

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

def embed_data(text_list, add_noise = False):
    encoder = AutoModel.from_pretrained("jxm/gtr__nq__32__correct").encoder.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("jxm/gtr__nq__32__correct")
    embeddings = get_gtr_embeddings(text_list, encoder, tokenizer)
    
    if add_noise:
        pass
    
    return embeddings
    
def BEIR_loop(dataset = "msmarco", corpus_batch_size = 64):
    corpus, queries, qrels = load_dataset(dataset)
    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    query_embeddings = embed_data(list(queries.values()))
    total_batches = len(corpus_ids) // corpus_batch_size
    for i in range(total_batches):
        batch = corpus_ids[i*corpus_batch_size : (i+1) * corpus_batch_size]
        batch_text = [corpus[id]["text"] for id in batch]
        corpus_embedding = embed_data(batch_text, True)
        print(query_embeddings.size())
        print(corpus_embedding.size())
        exit()
    # query_embeddings = embed_data()
    pass

# corpus, queries, qrels = load_dataset()
# print(len(queries))

BEIR_loop()