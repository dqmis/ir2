# I plan to have a function here where we give the name of a dataset and it returns the list of sentences.
# I want it in a separate file because we probably have to write a seperate function / code for each dataset
from datasets import load_dataset

def load_dataset(data_name):
    possible_list = ["quora", "signal1m", "msmarco", "climate-fever", "fever", "dbpedia-entity", "nq", "hotpotqa", "fiqa", "webis-touche2020", "cqadupstack", "arguana", "scidocs", "trec-covid", "robust04", "bioasq", "scifact", "nfcorpus", "trec-news"]
    if data_name not in possible_list:
        raise Exception(f"dataset {data_name} is not a suported dataset pick a dataset from \n {possible_list}")
    
    # if data_name == "quora":
    #     sent_list = load_dataset("BeIR/quora", "corpus")["corpus"]["text"]
        
    if data_name == "signal1m":
        # signal1m didn't seem to have a comparable dataset
        sent_list = load_dataset("BeIR/signal1m-generated-queries")["train"]["text"]
    else:
        sent_list = load_dataset(f"BeIR/{data_name}", "corpus")["corpus"]["text"] 
    # elif data_name == "msmarco":
    #     sent_list = load_dataset("BeIR/msmarco", "corpus")["corpus"]["text"]
        
    # elif data_name == "climate-fever":
    #     sent_list = load_dataset("BeIR/climate-fever", "corpus")["corpus"]["text"]
        
    # elif data_name == "fever":
    #     sent_list = load_dataset("BeIR/fever", "corpus")["corpus"]["text"]
        
    # elif data_name == "dbpedia-entity":
    #     sent_list = load_dataset("BeIR/dbpedia", "corpus")["corpus"]["text"]
        
    # elif data_name == "nq":
    #     sent_list = load_dataset("BeIR/nq", "corpus")["corpus"]["text"]
    return sent_list