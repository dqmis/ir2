import vec2text
from datasets import load_dataset
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from alive_progress import alive_bar
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel

"""Loop idea
    Load corrector,
    loop over all data,
    calc score
"""


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

def calc_bleu(real_data, pred_data):
    real_data = [sent.split() for sent in real_data]
    pred_data = [sent.split() for sent in pred_data]
    bleu_score = 0
    for idx, sent in enumerate(pred_data):
        bleu_score += sentence_bleu([real_data[idx]], sent, smoothing_function=SmoothingFunction().method1) # IDK if this is what they used in the paper
    
    return bleu_score / len(real_data)

def calc_F1(real_data, pred_data):
    real_data = [set(sent.split()) for sent in real_data]
    pred_data = [set(sent.split()) for sent in pred_data]
    f1_score = 0
    for idx, sent in enumerate(real_data):
        pred_sent = pred_data[idx]
        TP = len(sent.intersection(pred_sent)) # get true positives by all tokens in both sets
        FP = len(pred_sent - sent) # get false positves as tokens in the pred sent but not in the real sent
        FN = len(sent - pred_sent) # get false negatives as tokens in the real sent but not in the pred sent
        if TP == 0:
            f1_score += 0
            continue
        prec = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score += (2*prec*recall) / (prec + recall)
    
    return f1_score / len(real_data)
    
def calc_exact(real_data, pred_data):
    correct = 0
    for idx, sent in enumerate(real_data):
        if sent == pred_data[idx]:
            correct += 1
            
    return correct / len(pred_data)

def eval_metric(real_data, pred_data):
    bleu = calc_bleu(real_data, pred_data)
    F1 = calc_F1(real_data, pred_data)
    exact = calc_exact(real_data, pred_data)
    
    return bleu, F1, exact
    pass

def data_loader(data_name):
    pass


def infrence_loop(data_name, num_steps, batch_size = 64):
    corrector = vec2text.load_pretrained_corrector("gtr-base")
    encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
    ds = load_dataset("BeIR/quora", "corpus")["corpus"]["text"]
    bleu = 0
    F1 = 0
    exact = 0
    
    total_itters = len(ds) // batch_size
    total_itters = min(total_itters, 1000)
    print(total_itters)
    with alive_bar(total_itters) as bar:
        for itter in range(total_itters):
            sent_list = ds[itter * batch_size: (itter + 1) * batch_size]
            embeddings = get_gtr_embeddings(sent_list, encoder, tokenizer)
            results = vec2text.invert_embeddings(embeddings, corrector, num_steps=num_steps)
            temp_bleu, temp_F1, temp_exact = eval_metric(sent_list, results)
            bleu += temp_bleu / total_itters
            F1 += temp_F1 / total_itters
            exact += temp_exact / total_itters
            bar()
    print(bleu)
    print(F1)
    print(exact)
infrence_loop("", 0)