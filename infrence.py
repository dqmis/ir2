import vec2text
from datasets import load_dataset
import time


"""Loop idea
    Load corrector,
    loop over all data,
    calc score
"""

def calc_blue(real_data, pred_data):
    pass

def calc_F1(real_data, pred_data):
    pass

def calc_exact(real_data, pred_data):
    correct = 0
    for idx, sent in real_data:
        if sent == pred_data[idx]:
            correct += 1
            
    return correct / len(pred_data)

def eval_metric(real_data, pred_data):
    blue = None
    F1 = None
    print(real_data)
    print(pred_data)
    exact = calc_exact(real_data, pred_data)
    print(exact)
    pass

def data_loader(data_name):
    pass


def infrence_loop(data_name, num_steps):
    corrector = vec2text.load_pretrained_corrector("gtr-base")
    ds = load_dataset("BeIR/quora", "corpus")["corpus"]["text"][:10]
    
    results = vec2text.invert_strings(ds, corrector, num_steps=1)
    eval_metric(ds, results)
    
infrence_loop("", 0)