import os
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model, dataset_source="hf"):
    traindata = _load_wikitext2_split("train", dataset_source)
    testdata = _load_wikitext2_split("test", dataset_source)

    from transformers import AutoTokenizer 
    print('here1')
    try:
        print('here2')
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True, batched=True)
    except:
        print('here3')
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True, batched=True)
    print('here4')

    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    print('here5')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    print('here6')
    return trainloader, testenc


MODELSCOPE_WIKITEXT2_DATASET = os.environ.get("KVQUANT_MODELSCOPE_WIKITEXT2_DATASET", "modelscope/wikitext")


def _load_wikitext2_split(split: str, dataset_source: str):
    if dataset_source == "modelscope":
        # ModelScope's wikitext repository is backed by an old HuggingFace
        # datasets script. datasets>=3 removed support for script datasets, so
        # read the underlying ModelScope Open Data zip file directly. The file
        # is cached in a user-level cache and shared by all KVQuant stages.
        import urllib.request
        import zipfile

        dataset_name = "wikitext-2-raw-v1"
        cache_dir = Path.home() / ".cache" / "kvquant" / "modelscope" / dataset_name
        archive_path = cache_dir / f"{dataset_name}.zip"
        if not archive_path.is_file():
            cache_dir.mkdir(parents=True, exist_ok=True)
            temporary_path = cache_dir / f"{dataset_name}.zip.tmp"
            urllib.request.urlretrieve(
                f"https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/wikitext/{dataset_name}.zip",
                temporary_path,
            )
            temporary_path.replace(archive_path)

        split_file = {
            "train": "wiki.train.raw",
            "validation": "wiki.valid.raw",
            "test": "wiki.test.raw",
        }[split]
        with zipfile.ZipFile(archive_path, "r") as archive:
            lines = archive.read(f"wikitext-2-raw/{split_file}").decode("utf-8").splitlines()
        return {"text": lines}

    from datasets import load_dataset
    return load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    from transformers import AutoTokenizer 
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', use_auth_token=False
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', use_auth_token=False
    )

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        if trainenc.input_ids.shape[1] - seqlen - 1 < 0:
            i = 0
        else:
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        if tmp.input_ids.shape[1] - seqlen - 1 <= 0:
            i = 0
        else:
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 



def get_ptb_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc
def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', dataset_source="hf"
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, dataset_source=dataset_source)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)
