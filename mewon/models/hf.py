def loadlm(name,**kw):
    try:
        from transformers import AutoModelForCausalLM,AutoTokenizer
    except Exception as e:
        raise RuntimeError('install mewon[hf] for Hugging Face experiments') from e
    tok=AutoTokenizer.from_pretrained(name,**{k:v for k,v in kw.items() if k.startswith('tokenizer_')})
    model=AutoModelForCausalLM.from_pretrained(name,**{k:v for k,v in kw.items() if not k.startswith('tokenizer_')})
    return model,tok
