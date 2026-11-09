"""Opt-in real snapshot and independently answered Persian regression checks."""
import json,os,pickle
from pathlib import Path
import pytest
pytestmark=pytest.mark.integration


@pytest.mark.skipif(os.getenv('PROMPTSIEVE_TEST_HF')!='1',reason='enable PROMPTSIEVE_TEST_HF=1 for loaded tokenizer snapshot transfer')
def test_loaded_hf_snapshot_pickle_keeps_multilingual_counts():
    from promptsieve.counters import get_counter
    counter=get_counter('hf:Qwen/Qwen2.5-0.5B-Instruct')
    restored=pickle.loads(pickle.dumps(counter))
    for text in ('Keep 42 records.','هرگز حذف نکنید.','保留42条记录。','42件を保持','2 + 3 * 4','<|im_start|>user\n42<|im_end|>','```python\nreturn -5\n```'):
        assert restored.count(text)==counter.count(text)


@pytest.mark.skipif(os.getenv('PROMPTSIEVE_TEST_PHASE3_MODEL')!='1',reason='enable PROMPTSIEVE_TEST_PHASE3_MODEL=1 for cached Qwen Persian replay')
def test_real_persian_original_legacy_and_repaired_answers():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from promptsieve import optimize
    from promptsieve.counters import HeuristicCounter
    fixture=json.loads((Path(__file__).resolve().parents[1]/'review/phase3-legacy-fixture.json').read_text())
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(fixture['model'],revision=fixture['revision'],local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(fixture['model'],revision=fixture['revision'],local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32).eval()
    fixed=optimize(fixture['original'],counter=HeuristicCounter(),strategies=['lingua'],mode='full',cache=False)
    for text,expected in ((fixture['original'],'42'),(fixture['normalized'],'10'),(fixed.best.text,'42')):
        ids=tok.apply_chat_template([{'role':'user','content':text+'\n\n'+fixture['question']}],add_generation_prompt=True,return_tensors='pt')
        with torch.inference_mode():out=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=32,do_sample=False,pad_token_id=tok.eos_token_id)
        assert tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True).strip().strip('. ')==expected
