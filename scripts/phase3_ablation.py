"""Replay captured Phase 2 bytes and current fixes against cached local Qwen."""
import difflib,json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from tokopt import optimize
from tokopt.counters import get_counter
from tokopt.strategies.normalize import normalize


def main():
    torch.set_num_threads(2)
    fixture=json.loads(Path('review/phase3-legacy-fixture.json').read_text())
    tok=AutoTokenizer.from_pretrained(fixture['model'],local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(fixture['model'],local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32).eval()
    original=fixture['original'];legacy=fixture['normalized'];counter=get_counter('hf:'+fixture['model'])
    fast=optimize(original,counter=counter,cache=False)
    full=optimize(original,counter=counter,strategies=['lingua'],mode='full',cache=False)
    variants={'original':original,'legacy_single_space_removed':legacy,'current_normalization':normalize(original),'legacy_llmlingua':fixture['compressed'],'current_fast':fast.best.text,'current_lingua_policy':full.best.text,'fact_only':original.split('\n\n')[-1]}
    rows=[]
    for label,text in variants.items():
        ids=tok.apply_chat_template([{'role':'user','content':text+'\n\n'+fixture['question']}],add_generation_prompt=True,return_tensors='pt')
        start=time.perf_counter()
        with torch.inference_mode():out=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=32,do_sample=False,pad_token_id=tok.eos_token_id)
        answer=tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True).strip()
        rows.append(dict(variant=label,text=text,answer=answer,correct=answer.strip('. ')==fixture['expected'],input_tokens=ids.shape[1],output_tokens=out.shape[1]-ids.shape[1],seconds=time.perf_counter()-start))
        print(label,repr(answer),flush=True)
    before=tok.encode(original);after=tok.encode(legacy)
    token_changes=[dict(tag=t,original_ids=before[a:b],changed_ids=after[c:d],original_decoded=tok.decode(before[a:b]),changed_decoded=tok.decode(after[c:d])) for t,a,b,c,d in difflib.SequenceMatcher(a=before,b=after,autojunk=False).get_opcodes() if t!='equal']
    character_changes=[dict(tag=t,original_start=a,original=original[a:b],changed=legacy[c:d]) for t,a,b,c,d in difflib.SequenceMatcher(a=original,b=legacy,autojunk=False).get_opcodes() if t!='equal']
    report=dict(model=fixture['model'],revision=model.config._commit_hash,expected=fixture['expected'],original=original,legacy_normalized=legacy,legacy_compressed=fixture['compressed'],character_changes=character_changes,token_changes=token_changes,word_diff=fixture['word_diff'],variants=rows,paid_api_requests=0)
    Path('review/phase3-ablation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
