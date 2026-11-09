"""Opt-in real local integration measurements. No paid API or mock inference.

Run each mode in a separate process with OMP_NUM_THREADS=2 MKL_NUM_THREADS=2.
Downloads trusted public model/tokenizer files on first use. Outputs only
project examples and authored synthetic cases to review/phase2-*.json.
"""
import argparse
import importlib.metadata as md
import json
import platform
import resource
import sys
import time
from pathlib import Path

from tokopt.check import preservation
from tokopt.counters import get_counter
from tokopt.optimizer import optimize

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = {
    "en": "Never delete the database. Keep 42 records. Read `config.yaml` at /srv/app.py with --dry-run and user_id.",
    "fa": "هرگز پایگاه داده را حذف نکنید. ۴۲ رکورد را نگه دارید. فایل `config.yaml` و /srv/app.py و --dry-run و user_id را بررسی کنید.",
    "zh": "请勿删除数据库。保留42条记录。读取 `config.yaml` 和 /srv/app.py，使用 --dry-run 和 user_id。",
    "ja": "データベースを削除しないでください。42件を保持してください。`config.yaml` と /srv/app.py と --dry-run と user_id を確認してください。",
    "mixed": "Never delete production. داده را نگه دارید。保留42条记录。`config.yaml` と /srv/app.py を確認。",
    "code": "```python\ndef calculate_total(x):\n    return x * 2 + 5\n```\n",
    "logs": (ROOT / "examples/pytest_run.log").read_text(),
    "json": '{"user_id":42,"limit":-1.5,"safe":true}',
    "markdown": "# Heading\n\n- `value` must be 42.\n",
}


def peak():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def write(mode, payload):
    report = dict(mode=mode, python=sys.version, executable=sys.executable,
                  platform=platform.platform(), peak_process_rss_bytes=peak(),
                  gpu_vram_bytes=None, paid_api_requests=0, result=payload)
    report["dependencies"] = {}
    for name in ("tokopt", "tiktoken", "argostranslate", "llmlingua", "transformers", "torch", "tokenizers"):
        try: report["dependencies"][name] = md.version(name)
        except md.PackageNotFoundError: report["dependencies"][name] = None
    dest = ROOT / "review" / f"phase2-{mode}.json"
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved {dest.name}; peak RSS {peak()/2**20:.1f} MiB", flush=True)


def tokenizers():
    counters = [get_counter(s) for s in ("tiktoken:o200k_base", "tiktoken:cl100k_base", "hf:Qwen/Qwen2.5-0.5B-Instruct", "heuristic")]
    data = []
    for lang, text in SAMPLES.items():
        row = dict(sample=lang, characters=len(text), counts={c.name:c.count(text) for c in counters})
        row["heuristic_relative_error"] = {c.name: counters[-1].count(text)/c.count(text)-1 for c in counters[:-1]}
        data.append(row)
    write("tokenizers", data)


def translation():
    from tokopt.strategies.translate import ArgosTranslator, translate_text
    tr = ArgosTranslator()
    counter = get_counter("tiktoken:o200k_base")
    rows = []
    cases = [("example", (ROOT / "examples/prompt.md").read_text(), "en", ["fa","zh","ja"])]
    cases += [(lang, text, lang if lang != "mixed" else "en", ["en"] if lang in ("fa","zh","ja") else ["fa","zh","ja"]) for lang,text in SAMPLES.items() if lang in ("en","fa","zh","ja","mixed")]
    for label, text, src, langs in cases:
        start=time.perf_counter()
        result=optimize(text,counter=counter,langs=langs,src_lang=src,translator=tr,per_segment=True,measure_all=True)
        row=json.loads(result.to_json(include_text=True))
        row.update(sample=label, source_language=src, seconds=time.perf_counter()-start, peak_process_rss_bytes=peak())
        rows.append(row)
        write("translation", rows)
        print(label, [(c.name,c.tokens,c.passed,c.roundtrip_chrf) for c in result.candidates],flush=True)


def lingua():
    from tokopt.strategies.lingua import LinguaCompressor
    import torch
    torch.set_num_threads(2)
    counter=get_counter("tiktoken:o200k_base")
    rows=[]
    for lang in ("en","fa","zh","ja","mixed"):
        text=(SAMPLES[lang]+" ")*5 + "\n\n```sh\npython /srv/app.py --dry-run\n```"
        start=time.perf_counter()
        raw=LinguaCompressor(device="cpu").apply(text)
        raw_seconds=time.perf_counter()-start
        start=time.perf_counter()
        result=optimize(text,counter=counter,strategies=["lingua"])
        row=json.loads(result.to_json(include_text=True))
        row.update(sample=lang, raw_output=raw, raw_tokens=counter.count(raw), raw_preservation=vars(preservation(text,raw)), raw_seconds=raw_seconds, warm_optimizer_seconds=time.perf_counter()-start,peak_process_rss_bytes=peak(),cuda_available=torch.cuda.is_available())
        rows.append(row); write("lingua",rows)
        print(lang,row["raw_tokens"],result.best.name,flush=True)


def benchmark():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tokopt.strategies.translate import ArgosTranslator
    torch.set_num_threads(2)
    model_id="Qwen/Qwen2.5-0.5B-Instruct"
    start=time.perf_counter()
    tok=AutoTokenizer.from_pretrained(model_id,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(model_id,trust_remote_code=False,torch_dtype=torch.float32).eval()
    load_seconds=time.perf_counter()-start
    counter=get_counter("hf:"+model_id)
    tr=ArgosTranslator()
    facts={
        "en":"The access code is 42. Never delete the database. The approved file is `config.yaml`.",
        "fa":"کد دسترسی 42 است. هرگز پایگاه داده را حذف نکنید. فایل تاییدشده `config.yaml` است.",
        "zh":"访问代码是42。请勿删除数据库。批准的文件是 `config.yaml`。",
        "ja":"アクセスコードは42です。データベースを削除しないでください。承認されたファイルは `config.yaml` です。",
        "mixed":"Access code: 42. هرگز پایگاه داده را حذف نکنید。批准的文件是 `config.yaml`。",
    }
    questions=[("code","What is the access code? Reply only with the number.","42"), ("safety","Is deleting the database permitted? Reply only yes or no.","no"), ("file","What is the approved filename? Reply only with the filename.","config.yaml")]
    cases=[]
    filler="This is routine meeting background about scheduling and planning, with no additional task facts. "*5
    for lang,fact in facts.items():
        for suffix,q,expected in questions:
            cases.append(dict(id=lang+"-"+suffix,prompt=filler+"\n\n"+fact,question=q,expected=expected,strategies=["lingua"],langs=[]))
    for lang,fact in facts.items():
        cases.append(dict(id=lang+"-integrated-safety",prompt=filler+fact+" "+filler,
                          question=questions[1][1],expected="no",strategies=["lingua"],langs=[]))
    cases.append(dict(id="code",prompt=filler+"\n\n```python\ndef f(x):\n    return x * 2 + 5\n```",
                      question="What is f(2)? Reply only with the number.", expected="9",strategies=["lingua"],langs=[]))
    for suffix,q,expected in questions[:2]:
        cases.append(dict(id="translation-"+suffix,prompt=filler+"\n\n"+facts["en"],question=q,expected=expected,strategies=["normalize"],langs=["fa","zh","ja"]))
    noise="\n".join(f"tests/test_api.py::test_case_{i} PASSED [ 50%]" for i in range(50))
    cases.append(dict(id="logs",prompt=noise+"\nERROR: code 42 at /srv/app.py\nFAILED tests/test_api.py::test_connect\n",question="Which test failed? Reply only with the function name.",expected="test_connect",strategies=["logs"],langs=[]))
    rows=[]
    for index,case in enumerate(cases):
        start=time.perf_counter()
        opt=optimize(case["prompt"],counter=counter,strategies=case["strategies"],langs=case["langs"],translator=tr)
        elapsed=time.perf_counter()-start
        prompts={"original":case["prompt"]+"\n\n"+case["question"],"optimized":opt.best.text+"\n\n"+case["question"]}
        row=dict(id=case["id"],expected=case["expected"],strategy=opt.best.name,optimization_seconds=elapsed,candidates=[c.to_dict() for c in opt.candidates],arms={})
        # Alternate arm order to reduce the baseline-first warmup bias.
        for arm in (("original","optimized") if index%2==0 else ("optimized","original")):
            messages=[{"role":"user","content":prompts[arm]}]
            inputs=tok.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt")
            start=time.perf_counter()
            with torch.inference_mode():
                output=model.generate(inputs,attention_mask=torch.ones_like(inputs),max_new_tokens=32,do_sample=False,pad_token_id=tok.eos_token_id)
            seconds=time.perf_counter()-start
            answer=tok.decode(output[0,inputs.shape[1]:],skip_special_tokens=True).strip()
            # Exact independently authored expected answers; no agreement grading.
            correct=answer.strip().strip(".` ").lower()==case["expected"].lower()
            row["arms"][arm]=dict(answer=answer,correct=correct,seconds=seconds,text_tokens=counter.count(prompts[arm]),chat_tokens=inputs.shape[1],output_tokens=output.shape[1]-inputs.shape[1])
        rows.append(row)
        write("benchmark",dict(model=model_id,revision=getattr(model.config,"_commit_hash",None),load_seconds=load_seconds,generation=dict(do_sample=False,max_new_tokens=32,dtype="float32",threads=2),cases=rows))
        print(case["id"], opt.best.name,[(a,r["correct"]) for a,r in row["arms"].items()],flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("mode",choices=["tokenizers","translation","lingua","benchmark"])
    mode=parser.parse_args().mode
    globals()[mode]()
