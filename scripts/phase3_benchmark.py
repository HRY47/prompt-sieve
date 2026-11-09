"""120 independently answered cases, six strategy arms; cached local models.

Greedy identical serialized prompts reuse a recorded real generation. Reuse is
explicit and contributes zero actual generation time/CPU; reference timings
remain available for comparison. No simulated answers and no paid APIs.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import resource
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
FILLER={
'en':"The meeting covered routine scheduling and general planning for the coming season. "*5,
'fa':"جلسه درباره برنامه روزانه و برنامه‌ریزی عمومی برای فصل آینده بود. "*6,
'zh':"会议讨论了日常安排和下一季的一般规划。"*12,
'ja':"会議では日程と来季の一般的な計画について話しました。"*10,
}


def corpus():
    cases=[]
    def add(family,index,fact,question,expected,lang='en',noise=True):
        prompt=(FILLER[lang]+"\n\n" if noise else '')+fact
        cases.append(dict(id=f'{family}-{index}',family=family,source_language=lang,prompt=prompt,question=question,expected=expected,match='exact'))
    codefacts={
'en':"The access code is {n}. Never delete the database.",
'fa':"کد دسترسی {n} است. هرگز پایگاه داده را حذف نکنید.",
'zh':"访问代码是{n}。请勿删除数据库。",
'ja':"アクセスコードは{n}です。データベースを削除しないでください。",
}
    for i,n in enumerate((42,17,63,28,91,36,54,82,19,75)):
        lang=('en','fa','zh','ja','en')[i%5]
        add('access',i,codefacts[lang].format(n=n),'What is the access code? Reply only with the number.',str(n),lang)
    negatives=[
('Never delete backups.','Is deleting backups allowed?','no','en'),
('Only read the file. Do not modify it.','May the file be modified?','no','en'),
('Deleting temporary logs is allowed.','Is deleting temporary logs allowed?','yes','en'),
('هرگز نسخه پشتیبان را حذف نکنید.','Is deleting the backup permitted?','no','fa'),
('请勿删除备份。','Is deleting backups permitted?','no','zh'),
('バックアップを削除しないでください。','Is deleting backups permitted?','no','ja'),
('Do not disable encryption.','May encryption be disabled?','no','en'),
('The service must retain the archive.','May the archive be removed?','no','en'),
('Read access is allowed.','Is reading permitted?','yes','en'),
('Never publish the secret.','May the secret be published?','no','en')]
    for i,(f,q,e,l) in enumerate(negatives):add('negation',i,f,q+' Reply only yes or no.',e,l)
    measures=[('The dose is 5 mg, not 5 g.','What is the dose in mg?','5'),('The trip is 10 km and lasts 2 hours.','What is the distance in km?','10'),('The delay is 250 ms.','What is the delay in ms?','250'),('Memory limit is 512 MiB.','What is the memory limit in MiB?','512'),('The temperature is -5 °C.','What is the temperature in Celsius?','-5'),('Use 3 mL of liquid.','How many mL are required?','3'),('The power is 20 W.','What is the power in W?','20'),('The frequency is 60 Hz.','What is the frequency in Hz?','60'),('The cable is 25 cm long.','What is the cable length in cm?','25'),('The price is 19 USD.','What is the price in USD?','19')]
    for i,(f,q,e) in enumerate(measures):add('units',i,f,q+' Reply only with the number.',e)
    expressions=[('2 + 3 * 4','14'),('(2 + 3) * 4','20'),('18 / 3 + 2','8'),('9 - 4 * 2','1'),('7 * 6','42'),('5 * 5 - 1','24'),('12 + 8 / 2','16'),('3 * (4 + 2)','18'),('100 - 25','75'),('2 ** 5','32')]
    for i,(expr,e) in enumerate(expressions):add('math',i,'Evaluate '+expr+' using ordinary arithmetic precedence.','Reply only with the result.',e)
    programs=[('x * 2 + 5',2,'9'),('x - 3',10,'7'),('x ** 2',4,'16'),('x // 2',9,'4'),('x + 11',6,'17'),('x % 3',8,'2'),('x * 3',7,'21'),('x / 2',12,'6'),('x - 10',3,'-7'),('abs(x)',-5,'5')]
    for i,(expr,arg,e) in enumerate(programs):add('code',i,'```python\ndef f(x):\n    return '+expr+'\n```',f'What is f({arg})? Reply only with the numeric result.',e)
    files=['config.yaml','report.csv','settings.json','main.py','deploy.sh','limits.toml','README.md','audit.log','worker.ts','result.txt']
    for i,file in enumerate(files):add('paths',i,f'Read `/srv/app/{file}` with --dry-run. The approved filename is `{file}`.','What is the approved filename? Reply only with the filename.',file)
    orders=[('Run tests before deployment.','Which action comes first?','tests'),('Validate before saving.','Which action comes first?','validate'),('Back up before migration.','Which action comes first?','back up'),('Read the plan before editing.','Which action comes first?','read'),('After compiling, package the binary.','Which action comes first?','compiling'),('Authenticate before downloading.','Which action comes first?','authenticate'),('Install before starting.','Which action comes first?','install'),('Stop the worker before updating.','Which action comes first?','stop'),('Review before approval.','Which action comes first?','review'),('Open the archive after verifying it.','Which action comes first?','verifying')]
    for i,(f,q,e) in enumerate(orders):add('ordering',i,f,q+' Reply only with the action word or phrase.',e)
    for i in range(10):
        noise='\n'.join(f'tests/test_noise.py::test_ok_{j} PASSED [ 50%]' for j in range(60))
        add('logs',i,noise+f'\nERROR: connection refused at /srv/db.py:42\nFAILED tests/test_api.py::test_failure_{i}\n','Which test failed? Reply only with its function name.',f'test_failure_{i}',noise=False)
    for i,n in enumerate((11,22,33,44,55,66,77,88,99,101)):
        add('json',i,json.dumps({'user_id':n,'limit':-5,'enabled':False}),'What is user_id? Reply only with the number.',str(n))
    for i,(old,new) in enumerate(((10,42),(12,17),(21,63),(30,28),(14,91),(55,36),(71,54),(90,82),(88,19),(62,75))):
        add('distractors',i,f'The obsolete access code was {old}. The current access code is {new}. Use only the current code.','What is the current access code? Reply only with the number.',str(new))
    scopes=[('Never delete backups; temporary logs may be deleted.','May backups be deleted?','no'),('Never delete backups; temporary logs may be deleted.','May temporary logs be deleted?','yes'),('Upload reports, but never upload keys.','May keys be uploaded?','no'),('Upload reports, but never upload keys.','May reports be uploaded?','yes'),('Only administrators may deploy. Sam is a guest.','May Sam deploy?','no'),('Only administrators may deploy. Kim is an administrator.','May Kim deploy?','yes'),('Save drafts; never publish drafts.','May drafts be published?','no'),('Save drafts; never publish drafts.','May drafts be saved?','yes'),('Do not overwrite the source; writing a new destination is allowed.','May the source be overwritten?','no'),('Do not overwrite the source; writing a new destination is allowed.','May a new destination be written?','yes')]
    for i,(f,q,e) in enumerate(scopes):add('scope',i,f,q+' Reply only yes or no.',e)
    facts=[('Mira owns the blue bicycle.','Who owns the blue bicycle?','Mira'),('The workshop is in Oslo.','Where is the workshop?','Oslo'),('The shipment contains copper.','What material is in the shipment?','copper'),('The meeting is on Tuesday.','On which day is the meeting?','Tuesday'),('The project codename is Cedar.','What is the project codename?','Cedar'),('The librarian is named Hana.','What is the librarian named?','Hana'),('The approved color is green.','What is the approved color?','green'),('The server region is Canada.','What is the server region?','Canada'),('The package owner is Ravi.','Who owns the package?','Ravi'),('The destination city is Kyoto.','What is the destination city?','Kyoto')]
    for i,(f,q,e) in enumerate(facts):add('facts',i,f,q+' Reply only with the answer word.',e)
    # Include the exact historic mixed-language failure without adding a new task.
    cases[1].update(id='persian-phase2-42',source_language='en',prompt=("This is routine meeting background about scheduling and planning, with no additional task facts. "*5)+"\n\nکد دسترسی 42 است. هرگز پایگاه داده را حذف نکنید. فایل تاییدشده `config.yaml` است.",expected='42')
    assert len(cases)==120 and len({c['id'] for c in cases})==120
    return cases


def definition_hash(cases):
    return hashlib.sha256(json.dumps(cases,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def observation_signature(revision):
    from importlib.metadata import version
    from huggingface_hub import try_to_load_from_cache
    fingerprints={}
    for name in ('tokenizer.json','tokenizer_config.json'):
        path=try_to_load_from_cache(MODEL,name,revision=revision)
        if not isinstance(path,str):raise ValueError('missing pinned tokenizer snapshot')
        fingerprints[name]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return dict(tokenizer_files=fingerprints,dependencies={name:version(name) for name in ('torch','transformers','tokenizers')})


def validate_reuse_report(previous,revision,generation,cases,signature):
    if (previous.get('model')!=MODEL or previous.get('revision')!=revision
            or previous.get('generation')!=generation
            or previous.get('dataset_sha256')!=definition_hash(cases)
            or previous.get('observation_signature')!=signature):
        raise ValueError('reuse requires identical model, tokenizer, dependencies, settings, questions and expected answers')


class Meter:
    def __enter__(self):
        import psutil
        self.process=psutil.Process(); self.peak=self.process.memory_info().rss
        self.stop=threading.Event()
        def sample():
            while not self.stop.wait(.01):
                self.peak=max(self.peak,self.process.memory_info().rss)
        self.thread=threading.Thread(target=sample,daemon=True); self.thread.start()
        self.wall=time.perf_counter(); self.cpu=time.process_time(); return self
    def __exit__(self,*args):
        self.seconds=time.perf_counter()-self.wall; self.cpu_seconds=time.process_time()-self.cpu
        self.stop.set(); self.thread.join()
        self.peak=max(self.peak,self.process.memory_info().rss)
        self.metrics=dict(seconds=self.seconds,cpu_seconds=self.cpu_seconds,cpu_percent=100*self.cpu_seconds/self.seconds if self.seconds else 0,peak_process_rss_bytes=self.peak)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--strategies',default='fast,normalize,logs,lingua,translation,combined');parser.add_argument('--limit',type=int,default=120);parser.add_argument('--reuse-report');a=parser.parse_args()
    cases=corpus()[:a.limit]
    (ROOT/'review/phase3-cases.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases))
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from promptsieve.counters import get_counter
    from promptsieve.optimizer import optimize
    from promptsieve.strategies.translate import ArgosTranslator
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    with Meter() as load:
        model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32).eval()
    counter=get_counter('hf:'+MODEL); translator=ArgosTranslator(); generation_cache={}
    reused_source=None
    if a.reuse_report:
        previous=json.loads(Path(a.reuse_report).read_text())
        validate_reuse_report(previous,model.config._commit_hash,
                              {'do_sample':False,'max_new_tokens':32,'dtype':'float32','threads':2},
                              cases,observation_signature(model.config._commit_hash))
        by_id={c['id']:c for c in cases}
        def seed(prompt,arm):
            ids=tok.apply_chat_template([{'role':'user','content':prompt}],add_generation_prompt=True,return_tensors='pt')
            assert ids.shape[1]==arm['input_chat_tokens']
            if hashlib.sha256(prompt.encode()).hexdigest()!=arm.get('input_sha256'):
                raise ValueError('reused input bytes differ')
            generation_cache[hashlib.sha256(ids.numpy().tobytes()).hexdigest()]=dict(arm)
        for cid,arm in previous['baseline'].items():
            if cid in by_id:
                c=by_id[cid];seed(c['prompt']+'\n\n'+c['question'],arm)
        for previous_rows in previous['strategies'].values():
            for r in previous_rows:
                if r['id'] in by_id:
                    c=by_id[r['id']]
                    if hashlib.sha256(c['prompt'].encode()).hexdigest()!=r['original_sha256']:raise ValueError('case changed')
                    seed(r['selected_text']+'\n\n'+c['question'],r['optimized'])
        reused_source={'path':str(a.reuse_report),'sha256':hashlib.sha256(Path(a.reuse_report).read_bytes()).hexdigest(),'serialized_inputs_seeded':len(generation_cache)}

    def infer(prompt):
        ids=tok.apply_chat_template([{'role':'user','content':prompt}],add_generation_prompt=True,return_tensors='pt')
        key=hashlib.sha256(ids.numpy().tobytes()).hexdigest()
        if key in generation_cache:
            return dict(generation_cache[key],reused_generation=True,actual_generation_seconds=0,actual_generation_cpu_seconds=0)
        with Meter() as meter:
            with torch.inference_mode():
                out=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=32,do_sample=False,pad_token_id=tok.eos_token_id)
        answer=tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True).strip()
        result=dict(answer=answer,input_sha256=hashlib.sha256(prompt.encode()).hexdigest(),input_text_tokens=counter.count(prompt),input_chat_tokens=ids.shape[1],output_tokens=out.shape[1]-ids.shape[1],**meter.metrics,reused_generation=False,actual_generation_seconds=meter.seconds,actual_generation_cpu_seconds=meter.cpu_seconds)
        generation_cache[key]=result
        return result
    def correct(answer,expected):return answer.strip().strip('`. ').lower()==expected.lower()
    baselines={}
    for i,c in enumerate(cases):
        r=infer(c['prompt']+'\n\n'+c['question']);r['correct']=correct(r['answer'],c['expected']);baselines[c['id']]=r
        if i%20==0: print('baseline',i,len(cases),flush=True)
    report=dict(dataset_sha256=definition_hash(cases),observation_signature=observation_signature(model.config._commit_hash),reused_observation_source=reused_source,model=MODEL,revision=model.config._commit_hash,generation={'do_sample':False,'max_new_tokens':32,'dtype':'float32','threads':2},load=load.metrics,cases=len(cases),baseline=baselines,strategies={},gpu_vram_bytes=None,cuda_available=torch.cuda.is_available(),paid_api_requests=0,timing_notes='Same serialized greedy inputs reuse a recorded real generation with actual time/CPU zero; reference measurement fields are retained. Process RSS includes loaded shared models; no isolated target-memory saving is claimed.')
    for strategy in a.strategies.split(','):
        rows=[]
        for i,c in enumerate(cases):
            langs=[]
            if strategy=='fast':kw=dict(strategies=['auto'],mode='fast')
            elif strategy=='normalize':kw=dict(strategies=['normalize'],mode='full')
            elif strategy=='logs':kw=dict(strategies=['logs'],mode='full')
            elif strategy=='lingua':kw=dict(strategies=['lingua'],mode='full')
            elif strategy=='translation':kw=dict(strategies=['normalize'],mode='full');langs=['zh' if c['source_language']=='en' else 'en']
            elif strategy=='combined':kw=dict(strategies=['auto','logs','lingua'],mode='full');langs=['zh' if c['source_language']=='en' else 'en']
            else:raise ValueError(strategy)
            with Meter() as meter:
                result=optimize(c['prompt'],counter=counter,langs=langs,src_lang=c['source_language'],translator=translator,cache=False,**kw)
            prompt=result.best.text+'\n\n'+c['question']
            arm=infer(prompt);arm['correct']=correct(arm['answer'],c['expected'])
            baseline=baselines[c['id']]
            row=dict(id=c['id'],family=c['family'],expected=c['expected'],selected=result.best.name,optimization=meter.metrics,original=baseline,optimized=arm,paired_regression=baseline['correct'] and not arm['correct'],paired_improvement=not baseline['correct'] and arm['correct'],candidates=[x.to_dict() for x in result.candidates],original_sha256=hashlib.sha256(c['prompt'].encode()).hexdigest(),selected_text=result.best.text)
            rows.append(row)
            if i%20==0:print(strategy,i,len(cases),'regressions',sum(r['paired_regression'] for r in rows),flush=True)
            report['strategies'][strategy]=rows
            (ROOT/'review/phase3-benchmark.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print('DONE',strategy,'accuracy',sum(r['optimized']['correct'] for r in rows),'regressions',[r['id'] for r in rows if r['paired_regression']],flush=True)
    from promptsieve.runtime import OptimizationCache
    repeat=cases[-1]['prompt'];cache=OptimizationCache()
    with Meter() as cold: optimize(repeat,counter=counter,cache=cache)
    with Meter() as warm: hit=optimize(repeat,counter=counter,cache=cache)
    report['optimization_cache_probe']=dict(cold=cold.metrics,warm=warm.metrics,hit=hit.cache_hit,storage=cache.info())
    report['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    (ROOT/'review/phase3-benchmark.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
