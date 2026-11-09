"""Meaning-sensitive and lifecycle regressions; no paid or model inference."""
import json
import multiprocessing as mp
import time
import pytest
from promptsieve import optimize, OptimizationCache
from promptsieve.check import preservation
from promptsieve.counters import HeuristicCounter
from promptsieve.strategies.normalize import normalize

FILLER="This is routine meeting background about scheduling and planning, with no additional task facts. "*5
FACT="کد دسترسی 42 است. هرگز پایگاه داده را حذف نکنید. فایل تاییدشده `config.yaml` است."


def test_persian_regression_keeps_exact_separator():
    prompt=FILLER+"\n\n"+FACT
    assert normalize(prompt)==prompt
    result=optimize(prompt,counter=HeuristicCounter(),cache=False)
    assert result.best.text==prompt and result.original_text==prompt


def test_mixed_language_lingua_fails_before_model_load(monkeypatch):
    from promptsieve.strategies.lingua import LinguaCompressor
    monkeypatch.setattr(LinguaCompressor,'available',lambda: (_ for _ in ()).throw(AssertionError('model loaded')))
    prompt=FILLER+"\n\n"+FACT
    result=optimize(prompt,counter=HeuristicCounter(),strategies=['lingua'],cache=False)
    assert result.best.text==prompt
    assert any('mixed-language' in ' '.join(c.reasons) for c in result.candidates)


@pytest.mark.parametrize('original,changed',[
    ('Never delete the backup.','Delete the backup.'),
    ('The dose is 5 mg, not 5 g.','The dose is 5 g, not 5 mg.'),
    ('Travel 10 km in 2 hours.','Travel 10 hours in 2 km.'),
    ('Compute 2 + 3 * 4.','Compute 2 * 3 + 4.'),
    ('Use x <= 5.','Use x >= 5.'),
    ('Keep -5 records.','Keep 5 records.'),
    ('Use `a - b`, never `b - a`.','Use `b - a`, never `a - b`.'),
    ('```python\nreturn x - 2\n```','```python\nreturn x + 2\n```'),
])
def test_meaning_sensitive_edits_are_rejected(original,changed):
    assert not preservation(original,changed).ok


class MutableCounter:
    name='versioned-test-counter'
    def __init__(self): self.version=1; self.calls=0
    def cache_key(self): return ('versioned-test-counter',self.version)
    def count(self,text): self.calls+=1; return len(text)*self.version


class SlowCounter:
    name='slow-counter'
    def count(self,text): time.sleep(5); return len(text)


def test_cache_copy_bounds_invalidation_and_original():
    counter=MutableCounter(); cache=OptimizationCache(max_entries=2,max_bytes=4096)
    a=optimize('abc',counter=counter,cache=cache)
    calls=counter.calls
    a.best.text='corrupted caller copy'
    b=optimize('abc',counter=counter,cache=cache)
    assert b.cache_hit and b.best.text=='abc' and counter.calls==calls
    counter.version=2
    c=optimize('abc',counter=counter,cache=cache)
    assert not c.cache_hit and c.original_tokens==6
    optimize('abc',counter=counter,cache=cache,min_savings=.2)
    assert cache.info()['entries']<=2 and cache.info()['bytes']<=4096
    cache.clear(); assert cache.info()['entries']==0


def test_cache_ttl_and_oversized_entries():
    cache=OptimizationCache(ttl_seconds=.001)
    optimize('abc',counter=HeuristicCounter(),cache=cache)
    time.sleep(.01)
    assert not optimize('abc',counter=HeuristicCounter(),cache=cache).cache_hit
    tiny=OptimizationCache(max_bytes=1)
    optimize('abc',counter=HeuristicCounter(),cache=tiny)
    assert tiny.info()['entries']==0


def test_cache_namespaces_and_no_opaque_backend_caching():
    cache=OptimizationCache(); counter=MutableCounter()
    optimize('abc',counter=counter,cache=cache,cache_namespace='a')
    assert not optimize('abc',counter=counter,cache=cache,cache_namespace='b').cache_hit
    class Opaque:
        name='opaque'
        def count(self,text): return len(text)
    o=Opaque()
    optimize('abc',counter=o,cache=cache)
    assert not optimize('abc',counter=o,cache=cache).cache_hit


def test_timeout_returns_exact_original_unknown_count_and_no_live_worker():
    before={p.pid for p in mp.active_children()}
    start=time.monotonic()
    result=optimize('never lose this original',counter=SlowCounter(),timeout_seconds=.05,cache=False,max_tokens=1)
    assert time.monotonic()-start<2
    assert result.timed_out and result.best.text==result.original_text=='never lose this original'
    assert result.original_tokens is None and not result.best.passed
    assert not ({p.pid for p in mp.active_children()}-before)
    assert json.loads(result.to_json())['timed_out']
    assert 'unknown' in result.table()


def test_timeout_success_and_invalid_config():
    result=optimize('hello',counter=HeuristicCounter(),timeout_seconds=5,cache=False)
    assert not result.timed_out and result.original_tokens==1
    for value in (0,-1,float('nan'),float('inf')):
        with pytest.raises(ValueError): optimize('hello',timeout_seconds=value)


def test_unpicklable_backend_fails_safe():
    class Local:
        name='local'
        def count(self,text): return len(text)
    result=optimize('original',counter=Local(),timeout_seconds=.5,cache=False)
    assert result.best.text=='original' and result.warnings


def test_fast_mode_skips_short_ml_and_balanced_respects_rules(monkeypatch):
    from promptsieve.strategies.lingua import LinguaCompressor
    monkeypatch.setattr(LinguaCompressor,'available',lambda: (_ for _ in ()).throw(AssertionError('model loaded')))
    r=optimize('short text',counter=HeuristicCounter(),strategies=['lingua'],mode='fast',cache=False)
    assert not next(c for c in r.candidates if c.name.startswith('lingua')).passed
    prompt='\n'.join(f'tests/test_x.py::test_{i} PASSED [ 50%]' for i in range(500))+'\nERROR: denied'
    r=optimize(prompt,counter=HeuristicCounter(),mode='balanced',cache=False)
    assert r.best.name=='logs'


def test_paired_accuracy_reports_regressions_and_improvements():
    from promptsieve.bench import CaseResult,summarize
    rows=[CaseResult('worse','x',1,1,0,True,False),CaseResult('better','x',1,1,0,False,True)]
    s=summarize(rows)
    assert s['paired_regressions']==['worse'] and s['paired_improvements']==['better']



def test_cli_does_not_overwrite_original(tmp_path):
    import subprocess,sys
    p=tmp_path/'prompt.txt'; p.write_bytes(b'keep original\r\n')
    r=subprocess.run([sys.executable,'-m','promptsieve','compress',str(p),'-o',str(p),'-c','heuristic'],capture_output=True)
    assert r.returncode!=0 and p.read_bytes()==b'keep original\r\n'


def test_cli_timeout_returns_original(tmp_path):
    import subprocess,sys
    p=tmp_path/'prompt.txt'; p.write_text('keep original')
    r=subprocess.run([sys.executable,'-m','promptsieve','compress',str(p),'--timeout','0.001','-c','heuristic','--json'],capture_output=True,text=True)
    assert r.returncode==0 and r.stdout=='keep original'
    assert json.loads(r.stderr)['timed_out']



def test_fast_translation_cost_gate_never_constructs_backend(monkeypatch):
    import promptsieve.strategies.translate as module
    monkeypatch.setattr(module,'get_translator',lambda backend: (_ for _ in ()).throw(AssertionError('translator loaded')))
    result=optimize('short text',counter=HeuristicCounter(),langs=['zh'],mode='fast',cache=False)
    assert result.best.text=='short text'


def test_fast_default_avoids_minor_cosmetic_edits():
    text='Some ordinary prose.   \n\nAnother ordinary paragraph.'
    result=optimize(text,counter=HeuristicCounter(),cache=False)
    assert result.best.text==text


def test_unit_association_and_negation_scope_are_not_just_multisets():
    assert not preservation('Use 5 mg and 10 g.','Use 5 g and 10 mg.').ok
    assert not preservation('Never delete backups; delete logs.','Never delete logs; delete backups.').ok



def test_backend_runtime_error_returns_original_without_leaking(monkeypatch):
    import promptsieve.optimizer as module
    monkeypatch.setattr(module,'normalize',lambda text: (_ for _ in ()).throw(RuntimeError('PRIVATE_DETAILS')))
    result=optimize('recover this',counter=HeuristicCounter(),cache=False)
    assert result.best.text==result.original_text=='recover this'
    assert result.warnings and 'PRIVATE_DETAILS' not in result.to_json()
    assert 'warning:' in result.table()



def test_original_snapshot_survives_candidate_edits():
    result=optimize('recover the exact source',counter=HeuristicCounter(),cache=False)
    result.best.text='caller changed candidate'
    assert result.original_text=='recover the exact source'
    assert json.loads(result.to_json(include_text=True))['original_text']=='recover the exact source'
    assert 'original_text' not in json.loads(result.to_json())


def test_legacy_positional_source_language_is_unchanged():
    result=optimize('original',HeuristicCounter(),['normalize'],[],None,'en',cache=False)
    assert result.original_text=='original'



@pytest.mark.parametrize('fact',['Memory limit is 2048 MiB.','The temperature is 12 °C.','Compute 6 + 2 * 9.','Compute 2 ** 6.'])
def test_sensitive_context_skips_model_even_when_facts_would_survive(monkeypatch,fact):
    from promptsieve.strategies.lingua import LinguaCompressor
    monkeypatch.setattr(LinguaCompressor,'available',lambda: (_ for _ in ()).throw(AssertionError('model loaded')))
    prompt=('The meeting covered routine planning for the coming season. '*8)+'\n\n'+fact
    result=optimize(prompt,counter=HeuristicCounter(),strategies=['lingua'],mode='full',cache=False)
    assert result.best.text==prompt
    assert any('unit/math context' in ' '.join(c.reasons) for c in result.candidates)



def test_unit_associations_survive_identifier_masking():
    assert not preservation('Use 512 MiB and 2 GiB.','Use 512 GiB and 2 MiB.').ok



@pytest.mark.parametrize('fence',['```python\nx = "\x1b[31mred"\n```','```python\r\nx = "literal"  \r\n```'])
def test_normalization_keeps_code_controls_and_line_endings(fence):
    assert normalize(fence)==fence
    assert normalize('\x1b[32mIntro\x1b[0m\n\n'+fence)=='Intro\n\n'+fence
