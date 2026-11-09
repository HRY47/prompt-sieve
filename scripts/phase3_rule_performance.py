"""Fresh-process rule timing/RAM with exact o200k counts; no ML inference."""
import json,resource,subprocess,sys,time
from pathlib import Path
from tokopt import optimize,OptimizationCache,get_counter


def child(strategy):
    cases=[json.loads(s) for s in Path('review/phase3-cases.jsonl').read_text().splitlines()]
    start=time.perf_counter();counter=get_counter('tiktoken:o200k_base');initialization=time.perf_counter()-start
    results=[];start=time.perf_counter();cpu=time.process_time()
    for c in cases:
        r=optimize(c['prompt'],counter=counter,strategies=['auto'] if strategy=='fast' else [strategy],mode='fast' if strategy=='fast' else 'full',cache=False)
        results.append(dict(id=c['id'],selected=r.best.name,original_tokens=r.original_tokens,selected_tokens=r.best.tokens,selected_text=r.best.text))
    wall=time.perf_counter()-start;cpu=time.process_time()-cpu
    cache=OptimizationCache();sample=cases[80]['prompt']
    start=time.perf_counter();optimize(sample,counter=counter,cache=cache);cold=time.perf_counter()-start
    start=time.perf_counter();hit=optimize(sample,counter=counter,cache=cache);warm=time.perf_counter()-start
    return dict(strategy=strategy,counter=counter.name,counter_initialization_seconds=initialization,optimization_seconds=wall,cpu_seconds=cpu,cpu_percent=cpu/wall*100,peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,cache_probe=dict(cold_seconds=cold,warm_seconds=warm,cache_hit=hit.cache_hit,storage=cache.info()),cases=results,target_model_inference=False,gpu_vram_bytes=None)


def main():
    if len(sys.argv)>1:print(json.dumps(child(sys.argv[1])));return
    rows=[]
    for mode in ('fast','normalize','logs'):
        proc=subprocess.run([sys.executable,__file__,mode],capture_output=True,text=True,check=True)
        rows.append(json.loads(proc.stdout))
    benchmark=json.loads(Path('review/phase3-benchmark.json').read_text())
    known={r['id']:r['selected_text'] for r in benchmark['strategies']['fast']}
    assert all(r['selected_text']==known[r['id']] for r in rows[0]['cases'])
    Path('review/phase3-rule-performance.json').write_text(json.dumps(dict(fresh_processes=True,rows=rows,final_fast_selections_match_model_benchmark=True,paid_api_requests=0),ensure_ascii=False,indent=2))
    for row in rows:print(row['strategy'],row['optimization_seconds'],row['peak_process_rss_bytes']/2**20,flush=True)
if __name__=='__main__':main()
