"""Measure real worker cancellation and loaded-tokenizer snapshot transfer."""
import json,multiprocessing as mp,time
from pathlib import Path
from tokopt import optimize
from tokopt.counters import HeuristicCounter,get_counter


def main():
    import psutil
    rows=[]
    prompt="The meeting discussed schedules and planning for the coming season. "*40
    for budget in (.05,.5,2):
        start=time.perf_counter()
        r=optimize(prompt,counter=HeuristicCounter(),strategies=['lingua'],mode='full',timeout_seconds=budget,cache=False)
        rows.append(dict(budget_seconds=budget,actual_seconds=time.perf_counter()-start,timed_out=r.timed_out,original_returned=r.best.text==prompt,original_tokens=r.original_tokens,warnings=r.warnings,live_workers=[p.pid for p in mp.active_children()],child_processes=[p.pid for p in psutil.Process().children(recursive=True) if 'resource_tracker' not in ' '.join(p.cmdline())]))
        print(rows[-1],flush=True)
    counter=get_counter('hf:Qwen/Qwen2.5-0.5B-Instruct')
    start=time.perf_counter()
    r=optimize('Keep 42 records.',counter=counter,timeout_seconds=120,cache=False)
    snapshot=dict(seconds=time.perf_counter()-start,timed_out=r.timed_out,count=r.original_tokens,expected_count=counter.count('Keep 42 records.'),warnings=r.warnings)
    assert not r.timed_out and not r.warnings and r.original_tokens==snapshot['expected_count']
    Path('review/phase3-deadlines.json').write_text(json.dumps(dict(cancellations=rows,hf_snapshot=snapshot,paid_api_requests=0),indent=2))
if __name__=='__main__':main()
