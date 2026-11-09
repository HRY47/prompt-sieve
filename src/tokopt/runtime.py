"""Bounded private result cache and process-isolated optimization deadlines."""
from __future__ import annotations
import hashlib
import inspect
import json
import math
import os
import signal
import multiprocessing as mp
import pickle
import threading
import uuid
import time
from collections import OrderedDict


class OptimizationCache:
    """Process-local LRU: entry and serialized-byte caps, TTL, defensive copies.

    No disk writes. Mutable/remote ML backends are deliberately not cached.
    Custom counters may opt in via cache_key(), which must change with behavior.
    """
    def __init__(self, max_entries=128, max_bytes=8 * 1024 * 1024, ttl_seconds=300):
        if not isinstance(max_entries, int) or max_entries < 0 or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("cache bounds must be nonnegative integers")
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("cache TTL must be finite and positive")
        self.max_entries, self.max_bytes, self.ttl_seconds = max_entries, max_bytes, ttl_seconds
        self._data, self._bytes, self._lock = OrderedDict(), 0, threading.RLock()

    def clear(self):
        with self._lock:
            self._data.clear(); self._bytes = 0

    def info(self):
        with self._lock:
            self._expire()
            return {"entries": len(self._data), "bytes": self._bytes,
                    "max_entries": self.max_entries, "max_bytes": self.max_bytes}

    def _expire(self):
        now = time.monotonic()
        for key in list(self._data):
            if now - self._data[key][0] >= self.ttl_seconds:
                self._bytes -= len(self._data.pop(key)[1])

    def get(self, key):
        with self._lock:
            self._expire()
            entry = self._data.get(key)
            if entry is None: return None
            self._data.move_to_end(key)
            return pickle.loads(entry[1])  # only our own in-memory serialization

    def put(self, key, result):
        blob = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
        if len(blob) > self.max_bytes or not self.max_entries: return
        with self._lock:
            self._expire()
            if key in self._data: self._bytes -= len(self._data.pop(key)[1])
            self._data[key] = (time.monotonic(), blob); self._bytes += len(blob)
            while len(self._data) > self.max_entries or self._bytes > self.max_bytes:
                _, (_, removed) = self._data.popitem(last=False); self._bytes -= len(removed)


DEFAULT_CACHE = OptimizationCache()


def _cache_key(text, options, namespace):
    # Inference and translation depend on mutable models, packages and settings.
    # Cache only deterministic local rule strategies; do not guess ML identities.
    if options['langs'] or any(s == 'lingua' or s.startswith('translate:') for s in options['strategies']) or options['mode'] == 'balanced':
        return None
    counter = options['counter']
    from .counters import HeuristicCounter, TiktokenCounter, HFCounter
    if type(counter) is HeuristicCounter:
        identity = ('heuristic-v1',)
    elif type(counter) is TiktokenCounter:
        identity = ('tiktoken', counter.name, counter._enc.name)
    elif type(counter) is HFCounter:
        # Per-loaded-tokenizer identity avoids aliasing different snapshots or
        # local tokenizer files that happen to use the same model name.
        if not hasattr(counter, '_result_cache_token'):
            counter._result_cache_token = uuid.uuid4().hex
        identity = counter._result_cache_token
    elif callable(getattr(counter, 'cache_key', None)):
        identity = counter.cache_key()
        hash(identity)
    else:
        return None
    data = {k:v for k,v in options.items() if k not in ('counter','translator')}
    encoded = json.dumps({'text': text, 'options': data, 'counter': identity, 'namespace': namespace}, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode()).digest()


def _counter_wire(counter):
    from .counters import HeuristicCounter, TiktokenCounter, HFCounter
    if type(counter) in (HeuristicCounter, TiktokenCounter):
        if type(counter) is HeuristicCounter: return ('spec','heuristic')
        return ('spec', counter.name)
    return ('object',counter)


def _worker(pipe, text, options, counter_wire):
    try:
        if os.name != 'nt': os.setsid()
        from .optimizer import _optimize
        from .counters import get_counter
        options['counter'] = get_counter(counter_wire[1]) if counter_wire[0] == 'spec' else counter_wire[1]
        result = _optimize(text, **options, _progress=pipe.send)
        pipe.send(('result', result))
    except BaseException as exc:
        try: pipe.send(('error', type(exc).__name__))
        except (OSError, EOFError): pass
    finally:
        pipe.close()


def _fallback(text, count, name, mode, reason, timed_out, max_tokens):
    from .optimizer import Candidate, OptimizeResult
    within_limit = max_tokens is None or (count is not None and count <= max_tokens)
    c = Candidate('original',text,count,0,False,within_limit,[reason])
    result = OptimizeResult(name,count,[c],c,timed_out=timed_out,mode=mode,warnings=[reason])
    if not within_limit:
        result.warnings.append('original fallback is not verified within max_tokens; do not send without checking')
    return result


def _with_timeout(text, options, seconds):
    deadline = time.monotonic() + seconds
    ctx = mp.get_context('spawn')  # never fork a live torch/OpenMP runtime
    receiver, sender = ctx.Pipe(duplex=False)
    options = dict(options)
    counter = options.pop('counter')
    proc = ctx.Process(target=_worker,args=(sender,text,options,_counter_wire(counter)),daemon=True)
    count, name = None, getattr(counter,'name','unknown (deadline before tokenization)')
    try:
        # Fail safely for non-picklable user adapters; never fall back to a
        # background thread whose inference would continue after the deadline.
        proc.start(); sender.close()
        while True:
            remaining = deadline-time.monotonic()
            if remaining <= 0 or not receiver.poll(max(0,remaining)):
                return _fallback(text,count,name,options['mode'],'optimization timeout; original returned',True,options['max_tokens'])
            message = receiver.recv()
            if message[0] == 'count': count,name = message[1:]
            elif message[0] == 'result': return message[1]
            elif message[0] == 'error':
                return _fallback(text,count,name,options['mode'],f'optimization worker failed ({message[1]}); original returned',False,options['max_tokens'])
    except (Exception,) as exc:
        return _fallback(text,count,name,options['mode'],f'optimization isolation failed ({type(exc).__name__}); original returned',False,options['max_tokens'])
    finally:
        sender.close(); receiver.close()
        if proc.pid is not None:
            if os.name != 'nt':
                try:
                    # Worker owns this group (setsid); never signal our group.
                    if os.getpgid(proc.pid) == proc.pid: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            if proc.is_alive(): proc.terminate()
            proc.join(0.2)
            if proc.is_alive(): proc.kill(); proc.join(0.2)
            if not proc.is_alive(): proc.close()


def optimize(text, *args, mode=None, timeout_seconds=None, cache=True,
             cache_namespace='', ml_min_tokens=1024, ml_min_saved_tokens=256, **kwargs):
    """Optimize with an original fallback and optional hard process deadline.

    Omitted mode chooses fast for auto/rules; explicit ML requests retain legacy
    full behavior. mode='fast' enforces cost gates even for explicit ML requests.
    timeout_seconds includes child startup and counting; unknown fallback counts
    are None, never fabricated. Startup/cleanup has small OS scheduling overhead.
    """
    started = time.monotonic()
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    from .optimizer import _optimize
    from .counters import get_counter
    bound = inspect.signature(_optimize).bind(text,*args,**kwargs)
    bound.apply_defaults(); options = dict(bound.arguments); options.pop('text'); options.pop('_progress')
    options['langs'] = list(options['langs']); options['strategies'] = list(options['strategies'])
    if mode is None:
        mode = 'full' if options['langs'] or any(s=='lingua' or s.startswith('translate:') for s in options['strategies']) else 'fast'
    if mode not in ('fast','balanced','full'): raise ValueError('mode must be fast, balanced or full')
    for value in (ml_min_tokens, ml_min_saved_tokens):
        if not isinstance(value,int) or value < 0: raise ValueError('ML token thresholds must be nonnegative integers')
    if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError('timeout_seconds must be finite and positive')
    if not isinstance(cache_namespace,str) or len(cache_namespace)>256: raise ValueError('cache_namespace must be a string of at most 256 characters')
    if options['translation_backend'] not in ('argos','nllb'): raise ValueError('translation_backend')
    if cache is not True and cache is not False and not isinstance(cache,OptimizationCache):
        raise ValueError('cache must be bool or OptimizationCache')
    options.update(mode=mode,ml_min_tokens=ml_min_tokens,ml_min_saved_tokens=ml_min_saved_tokens)
    # Validate user configuration before creating a worker, loading a tokenizer,
    # or looking in the result cache. Errors are not hidden as safe fallbacks.
    for key,low,high in (('min_savings',0,1),('min_preservation',0,1),('min_chrf',0,100),('lingua_rate',0,1)):
        v=options[key]
        if not math.isfinite(v) or not low<=v<=high or (key=='lingua_rate' and v==0): raise ValueError(key)
    limit=options['max_tokens']
    if limit is not None and (not isinstance(limit,int) or limit<0): raise ValueError('max_tokens')
    for strategy in options['strategies']:
        if strategy not in ('auto','logs','normalize','lingua') and not (strategy.startswith('translate:') and strategy.partition(':')[2]): raise ValueError('unknown strategy')
    # With a deadline, even default tokenizer construction belongs in the child.
    if options['counter'] is None and timeout_seconds is None: options['counter']=get_counter()
    store=(DEFAULT_CACHE if cache is True else cache if isinstance(cache,OptimizationCache) else None) if timeout_seconds is None else None
    key=None
    if store is not None:
        try: key=_cache_key(text,options,cache_namespace)
        except Exception: key=None
        if key is not None:
            hit=store.get(key)
            if hit is not None: hit.cache_hit=True; return hit
    if timeout_seconds is not None:
        remaining=timeout_seconds-(time.monotonic()-started)
        result=(_with_timeout(text,options,remaining) if remaining>0 else
                _fallback(text,None,getattr(options['counter'],'name','unknown'),mode,
                          'optimization timeout; original returned',True,limit))
    else:
        try:
            result=_optimize(text,**options)
        except (OSError,RuntimeError,TypeError) as exc:
            result=_fallback(text,None,getattr(options['counter'],'name','unknown'),mode,
                             f'optimization failed ({type(exc).__name__}); original returned',False,limit)
    if store is not None and key is not None and not result.timed_out and not result.warnings:
        store.put(key,result)
    return result
