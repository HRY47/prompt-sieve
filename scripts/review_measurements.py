"""Reproduce local review measurements without APIs or model downloads.

Run with PYTHONPATH=src if tokopt is not installed. Output contains generated
fixtures only, never user prompts, environment variables or credentials.
"""
from __future__ import annotations

import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import tracemalloc

from tokopt.bench import CommandRunner, run_bench, summarize, results_jsonl
from tokopt.counters import HeuristicCounter, get_counter
from tokopt.optimizer import optimize


def peak_rss_bytes():
    if os.name == "nt":
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not get_memory(get_process(), ctypes.byref(counters), counters.cb):
            return None
        return counters.PeakWorkingSetSize
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def child(n, transform):
    counter = HeuristicCounter()
    text = "\n".join(f"tests/test_api.py::test_case_{i} PASSED [ 50%]" for i in range(n))
    text += "\nERROR: connection refused at /app/db.py:42\nFAILED tests/test_api.py::test_connect\n"
    before = peak_rss_bytes()
    start_cpu, start = time.process_time(), time.perf_counter()
    tracemalloc.start()
    if transform:
        res = optimize(text, counter=counter, strategies=["logs"])
        original_n, optimized_n, strategy = res.original_tokens, res.best.tokens, res.best.name
        optimized_bytes = len(res.best.text.encode())
    else:
        original_n = optimized_n = counter.count(text)
        strategy = "count_only"
        optimized_bytes = len(text.encode())
    elapsed, cpu = time.perf_counter() - start, time.process_time() - start_cpu
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dict(lines=n+2, input_bytes=len(text.encode()), optimized_bytes=optimized_bytes,
                byte_compression_ratio=optimized_bytes/len(text.encode()), counter=counter.name,
                original_tokens_estimated=original_n, optimized_tokens_estimated=optimized_n,
                savings_estimated=1-optimized_n/original_n, strategy=strategy,
                wall_seconds=elapsed, cpu_seconds=cpu, python_allocation_peak_bytes=python_peak,
                process_peak_working_set_bytes=peak_rss_bytes(), process_peak_before_bytes=before,
                gpu_vram_bytes=None, target_model_inference=False)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        print(json.dumps(child(int(sys.argv[2]), sys.argv[3] == "optimize")))
        return
    out = Path("review"); out.mkdir(exist_ok=True)
    if "--benchmark-only" in sys.argv:
        measurements = json.loads((out / "measurements.json").read_text())["log_measurements"]
    else:
        measurements = []
        for n in (500, 10000, 100000):
            for mode in ("count", "optimize"):
                p = subprocess.run([sys.executable, __file__, "--child", str(n), mode], capture_output=True, text=True, check=True)
                measurements.append(json.loads(p.stdout))
    with tempfile.TemporaryDirectory() as temp:
        model = Path(temp) / "model.py"
        model.write_text("""import sys
p = sys.stdin.read()
noisy = 'PASSED' in p
if 'BOTH_WRONG' in p or 'AGREEMENT_ONLY' in p: print('41')
elif 'BETTER' in p: print('41' if noisy else '42')
elif 'WORSE' in p: print('42' if noisy else '41')
elif 'FORMAT' in p: print('Answer: 42.' if noisy else '42')
else: print('42')
""", encoding="utf-8")
        cases = []
        for name in ("CORRECT", "BOTH_WRONG", "BETTER", "WORSE", "FORMAT", "AGREEMENT_ONLY"):
            prompt = "\n".join(f"tests/a.py::test_{i} PASSED [ 50%]" for i in range(100)) + "\nERROR: preserve " + name
            cases.append(dict(id=name, prompt=prompt, question="What is six times seven?", expected="42", match="same_as_original" if name == "AGREEMENT_ONLY" else "contains"))
        bench_start = time.perf_counter()
        results = run_bench(cases, CommandRunner(f'"{sys.executable}" "{model}"'), {"counter": HeuristicCounter()})
        bench_seconds = time.perf_counter() - bench_start
    versions = {}
    for pkg in ("pip", "pytest", "pluggy", "packaging", "iniconfig", "pygments", "colorama", "setuptools", "tiktoken", "openai", "argostranslate", "llmlingua", "torch", "transformers"):
        try: versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError: versions[pkg] = None
    report = dict(python=sys.version, platform=platform.platform(), dependencies=versions,
                  log_measurements=measurements, benchmark=summarize(results),
                  benchmark_total_seconds=bench_seconds,
                  benchmark_backend="deterministic local Python command, not an LLM",
                  exact_token_measurements_available=False,
                  optimization_model_tokens=0, optimization_api_requests=0,
                  timing_includes_tracemalloc_overhead=True,
                  measured_target_model_memory_savings=None)
    (out / "measurements.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "local-benchmark.jsonl").write_text(results_jsonl(results, include_answers=True), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
