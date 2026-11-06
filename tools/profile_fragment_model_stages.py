"""Diagnostic Graph ranges and isolated operators using real D256 model tensors.

No production edits. External event nodes are deliberate instrumentation; their
overhead is measured against an otherwise identical uninstrumented Graph.
"""
import argparse
from contextlib import contextmanager
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import threading
import time
import types

from _common import ROOT, fingerprint, model_identity, write_json
from fragment_d256_common import prepare, complete, strict_equal


class GraphRanges:
    """PyTorch 2.6 events plus CUDA's explicit external record-node API.

    cudaEventRecordExternal=1 is valid only in capture. Event handles are
    initialized before capture and retained for the entire Graph lifetime.
    https://docs.nvidia.com/cuda/archive/12.4.1/pdf/CUDA_Runtime_API.pdf
    """
    def __init__(self, torch, labels):
        self.torch = torch
        self.enabled = False
        self.events = {k: (torch.cuda.Event(enable_timing=True),
                           torch.cuda.Event(enable_timing=True)) for k in labels}
        libraries = sorted(Path('/proc/self/maps').read_text().splitlines())
        candidates = {line.split()[-1] for line in libraries if 'libcudart.so' in line}
        if len(candidates) != 1:
            raise RuntimeError(f'need one loaded CUDA runtime, found {candidates}')
        self.library = ctypes.CDLL(next(iter(candidates)))
        self.record_external = self.library.cudaEventRecordWithFlags
        self.record_external.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
        self.record_external.restype = ctypes.c_int
        for pair in self.events.values():
            for event in pair:
                event.record()
        torch.cuda.synchronize()

    def record(self, event):
        if self.torch.cuda.is_current_stream_capturing():
            code = self.record_external(event.cuda_event,
                                       self.torch.cuda.current_stream().cuda_stream, 1)
            if code:
                raise RuntimeError(f'cudaEventRecordWithFlags failed: {code}')
        else:
            event.record()

    @contextmanager
    def span(self, label, active=True):
        measured = self.enabled and active
        if measured:
            self.record(self.events[label][0])
        try:
            yield
        finally:
            if measured:
                self.record(self.events[label][1])

    def values(self):
        return {k: a.elapsed_time(b) for k, (a, b) in self.events.items()}


def query(fields, kind='gpu'):
    return subprocess.check_output(['nvidia-smi', f'--query-{kind}='+fields,
                                    '--format=csv,noheader'], text=True, timeout=20)


class ExclusiveGPU:
    def __init__(self, output):
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        self.uuid = next(u for row in query('index,uuid').splitlines()
                         for i, u in [list(map(str.strip, row.split(',')))]
                         if i == visible or u == visible)
        self.output = output
        if self.pids():
            raise RuntimeError('selected GPU is not idle; no process will be stopped')
        import torch
        self.allocation = torch.empty(1, device='cuda')
        torch.cuda.synchronize()
        self.own = self.pids()
        if len(self.own) != 1:
            raise RuntimeError('cannot establish exclusive host GPU PID')
        self.stop = threading.Event()
        self.rows, self.violations = [], []
        self.worker = threading.Thread(target=self.monitor, daemon=True)
        self.worker.start()

    def pids(self):
        return {int(p) for row in query('pid,gpu_uuid', 'compute-apps').splitlines()
                for p, u in [list(map(str.strip, row.split(',')))] if u == self.uuid}

    def monitor(self):
        while not self.stop.wait(1):
            try:
                found = self.pids()
                self.rows.append({'time': time.monotonic(), 'pids': sorted(found)})
                if found != self.own:
                    self.violations.append({'pids': sorted(found)})
            except Exception as exc:
                self.violations.append({'error': repr(exc)})

    def guard(self):
        if self.pids() != self.own or self.violations:
            raise RuntimeError('GPU ownership changed: campaign invalid')
        return query('index,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw')

    def close(self):
        self.stop.set()
        self.worker.join(timeout=25)
        write_json(self.output/'gpu-monitor.json',
                   {'rows': self.rows, 'violations': self.violations})


def event_selftest(torch):
    timer = GraphRanges(torch, ['sleep'])
    timer.enabled = True
    results = []
    for cycles in (10000, 1000000):
        x = torch.zeros(1, device='cuda')
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with timer.span('sleep'):
                torch.cuda._sleep(cycles)
                x.add_(1)
        readings = []
        for _ in range(3):
            graph.replay()
            torch.cuda.synchronize()
            readings.append(timer.values()['sleep'])
        assert x.item() == 3, 'selftest must execute three replays'
        assert all(v > 0 for v in readings)
        results.append({'cycles': cycles, 'replay_ms': readings, 'value': x.item()})
        del graph
    assert statistics.median(results[1]['replay_ms']) > 3*statistics.median(results[0]['replay_ms'])
    return results


def run(args, torch, gpu):
    from dataclasses import asdict
    from transformers import AttentionInterface
    from paged_decode.hybrid import HybridRunner, _attention
    from paged_decode.matched_splitkv import allocate_matched_resources, matched_decode
    from warp_ablation_common import partition as original_partition, merge
    from fragment_d256_common import partition as shared_partition
    from flash_attn import flash_attn_with_kvcache

    write_json(args.output/'environment.json', fingerprint())
    write_json(args.output/'event-selftest.json', event_selftest(torch))
    identity = model_identity(args.model)
    runner = HybridRunner(args.model, backend='flash_attn', mode='eager', batch=1, max_context=2112)
    full_layers = list(runner.cache.pages)
    if runner.spec.head_dim != 256:
        raise ValueError('this diagnostic targets the audited D256 geometry')
    labels = ['forward'] + [f'{kind}.{i}' for i in full_layers for kind in ('module', 'operator')]
    timer = GraphRanges(torch, labels)
    qbuffer = torch.empty((1, runner.spec.num_heads, 256), device='cuda', dtype=torch.bfloat16)
    resources = {i: allocate_matched_resources(runner.spec, runner.device) for i in full_layers}
    for i, (k, v) in runner.cache.pages.items():
        prepare(qbuffer, k, v, runner.cache.tables, runner.cache.lengths, resources[i], 'shared')
    active, snapshots, snapshot_mode = ['native'], {}, [None]

    def attention(module, query, key, value, mask, scaling, dropout=0., **kwargs):
        if runner.cache.is_prefill:
            return _attention(module, query, key, value, mask, scaling, dropout, **kwargs)
        i = module.layer_idx
        q = query.squeeze(2)
        k, v = runner.cache.pages[i]
        if snapshot_mode[0] is not None:
            snapshots[(snapshot_mode[0], i)] = (q.detach().clone(), k.clone(), v.clone(),
                                               runner.cache.tables.clone(), runner.cache.lengths.clone())
        with timer.span(f'operator.{i}'):
            if active[0] == 'native':
                return _attention(module, query, key, value, mask, scaling, dropout, **kwargs)
            if scaling != 256**-.5:
                raise ValueError('unsupported scale')
            qbuffer.copy_(q)
            r = resources[i]
            out = (matched_decode(qbuffer, k, v, runner.cache.tables, runner.cache.lengths,
                                  resources=r, scale=scaling) if active[0] == 'original' else
                   complete(qbuffer, k, v, runner.cache.tables, runner.cache.lengths, r, 'shared'))
            return out.unsqueeze(1), None

    name = 'paged_decode_lab_stage_profile'
    AttentionInterface.register(name, attention)
    hooks = []
    for i in full_layers:
        module = runner.text.layers[i].self_attn
        module.config._attn_implementation = name
        def before(module, inputs, index=i):
            if timer.enabled and not runner.cache.is_prefill:
                timer.record(timer.events[f'module.{index}'][0])
        def after(module, inputs, output, index=i):
            if timer.enabled and not runner.cache.is_prefill:
                timer.record(timer.events[f'module.{index}'][1])
        hooks += [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
    original_forward = runner._forward_impl
    def profiled_forward(self, tokens, positions):
        with timer.span('forward', not self.cache.is_prefill):
            return original_forward(tokens, positions)
    runner._forward_impl = types.MethodType(profiled_forward, runner)

    rng = random.Random(20261105)
    inputs = {n: [rng.choices(range(100, 30000), k=n)] for n in (512, 2048)}
    write_json(args.output/'inputs.json', inputs)
    sources = [Path(__file__), ROOT/'tools/fragment_d256_common.py', ROOT/'tools/warp_ablation_common.py',
               ROOT/'tools/fragment_partition_asm_d256.py', ROOT/'tools/fragment_asm_d256.py',
               ROOT/'src/paged_decode/experimental_fragment_partition.py',
               ROOT/'src/paged_decode/matched_splitkv.py', ROOT/'src/paged_decode/hybrid.py']
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for p in sources:
        dest = args.output/'sources'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    write_json(args.output/'arguments.json', {'model': identity, 'gpu_uuid': gpu.uuid,
        'host_pids': sorted(gpu.own), 'source_sha256': hashes, 'batch': 1, 'max_context': 2112,
        'repeats': args.repeats, 'decode_steps': args.steps, 'description': runner.description(),
        'plan': asdict(next(iter(resources.values())).plan), 'full_layers': full_layers,
        'scope': 'Graph-only diagnostic; per-step synchronization and logit reads outside intervals. '
                 'External event nodes perturb execution; matched plain runs quantify that overhead. '
                 'No stage estimate is a primary uninstrumented end-to-end performance result.'})
    gold = {n: runner.generate(inputs[n], args.steps+1, observe=True) for n in inputs}
    for n in inputs:
        write_json(args.output/f'reference-n{n}.json', {'token_ids': gold[n]['token_ids']})
    outer = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
             for _ in range(args.steps)]

    def configure(backend, instrumented):
        active[0] = backend
        runner.graph = None
        for attr in ('graph_logits', 'graph_tokens', 'graph_position'):
            if hasattr(runner, attr):
                delattr(runner, attr)
        timer.enabled = instrumented
        gc.collect()
        torch.cuda.empty_cache()
        runner._capture()

    @torch.inference_mode()
    def sequence(n, collect=False, snapshot=False):
        runner.cache.reset()
        logits = runner._forward(torch.tensor(inputs[n], device='cuda'), torch.arange(n, device='cuda'))
        assert strict_equal(torch, logits.cpu(), gold[n]['logits'][0]), 'prefill logits differ'
        runner.cache.prefilled, runner.cache.is_prefill = True, False
        tokens = torch.tensor(gold[n]['token_ids'][0], device='cuda', dtype=torch.long)
        observations = []
        for step in range(1, args.steps+1):
            if snapshot and step == args.steps:
                # Python callbacks execute in eager, not during replay.
                snapshot_mode[0] = n
                eager = runner._forward(tokens[step-1:step].view(1, 1),
                                        torch.tensor([n+step-1], device='cuda'))
                snapshot_mode[0] = None
                assert strict_equal(torch, eager.cpu(), gold[n]['logits'][step])
                continue
            start, end = outer[step-1]
            torch.cuda.synchronize()
            start.record()
            runner.graph_tokens.copy_(tokens[step-1:step].view(1, 1))
            runner.graph_position.fill_(n+step-1)
            runner.graph.replay()
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end)
            stages = timer.values() if timer.enabled else None
            assert strict_equal(torch, runner.graph_logits.cpu(), gold[n]['logits'][step]), \
                f'logits diverged at {active[0]} n={n} step={step}'
            if collect:
                observations.append({'step': step, 'context': n+step, 'outer_ms': elapsed,
                                     'stages_ms': stages, 'logits_bytes_equal': True})
        return observations

    for n in inputs:
        configure('native', False)
        sequence(n, snapshot=True)
    rows = []
    configs = [(b, flag, n) for b in ('native', 'original', 'shared')
               for flag in (False, True) for n in inputs]
    for repeat in range(args.repeats):
        order = configs.copy()
        rng.shuffle(order)
        for ordinal, (backend, flag, n) in enumerate(order):
            before = gpu.guard()
            configure(backend, flag)
            sequence(n)
            observed = sequence(n, collect=True)
            after = gpu.guard()
            row = {'backend': backend, 'instrumented': flag, 'input_tokens': n, 'repeat': repeat,
                   'order': ordinal, 'steps': observed, 'before': before, 'after': after}
            rows.append(row)
            write_json(args.output/f'r{repeat}-{backend}-tags{int(flag)}-n{n}.json', row)
            print('STAGES', repeat, backend, flag, n,
                  statistics.median(s['outer_ms'] for s in observed), flush=True)
    timer.enabled = False
    runner.graph = None
    gc.collect()
    torch.cuda.synchronize()
    micro = []
    metadata = []
    for (n, i), (q, k, v, table, lengths) in sorted(snapshots.items()):
        gpu.guard()
        native = lambda: flash_attn_with_kvcache(q.unsqueeze(1), k, v, cache_seqlens=lengths,
                            block_table=table, softmax_scale=256**-.5, causal=True).squeeze(1)
        reference = native().clone()
        funcs = {'native': {'full': native}}
        rs = {}
        for b, variant in (('original', 'baseline'), ('shared', 'shared')):
            r = allocate_matched_resources(runner.spec, runner.device)
            rs[b] = r
            prepare(q, k, v, table, lengths, r, variant)
            if b == 'original':
                whole = lambda r=r: matched_decode(q, k, v, table, lengths, resources=r)
                part = lambda r=r: original_partition(q, k, v, table, lengths, r, 4)
            else:
                whole = lambda r=r: complete(q, k, v, table, lengths, r, 'shared')
                part = lambda r=r: shared_partition(q, k, v, table, lengths, r, 'shared')
            candidate = whole().clone()
            if not strict_equal(torch, candidate, reference):
                raise RuntimeError(f'real-tensor native byte gate failed at n={n}, layer={i}, {b}')
            funcs[b] = {'full': whole, 'partition': part, 'merge': lambda r=r: merge(r)}
        meta = {'input_tokens': n, 'layer': i, 'context': int(lengths.item()), 'q_shape': list(q.shape),
                'kv_shape': list(k.shape), 'q_stride': list(q.stride()),
                'all_outputs_bytes_equal_native': True,
                'tensor_sha256': {name: hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                                  for name, t in zip(('q', 'k', 'v', 'table', 'lengths'), (q,k,v,table,lengths))}}
        metadata.append(meta)
        graphs = {}
        for b, stages in funcs.items():
            for stage, fn in stages.items():
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(args.unroll):
                        fn()
                graphs[(b, stage)] = graph
        pairs = list(graphs)
        for repeat in range(args.repeats):
            rng.shuffle(pairs)
            for b, stage in pairs:
                graph = graphs[(b, stage)]
                for _ in range(3):
                    graph.replay()
                start, end = outer[0]
                torch.cuda.synchronize()
                start.record()
                for _ in range(args.replays):
                    graph.replay()
                end.record()
                end.synchronize()
                micro.append({'input_tokens': n, 'context': n+args.steps, 'layer': i, 'backend': b,
                    'stage': stage, 'repeat': repeat, 'us_per_call': start.elapsed_time(end)*1000/(args.unroll*args.replays),
                    'unroll': args.unroll, 'replays': args.replays})
        del graphs, rs
        gpu.guard()
        print('MICRO', n, i, meta['q_shape'], flush=True)
    write_json(args.output/'real-tensor-metadata.json', metadata)
    write_json(args.output/'micro.json', micro)
    write_json(args.output/'summary.json', {'status': 'passed', 'model': identity, 'rows': rows,
               'micro': micro, 'metadata': metadata, 'production_changed': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--unroll', type=int, default=64)
    parser.add_argument('--replays', type=int, default=20)
    args = parser.parse_args()
    if min(args.repeats, args.steps, args.unroll, args.replays) < 1 or args.steps > 63:
        parser.error('positive counts required, with at most 63 decode steps')
    args.output.mkdir(parents=True, exist_ok=False)
    gpu = ExclusiveGPU(args.output)
    try:
        import torch
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.inference_mode():
            run(args, torch, gpu)
    finally:
        gpu.close()


if __name__ == '__main__':
    main()
