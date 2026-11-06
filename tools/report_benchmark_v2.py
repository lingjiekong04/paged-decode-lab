"""Fixed-primary unroll128 report with convergence flags and observed GPU sharing."""
import json,statistics as st,math
from _common import ROOT
from paged_decode.benchmark_protocol import convergence_delta,CONVERGENCE_LIMIT


def main():
    rows=[];counts={};sharing=[]
    for geometry,expected in [('small',540),('large',540),('standard',1080)]:
        folder=ROOT/f'results/benchv2-isolated-{geometry}'
        d=json.loads((folder/'summary.json').read_text())
        assert len(d['rows'])==expected
        args=json.loads((folder/'arguments.json').read_text());pid=args['host_gpu_pid']
        visible=json.loads((folder/'environment.json').read_text())['cuda_visible_devices'].split(',')[0]
        for p in folder.glob('b*-l*-d*.json'):
            for snap in [point for before in json.loads(p.read_text())['gpu_snapshots'] for point in (before,before['after'])]:
                device=[line.split(',') for line in snap['devices'].splitlines() if line.strip()]
                uuid=next(v[1].strip() for v in device if (v[0].strip()==visible if visible.isdigit() else v[1].strip().startswith(visible)))
                for line in snap['apps'].splitlines():
                    values=[v.strip() for v in line.split(',')]
                    if len(values)>=2 and values[1]==uuid and int(values[0])!=pid:sharing.append({'geometry':geometry,'file':p.name,'pid':int(values[0])})
        counts[geometry]={'graph_records':len(d['rows']),'host_records':len(d['host_rows']),
            'oracle_passed':len(d['checks']),'matched_native_equal':sum(c['native_equal'] for c in d['checks'] if c['backend']=='triton_matched')}
        for key in sorted({(r['batch'],r['context'],r['dim'],r['backend']) for r in d['rows']}):
            b,n,dim,backend=key
            groups={u:[r['device_interval_us'] for r in d['rows'] if (r['batch'],r['context'],r['dim'],r['backend'],r['unroll'])==(*key,u)] for u in (1,8,32,128)}
            assert all(len(v)==5 for v in groups.values())
            med={str(u):st.median(v) for u,v in groups.items()}
            delta=convergence_delta(med['32'],med['128'])
            host=[r for r in d['host_rows'] if (r['batch'],r['context'],r['dim'],r['backend'])==key]
            check=next(c for c in d['checks'] if (c['batch'],c['context'],c['dim'],c['backend'])==key)
            rows.append({'geometry':geometry,'batch':b,'context':n,'dim':dim,'backend':backend,
                'median_us_by_unroll':med,'primary_us':med['128'],'primary_range':[min(groups[128]),max(groups[128])],
                'delta_32_128':delta,'converged_5pct':delta<=CONVERGENCE_LIMIT,
                'full_call_wall_us':st.median(st.median(r['allocation']) for r in host),
                'preallocated_call_wall_us':st.median(st.median(r['preallocated']) for r in host),
                'workspace_bytes':next(r['workspace_bytes'] for r in d['rows'] if (r['batch'],r['context'],r['dim'],r['backend'])==key),
                'max_abs_fp32':check['max_abs_fp32'],'native_equal':check['native_equal'],'plan':check['plan']})
    out=ROOT/'docs/benchmark-v2';out.mkdir(parents=True,exist_ok=True)
    data={'protocol':'fixed unroll128; 5 randomized rounds; 20 replays; 5pct 32-to128 convergence flag',
        'counts':counts,'rows':rows,'observed_foreign_gpu0_processes':sharing,
        'nonconverged_configs':sum(not r['converged_5pct'] for r in rows)}
    (out/'summary.json').write_bytes((json.dumps(data,indent=2)+'\n').encode())
    lines=['# 修正计时方法后的正式算子基准','',
        '36个形状：两档Qwen3.5的D256几何各9个，以及Hq32/Hkv8、D64/128的18个。三条路径为原生FlashAttention、固定1024分区TC split-K、正式triton_matched。每配置5轮，每轮随机交错后端和unroll，形状顺序也固定seed随机化。共2160条Graph区间及540条完整调用重复记录。','',
        '**主指标预先固定为Graph128区间除以128。** 每轮20次replay；同时记录Graph1/8/32，32与128中位数相差超过5%的配置标记为未收敛。该标记用于识别测量边界，不选择最快unroll，也不用于调优kernel。区间仍包含设备graph节点间隙，不称为纯kernel busy time。','',
        f"108个形状/后端配置中，未收敛{data['nonconverged_configs']}个；GPU0采样中发现其他进程{len(sharing)}条。",'',
        '| 几何 | B / KV / D | 原生 μs | 固定1024 μs | 对齐 μs | 原生/对齐 | 三条路径均收敛 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for geom in ('small','large','standard'):
        for b,n,dim in ([(1,512,256),(1,8192,256),(8,2048,256),(32,8192,256)] if geom!='standard' else [(1,512,64),(1,8192,128),(32,8192,128)]):
            v={r['backend']:r for r in rows if (r['geometry'],r['batch'],r['context'],r['dim'])==(geom,b,n,dim)}
            a,t,m=(v[k]['primary_us'] for k in ('flash_attn','triton_tc_splitk','triton_matched'))
            lines.append(f"| {geom} | {b}/{n}/{dim} | {a:.3f} | {t:.3f} | {m:.3f} | {a/m:.3f} | {'是' if all(r['converged_5pct'] for r in v.values()) else '否'} |")
    lines+=['','![Measured matrix](matrix.png)','','收益集中于本次D256小batch形状；较大batch存在退化，D64/128通用矩阵整体落后于原生。当前实现的数值协议对齐已验证，性能优化范围仍有限，不构成通用更快后端的结论。D64/128需要另做专门开销分析，不能沿用D256的收益解释。','',
        '完整矩阵、五次范围、每个unroll中位数、包含分配的同步wall time、预分配调用wall time、workspace及FP32误差均见summary.json。原生API自行管理输出/workspace，因此其workspace字段为空，不将未知大小写成零。','',
        '## 正确性与边界','',
        f"独立gather+FP32参考通过108次比较（原始atol=rtol=2e-2）；每个捕获图另检查输出。matched与原生torch.equal的形状数：small {counts['small']['matched_native_equal']}/9，large {counts['large']['matched_native_equal']}/9，standard {counts['standard']['matched_native_equal']}/18。随机算子逐位结果不替代正式模型回归。",'',
        'Qwen3.5正式入口上一阶段的161项测试、1024步固定logits和4096个B8生成token结果继续保留；本轮不修改kernel或模型执行路径，不重新宣称模型吞吐提升。','',
        '同一Graph重复读取固定KV/元数据，用于测量稳定算子吞吐；不是连续生成128个token。没有锁定GPU时钟，GPU快照保留每形状每轮前后的时钟、温度、功耗及进程。工具先验证所选GPU空闲，再建立容器/宿主PID映射，发现共享即停止。5轮范围是描述性波动，不是置信区间。对未收敛配置，应先复测再作性能结论。首次共享GPU试跑仅作诊断保留，见[PILOT.md](PILOT.md)。','',
        '独立kernel profiling和host提交限制的因果诊断见[短上下文报告](../short-overhead/RESULTS.md)。旧数据保留为不同计时协议的历史观测，不能跨协议直接拼接提速比例。','',
        '## 复现','',
        '```bash','source tools/env.sh','export CUDA_VISIBLE_DEVICES=0',
        '.venv35/bin/python tools/benchmark_operator_v2.py --geometry small --output results/new-v2-small',
        '# geometry可选large或standard；默认5轮、20次replay、固定四个unroll。',
        '.venv35/bin/python tools/report_benchmark_v2.py','```','',
        '报告生成器默认读取交付包内results/benchv2-isolated-small、large、standard三组记录。所有JIT、warmup、capture、正确性检查及GPU状态查询均在正式区间计时外。','']
    (out/'RESULTS.md').write_bytes(('\n'.join(lines)).encode())
    print(json.dumps({'configs':len(rows),'nonconverged':data['nonconverged_configs'],'sharing':len(sharing)}))


if __name__=='__main__':main()
