"""Build the Chinese project document. Requires reportlab and a licensed CJK TTF."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon

ROOT = Path(__file__).resolve().parents[1]

def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--font', required=True, type=Path)
    a.add_argument('--output', type=Path, default=ROOT/'docs/Paged_Decode_Lab_项目说明.pdf')
    args = a.parse_args()
    if args.output.exists(): raise FileExistsError('Use a new output path; do not overwrite released PDFs')
    pdfmetrics.registerFont(TTFont('CJK', str(args.font)))
    blue=colors.HexColor('#133955'); green=colors.HexColor('#187e75'); gray=colors.HexColor('#526271')
    body=ParagraphStyle('body',fontName='CJK',fontSize=10.3,leading=17,wordWrap='CJK',textColor=blue,spaceAfter=10)
    small=ParagraphStyle('small',parent=body,fontSize=8.5,leading=13,spaceAfter=4)
    title=ParagraphStyle('title',parent=body,fontSize=25,leading=36,spaceAfter=22)
    h=ParagraphStyle('h',parent=body,fontSize=17,leading=25,spaceAfter=18)
    sub=ParagraphStyle('sub',parent=body,fontSize=12,leading=18,textColor=green,spaceBefore=8,spaceAfter=9)
    code=ParagraphStyle('code',fontName='Courier',fontSize=7.8,leading=12,spaceAfter=10,textColor=blue)
    story=[]
    def p(t,style=body):story.append(Paragraph(escape(t).replace('\n','<br/>'),style))
    def heading(t):p(t,h)
    def table(rows,widths):
        data=[[Paragraph(escape(str(c)),small) for c in r] for r in rows]
        t=Table(data,colWidths=widths,hAlign='LEFT',repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eff3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.8,green),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d5dfe4')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        story.append(t);story.append(Spacer(1,12))
    def nextpage():story.append(PageBreak())
    def commands(lines):
        for line in lines:p(line,code)
    def architecture():
        d=Drawing(500,230)
        boxes=[(4,159,144,53,'官方模型块 / KV 写入'),(177,159,145,53,'Query + table + lengths'),(351,159,144,53,'NHD 物理 KV 页'),(177,83,145,47,'Triton paged decode'),(177,8,145,47,'稳定 merge / BF16 输出')]
        for x,y,w,hh,label in boxes:
            d.add(Rect(x,y,w,hh,rx=5,ry=5,fillColor=colors.HexColor('#e9f1f4'),strokeColor=green))
            d.add(String(x+w/2,y+hh/2-4,label,fontName='CJK',fontSize=9.5,textAnchor='middle',fillColor=blue))
        for x1,y1,x2,y2 in [(148,184,177,184),(423,159,322,110),(249,159,249,130),(249,83,249,55)]:
            d.add(Line(x1,y1,x2,y2,strokeColor=green,strokeWidth=1.3))
        return d

    # 1 Cover + abstract
    story.append(Spacer(1,42));p('Paged Decode Lab',title)
    p('基于 Triton 的分页 Decode Attention\n实现、推理接入与性能边界分析',h)
    p('求职展示与 GitHub 开源候选说明',sub)
    p('整理日期：2026-11-06\n来源：本次远程只读采集快照；未重新运行 GPU 或模型。')
    story.append(Spacer(1,24));p('摘要',sub)
    p('项目研究单 token decode 如何访问分页 KV Cache、如何通过 split-K 提高并行度，以及如何在固定容量和 CUDA Graph 中保持数值与资源正确。实现基础、split-K、Tensor Core 和容量／LSE 对齐路径，并在固定 Qwen3.5-0.8B／9B 上进行历史真实模型回归。')
    p('最新 D256 实验通过减少流水线阶段降低 shared 占用，恢复驱动允许的双 CTA 驻留上限。固定长上下文完整算子约加速1.25倍，短上下文约退化2.44%，模型端到端收益很小。本文同时保留收益、失败和复现缺口，避免把局部算子结果泛化成工业服务能力。')
    table([['核心交付','范围'],['源码','自定义 kernel、运行时、官方模型适配、上游覆盖层'],['验证','历史 GPU／logits 证据；本轮 CPU、重建与统计重算'],['适用范围','单 GPU、文本、固定无 padding batch；学习与研究']], [95,405])
    nextpage()
    # 2 problem architecture
    heading('01  问题、目标与系统架构')
    p('自回归生成每次只有一个 query token，却要读取完整历史 KV。连续数组的地址假设不能直接应用于分页缓存；小 batch 又可能没有足够 CTA。项目把分页寻址、归约并行度、浮点计算顺序和执行开销放在同一条可验证路径中。')
    story.append(architecture());story.append(Spacer(1,10))
    table([['路径','上游承担','本项目承担'],['nano-vLLM','模型、权重、prefill、KV写入、调度及Graph框架','固定基线的decode后端、workspace及接入校验'],['Qwen3.5 hybrid','官方模型块、Gated DeltaNet、RoPE、norm、MLP','full attention分页执行、稳定状态、Graph与实验callback']], [95,200,205])
    p('两个路径并列存在。Qwen3.5 runner 不是把模型完整移植进 nano-vLLM，也不提供连续批处理、多卡通信、抢占或图像输入。不能把上游能力作为个人原创成果。')
    nextpage()
    # 3 algorithm
    heading('02  分页寻址与稳定归约')
    p('输入 Q 的形状为[B,Hq,D]；KV为[Npages,256,Hkv,D]；table和lengths为int32。GQA把多个query heads映射至同一个KV head。')
    commands(['page = table[b, t // 256]', 'slot = t % 256', 'kv_head = h // (Hq // Hkv)', 'offset = ((page * 256 + slot) * Hkv + kv_head) * D + d'])
    p('基础版由一个 CTA 处理序列中的一个query head。每次读取一块token，更新online softmax最大值m、分母l和加权和a：先用新最大值缩放历史状态，再累加当前块，保持FP32累加，最后转回BF16。零长度行返回零。')
    table([['方案','中间状态','合并与代价'],['基础版','m / l / acc','无额外merge，但小batch长KV的并行度有限'],['固定1024 split-K','每区m / l / acc','稳定重缩放后合并；增加并行也增加workspace和kernel开销'],['容量匹配split-KV','normalized partial + LSE','固定容量决定分区；协议不同，不能复用其他后端成绩']], [103,155,242])
    p('逻辑位置与物理页不连续；分区边界也未必与256-token页边界重合。所有有效访问必须按逻辑位置查表，并用长度mask保护无效表项和NaN尾部。空分区不贡献概率，全部为空时不能产生NaN。')
    p('主要源码：kernels.py实现基础／split／merge；runtime.py管理借用输出与预分配workspace；matched_splitkv.py和splitkv_plan.py实现容量与LSE协议。外部页表内容校验需显式进行，热路径的shape检查不能代替内容安全。')
    nextpage()
    # 4 numeric graph
    heading('03  数值一致性与 CUDA Graph')
    p('attention容差内误差不保证greedy token一致。真实激活诊断发现：BF16 partial已经相同，FP32 local LSE仍可能因log实现与舍入顺序而不同，最终改变BF16输出或接近的logits argmax。')
    table([['阶段','关键发现','处理'],['matched v1','容量／normalized partial已对齐，local LSE仍有差异','保留失败并定位log协议'],['matched v2','控制快速log指令及合并顺序','已记录用例恢复对齐；非任意输入保证'],['四状态微内核','预先缩放状态阻断首项FMA收缩','显式传递未缩放state和alpha，保留FMA'],['S2实验','数值协议不变，只手动调整stages','重跑同套分页、模型与Graph回归']], [95,205,200])
    p('Graph生命周期',sub)
    p('先固定batch、完整table宽度和最大容量；分配KV、输出、workspace及linear state；在capture之前编译、预热和布局预检。捕获完整文本decode forward，包含KV写入、linear state更新、lm_head。prefill和token选择仍在Graph外。')
    p('replay只更新稳定地址中的内容，不读取GPU标量决定Python分支。回归覆盖零上下文、无效页、NaN padding、内容修改、页映射翻转、跨页增长和多次replay；输出、partials和LSE按形状与dtype检查后比较字节。')
    p('最新shared_s2只在独立实验入口中启用，正式CLI的默认后端仍是FlashAttention；不支持的sm80／D256／布局配置会报错，避免静默回退掩盖实验。')
    nextpage()
    # 5 experiment
    heading('04  实验设置与验证层级')
    table([['项目','设置与限制'],['硬件','A100-SXM4-80GB，sm80，108 SM；两模型分用两张GPU，CPU共享'],['软件','Torch2.6.0 / Triton3.2.0 / FA2.7.4.post1 / Transformers5.3.0；历史CUDA12.4'],['模型','Qwen3.5-0.8B与9B，revision固定；full attention D256；linear attention官方Torch fallback'],['算子矩阵','真实9B第3层第16步激活；上下文528/2064；36splits；两种unroll、五轮随机顺序，共480条'],['模型矩阵','batch1；输入512/2048；输出64；4后端×eager/Graph×5轮×2模型，共160条'],['数值','72项分页输出；每模型1920完整logits步骤和60次自由生成；性能输入各另1024步骤']], [100,400])
    p('计时区间',sub)
    p('算子采用Graph unroll32/128、20次replay、五轮重复，主表取unroll128；包含partition+merge，workspace提前分配。固定真实张量反复访问为hot-cache诊断，不含模型或KV写入，不等同于kernel busy time。缩减grid只用于局部分区诊断。')
    p('模型wall调用没有逐步CUDA Event；另一次生成采集63个decode事件区间。JIT、预热与capture不计入。三组自定义workspace在native等全部配置中保留，因此峰值allocated显存不能解释为原生最小显存。')
    p('验证层级依次为CPU算法／协议、GPU算子、Graph replay、完整logits、独立greedy和性能；没有训练、模型效果排行榜或真实业务验收。本次公开整理只执行无GPU验证，历史GPU结果单独标注。')
    nextpage()
    # 6 results
    heading('05  资源变化、收益与退化')
    table([['配置','dynamic B','static B','regs','spill字段','CTA/SM上限'],['original','79232','0','255','16','2'],['shared_s3','79232','4352','255','8','1'],['shared_s2','77952','4352','192','0','2']], [90,86,74,58,85,107])
    p('S2 dynamic shared减少1280B。每CTA含驱动预留共83328B，两CTA为166656B，小于每SM167936B。驱动查询支持双CTA驻留；这不是achieved occupancy实测。寄存器和spill也随编译调度变化，不能把全部性能变化仅归因于容量。')
    table([['上下文','完整S3 us','完整S2 us','同轮S3/S2中位数'],['528','10.7780','11.0369','0.9762（S2退化约2.44%）'],['2064','16.6610','13.3474','1.2483（S2获益）']], [85,113,113,189])
    story.append(Image(str(ROOT/'docs/assets/pipeline.png'),width=500,height=200))
    p('图：同一真实张量的partition grid对照，五轮中位数及min/max，非置信区间。完整算子比值取同轮配对，不是两个独立中位数之比。Graph模型wall比值仅0.9984–1.0015，不能把算子约1.25倍写成模型吞吐提升。',small)
    nextpage()
    # 7 iteration
    heading('06  迭代过程与未采用方案')
    table([['阶段','相对变化／解决的问题','验证与最终状态'],['基础 → split-K','增加KV分区并行，减少单CTA长循环','FP32参考；历史模型仍有argmax分歧，显式保留'],['TC ordered → TC split-K','引入Tensor Core及有序分母','v2-gather因API缺失编译失败；固定1024版9B长输入仍失败'],['matched v1 → v2','容量规划、normalized partial/LSE及快速log','真实激活诊断＋logits；v2正式CLI，v1快照保留'],['warps / late V / IR','检查并行度、临时量、布局与存活','独立实验；部分计时不收敛，未全部采用'],['四状态 → D256 S3','保序归约微内核接入分页partition','槽位／FMA／数值通过；长上下文因资源门槛退化'],['D256 S3 → S2','stages3→2降低dynamic shared','同套回归，长输入改善、短输入退化，保留显式配置'],['本次候选整理','来源映射、派生摘要、重建、README和PDF','发布工具与文档变化；核心算法变化为无']], [110,195,195])
    p('完整迭代文档按模型／适配器、输入、工具执行层、核心算法、评测及发布分别记录。Qwen3到Qwen3.5是模型适配路线变化，不能把不同模型或不同测试集成绩拼成升级曲线。发布包修订不是算法版本升级。')
    p('没有项目Git历史可供恢复，因此不虚构tag、commit或迭代日期。源码快照只用于审计，不是可独立运行的全部旧版本。早期负结果按远程报告复核，本次没有重跑全部历史实验。')
    p('Nsight多种探测超时／未捕获有效kernel计数器，不能写成已证实权限拒绝。旧控制路径新编译.text不完全相同，原因未定；禁止跨campaign绝对时延归因。')
    nextpage()
    # 8 demo repro
    heading('07  演示与分层复现')
    p('演示先从逻辑页映射和NumPy reference开始，再解释kernel、split合并、Graph和资源门槛。没有GPU时明确展示历史证据，不用模拟输出冒充真实推理。全部命令从候选根目录执行。')
    p('Windows PowerShell：无GPU示例',sub)
    commands(['python -m venv .venv-cpu', r'.\.venv-cpu\Scripts\python.exe -m pip install -e ".[test]"', r'.\.venv-cpu\Scripts\python.exe examples/cpu_attention.py', r'.\.venv-cpu\Scripts\python.exe release/rebuild_upstream.py', r'.\.venv-cpu\Scripts\python.exe release/check_cpu.py'])
    p('完整可复制命令及每步退出码检查见README。基线重建默认联网；已有固定ZIP时用--archive离线。候选不重复分发完整上游源码。')
    p('Linux终端：已准备GPU依赖和模型后',sub)
    commands(['source tools/env.sh', 'python tools/qwen35.py --model models/Qwen3.5-0.8B \\', '  --backend triton_matched --mode graph --output results/demo-new', 'python tools/resident_experiment.py --task correctness \\', '  --output results/correct-new'])
    p('下载模型必须先于source env.sh，因为该脚本开启离线模式。模型revision及权重SHA256在configs/models.json，安装版本在requirements-qwen35.txt。S2实验另用resident_experiment或benchmark_fragment_resident_models，不是qwen35的后端选项。')
    p('输出路径必须不存在。真实GPU命令由使用者主动执行；本轮未安装系统依赖、运行模型、训练或启动GPU任务。更完整的Linux安装、两模型和五轮性能命令见docs/REPRODUCE.md。')
    nextpage()
    # 9 provenance limitations
    heading('08  来源、公开范围与检查记录')
    p('远程为事实来源：只读盘点5857文件，已扫描文件合计2438308723B；模型、环境、缓存等未遍历，因此不知道项目总大小。按白名单采集725文件、10902819B，逐文件SHA256通过，与旧本地同路径全部一致。')
    p('公开版保留原字节源码、必要快照、配置、测试和上游覆盖层；不公开原日志、UUID/PID、服务器路径、生成响应、权重或真实KV大张量。派生摘要通过字段白名单提取，记录原SHA、提取规则及公开文件SHA；不是原始证据的替身。')
    table([['检查层','本次结论'],['原字节与版本','私有审查包及725文件核验；核心源码保持字节'],['上游与CPU','固定ZIP离线重建26文件；176项CPU检查通过'],['统计','480条算子和160条模型公开记录重算，与远程报告一致'],['GPU／模型','本轮未运行，使用有来源的历史结果'],['新环境／业务','未安装完整新GPU环境，无真实业务／生产验收']], [112,388])
    p('重要缺口',sub)
    p('原真实张量算子计时需要未公开KV快照，旧采集命令还依赖历史输入记录；不能宣称外部读者可直接完全复跑。可以离线重算统计，并在自己环境重新采集新实验。没有新的硬件计数器、新模型下载可用性检查或当前GitHub URL。')
    p('首轮CPU检查遇到Windows临时目录权限问题，修正的是新增release检查脚本；最终176项通过。继承源码的空白、换行及补丁上下文没有为消除工具提示而修改。PDF和最终ZIP检查详见docs/VALIDATION.md及私有审查记录。')
    nextpage()
    # 10 references and contribution
    heading('09  贡献归属、参考资料与展示要点')
    p('贡献与许可',sub)
    p('新增kernel、runtime、后端集成和实验工具使用AI辅助完成，延用原项目MIT许可。nano-vLLM保留MIT版权；Qwen模型及Transformers模型块遵循各自上游许可。未复制分发FlashInfer代码，不把官方linear attention、prefill或调度能力归为自研。')
    p('参考资料（上游固定来源与学习资料）',sub)
    refs=[
      'nano-vLLM: https://github.com/GeeeekExplorer/nano-vllm ; 固定提交 bb823b3e06983d71485a8e1f23715ebd87d98ef8。',
      'Triton fused attention tutorial: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html 。',
      'FlashAttention论文: https://arxiv.org/abs/2205.14135 。',
      'FlashAttention源码参考版本: v2.7.4.post1；softmax.h、utils.h、flash_api.cpp；BSD-3-Clause。',
      'FlashInfer: https://github.com/flashinfer-ai/flashinfer 。用于理解工程取舍，不随项目分发。',
      'Qwen官方仓库: https://huggingface.co/Qwen/Qwen3.5-0.8B 与 https://huggingface.co/Qwen/Qwen3.5-9B 。revision和权重摘要见configs/models.json。',
      'Transformers官方Qwen3.5模型块: transformers/models/qwen3_5/modeling_qwen3_5.py，5.3.0，Apache-2.0。',
      '选题流程参考: https://github.com/LiuMengxuan04/shushu-internship-tool 。未复制其脚本或虚构评分。']
    for r in refs:p(r,small)
    p('面试应能解释',sub)
    p('逻辑token如何映射到物理KV页；split-K何时增加并行、何时多出归约成本；为什么log与FMA改变数值协议；Graph需要稳定哪些地址；驱动驻留上限与实际occupancy有什么区别；为什么attention算子加速没有等比例改善端到端。')
    p('GitHub仓库由用户手动创建和上传。本次没有自动commit、push或发布；真实姓名、邮箱和仓库地址未猜测。上传工具保留现有Git历史和修改，检查失败时停止，不使用reset --hard或force push。')

    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#d1dee5'));canvas.line(42,38,553,38)
        canvas.setFont('CJK',8);canvas.setFillColor(gray)
        canvas.drawString(42,24,'Paged Decode Lab | 公开候选说明 | 历史实验与本轮检查分开记录')
        canvas.drawRightString(553,24,str(doc.page));canvas.restoreState()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(args.output),pagesize=(595.28,841.89),leftMargin=42,rightMargin=42,topMargin=42,bottomMargin=52,
                          title='Paged Decode Lab：分页Decode Attention实现与性能边界',author='Paged Decode Lab contributors')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(args.output)

if __name__=='__main__':main()
