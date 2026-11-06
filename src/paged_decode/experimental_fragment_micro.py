"""Standalone QK/P/ordered-denominator/PV harness; not an engine backend.

The inline PTX candidate requires a separately audited MMA-v2 layout [1,4].
Never launch it on another layout based only on logical tensor shape.
"""
import triton
import triton.language as tl


@triton.jit
def fragment_micro(Q, K, V, Initial, Denom, PV, Tags, Prob, Lengths,
                   D: tl.constexpr, TILE: tl.constexpr, MODE: tl.constexpr,
                   ASM: tl.constexpr, CONSTRAINTS: tl.constexpr, PACK: tl.constexpr):
    r, d, t = tl.arange(0, 16), tl.arange(0, D), tl.arange(0, TILE)
    length = tl.load(Lengths)
    q = tl.load(Q + r[:, None]*D + d[None, :])
    k = tl.load(K + d[:, None]*TILE + t[None, :], t[None, :] < length, 0)
    v = tl.load(V + t[:, None]*D + d[None, :], t[:, None] < length, 0)
    scores = tl.where(t[None, :] < length, tl.dot(q, k) * (D ** -0.5), -float('inf'))
    maximum = tl.max(scores, 1)
    maximum = tl.where(length > 0, maximum, 0.)
    p = tl.exp2((scores - maximum[:, None]) * 1.4426950408889634)
    if MODE == 'map':
        tags = tl.inline_asm_elementwise(ASM, CONSTRAINTS, [p], dtype=tl.int32,
                                        is_pure=False, pack=PACK)
        tl.store(Tags + r[:, None]*TILE + t[None, :], tags)
        denom = tl.sum(p, 1)
    elif MODE == 'baseline':
        lanes = tl.arange(0, 4)
        blocks = tl.arange(0, TILE // 8)
        state = tl.load(Initial + r[:, None]*4 + lanes[None, :])
        even, odd = tl.split(tl.reshape(p, (16, TILE // 8, 4, 2)))
        for block in tl.static_range(TILE // 8):
            state = state + tl.sum(tl.where(blocks[None, :, None] == block, even, 0.), 1)
            state = state + tl.sum(tl.where(blocks[None, :, None] == block, odd, 0.), 1)
        s0 = tl.sum(tl.where(lanes[None, :] == 0, state, 0.), 1)
        s1 = tl.sum(tl.where(lanes[None, :] == 1, state, 0.), 1)
        s2 = tl.sum(tl.where(lanes[None, :] == 2, state, 0.), 1)
        s3 = tl.sum(tl.where(lanes[None, :] == 3, state, 0.), 1)
        denom = (s0+s2)+(s1+s3)
    elif MODE == 'shared':
        initial = tl.load(Initial + r[:, None]*4 + ((t[None, :] // 2) % 4))
        repeated = tl.inline_asm_elementwise(ASM, CONSTRAINTS, [p, initial], dtype=tl.float32,
                                            is_pure=False, pack=PACK)
        # Every column of one row holds the same already-ordered row sum.
        denom = tl.max(repeated, 1)
    else:
        tl.static_assert(False, 'unknown microkernel mode')
    tl.store(Denom+r, denom)
    if MODE == 'map':
        tl.store(Prob + r[:, None]*TILE + t[None, :], p)
    acc = tl.dot(p.to(tl.bfloat16), v)
    tl.store(PV + r[:, None]*D + d[None, :], acc)


@triton.jit
def fragment_lane_trace(Q, K, V, Initial, State, PV, Prob, Lengths,
                        D: tl.constexpr, TILE: tl.constexpr, MODE: tl.constexpr,
                        ASM: tl.constexpr, CONSTRAINTS: tl.constexpr, PACK: tl.constexpr):
    """Debug/recurrence harness: return all four states instead of merging them."""
    r, d, t = tl.arange(0,16), tl.arange(0,D), tl.arange(0,TILE)
    lanes=tl.arange(0,4);length=tl.load(Lengths)
    q=tl.load(Q+r[:,None]*D+d[None,:])
    k=tl.load(K+d[:,None]*TILE+t[None,:],t[None,:]<length,0)
    v=tl.load(V+t[:,None]*D+d[None,:],t[:,None]<length,0)
    scores=tl.where(t[None,:]<length,tl.dot(q,k)*(D**-0.5),-float('inf'))
    maximum=tl.where(length>0,tl.max(scores,1),0.)
    p=tl.exp2((scores-maximum[:,None])*1.4426950408889634)
    if MODE=='baseline':
        state=tl.load(Initial+r[:,None]*4+lanes[None,:])
        blocks=tl.arange(0,TILE//8)
        even,odd=tl.split(tl.reshape(p,(16,TILE//8,4,2)))
        for block in tl.static_range(TILE//8):
            state=state+tl.sum(tl.where(blocks[None,:,None]==block,even,0.),1)
            state=state+tl.sum(tl.where(blocks[None,:,None]==block,odd,0.),1)
        tl.store(State+r[:,None]*4+lanes[None,:],state)
    elif MODE=='shared':
        initial=tl.load(Initial+r[:,None]*4+((t[None,:]//2)%4))
        repeated=tl.inline_asm_elementwise(ASM,CONSTRAINTS,[p,initial],dtype=tl.float32,is_pure=False,pack=PACK)
        tl.store(State+r[:,None]*4+(t[None,:]//2)%4,repeated,(t[None,:]<8)&(t[None,:]%2==0))
    else:
        tl.static_assert(False,'unknown lane trace mode')
    tl.store(Prob+r[:,None]*TILE+t[None,:],p)
    acc=tl.dot(p.to(tl.bfloat16),v)
    tl.store(PV+r[:,None]*D+d[None,:],acc)
