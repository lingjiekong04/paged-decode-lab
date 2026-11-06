"""Explicit experimental MMA-v2 [1,4] register mapping, gated by a GPU probe.

Not a portable Triton layout API. Only 16x{128,256}, four warps, sm80 is audited.
"""


def coordinates(thread, slot, tile):
    if tile not in (128, 256) or not 0 <= thread < 128 or not 0 <= slot < tile // 8:
        raise ValueError('unsupported MMA fragment geometry')
    warp, lane = divmod(thread, 32)
    repeat, inner = divmod(slot, 4)
    return lane // 4 + (inner // 2) * 8, warp * 8 + repeat * 32 + (lane % 4) * 2 + inner % 2


def marker_asm(tile):
    if tile not in (128, 256):
        raise ValueError('tile must be 128 or 256')
    pack = tile // 8
    code = ['{', '.reg .u32 tid;', 'mov.u32 tid, %tid.x;']
    code += [f'mad.lo.u32 ${i}, tid, {pack}, {i};' for i in range(pack)]
    code.append('}')
    return '\n'.join(code), ','.join(['=r'] * pack + ['f'] * pack), pack


def shared_ordered_asm(tile, *, states=False):
    """All 128 threads stage P once; warp0 accumulates two rows per lane.

    Each row's four lanes preserve block order and even-before-odd additions.
    Final row sum is exactly (s0+s2)+(s1+s3). Two CTA barriers enclose staging.
    The input initial state is broadcast into the same MMA layout as P.
    """
    if tile not in (128, 256):
        raise ValueError('tile must be 128 or 256')
    n = tile // 8
    code = ['{', f'.shared .align 4 .b8 fragment_p[{16 * tile * 4}];',
            f'.shared .align 4 .b8 fragment_sum[{256 if states else 64}];',
            '.reg .u32 tid, lane, warp, row, col, addr, outaddr, src0, src1, basep, bases;',
            '.reg .f32 a, b, x, y, pair, u, v;', '.reg .pred leader, writer;',
            'mov.u32 basep, fragment_p;', 'mov.u32 bases, fragment_sum;',
            'mov.u32 tid, %tid.x;', 'and.b32 lane, tid, 31;', 'shr.u32 warp, tid, 5;',
            'shr.u32 row, lane, 2;',
            'and.b32 col, lane, 3;', 'shl.b32 col, col, 1;', 'mad.lo.u32 col, warp, 8, col;',
            f'mad.lo.u32 addr, row, {tile}, col;', 'shl.b32 addr, addr, 2;',
            'add.u32 addr, addr, basep;']
    for slot in range(n):
        repeat, inner = divmod(slot, 4)
        offset = ((inner // 2) * 8 * tile + repeat * 32 + inner % 2) * 4
        code.append(f'st.shared.f32 [addr+{offset}], ${n + slot};')
    code += ['bar.sync 0;', 'setp.eq.u32 leader, warp, 0;', '@!leader bra FRAGMENT_JOIN;',
             f'mov.f32 a, ${2*n};', f'mov.f32 b, ${2*n+2};',
             'and.b32 col, lane, 3;', 'shl.b32 col, col, 1;',
             f'mad.lo.u32 addr, row, {tile}, col;', 'shl.b32 addr, addr, 2;',
             'add.u32 addr, addr, basep;']
    for block in range(tile // 8):
        for parity in range(2):
            offset=(block * 8 + parity)*4
            code += [f'ld.shared.f32 x, [addr+{offset}];', 'add.rn.f32 a, a, x;',
                     f'ld.shared.f32 y, [addr+{offset+8*tile*4}];', 'add.rn.f32 b, b, y;']
    if states:
        # All 32 warp0 lanes write their own states, before any row merge.
        code += ['and.b32 col, lane, 3;', 'mad.lo.u32 outaddr, row, 4, col;',
                 'shl.b32 outaddr, outaddr, 2;', 'add.u32 outaddr, outaddr, bases;',
                 'st.shared.f32 [outaddr], a;', 'st.shared.f32 [outaddr+128], b;',
                 'FRAGMENT_JOIN:', 'bar.sync 0;', 'and.b32 col, lane, 3;',
                 'mad.lo.u32 outaddr, row, 4, col;', 'shl.b32 outaddr, outaddr, 2;',
                 'add.u32 outaddr, outaddr, bases;']
        for slot in range(n):
            offset = 128 if (slot % 4) // 2 else 0
            code.append(f'ld.shared.f32 ${slot}, [outaddr+{offset}];')
        code.append('}')
        return '\n'.join(code), ','.join(['=f'] * n + ['f'] * (2*n) + ['~{memory}']), n
    # Four lanes for a row are consecutive in warp0. Fetch pair0 and pair1 in
    # this explicit order; a generic butterfly's lane-dependent order is avoided.
    code += ['and.b32 src0, lane, 28;', 'add.u32 src1, src0, 1;']
    for value in ('a','b'):
        code += [f'shfl.sync.bfly.b32 pair, {value}, 2, 31, -1;', f'add.rn.f32 pair, {value}, pair;',
                 'shfl.sync.idx.b32 u, pair, src0, 31, -1;',
                 'shfl.sync.idx.b32 v, pair, src1, 31, -1;', f'add.rn.f32 {value}, u, v;']
    code += ['and.b32 col, lane, 3;', 'setp.eq.u32 writer, col, 0;',
             'shl.b32 outaddr, row, 2;', 'add.u32 outaddr, outaddr, bases;',
             '@writer st.shared.f32 [outaddr], a;', '@writer st.shared.f32 [outaddr+32], b;',
             'FRAGMENT_JOIN:', 'bar.sync 0;', 'shl.b32 outaddr, row, 2;',
             'add.u32 outaddr, outaddr, bases;']
    for slot in range(n):
        offset = 32 if (slot % 4) // 2 else 0
        code.append(f'ld.shared.f32 ${slot}, [outaddr+{offset}];')
    code.append('}')
    return '\n'.join(code), ','.join(['=f'] * n + ['f'] * (2*n) + ['~{memory}']), n
