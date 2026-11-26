//! Group barriers for a program whose thread group is one console wave,
//! run on a host of 32-lane waves as two host waves (`Parts::host_wave32`).

use std::collections::HashMap;

use super::super::bitcode::{Attr, Term};
use super::{Lower, Why};
use crate::gfx10::Fields;
use crate::ir::Program;
use crate::structure::LaneExit;

/// Where the group's threads meet ([`barriers`]).
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Plan {
    /// Before instruction `i` of block `b` (`i` the count: at its end).
    pub at: Vec<(usize, usize)>,
    /// Loop latches whose unconditional back edge a lane past the iteration
    /// limit leaves the program through, and how many of the barriers lie
    /// ahead of them: that lane passes them on its way out, so the others
    /// do not wait for it.
    pub drains: HashMap<usize, usize>,
}

/// Where a program whose thread group is one console wave, run as two
/// host waves, orders its LDS accesses: the positions (block, instruction;
/// the instruction count for the block's end) before which a group barrier
/// goes. The console compiler leaves out barriers within one wave, whose
/// lanes run in step; the host's two waves do not, so a lane reading what a
/// lane of the other half wrote, or writing where it read, must wait at a
/// barrier between the two. The program's accesses must come in phases of
/// reads only, writes only (each write ordered after the last) or atomics
/// only, and between two of them there must be a point
/// every thread reaches exactly once (on every path to every exit, outside
/// loops, where no instruction is half done): a barrier anywhere else could
/// hang the group. The first such point after the last access of a phase
/// takes the barrier. `inside(b, i)` tells an instruction the lowering
/// handles as part of an earlier one (no barrier before it).
pub(super) fn barriers(
    p: &Program,
    exits: &[LaneExit],
    inside: impl Fn(usize, usize) -> bool,
) -> Result<Plan, Why> {
    const READ: u8 = 0;
    const WRITE: u8 = 1;
    const ATOMIC: u8 = 2;
    const NO_POINT: Why =
        "one-wave LDS on a 32-lane host: no point all threads reach between a write and a read";
    let n = p.blocks.len();
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut rpo = vec![usize::MAX; n];
    for (k, &b) in order.iter().enumerate() {
        rpo[b] = k;
    }
    let mut succs = vec![Vec::new(); n];
    // Where a lane can leave: the end. (A loop's unconditional back edge,
    // where a lane past the iteration limit returns, passes the barriers
    // ahead first: `Plan::drains`.)
    let mut leaves = vec![false; n];
    let mut latches = Vec::new();
    for &b in &order {
        match exits[b] {
            LaneExit::Jump(t) => {
                succs[b].push(t);
                if rpo[t] <= rpo[b] {
                    latches.push(b);
                }
            }
            LaneExit::Branch {
                taken, not_taken, ..
            } => succs[b].extend([taken, not_taken]),
            LaneExit::End => leaves[b] = true,
            LaneExit::Indirect => return Err("one-wave LDS on a 32-lane host: an indirect jump"),
        }
    }
    // Blocks reachable from `from` over at least one edge, without
    // entering `avoid`.
    let reach = |from: usize, avoid: Option<usize>| {
        let mut seen = vec![false; n];
        let mut stack: Vec<usize> = succs[from].clone();
        while let Some(b) = stack.pop() {
            if Some(b) == avoid || seen[b] {
                continue;
            }
            seen[b] = true;
            stack.extend(&succs[b]);
        }
        seen
    };
    let entry = order[0];
    // Blocks every thread runs exactly once, in program order.
    let spine: Vec<usize> = order
        .iter()
        .copied()
        .filter(|&s| {
            let in_loop = reach(s, None)[s];
            let mut bypass = if s == entry {
                vec![false; n]
            } else {
                reach(entry, Some(s))
            };
            if s != entry {
                bypass[entry] = true;
            }
            !in_loop && !(0..n).any(|b| bypass[b] && leaves[b])
        })
        .collect();
    let after: Vec<Vec<bool>> = spine.iter().map(|&s| reach(s, None)).collect();
    // A place in program order: (j, i) is before instruction `i` of spine
    // block j - 1; (j, MAX) anywhere between spine blocks j - 1 and j.
    let place = |b: usize, i: usize| match spine.iter().position(|&s| s == b) {
        Some(j) => (j + 1, i),
        None => (after.iter().filter(|a| a[b]).count(), usize::MAX),
    };
    let mut accesses = Vec::new();
    for &b in &order {
        for (i, inst) in p.blocks[b].insts.iter().enumerate() {
            if matches!(
                inst.fields,
                Fields::Buffer { lds: true, .. } | Fields::Flat { lds: true, .. }
            ) {
                return Err("one-wave LDS on a 32-lane host: a memory load into LDS");
            }
            let Fields::Ds { gds, .. } = inst.fields else {
                continue;
            };
            let name = inst.name();
            if gds || name.starts_with("ds_swizzle") || name.contains("permute") {
                continue;
            }
            // Atomics need no order among themselves (any order of them is
            // one the console could run in a group of several waves), only
            // against plain reads and writes.
            let kind = if name.starts_with("ds_read") {
                READ
            } else if name.starts_with("ds_write") {
                WRITE
            } else if name.starts_with("ds_gws") {
                return Err("one-wave LDS on a 32-lane host: a GWS operation");
            } else {
                ATOMIC
            };
            accesses.push((place(b, i), kind));
        }
    }
    accesses.sort_unstable();
    // The places a barrier can take, in program order.
    let points: Vec<(usize, usize)> = spine
        .iter()
        .enumerate()
        .flat_map(|(j, &s)| (0..=p.blocks[s].insts.len()).map(move |i| (j, s, i)))
        .filter(|&(_, s, i)| i == p.blocks[s].insts.len() || !inside(s, i))
        .map(|(j, _, i)| (j + 1, i))
        .collect();
    let mut barriers = Vec::new();
    for w in accesses.windows(2) {
        let ((from, a), (to, b)) = (w[0], w[1]);
        if a == b && a != WRITE {
            continue;
        }
        if from == to {
            return Err(NO_POINT);
        }
        // The first place past `from` and no later than `to`.
        let past = |q: &(usize, usize)| match from.1 {
            usize::MAX => q.0 > from.0,
            i => *q > (from.0, i),
        };
        let Some(point) = points.iter().find(|q| past(q) && **q <= to) else {
            return Err(NO_POINT);
        };
        barriers.push((spine[point.0 - 1], point.1));
    }
    barriers.dedup();
    // A latch between spine blocks j - 1 and j has the barriers from spine
    // block j on ahead.
    let drains = latches
        .into_iter()
        .map(|l| {
            let (j, _) = place(l, 0);
            let ahead = barriers
                .iter()
                .filter(|&&(b, _)| {
                    spine
                        .iter()
                        .position(|&s| s == b)
                        .is_some_and(|k| k + 1 > j)
                })
                .count();
            (l, ahead)
        })
        .filter(|&(_, ahead)| ahead > 0)
        .collect();
    Ok(Plan {
        at: barriers,
        drains,
    })
}

impl Lower<'_> {
    /// A group barrier ordering LDS accesses ([`barriers`]): every thread
    /// waits, and LDS writes before it are seen after it.
    pub(super) fn lds_barrier(&mut self) {
        // Mode: sync the group, fence group-shared memory.
        let mode = self.s.m.i32(1 | 8);
        let void = self.s.t.void;
        self.s
            .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }

    /// Where a lane past the iteration limit at loop latch `b` leaves: the
    /// program's abort block, or one of its own that first passes the
    /// barriers still ahead (`Plan::drains`).
    pub(super) fn lds_abort(&mut self, b: usize) -> usize {
        let Some(&ahead) = self.drains.get(&b) else {
            return self.abort_block();
        };
        let (cur, a) = (self.cur, self.s.m.new_block());
        self.cur = a;
        for _ in 0..ahead {
            self.lds_barrier();
        }
        self.s.m.terminate(a, Term::Ret);
        self.cur = cur;
        a
    }
}

#[cfg(test)]
mod tests {
    use super::{Why, barriers};

    type Plan = Result<Vec<(usize, usize)>, Why>;

    fn at(plan: Result<super::Plan, Why>) -> Plan {
        plan.map(|p| p.at)
    }
    const S_ENDPGM: u32 = 0xBF81_0000;
    // GFX10 words for the one-wave LDS tests.
    const DS_WRITE_V0_V1: [u32; 2] = [0xD834_0000, 0x0000_0100];
    const DS_READ_V2_V0: [u32; 2] = [0xD8D8_0000, 0x0200_0000];
    const DS_ADD_V0_V1: [u32; 2] = [0xD800_0000, 0x0000_0100];
    const S_MOV_S0_4: u32 = 0xBE80_0384;
    const S_SUB_S0_1: u32 = 0x8180_8100;
    const S_CMP_LG_S0_0: u32 = 0xBF07_8000;

    /// `lds_barriers` of a wave64 program, and where its first `ds_write` is.
    fn plan_lds(words: &[u32]) -> (Plan, (usize, usize)) {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, true, 4).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let write = p
            .blocks
            .iter()
            .enumerate()
            .find_map(|(b, k)| {
                k.insts
                    .iter()
                    .position(|i| i.name() == "ds_write_b32")
                    .map(|i| (b, i))
            })
            .unwrap_or((usize::MAX, usize::MAX));
        (at(barriers(&p, &st.exits, |_, _| false)), write)
    }

    /// A read after a write in the same straight code: the barrier goes
    /// right after the write, where every thread passes once.
    #[test]
    fn a_one_wave_lds_read_after_a_write_waits_at_a_barrier_between() {
        let words = [
            DS_WRITE_V0_V1[0],
            DS_WRITE_V0_V1[1],
            DS_READ_V2_V0[0],
            DS_READ_V2_V0[1],
            S_ENDPGM,
        ];
        let (plan, (b, i)) = plan_lds(&words);
        assert_eq!(plan.unwrap(), vec![(b, i + 1)]);
    }

    /// Writes only: nothing to order between the halves.
    #[test]
    fn one_wave_lds_writes_alone_need_no_barrier() {
        let words = [DS_WRITE_V0_V1[0], DS_WRITE_V0_V1[1], S_ENDPGM];
        assert_eq!(plan_lds(&words).0.unwrap(), vec![]);
    }

    /// Reads in a loop after the writes: the barrier goes before the loop,
    /// not in it (where a thread could pass it another number of times).
    #[test]
    fn one_wave_lds_reads_in_a_loop_wait_once_before_it() {
        // ds_write; s_mov s0, 4; loop: ds_read; s_sub s0, 1; s_cmp_lg s0, 0;
        // s_cbranch_scc1 loop; s_endpgm
        let words = [
            DS_WRITE_V0_V1[0],
            DS_WRITE_V0_V1[1],
            S_MOV_S0_4,
            DS_READ_V2_V0[0],
            DS_READ_V2_V0[1],
            S_SUB_S0_1,
            S_CMP_LG_S0_0,
            0xBF85_FFFB, // s_cbranch_scc1 -5 dwords: the ds_read
            S_ENDPGM,
        ];
        let (plan, (b, i)) = plan_lds(&words);
        assert_eq!(plan.unwrap(), vec![(b, i + 1)]);
    }

    /// A loop that both writes and reads LDS has no point between the two
    /// that every thread passes once: refused.
    #[test]
    fn one_wave_lds_written_and_read_in_one_loop_is_refused() {
        // s_mov s0, 4; loop: ds_write; ds_read; s_sub; s_cmp_lg; s_cbranch_scc1
        // loop; s_endpgm
        let words = [
            S_MOV_S0_4,
            DS_WRITE_V0_V1[0],
            DS_WRITE_V0_V1[1],
            DS_READ_V2_V0[0],
            DS_READ_V2_V0[1],
            S_SUB_S0_1,
            S_CMP_LG_S0_0,
            0xBF85_FFF9, // s_cbranch_scc1 -7 dwords: the ds_write
            S_ENDPGM,
        ];
        assert!(plan_lds(&words).0.is_err());
    }

    /// Two writes in straight code: the second waits for the first, as the
    /// console orders them.
    #[test]
    fn one_wave_lds_writes_in_straight_code_are_ordered() {
        let words = [
            DS_WRITE_V0_V1[0],
            DS_WRITE_V0_V1[1],
            DS_WRITE_V0_V1[0],
            DS_WRITE_V0_V1[1],
            S_ENDPGM,
        ];
        let (plan, (b, i)) = plan_lds(&words);
        assert_eq!(plan.unwrap(), vec![(b, i + 1)]);
    }

    /// Two writes in one loop have no spine point between them: refused.
    #[test]
    fn one_wave_lds_writes_in_one_loop_are_refused() {
        let words = [
            S_MOV_S0_4,
            DS_WRITE_V0_V1[0],
            DS_WRITE_V0_V1[1],
            DS_WRITE_V0_V1[0],
            DS_WRITE_V0_V1[1],
            S_SUB_S0_1,
            S_CMP_LG_S0_0,
            0xBF85_FFF9, // s_cbranch_scc1 -7 dwords: the first ds_write
            S_ENDPGM,
        ];
        assert!(plan_lds(&words).0.is_err());
    }

    /// A write some lanes skip, then a read where all meet: the barrier
    /// goes at the join, before the read.
    #[test]
    fn one_wave_lds_read_after_a_skipped_write_waits_at_the_join() {
        let words = [
            0xBF85_0002, // s_cbranch_scc1 past the write
            DS_WRITE_V0_V1[0],
            DS_WRITE_V0_V1[1],
            DS_READ_V2_V0[0],
            DS_READ_V2_V0[1],
            S_ENDPGM,
        ];
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, true, 4).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let read = p
            .blocks
            .iter()
            .enumerate()
            .find_map(|(b, k)| {
                k.insts
                    .iter()
                    .position(|i| i.name() == "ds_read_b32")
                    .map(|i| (b, i))
            })
            .unwrap();
        let plan = barriers(&p, &st.exits, |_, _| false).unwrap().at;
        assert_eq!(plan.len(), 1);
        let (b, i) = plan[0];
        // In the read's block, at or before it, and not in the write's.
        assert_eq!(b, read.0);
        assert!(i <= read.1);
    }

    /// A buffer load into LDS is not lowered: refused, not dropped.
    #[test]
    fn one_wave_lds_filled_by_a_buffer_load_is_refused() {
        // buffer_load_dword v0, off, s[0:3], 0 lds; s_endpgm
        let words = [0xE031_0000, 0x8000_0000, S_ENDPGM];
        assert_eq!(
            plan_lds(&words).0,
            Err("one-wave LDS on a 32-lane host: a memory load into LDS")
        );
    }

    /// A barrier after a loop whose back edge is unconditional: a lane the
    /// iteration limit stops there passes that barrier on its way out.
    #[test]
    fn a_lane_the_loop_limit_stops_passes_the_barriers_ahead() {
        // s_mov s0, 4; loop: s_sub; s_cmp_lg; s_cbranch_scc0 out; s_branch
        // loop; out: ds_write; ds_read; s_endpgm
        let words = [
            S_MOV_S0_4,
            S_SUB_S0_1,
            S_CMP_LG_S0_0,
            0xBF84_0001, // s_cbranch_scc0 +1: out
            0xBF82_FFFC, // s_branch -4: the s_sub
            DS_WRITE_V0_V1[0],
            DS_WRITE_V0_V1[1],
            DS_READ_V2_V0[0],
            DS_READ_V2_V0[1],
            S_ENDPGM,
        ];
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, true, 4).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let plan = barriers(&p, &st.exits, |_, _| false).unwrap();
        assert_eq!(plan.at.len(), 1);
        assert_eq!(plan.drains.values().copied().collect::<Vec<_>>(), vec![1]);
    }

    /// Atomics alone need no barrier; a read of what they left waits for
    /// them.
    #[test]
    fn one_wave_lds_atomics_wait_only_for_reads() {
        let words = [
            DS_ADD_V0_V1[0],
            DS_ADD_V0_V1[1],
            DS_ADD_V0_V1[0],
            DS_ADD_V0_V1[1],
            S_ENDPGM,
        ];
        assert_eq!(plan_lds(&words).0.unwrap(), vec![]);
        let words = [
            DS_ADD_V0_V1[0],
            DS_ADD_V0_V1[1],
            DS_READ_V2_V0[0],
            DS_READ_V2_V0[1],
            S_ENDPGM,
        ];
        let plan = plan_lds(&words).0.unwrap();
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].1, 1);
    }
}
