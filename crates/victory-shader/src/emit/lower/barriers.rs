//! Where group barriers can go when the host splits a compute group into
//! waves the console's lockstep did not have: the blocks every lane of the
//! group runs together ([`in_step`]), and the barriers a group that is one
//! console wave needs between its lanes' LDS accesses
//! ([`split_wave_barriers`]).

use super::*;

/// Where a compute group that is one console wave needs group barriers
/// when the host splits it into narrower waves: (block, instruction) to
/// sync before, instruction index = the block's length for its end.
///
/// The console runs such a group in lockstep, so its compiler leaves out
/// the barriers between one lane's LDS access and another's. Here a read
/// (or atomic) of LDS that a write or atomic may have changed since the last
/// barrier gets one before it, and so does a write (or atomic) after a read
/// or atomic since then, so that it cannot overwrite what another lane has
/// yet to read. (A write after a write, which the console orders too, gets
/// none.) Barriers need every lane of the group, so they go only in blocks
/// every lane runs together, the same number of times ([`in_step`]): once,
/// or in a loop on a count every host wave computes alike. Such a block that
/// leads into code only some lanes run syncs at its end where that code
/// could otherwise read a write, or overwrite a read, made before it.
///
/// `None`: a read in divergent code may see a write made in divergent code
/// since the last barrier. A write in divergent code after a read there gets
/// no barrier, as before.
#[cfg(test)]
pub(crate) fn split_wave_barriers(p: &Program, st: &Structured) -> Option<Set<(usize, usize)>> {
    let step = in_step(p, st);
    split_wave_barriers_with_step(p, &step)
}

/// [`split_wave_barriers`] using a precomputed `step` vector (which may
/// include blocks legalized by acyclic predication in `halves`).
pub(crate) fn split_wave_barriers_with_step(
    p: &Program,
    step: &[bool],
) -> Option<Set<(usize, usize)>> {
    /// LDS accesses since the last barrier.
    #[derive(Clone, Copy, Default, PartialEq)]
    struct Since {
        read: bool,
        written: bool,
    }
    impl Since {
        fn or(self, o: Since) -> Since {
            Since {
                read: self.read || o.read,
                written: self.written || o.written,
            }
        }
    }
    // (reads, writes) of an LDS access; an atomic does both.
    let access = |inst: &Inst| {
        let name = inst.name();
        if !matches!(inst.fields, Fields::Ds { .. })
            || name.starts_with("ds_swizzle")
            || name.contains("permute")
        {
            return None;
        }
        Some((
            !name.starts_with("ds_write"),
            !name.starts_with("ds_read"),
        ))
    };
    let n = p.blocks.len();
    let succ = |b: usize| p.blocks[b].exit.successors();
    let divergent_writes = (0..n).any(|b| {
        !step[b]
            && p.blocks[b]
                .insts
                .iter()
                .any(|i| access(i).is_some_and(|(_, w)| w))
    });
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut out = vec![Since::default(); n];
    loop {
        let mut at = Set::default();
        let mut changed = false;
        for &b in &order {
            let mut since = p.blocks[b]
                .preds
                .iter()
                .fold(Since::default(), |a, &q| a.or(out[q]));
            for (i, inst) in p.blocks[b].insts.iter().enumerate() {
                if inst.name() == "s_barrier" {
                    since = Since::default();
                }
                let Some((reads, writes)) = access(inst) else {
                    continue;
                };
                // A read of what another lane may have written; a write over
                // what another lane may have yet to read.
                let stale = reads && since.written;
                let early = writes && since.read;
                if stale || early {
                    if step[b] {
                        at.insert((b, i));
                        since = Since::default();
                    } else if stale {
                        return None;
                    }
                }
                since = since.or(Since {
                    read: reads,
                    written: writes,
                });
            }
            if step[b]
                && succ(b).iter().any(|&s| !step[s])
                && (since.written || since.read && divergent_writes)
            {
                at.insert((b, p.blocks[b].insts.len()));
                since = Since::default();
            }
            // What reaches a block only grows, so this ends; barriers placed
            // for more than reaches it are still right.
            let since = out[b].or(since);
            changed |= std::mem::replace(&mut out[b], since) != since;
        }
        if !changed {
            return Some(at);
        }
    }
}

/// Per block, whether every lane of the group runs it together, the same
/// number of times, however the host splits the group into waves, so that a
/// group barrier can go there: outside the region of every branch whose
/// condition may differ between waves ([`halves::branch_regions`]). That is
/// a per-lane branch, or one on a scalar not computed alike in every wave.
/// Every block each lane runs once ([`once_per_lane`]) is one; so is a loop
/// whose count every wave computes alike, from the group's inputs,
/// constants and scalar loads through scalar code (not from lane masks,
/// other lanes' values or hardware state). Such a scalar is the same in
/// every console wave of the group too.
///
/// Scalars are assumed alike until shown otherwise (a loop counter needs
/// itself), and the regions and the scalars are narrowed together to a
/// fixed point. With an indirect branch, the blocks each lane runs once.
pub(crate) fn in_step(p: &Program, st: &Structured) -> Vec<bool> {
    in_step_with(p, st, &st.exits)
}

/// [`in_step`] evaluated over `exits` (either `st.exits` or exits after
/// selective acyclic predication), tracking both group-uniform SGPRs and
/// group-uniform VGPRs (so a uniform loop bound loaded via `buffer_load` /
/// `global_load` / `ds_read` and moved through `v_readfirstlane_b32` stays
/// group-uniform).
pub(crate) fn in_step_with(p: &Program, st: &Structured, exits: &[LaneExit]) -> Vec<bool> {
    let Some(regions) = halves::branch_regions_with(exits) else {
        return once_per_lane(p);
    };
    let n = p.blocks.len();
    let scalar = |r: Reg| matches!(r, Reg::S(_) | Reg::Scc);
    let vector = |r: Reg| matches!(r, Reg::V(_));
    // Not EXEC or VCC: lane masks.
    let mask = |r: Reg| matches!(r, Reg::S(k) if [EXEC, EXEC + 1, VCC, VCC + 1].contains(&k));
    // Scalar writes no host wave can compute alike: other lanes' values,
    // hardware state, atomics, coherent loads (the group may write them).
    let own = |inst: &Inst| {
        let name = inst.name();
        match &inst.fields {
            Fields::Sop { .. } => {
                name.starts_with("s_getreg")
                    || name.starts_with("s_sendmsg")
                    || name.contains("waveid")
            }
            Fields::Smem { cache, .. } => {
                !(name.starts_with("s_load") || name.starts_with("s_buffer_load")) || cache.glc
            }
            _ => true,
        }
    };
    let mut alike: Set<Value> = p
        .inputs
        .iter()
        .filter(|(r, _)| scalar(*r))
        .map(|(_, v)| *v)
        .chain(p.blocks.iter().flat_map(|b| {
            b.phis
                .iter()
                .filter(|f| scalar(f.reg))
                .map(|f| f.value)
                .chain(
                    b.insts
                        .iter()
                        .flat_map(|i| i.writes.iter().filter(|w| scalar(w.0)).map(|w| w.1)),
                )
        }))
        .collect();
    // Group-uniform VGPRs: optimistically all non-entry VGPRs, narrowed with
    // `alike` in lockstep. Entry VGPRs (`v0`/`v1`/`v2` thread ids) are never
    // group-uniform.
    let mut alike_v: Set<Value> = p
        .blocks
        .iter()
        .flat_map(|b| {
            b.phis
                .iter()
                .filter(|f| vector(f.reg))
                .map(|f| f.value)
                .chain(
                    b.insts
                        .iter()
                        .flat_map(|i| i.writes.iter().filter(|w| vector(w.0)).map(|w| w.1)),
                )
        })
        .collect();
    loop {
        let mut step = vec![true; n];
        for (d, region) in &regions {
            let LaneExit::Branch { when, .. } = exits[*d] else {
                continue;
            };
            let together = st
                .bools
                .support(when)
                .iter()
                .all(|a| matches!(a, Atom::Uniform(v) if alike.contains(v)));
            if !together {
                for (b, &r) in region.iter().enumerate() {
                    step[b] &= !r;
                }
            }
        }
        let mut next: Set<Value> = p
            .inputs
            .iter()
            .filter(|(r, _)| scalar(*r))
            .map(|(_, v)| *v)
            .collect();
        let mut next_v: Set<Value> = Set::default();
        for (b, block) in p.blocks.iter().enumerate() {
            // Every host wave enters by the same edge.
            let entered = step[b] && block.preds.iter().all(|&q| step[q]);
            for phi in &block.phis {
                if scalar(phi.reg) && entered && phi.args.iter().all(|a| alike.contains(a)) {
                    next.insert(phi.value);
                } else if vector(phi.reg) && entered && phi.args.iter().all(|a| alike_v.contains(a))
                {
                    next_v.insert(phi.value);
                }
            }
            for (i, inst) in block.insts.iter().enumerate() {
                let name = inst.name();
                // Group-uniform scalar ops.
                let from_scalar = inst
                    .reads
                    .iter()
                    .all(|&(r, v)| scalar(r) && !mask(r) && alike.contains(&v));
                if from_scalar && !own(inst) {
                    next.extend(inst.writes.iter().filter(|w| scalar(w.0)).map(|w| w.1));
                    continue;
                }
                // `v_readfirstlane_b32` of a group-uniform VGPR yields a
                // group-uniform SGPR.
                if name == "v_readfirstlane_b32"
                    && let Fields::Valu(v) = &inst.fields
                    && let Some(&Operand::Vector { reg, .. }) = v.src.first()
                    && inst.read(Reg::V(reg)).is_some_and(|x| alike_v.contains(&x))
                {
                    next.extend(inst.writes.iter().filter(|w| scalar(w.0)).map(|w| w.1));
                    continue;
                }
                // Guarded VGPR writes only stay group-uniform when the guard
                // itself is group-uniform (for simplicity, require no guard).
                if st.guards[b][i].is_some() {
                    continue;
                }
                // Buffer / global / LDS loads at group-uniform addresses.
                match &inst.fields {
                    Fields::Buffer {
                        addr,
                        rsrc,
                        soffset,
                        offen,
                        idxen,
                        cache,
                        ..
                    } if (name.starts_with("buffer_load") || name.starts_with("tbuffer_load"))
                        && !cache.glc =>
                    {
                        let addr_ok = !(*offen || *idxen)
                            || match *addr {
                                Operand::Vector { reg, dwords } => (0..dwords).all(|k| {
                                    inst.read(Reg::V(reg + k))
                                        .is_some_and(|v| alike_v.contains(&v))
                                }),
                                _ => true,
                            };
                        let sc_ok = |op: &Operand| match *op {
                            Operand::Scalar { code, dwords } if code != NULL => (0..dwords)
                                .all(|k| {
                                    let r = Reg::S(code + k);
                                    !mask(r) && inst.read(r).is_none_or(|v| alike.contains(&v))
                                }),
                            _ => true,
                        };
                        if addr_ok && sc_ok(rsrc) && sc_ok(soffset) {
                            next_v.extend(inst.writes.iter().filter(|w| vector(w.0)).map(|w| w.1));
                        }
                    }
                    Fields::Flat {
                        addr: Some(Operand::Vector { reg, dwords }),
                        saddr,
                        cache,
                        ..
                    } if name.starts_with("global_load") && !cache.glc => {
                        let addr_ok = (0..*dwords).all(|k| {
                            inst.read(Reg::V(reg + k))
                                .is_some_and(|v| alike_v.contains(&v))
                        });
                        let saddr_ok = match saddr {
                            Some(Operand::Scalar { code, dwords }) if *code != NULL => (0..*dwords)
                                .all(|k| {
                                    let r = Reg::S(code + k);
                                    !mask(r) && inst.read(r).is_none_or(|v| alike.contains(&v))
                                }),
                            _ => true,
                        };
                        if addr_ok && saddr_ok {
                            next_v.extend(inst.writes.iter().filter(|w| vector(w.0)).map(|w| w.1));
                        }
                    }
                    Fields::Ds {
                        addr: Some(Operand::Vector { reg, .. }),
                        dst: Some(_),
                        data,
                        ..
                    } if step[b] && name.starts_with("ds_read") && data.is_empty() => {
                        if inst.read(Reg::V(*reg)).is_some_and(|v| alike_v.contains(&v)) {
                            next_v.extend(inst.writes.iter().filter(|w| vector(w.0)).map(|w| w.1));
                        }
                    }
                    Fields::Valu(v)
                        if v.dpp.is_none()
                            && v.sdwa.is_none()
                            && !name.starts_with("v_mbcnt")
                            && !name.starts_with("v_cmpx")
                            && !name.contains("lane")
                            && !name.contains("cndmask")
                            && !name.contains("_co_ci_")
                            && !name.contains("mac")
                            && !name.contains("dot") =>
                    {
                        let ops_ok = v.src.iter().all(|op| match *op {
                            Operand::Vector { reg, dwords } => (0..dwords).all(|k| {
                                inst.read(Reg::V(reg + k))
                                    .is_some_and(|val| alike_v.contains(&val))
                            }),
                            Operand::Scalar { code, dwords } if code != NULL => {
                                (0..dwords).all(|k| {
                                    let r = Reg::S(code + k);
                                    !mask(r) && inst.read(r).is_none_or(|val| alike.contains(&val))
                                })
                            }
                            Operand::Const(_) | Operand::Literal(_) | Operand::Scalar { .. } => {
                                true
                            }
                            Operand::Attr { .. } => false,
                        });
                        if ops_ok {
                            next_v.extend(inst.writes.iter().filter(|w| vector(w.0)).map(|w| w.1));
                        }
                    }
                    _ => {}
                }
            }
        }
        if next.len() == alike.len() && next_v.len() == alike_v.len() {
            return step;
        }
        alike = next;
        alike_v = next_v;
    }
}

/// Per block, whether each lane of the group runs it exactly once: no cycle
/// passes through it, and every path from the entry to an end does. A group
/// barrier needs every lane of the group, so it can go only in these.
pub(crate) fn once_per_lane(p: &Program) -> Vec<bool> {
    let n = p.blocks.len();
    let succ = |b: usize| p.blocks[b].exit.successors();
    // Blocks reachable from `from`'s successors (or from `from` itself when
    // `start`), never entering `avoid`.
    let reach = |from: usize, start: bool, avoid: Option<usize>| {
        let mut seen = vec![false; n];
        let mut stack = if start { vec![from] } else { succ(from) };
        while let Some(b) = stack.pop() {
            if Some(b) == avoid || std::mem::replace(&mut seen[b], true) {
                continue;
            }
            stack.extend(succ(b));
        }
        seen
    };
    (0..n)
        .map(|b| {
            let ends_around = b != 0
                && reach(0, true, Some(b))
                    .iter()
                    .enumerate()
                    .any(|(q, &r)| r && succ(q).is_empty());
            !reach(b, false, None)[b] && !ends_around
        })
        .collect()
}
