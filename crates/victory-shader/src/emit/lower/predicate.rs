//! Selective acyclic divergent-branch predication for wave64-on-32-lane
//! (`halves`) compute shaders.
//!
//! Cross-half exchanges (`exchange_halves`) and split-wave LDS barriers
//! (`split_wave_barriers`) emit group barriers (`dx.op.barrier`), which
//! require every thread in the threadgroup to reach the block together
//! (`in_step`). Standard compute patterns such as an early-out bounds check
//! (`if (dtid >= limit) return;`) or an acyclic `if (cond) { ... } else { ... }`
//! diamond put their body outside `in_step`.
//!
//! When a wave64-on-32-lane compute shader has such an acyclic divergent
//! branch whose body needs `in_step` (a cross-half wave op or split-wave LDS
//! access), this pass linearizes that branch into predicated lockstep
//! execution:
//! * **Early-out (`if (cond) return;`)**: replaces `CondBr(cond, end, cont)`
//!   with `Br(cont)` and carries `active &= !cond` into every block dominated
//!   by `cont` (including straight through subsequent uniform loops).
//! * **Acyclic SESE diamond (`if (cond) { A } [else { B }] -> J`)**:
//!   linearizes the strictly acyclic region `R` between `d` and its immediate
//!   post-dominator `J` in reverse-postorder (`d -> r_0 -> ... -> r_k -> J`),
//!   tracking per-edge active predicates and lowering phis at linearized
//!   merges (`R ∪ {J}`) into `select` chains.
//!
//! Every side effect and potentially faulting load in a predicated block is
//! guarded by the block's `active` predicate through `Lower::guard`, while
//! barriers and cross-half exchanges execute at `self.level` in 100%
//! group-uniform control flow.

use super::*;

/// Maximum number of blocks inside one linearized SESE diamond.
const MAX_DIAMOND_BLOCKS: usize = 64;

/// How a predicated block's CFG exit is rewritten.
#[derive(Clone, Copy, Debug)]
pub(super) enum FlatExit {
    /// Early-out branch `d -> (end, cont)` rewritten to unconditional
    /// `Br(cont)`, with lanes staying active along `(d, cont)`.
    EarlyOut { cont: usize },
    /// Block inside (or at the head of) a linearized SESE diamond jumping
    /// unconditionally to the next block in topological RPO order.
    Linear { next: usize },
}

/// Predication plan computed before `split_wave_barriers` and `Lower::run`.
pub(super) struct PredicatePlan {
    /// Rewritten exits for predicated branches and blocks inside linearized
    /// diamonds.
    pub(super) exits: Map<usize, FlatExit>,
    /// Blocks inside linearized diamonds (strictly between `d` and `join`).
    pub(super) in_diamond: Set<usize>,
    /// Join blocks of linearized diamonds (`join -> head branch block d`).
    pub(super) diamond_joins: Map<usize, usize>,
    /// Blocks whose DXIL predecessors were linearized (internal diamond merges
    /// and diamond joins), so their IR phis lower to `select` chains.
    pub(super) linear_merges: Set<usize>,
    /// Blocks bypassed by `EarlyOut` that have no remaining live predecessors.
    pub(super) dead_blocks: Set<usize>,
    /// Immediate dominator per IR block (on the original CFG).
    pub(super) idom: Vec<usize>,
    /// Continuation blocks of `EarlyOut` branches (`cont -> branch block d`).
    pub(super) early_conts: Map<usize, usize>,
    /// Trivial `End` blocks bypassed by at least one `EarlyOut`.
    pub(super) skipped_ends: Set<usize>,
    /// Updated `in_step` vector reflecting linearized control flow.
    pub(super) step: Vec<bool>,
}

impl PredicatePlan {
    pub(super) fn empty(step: Vec<bool>) -> Self {
        PredicatePlan {
            exits: Map::default(),
            in_diamond: Set::default(),
            diamond_joins: Map::default(),
            linear_merges: Set::default(),
            dead_blocks: Set::default(),
            idom: Vec::new(),
            early_conts: Map::default(),
            skipped_ends: Set::default(),
            step,
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.exits.is_empty()
    }

    /// Computes a selective predication plan for a wave64 compute program on a
    /// 32-lane host. Returns an empty plan (zero codegen overhead) when no
    /// non-`in_step` block needs cross-half or split-wave LDS execution.
    pub(super) fn build(
        p: &Program,
        st: &Structured,
        base_step: Vec<bool>,
        one_wave_lds: bool,
    ) -> Self {
        let n = p.blocks.len();
        if n <= 1 {
            return Self::empty(base_step);
        }
        let needs = candidate_blocks(p, st, one_wave_lds);
        if !(0..n).any(|b| needs[b] && !base_step[b]) {
            return Self::empty(base_step);
        }

        let order = crate::ir::reverse_postorder(&p.blocks);
        let mut rpo = vec![usize::MAX; n];
        for (idx, &b) in order.iter().enumerate() {
            rpo[b] = idx;
        }
        let idom = crate::ir::idoms(&p.blocks, &order, &rpo);

        let mut plan = PredicatePlan {
            exits: Map::default(),
            in_diamond: Set::default(),
            diamond_joins: Map::default(),
            linear_merges: Set::default(),
            dead_blocks: Set::default(),
            idom,
            early_conts: Map::default(),
            skipped_ends: Set::default(),
            step: base_step,
        };
        let mut eff_exits = st.exits.clone();

        // Iteratively flatten eligible acyclic divergent branches whose
        // regions contain candidate blocks needing `in_step`.
        loop {
            let mut changed = false;
            let pdom = post_dominators(&eff_exits);
            let ipdom = immediate_post_dominators(&eff_exits, &pdom);

            for &d in &order {
                if !plan.step[d] || plan.exits.contains_key(&d) || plan.dead_blocks.contains(&d) {
                    continue;
                }
                let LaneExit::Branch {
                    taken, not_taken, ..
                } = eff_exits[d]
                else {
                    continue;
                };
                if taken == not_taken || rpo[taken] <= rpo[d] || rpo[not_taken] <= rpo[d] {
                    continue;
                }
                // Already uniform if both targets stay in_step (or neither side
                // has any candidate outside `step`).
                if plan.step[taken] && plan.step[not_taken] {
                    continue;
                }

                // Shape 1: acyclic early-out `if (cond) return;`.
                if let Some((cont, end_blk)) =
                    match_early_out(p, &eff_exits, d, taken, not_taken, &rpo, &needs, &plan.step)
                {
                    plan.exits.insert(d, FlatExit::EarlyOut { cont });
                    plan.early_conts.insert(cont, d);
                    plan.skipped_ends.insert(end_blk);
                    eff_exits[d] = LaneExit::Jump(cont);
                    if p.blocks[end_blk].preds.iter().all(|&q| {
                        plan.dead_blocks.contains(&q)
                            || matches!(plan.exits.get(&q), Some(FlatExit::EarlyOut { .. }))
                    }) {
                        plan.dead_blocks.insert(end_blk);
                    }
                    plan.step = barriers::in_step_with(p, st, &eff_exits);
                    changed = true;
                    break;
                }

                // Shape 2: acyclic single-entry single-exit diamond `d -> J`.
                if let Some(join) = ipdom[d]
                    && let Some(region) = match_sese_diamond(
                        p,
                        &eff_exits,
                        d,
                        taken,
                        not_taken,
                        join,
                        &rpo,
                        &needs,
                        &plan,
                    )
                {
                    // Linearize `d -> region[0] -> ... -> region[k-1] -> join`.
                    plan.exits.insert(d, FlatExit::Linear { next: region[0] });
                    eff_exits[d] = LaneExit::Jump(region[0]);
                    for (idx, &b) in region.iter().enumerate() {
                        let next = region.get(idx + 1).copied().unwrap_or(join);
                        plan.in_diamond.insert(b);
                        plan.exits.insert(b, FlatExit::Linear { next });
                        eff_exits[b] = LaneExit::Jump(next);
                        if !p.blocks[b].phis.is_empty() || p.blocks[b].preds.len() > 1 {
                            plan.linear_merges.insert(b);
                        }
                    }
                    plan.diamond_joins.insert(join, d);
                    plan.linear_merges.insert(join);
                    plan.step = barriers::in_step_with(p, st, &eff_exits);
                    changed = true;
                    break;
                }
            }
            if !changed {
                break;
            }
        }

        // If any dead block is unreachable from entry in `eff_exits`, keep it
        // out of `step`.
        for &b in &plan.dead_blocks {
            plan.step[b] = false;
        }
        plan
    }
}

/// Blocks that contain an operation needing `in_step` in `halves` mode
/// (cross-half wave ops, mask-as-data reads, `v_readlane`/`v_readfirstlane`,
/// or split-wave LDS accesses).
fn candidate_blocks(p: &Program, st: &Structured, one_wave_lds: bool) -> Vec<bool> {
    let mut needs = vec![false; p.blocks.len()];
    let mut mask_def: Map<Value, usize> = Map::default();
    for (b, block) in p.blocks.iter().enumerate() {
        for phi in &block.phis {
            if st.masks.contains_key(&phi.value) || st.highs.contains_key(&phi.value) {
                mask_def.insert(phi.value, b);
            }
        }
        for inst in &block.insts {
            for &(_, w) in &inst.writes {
                if st.masks.contains_key(&w) || st.highs.contains_key(&w) {
                    mask_def.insert(w, b);
                }
            }
        }
    }

    for (b, block) in p.blocks.iter().enumerate() {
        for inst in &block.insts {
            let name = inst.name();
            if matches!(
                name,
                "v_readfirstlane_b32"
                    | "v_readlane_b32"
                    | "s_or_saveexec_b64"
                    | "v_mbcnt_lo_u32_b32"
                    | "v_mbcnt_hi_u32_b32"
                    | "s_bcnt1_i32_b64"
                    | "s_ff1_i32_b64"
                    | "s_flbit_i32_b64"
            ) {
                needs[b] = true;
            }
            // Any instruction reading a tracked lane mask as data (not a mask
            // operation or branch): mark both this block and the mask's
            // defining block.
            let is_mask_user = name.starts_with("s_cbranch")
                || name.starts_with("v_cndmask")
                || matches!(
                    name,
                    "s_and_b64"
                        | "s_or_b64"
                        | "s_xor_b64"
                        | "s_andn2_b64"
                        | "s_orn2_b64"
                        | "s_nand_b64"
                        | "s_nor_b64"
                        | "s_xnor_b64"
                        | "s_not_b64"
                        | "s_mov_b64"
                        | "s_and_saveexec_b64"
                        | "s_or_saveexec_b64"
                        | "s_xor_saveexec_b64"
                        | "s_andn2_saveexec_b64"
                        | "s_orn2_saveexec_b64"
                        | "s_nand_saveexec_b64"
                        | "s_nor_saveexec_b64"
                        | "s_xnor_saveexec_b64"
                );
            if !is_mask_user {
                for &(_, v) in &inst.reads {
                    if st.masks.contains_key(&v) || st.highs.contains_key(&v) {
                        needs[b] = true;
                        if let Some(&def_b) = mask_def.get(&v) {
                            needs[def_b] = true;
                        }
                    }
                }
            }
            if one_wave_lds
                && matches!(inst.fields, Fields::Ds { .. })
                && !name.starts_with("ds_swizzle")
                && !name.contains("permute")
            {
                needs[b] = true;
            }
        }
    }
    needs
}

fn is_trivial_end(p: &Program, exits: &[LaneExit], b: usize) -> bool {
    exits[b] == LaneExit::End
        && p.blocks[b].phis.is_empty()
        && p.blocks[b]
            .insts
            .iter()
            .all(|i| matches!(i.name(), "s_endpgm" | "s_nop" | "s_waitcnt"))
}

#[allow(clippy::too_many_arguments)]
fn match_early_out(
    p: &Program,
    exits: &[LaneExit],
    d: usize,
    taken: usize,
    not_taken: usize,
    rpo: &[usize],
    needs: &[bool],
    step: &[bool],
) -> Option<(usize, usize)> {
    let (end_blk, cont) = if is_trivial_end(p, exits, taken) {
        (taken, not_taken)
    } else if is_trivial_end(p, exits, not_taken) {
        (not_taken, taken)
    } else {
        return None;
    };
    if cont == end_blk || rpo[cont] <= rpo[d] || p.blocks[cont].preds != [d] {
        return None;
    }
    // Only predicate if some block reachable from `cont` (without going
    // through `end_blk`) needs `in_step` and is not currently `in_step`.
    let mut seen = vec![false; p.blocks.len()];
    let mut stack = vec![cont];
    let mut useful = false;
    while let Some(b) = stack.pop() {
        if b == end_blk || std::mem::replace(&mut seen[b], true) {
            continue;
        }
        if needs[b] && !step[b] {
            useful = true;
            break;
        }
        match exits[b] {
            LaneExit::Jump(t) => stack.push(t),
            LaneExit::Branch {
                taken: t,
                not_taken: f,
                ..
            } => stack.extend([t, f]),
            LaneExit::End | LaneExit::Indirect => {}
        }
    }
    useful.then_some((cont, end_blk))
}

#[allow(clippy::too_many_arguments)]
fn match_sese_diamond(
    p: &Program,
    exits: &[LaneExit],
    d: usize,
    taken: usize,
    not_taken: usize,
    join: usize,
    rpo: &[usize],
    needs: &[bool],
    plan: &PredicatePlan,
) -> Option<Vec<usize>> {
    if join == d || rpo[join] <= rpo[d] {
        return None;
    }
    // Do not linearize if `join` is already a diamond join or a loop header.
    if plan.diamond_joins.contains_key(&join)
        || p.blocks[join].preds.iter().any(|&q| rpo[q] >= rpo[join])
    {
        return None;
    }
    let mut region_set = Set::default();
    let mut stack = Vec::new();
    for t in [taken, not_taken] {
        if t != join {
            stack.push(t);
        }
    }
    while let Some(b) = stack.pop() {
        if b == join || !region_set.insert(b) {
            continue;
        }
        if region_set.len() > MAX_DIAMOND_BLOCKS
            || plan.in_diamond.contains(&b)
            || plan.exits.contains_key(&b)
            || plan.dead_blocks.contains(&b)
        {
            return None;
        }
        match exits[b] {
            LaneExit::Jump(t) => {
                if rpo[t] <= rpo[b] {
                    return None; // cycle inside region
                }
                if t != join {
                    stack.push(t);
                }
            }
            LaneExit::Branch {
                taken: t,
                not_taken: f,
                ..
            } => {
                if rpo[t] <= rpo[b] || rpo[f] <= rpo[b] {
                    return None; // cycle inside region
                }
                for s in [t, f] {
                    if s != join {
                        stack.push(s);
                    }
                }
            }
            LaneExit::End | LaneExit::Indirect => return None,
        }
    }
    if region_set.is_empty() {
        return None;
    }
    // Single-entry: every predecessor of every block in `region_set` must be
    // `d` or inside `region_set`.
    for &b in &region_set {
        if !p.blocks[b]
            .preds
            .iter()
            .all(|&q| q == d || region_set.contains(&q))
        {
            return None;
        }
    }
    // Single-exit: every predecessor of `join` must be `d` or inside
    // `region_set` (excluding dead blocks).
    if !p.blocks[join]
        .preds
        .iter()
        .filter(|q| !plan.dead_blocks.contains(q))
        .all(|&q| q == d || region_set.contains(&q))
    {
        return None;
    }
    // Only predicate when at least one block in the diamond needs `in_step`.
    if !region_set.iter().any(|&b| needs[b] && !plan.step[b]) {
        return None;
    }
    let mut region: Vec<usize> = region_set.into_iter().collect();
    region.sort_unstable_by_key(|&b| rpo[b]);
    Some(region)
}

fn post_dominators(exits: &[LaneExit]) -> Vec<Vec<u64>> {
    let n = exits.len();
    let words = n.div_ceil(64);
    let full = vec![u64::MAX; words];
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, e) in exits.iter().enumerate() {
        match *e {
            LaneExit::Jump(t) => succ[b].push(t),
            LaneExit::Branch {
                taken, not_taken, ..
            } => {
                succ[b].push(taken);
                succ[b].push(not_taken);
            }
            LaneExit::End | LaneExit::Indirect => {}
        }
    }
    let mut pdom: Vec<Vec<u64>> = (0..n)
        .map(|b| {
            if succ[b].is_empty() {
                let mut v = vec![0; words];
                v[b / 64] |= 1 << (b % 64);
                v
            } else {
                full.clone()
            }
        })
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            if succ[b].is_empty() {
                continue;
            }
            let mut v = full.clone();
            for &t in &succ[b] {
                for (w, p) in v.iter_mut().zip(&pdom[t]) {
                    *w &= p;
                }
            }
            v[b / 64] |= 1 << (b % 64);
            if v != pdom[b] {
                pdom[b] = v;
                changed = true;
            }
        }
    }
    pdom
}

/// Immediate strict post-dominator per block (none for exits or unreachable).
fn immediate_post_dominators(exits: &[LaneExit], pdom: &[Vec<u64>]) -> Vec<Option<usize>> {
    let n = exits.len();
    let contains = |set: &[u64], x: usize| (set[x / 64] >> (x % 64)) & 1 == 1;
    let count = |set: &[u64]| -> u32 {
        let mut c = 0;
        for b in 0..n {
            if contains(set, b) {
                c += 1;
            }
        }
        c
    };
    let sizes: Vec<u32> = (0..n).map(|b| count(&pdom[b])).collect();
    (0..n)
        .map(|b| {
            let mut best: Option<(u32, usize)> = None;
            for (cand, &sz) in sizes.iter().enumerate() {
                if cand != b && contains(&pdom[b], cand) {
                    // The immediate post-dominator is the strict post-dominator
                    // with the largest pdom set (closest to `b`).
                    if best.is_none_or(|(best_sz, _)| sz > best_sz) {
                        best = Some((sz, cand));
                    }
                }
            }
            best.map(|(_, cand)| cand)
        })
        .collect()
}

impl Lower<'_> {
    /// Per-lane active predicate for IR block `b`, if `b` executes under
    /// predication (`None` means all lanes in the workgroup are active).
    pub(super) fn pred_active(&self, b: usize) -> Option<V> {
        self.block_active.get(b).copied().flatten()
    }

    /// Computes `self.block_active[b]` at the start of IR block `b` (when
    /// `self.pred` is non-empty) and lowers linearized phis at `b` into
    /// `select` chains instead of DXIL phis.
    pub(super) fn enter_predicated_block(&mut self, b: usize) -> Result<bool, Stop> {
        if self.pred.is_empty() {
            return Ok(false);
        }
        // 1. Compute `block_active[b]`.
        let active = if let Some(&d) = self.pred.diamond_joins.get(&b) {
            self.pred_active(d)
        } else if self.pred.in_diamond.contains(&b) {
            let mut acc: Option<V> = None;
            for &q in &self.x.p.blocks[b].preds {
                if let Some(&e) = self.pred_edge_active.get(&(q, b)) {
                    acc = Some(match acc {
                        None => e,
                        Some(prev) => self.s.m.bin(self.cur, Bin::Or, prev, e),
                    });
                }
            }
            acc
        } else if let Some(&d) = self.pred.early_conts.get(&b) {
            self.pred_edge_active.get(&(d, b)).copied()
        } else if self.pred.skipped_ends.contains(&b) {
            None
        } else {
            let idom = self.pred.idom.get(b).copied().unwrap_or(b);
            if idom != b && idom != usize::MAX {
                self.pred_active(idom)
            } else {
                None
            }
        };
        self.block_active[b] = active;

        if !self.pred.linear_merges.contains(&b) {
            return Ok(false);
        }

        // 2. Lower IR phis in linearized merge block `b` directly as `select`
        // chains over `b`'s original IR predecessors (all already lowered in
        // RPO order and strictly dominating `b`).
        let preds = &self.x.p.blocks[b].preds;
        for (i, phi) in self.x.p.blocks[b].phis.iter().enumerate() {
            if matches!(phi.reg, Reg::S(c) if c == EXEC || c == EXEC + 1) {
                self.vals.insert(phi.value, Val::Missing("a lane mask"));
                continue;
            }
            if let Some(&why) = self.bad.get(&phi.value) {
                self.vals.insert(phi.value, Val::Missing(why));
                continue;
            }
            let bit = phi.reg == Reg::Scc;
            let mut acc: Option<V> = None;
            let mut missing: Option<Why> = None;
            for (n, &q) in preds.iter().enumerate() {
                if self.pred.dead_blocks.contains(&q) || self.tail[q].is_none() {
                    continue;
                }
                let Some(&edge_act) = self.pred_edge_active.get(&(q, b)) else {
                    continue;
                };
                let arg = phi.args[n];
                let val = match (self.vals.get(&arg).copied(), bit) {
                    (Some(Val::I32(x)), false) | (Some(Val::Bit(x)), true) => Some(x),
                    _ if self.never_taken(q, b) || self.dead_on_edge(b, n, phi.reg) => Some(if bit {
                        self.s.m.bool(false)
                    } else {
                        self.s.m.i32(0)
                    }),
                    (Some(Val::Missing(why)), _) => {
                        missing.get_or_insert(why);
                        None
                    }
                    _ => {
                        missing.get_or_insert("a join of mismatched values");
                        None
                    }
                };
                if let Some(x) = val {
                    acc = Some(match acc {
                        None => x,
                        Some(prev) => self.s.m.select(self.cur, edge_act, x, prev),
                    });
                }
            }
            let result = match (acc, missing) {
                (Some(x), None) => {
                    if bit {
                        Val::Bit(x)
                    } else {
                        Val::I32(x)
                    }
                }
                (_, Some(why)) => Val::Missing(why),
                (None, None) => {
                    if bit {
                        Val::Bit(self.s.m.bool(false))
                    } else {
                        Val::I32(self.s.m.i32(0))
                    }
                }
            };
            let _ = i;
            self.vals.insert(phi.value, result);
        }
        Ok(true)
    }

    /// Evaluates a mask phi (`Atom::Phi`) at a linearized merge block `b` as a
    /// `select` chain over `b`'s original predecessors.
    pub(super) fn predicated_mask_phi(
        &mut self,
        id: Value,
        b: usize,
        i: usize,
        at: usize,
    ) -> Result<V, Why> {
        let phi = &self.x.p.blocks[b].phis[i];
        let preds = &self.x.p.blocks[b].preds;
        let mut acc: Option<V> = None;
        for (n, &q) in preds.iter().enumerate() {
            if self.pred.dead_blocks.contains(&q) {
                continue;
            }
            let Some(qb) = self.tail[q] else { continue };
            let Some(&edge_act) = self.pred_edge_active.get(&(q, b)) else {
                continue;
            };
            let arg = phi.args[n];
            let m_val = match self.x.st.masks.get(&arg) {
                Some(&m) => self.boolean(m, qb)?,
                None => self.s.m.bool(false),
            };
            acc = Some(match acc {
                None => m_val,
                Some(prev) => self.s.m.select(at, edge_act, m_val, prev),
            });
        }
        let v = acc.unwrap_or_else(|| self.s.m.bool(false));
        self.mask_phis.insert(id, v);
        Ok(v)
    }

    /// Records per-edge active predicates at the end of IR block `b` and, if
    /// `b` has a flattened exit in `self.pred`, returns its replacement
    /// unconditional terminator `Some(Term::Br(next))`.
    pub(super) fn exit_predicated_block(&mut self, b: usize) -> Result<Option<Term>, Stop> {
        let Some(&flat) = self.pred.exits.get(&b) else {
            return Ok(None);
        };
        let offset = self.x.p.blocks[b]
            .insts
            .last()
            .map_or(self.x.p.blocks[b].offset, |i| i.machine.offset);
        let cur = self.cur;
        let base_act = self.pred_active(b);
        let and_base = |l: &mut Self, cond: V| match base_act {
            Some(a) => l.s.m.bin(cur, Bin::And, a, cond),
            None => cond,
        };
        match self.x.st.exits[b] {
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => {
                let cond = self
                    .boolean(when, cur)
                    .map_err(|w| unsupported(offset, w))?;
                let t = self.s.m.bool(true);
                let not_cond = self.s.m.bin(cur, Bin::Xor, cond, t);
                let act_taken = and_base(self, cond);
                let act_not_taken = and_base(self, not_cond);
                self.pred_edge_active.insert((b, taken), act_taken);
                self.pred_edge_active.insert((b, not_taken), act_not_taken);
            }
            LaneExit::Jump(target) => {
                let act = base_act.unwrap_or_else(|| self.s.m.bool(true));
                self.pred_edge_active.insert((b, target), act);
            }
            LaneExit::End | LaneExit::Indirect => {}
        }
        let target = match flat {
            FlatExit::EarlyOut { cont } => cont,
            FlatExit::Linear { next } => next,
        };
        self.enter_loop(b, target, cur);
        Ok(Some(Term::Br(self.block[target].expect("reachable"))))
    }
}
