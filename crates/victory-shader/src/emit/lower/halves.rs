//! Console wave64 compute programs on a host whose waves have at most 32
//! lanes (`WaveLaneCountMax` 32: Apple GPUs under D3DMetal, NVIDIA, Intel).
//! Such a host refuses or mishandles `WaveSize(64)`, which a wave64 program
//! that needs the console's lanes ([`Lower::console_lanes`]) declares.
//!
//! Each console wave runs as two host waves of 32 lanes ("halves"): the
//! thread relabelling ([`Lower::thread_id`]) gives each host wave a 32-aligned
//! run of flat thread ids, so host wave `2w + h` holds console wave `w`'s
//! lanes `32h..32h + 31`, host lane `n` being console lane `32h + n`:
//!
//! * within a half, exact: DPP (rows of 16, DPP8 groups of 8), `ds_swizzle`
//!   (groups of 32 or quads), `v_permlanex16_b32`, and `ds_bpermute_b32`,
//!   which in wave64 permutes each half on its own. A lane's own bit of a
//!   mask held as a number (a constant, data loaded or computed by scalar
//!   code), `v_writelane_b32` (the console lane named takes the scalar; SGPR
//!   spills read back in the same block or SSA value are resolved by
//!   `resource::Tracer::restored`) and the row a DPP row mask names (rows 2
//!   and 3 are the upper half's) are decided by [`Lower::console_lane`];
//! * per-half on 32-lane hosts (`Marvisak/VIctory#2711`): wave ballots read
//!   as 64-bit numbers hold the host wave's 32 lanes in their low dword and
//!   0 in their high dword; `v_readlane_b32` masks its lane index to the
//!   32-lane half ([`Lower::host_lane`]); `v_readfirstlane_b32` reads the
//!   host wave's first active lane (exact for wave-uniform values, half-local
//!   otherwise); and wave reductions and prefix scans run across each 32-lane
//!   host wave;
//! * LDS: a group of 33 to 64 threads (one console wave64 split across two
//!   host waves) meets at group barriers between its LDS phases
//!   (`lds::barriers`), and is refused where no point every thread reaches
//!   separates them; groups of several console waves order their LDS with
//!   barriers, and those of 32 threads or fewer are one host wave.

use super::*;

/// Console lanes per host wave in halves.
const HALF: u32 = 32;

/// The ballot word and bit of console lane `lane` on a host wave of
/// `host_wave` lanes: the host wave's own lane, which [`Lower::read_lane`]
/// reads (on a host of 32-lane waves a lane 32-63 names that lane of the
/// half the host wave stands for).
pub(super) fn ballot_bit(lane: u32, host_wave: u32) -> (u32, u32) {
    let local = lane & (host_wave - 1);
    (local / 32, local % 32)
}

impl Lower<'_> {
    /// Whether a wave64 program whose lanes follow the console's runs in
    /// halves: the host's waves are 32 lanes at most.
    pub(super) fn halves_wanted(x: &Parts, needs_wave: bool) -> bool {
        needs_wave && x.native.wave64 && x.host_wave32
    }

    /// The lanes of a host wave running this program: 32 in halves, else
    /// the console's wave width.
    pub(super) fn host_wave(&self) -> u32 {
        if self.halves { HALF } else { self.wave() }
    }

    /// A lane number as the host wave can take it: in halves, console lanes
    /// 32-63 name the corresponding lane `0..31` of the half this host wave
    /// stands for.
    pub(super) fn host_lane(&mut self, lane: V) -> V {
        if !self.halves {
            return lane;
        }
        let m = self.s.m.i32(HALF - 1);
        self.s.m.bin(self.cur, Bin::And, lane, m)
    }

    /// The flat thread id, relabelled so that each host wave holds a run of
    /// [`Shader::wave_size`]-aligned ids from its first lane (see
    /// [`Lower::thread_id`]); made once, in the entry block.
    pub(super) fn relabelled_flat(&mut self) -> V {
        if let Some(f) = self.relabelled {
            return f;
        }
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        let flat = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            i32t,
            96,
            &[],
        );
        let first = self.s.op(
            e,
            "waveReadLaneFirst.i32",
            Attr::NoUnwind,
            i32t,
            118,
            &[flat],
        );
        let wave = self.s.wave_size.expect("set with console lanes");
        let mask = self.s.m.i32(!(wave - 1));
        let base = self.s.m.bin(e, Bin::And, first, mask);
        let lane = self
            .s
            .op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let f = self.s.m.bin(e, Bin::Add, base, lane);
        self.s.flags |= flags::WAVE_OPS;
        self.relabelled = Some(f);
        f
    }

    /// This lane's index in its console wave: the host lane, or in halves
    /// the host lane in the half the host wave holds.
    pub(super) fn console_lane(&mut self) -> V {
        if !self.halves {
            return self.lane_index();
        }
        let f = self.relabelled_flat();
        let m = self.s.m.i32(self.wave() - 1);
        self.s.m.bin(self.s.entry, Bin::And, f, m)
    }

    /// Whether the group fits one host wave: a console wave64 of 32 threads
    /// or fewer has no lane in the upper half.
    pub(super) fn one_host_wave(&self) -> bool {
        self.s
            .threads
            .is_some_and(|t| t.iter().product::<u32>() <= HALF)
    }
}
