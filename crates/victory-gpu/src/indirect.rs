//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either. By default (`VICTORY_INDIRECT_REST=0`), a record that does not
//! fit in the remaining budget runs empty without taking the rest, so
//! smaller draws after it still run.
//!
//! With a count buffer ExecuteIndirect runs only the first min(max, count)
//! records, so the clamp reads the count word and stops there: the maximum
//! can be far above the count, and the records between never run.
//!
//! shortcut: the bounds (below 2^23 vertices per instance, 2^26 vertex
//! instances, 65,535 groups per dimension except in x where [`widest`]
//! allows more, 2^20 per dispatch, one-dimensional mesh dispatches up to
//! 65,535 groups, the budget) cut
//! legitimate work beyond them; raise them if the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use std::sync::atomic::{AtomicU64, Ordering};

use crate::host::{HostList, host, transition, uav_barrier};

const SOURCE: &str = include_str!("indirect-clamp.hlsl");

/// The most groups in x the clamp lets a dispatch run (2^20 groups in all
/// still bound it). The title's tile passes are dispatches of one group per
/// 8x4 tile of the picture, 67,440 and 47,760 for its two tile classes in a
/// frame at 2560x1440 (measured on Windows in VIctory-7900XTX, 84eae35);
/// cut to 65,535, the last tiles, the lowest of the picture, never run.
/// The console runs what it is asked, and so does D3DMetal: an M5 Pro ran
/// every group of indirect dispatches up to 2^20 wide. D3D12's own limit,
/// 65,535, stays for other hosts. `VICTORY_INDIRECT_WIDE=0` or `1`
/// overrides.
pub(crate) fn widest() -> u32 {
    static WIDEST: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *WIDEST.get_or_init(|| {
        let wide = match std::env::var("VICTORY_INDIRECT_WIDE").as_deref() {
            Ok("0") => false,
            Ok("1") => true,
            _ => crate::shader_cache::under_wine(),
        };
        if wide { 1 << 20 } else { 65_535 }
    })
}

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// Built from [`WORDS_SOURCE`] or [`WORDS_1_SOURCE`], which also reads
    /// the budget as words through root view 5 (the other views are those of
    /// [`SOURCE`]).
    words: bool,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    #[cfg(victory_dxc_clamp)]
    match precompiled() {
        Ok(clamp) => {
            tracing::info!("argument clamp: using precompiled 64-thread shader");
            return Ok(clamp);
        }
        Err(e) => tracing::warn!(error = %e, "precompiled argument clamp unavailable"),
    }
    make_fxc()
        .or_else(|e| {
            tracing::info!(error = %e, "argument clamp: building the 64-lane words form");
            make_words()
        })
        .or_else(|e| {
            tracing::info!(error = %e, "argument clamp: building the one-thread words form");
            make_words_1()
        })
}

/// Whether a record that does not fit in what is left of its list's budget
/// takes the rest (`VICTORY_INDIRECT_REST=1`), emptying every record after
/// it. Off (`0`) by default so one oversized record zeroes only itself.
fn indirect_rest() -> bool {
    #[cfg(test)]
    if let Some(rest) = TEST_INDIRECT_REST.with(std::cell::Cell::get) {
        return rest;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_INDIRECT_REST"))
}

#[cfg(test)]
thread_local! {
    static TEST_INDIRECT_REST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn with_indirect_rest<R>(rest: bool, f: impl FnOnce() -> R) -> R {
    struct Guard(Option<bool>);
    impl Drop for Guard {
        fn drop(&mut self) {
            TEST_INDIRECT_REST.with(|cell| cell.set(self.0));
        }
    }
    let prev = TEST_INDIRECT_REST.with(|cell| cell.replace(Some(rest)));
    let _guard = Guard(prev);
    f()
}

/// The clamp compiled by DXC at build time. D3DMetal finds a shader by the
/// `HASH` part of its container, and DXC leaves that and the container
/// digest as zeros when it cannot find `dxil.dll` to sign with, which would
/// make this shader share a cache entry with every other unsigned one (issue
/// #4). So both are checked here and set when they are zero.
#[cfg(victory_dxc_clamp)]
fn precompiled() -> windows::core::Result<Clamp> {
    let mut bytes = include_bytes!(concat!(env!("OUT_DIR"), "/indirect-clamp.dxil")).to_vec();
    match victory_shader::emit::sign_container(&mut bytes) {
        Ok(true) => tracing::info!("argument clamp: the compiled shader was unsigned, its digests are set"),
        Ok(false) => {}
        Err(e) => tracing::warn!(error = %e, "argument clamp: could not check the compiled shader's digests"),
    }
    from_bytes(&bytes)
}

/// The 64-lane clamp for Wine's HLSL compiler (vkd3d-shader), which has no
/// `Load` on `RWByteAddressBuffer`, no stores to indexed group-shared memory
/// and no `[RootSignature]`: [`SOURCE`]'s 64-lane tile clamp, read through
/// structured views of the same buffers, with per-lane scratch slots in
/// `budget` bytes `64..320` (`budgetWords[16..80]`).
const WORDS_SOURCE: &str = r#"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint flags; uint widest; };
StructuredBuffer<uint> argWords : register(t0);
RWByteAddressBuffer budget : register(u1);
RWByteAddressBuffer dest : register(u0);
StructuredBuffer<uint> countWords : register(t1);
RWStructuredBuffer<uint> budgetWords : register(u2);
static const uint BUDGET = 1u << 27;
static const uint MOST_DRAW = 1u << 23;
static const uint GROUPS = 1u << 20;
static const uint PAST = 0xffffffffu;
uint3 load3(uint at) { return uint3(argWords[at / 4u], argWords[at / 4u + 1u], argWords[at / 4u + 2u]); }
uint clamped(uint at) {
    uint cost;
    uint base_bytes;
    if (kind <= 1u) {
        uint raw_count = argWords[at / 4u];
        uint raw_instances = argWords[at / 4u + 1u];
        uint count = raw_count;
        uint instances = 0u;
        if (raw_count >= MOST_DRAW) {
            count = 0u;
        } else {
            instances = min(raw_instances, (1u << 26) / max(count, 1u));
        }
        cost = count * instances;
        dest.Store2(at, uint2(count, instances));
        if (kind == 1u) {
            dest.Store3(at + 8u, uint3(argWords[at / 4u + 2u], argWords[at / 4u + 3u], argWords[at / 4u + 4u]));
            base_bytes = 20u;
        } else {
            dest.Store2(at + 8u, uint2(argWords[at / 4u + 2u], argWords[at / 4u + 3u]));
            base_bytes = 16u;
        }
    } else if (kind == 2u) {
        uint3 g = load3(at);
        g = uint3(g.x <= widest ? g.x : 65535u, min(g.y, 65535u), min(g.z, 65535u));
        if ((g.y != 0u && g.x > GROUPS / g.y) || (g.z != 0u && g.x * g.y > GROUPS / g.z)) {
            g = uint3(0u, 0u, 0u);
        }
        cost = g.x * g.y * g.z * 64u;
        dest.Store3(at, g);
        base_bytes = 12u;
    } else {
        uint3 raw_g = load3(at);
        uint3 g = uint3(raw_g.x, min(raw_g.y, 1u), min(raw_g.z, 1u));
        if (raw_g.x > 65535u) {
            g = uint3(0u, 0u, 0u);
        }
        cost = g.x * g.y * g.z * 128u;
        dest.Store3(at, g);
        base_bytes = 12u;
    }
    for (uint extra = base_bytes; extra < stride; extra += 4u) {
        dest.Store(at + extra, argWords[(at + extra) / 4u]);
    }
    return cost;
}
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint lane = id.x;
    uint left = 0u;
    if (lane == 0u) {
        left = BUDGET - min(budgetWords[0], BUDGET);
    }
    uint walked = records;
    if ((flags & 1u) != 0u) {
        walked = min(records, countWords[0]);
    }
    uint take_rest = flags & 2u;
    uint tiles = (records + 63u) / 64u;
    for (uint tile = 0u; tile < tiles; tile++) {
        uint record = tile * 64u + lane;
        uint at = record * stride;
        uint cost = 0u;
        if (record < walked) {
            cost = clamped(at);
        }
        budget.Store(64u + lane * 4u, cost);
        AllMemoryBarrierWithGroupSync();
        if (lane == 0u && tile * 64u < walked) {
            for (uint i = 0u; i < 64u; i++) {
                uint c = budgetWords[16u + i];
                if (c > left) {
                    budget.Store(64u + i * 4u, PAST);
                    if (take_rest != 0u) {
                        left = 0u;
                    }
                } else {
                    left -= c;
                }
            }
        }
        AllMemoryBarrierWithGroupSync();
        if (record < walked && budgetWords[16u + lane] == PAST) {
            if (kind <= 1u) {
                dest.Store2(at, uint2(0u, 0u));
            } else {
                dest.Store3(at, uint3(0u, 0u, 0u));
            }
        }
        if (tile + 1u < tiles) {
            AllMemoryBarrierWithGroupSync();
        }
    }
    if (lane == 0u) {
        budget.Store(0u, BUDGET - left);
    }
}
"#;

/// One-thread fallback for compilers that refuse [`WORDS_SOURCE`].
const WORDS_1_SOURCE: &str = r#"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint flags; uint widest; };
StructuredBuffer<uint> argWords : register(t0);
RWByteAddressBuffer budget : register(u1);
RWByteAddressBuffer dest : register(u0);
StructuredBuffer<uint> countWords : register(t1);
RWStructuredBuffer<uint> budgetWords : register(u2);
static const uint BUDGET = 1u << 27;
static const uint MOST_DRAW = 1u << 23;
static const uint GROUPS = 1u << 20;
uint3 load3(uint at) { return uint3(argWords[at / 4u], argWords[at / 4u + 1u], argWords[at / 4u + 2u]); }
uint clamped(uint at) {
    uint cost;
    uint base_bytes;
    if (kind <= 1u) {
        uint raw_count = argWords[at / 4u];
        uint raw_instances = argWords[at / 4u + 1u];
        uint count = raw_count;
        uint instances = 0u;
        if (raw_count >= MOST_DRAW) {
            count = 0u;
        } else {
            instances = min(raw_instances, (1u << 26) / max(count, 1u));
        }
        cost = count * instances;
        dest.Store2(at, uint2(count, instances));
        if (kind == 1u) {
            dest.Store3(at + 8u, uint3(argWords[at / 4u + 2u], argWords[at / 4u + 3u], argWords[at / 4u + 4u]));
            base_bytes = 20u;
        } else {
            dest.Store2(at + 8u, uint2(argWords[at / 4u + 2u], argWords[at / 4u + 3u]));
            base_bytes = 16u;
        }
    } else if (kind == 2u) {
        uint3 g = load3(at);
        g = uint3(g.x <= widest ? g.x : 65535u, min(g.y, 65535u), min(g.z, 65535u));
        if ((g.y != 0u && g.x > GROUPS / g.y) || (g.z != 0u && g.x * g.y > GROUPS / g.z)) {
            g = uint3(0u, 0u, 0u);
        }
        cost = g.x * g.y * g.z * 64u;
        dest.Store3(at, g);
        base_bytes = 12u;
    } else {
        uint3 raw_g = load3(at);
        uint3 g = uint3(raw_g.x, min(raw_g.y, 1u), min(raw_g.z, 1u));
        if (raw_g.x > 65535u) {
            g = uint3(0u, 0u, 0u);
        }
        cost = g.x * g.y * g.z * 128u;
        dest.Store3(at, g);
        base_bytes = 12u;
    }
    for (uint extra = base_bytes; extra < stride; extra += 4u) {
        dest.Store(at + extra, argWords[(at + extra) / 4u]);
    }
    return cost;
}
[numthreads(1, 1, 1)]
void main() {
    uint left = BUDGET - min(budgetWords[0], BUDGET);
    uint walked = records;
    if ((flags & 1u) != 0u) {
        walked = min(records, countWords[0]);
    }
    uint take_rest = flags & 2u;
    for (uint record = 0u; record < walked; record++) {
        uint at = record * stride;
        uint cost = clamped(at);
        if (cost > left) {
            if (kind <= 1u) {
                dest.Store2(at, uint2(0u, 0u));
            } else {
                dest.Store3(at, uint3(0u, 0u, 0u));
            }
            if (take_rest != 0u) {
                left = 0u;
            }
        } else {
            left -= cost;
        }
    }
    budget.Store(0u, BUDGET - left);
}
"#;

fn make_words() -> windows::core::Result<Clamp> {
    make_words_from(WORDS_SOURCE)
}

fn make_words_1() -> windows::core::Result<Clamp> {
    make_words_from(WORDS_1_SOURCE)
}

fn make_words_from(source: &str) -> windows::core::Result<Clamp> {
    use crate::compute::Param;
    let params = [
        Param::Constants { register: 0, count: 5 },
        Param::Srv { register: 0, space: 0 },
        Param::Uav { register: 1 },
        Param::Uav { register: 0 },
        Param::Srv { register: 1, space: 0 },
        Param::Uav { register: 2 },
    ];
    let (root, pso) = crate::compute::pipeline(source, s!("clamp"), 0, &params)?;
    Ok(Clamp { root, pso, words: true })
}

fn make_fxc() -> windows::core::Result<Clamp> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the compiler owns the blob until this call returns.
    let bytes = unsafe { std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize()) };
    from_bytes(bytes)
}

fn from_bytes(bytes: &[u8]) -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    // SAFETY: D3D12 validates the bytecode and its embedded root signature.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso, words: false })
    }
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` (a title
/// buffer's copy, only read) into the list's own argument buffer, left
/// ready to be read as arguments: that buffer, for [`done`] after the
/// execution. Replaces the list's compute root signature and pipeline
/// (callers bind theirs after). `None` if the clamp is not available.
///
/// `count` is the host address of the title's record count (a title
/// buffer's copy, only read): the clamp then stops at the lesser of it and
/// `records`, which is where ExecuteIndirect stops. A count that is not
/// 4-byte aligned is not read: every record is clamped.
///
/// Title buffers' copies are never transitioned: they are common between
/// submissions and every use promotes them (shader and root views, the
/// input assembler, these reads), with unordered-access barriers for order.
/// Only the list's argument buffer moves, common, unordered access,
/// indirect argument, common: its states are known whatever order lists
/// run in.
// shortcut: a buffer a shader writes through an unordered view and the
// input assembler then reads in one submission is, by the letter of D3D12,
// promoted to a write state the input assembler may not use (the debug
// layer cannot see view accesses; GPU-based validation might); upgrade to
// per-list tracked buffer states if a driver or that validation minds.
pub(crate) fn bound(
    list: &HostList,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
    count: Option<u64>,
) -> Option<ID3D12Resource> {
    bound_with(clamp()?, list, gpu, (kind, stride, records), count)
}

/// [`bound`] with a given clamp.
fn bound_with(
    c: &Clamp,
    list: &HostList,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
    count: Option<u64>,
) -> Option<ID3D12Resource> {
    let budget = list.budget()?;
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return None;
    }
    let count = count.filter(|at| at.is_multiple_of(4));
    RECORDS[0].fetch_add(u64::from(records), Ordering::Relaxed);
    if count.is_some() {
        RECORDS[1].fetch_add(u64::from(records), Ordering::Relaxed);
    }
    let out = list.arguments(u64::from(stride) * u64::from(records.max(1)))?;
    let l = &list.list;
    let flags = u32::from(count.is_some()) | (u32::from(indirect_rest()) << 1);
    let constants = [kind, stride, records, flags, widest()];
    // SAFETY: an open list; our root signature, pipeline and buffers, and
    // title buffers' copies the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, constants.len() as u32, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.SetComputeRootUnorderedAccessView(3, out.GetGPUVirtualAddress());
        // Set even when not read: a root view has to be set.
        l.SetComputeRootShaderResourceView(4, count.unwrap_or(gpu));
        if c.words {
            l.SetComputeRootUnorderedAccessView(5, budget);
        }
        // One group walks every record (up to the count).
        l.Dispatch(records.min(1), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        ]);
    }
    Some(out)
}

/// Records the clamps were asked to walk (the maximum of each execution)
/// since the last call, and the part of them under a count buffer, where
/// the walk stops at the count.
pub(crate) fn take_records() -> [u64; 2] {
    RECORDS.each_ref().map(|n| n.swap(0, Ordering::Relaxed))
}
static RECORDS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// After the execution: the list's argument buffer back to common.
pub(crate) fn done(list: &HostList, arguments: &ID3D12Resource) {
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            arguments,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
}

#[cfg(test)]
mod tests {
    use crate::backend::{HostGpu, HostMemory};

    /// The clamp compiles with the system's compiler (including both 64-lane
    /// and 1-lane words forms) and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
        assert!(super::make_words().is_ok());
        assert!(super::make_words_1().is_ok());
    }

    /// Every word the clamp `c` writes for `records` records of `kind` made
    /// from `seed` by a small generator, in one list with no count buffer.
    /// `None` without a host GPU.
    fn clamp_all(c: &super::Clamp, kind: u32, stride_words: usize, seed: u32, records: u32) -> Option<Vec<u32>> {
        let host = super::host()?;
        let all = stride_words * records as usize;
        let (buffer, data) = host.shared_buffer(all as u64 * 4 + 16).unwrap();
        let data = data.cast::<u32>();
        let list = host.new_list().unwrap();
        let mut state = seed;
        // SAFETY: our mapped buffer, written before the list runs and read
        // after it has; a live buffer.
        unsafe {
            for at in 0..all {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                // Mostly small values, now and then a wild or all-ones one.
                let word = match state >> 28 {
                    0 => state,
                    1 => u32::MAX,
                    _ => (state >> 8) % 9000,
                };
                data.add(at).write_volatile(word);
            }
            let spec = (kind, stride_words as u32 * 4, records);
            let out = super::bound_with(c, &list, buffer.GetGPUVirtualAddress(), spec, None).unwrap();
            super::done(&list, &out);
            list.close();
            host.execute(&[&list]);
            let mut out_data = std::ptr::null_mut();
            out.Map(0, None, Some(&mut out_data)).unwrap();
            Some((0..all).map(|i| out_data.cast::<u32>().add(i).read_volatile()).collect())
        }
    }

    /// The precompiled clamp is the one in use when it builds, and it writes
    /// what the clamp compiled at run time writes for the same records of
    /// every kind.
    #[cfg(victory_dxc_clamp)]
    #[test]
    #[ignore = "requires a hardware D3D12 queue and the build-time DXC shader"]
    fn hardware_uses_precompiled_clamp() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        let precompiled = super::precompiled().expect("create pipeline from the compiled clamp");
        assert!(!precompiled.words);
        assert!(!super::clamp().expect("argument clamp").words);
        let reference = super::make_fxc().or_else(|_| super::make_words()).expect("the existing clamp");
        for (kind, per, seed, records) in [(0, 4, 1, 200), (1, 5, 2, 200), (2, 3, 3, 130), (10, 3, 4, 70), (0, 8, 5, 135)] {
            let new = clamp_all(&precompiled, kind, per, seed, records).unwrap();
            let old = clamp_all(&reference, kind, per, seed, records).unwrap();
            assert_eq!(new, old, "kind {kind}, {records} records");
        }
    }

    /// The 64-lane `WORDS_SOURCE` clamp and the 1-lane `WORDS_1_SOURCE`
    /// fallback produce identical words across multi-tile batches and padded
    /// strides, under both `VICTORY_INDIRECT_REST=0` and `=1`.
    #[test]
    fn words_64_lane_matches_1_lane_across_tiles_and_strides() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        let wide = super::make_words().expect("64-lane words clamp");
        let single = super::make_words_1().expect("1-lane words clamp");
        for rest in [false, true] {
            super::with_indirect_rest(rest, || {
                for (kind, per, seed, records) in [
                    (0, 4, 11, 150),
                    (0, 8, 12, 135),
                    (1, 5, 13, 140),
                    (2, 3, 14, 130),
                    (10, 3, 15, 145),
                    (10, 8, 16, 132),
                ] {
                    let a = clamp_all(&wide, kind, per, seed, records).unwrap();
                    let b = clamp_all(&single, kind, per, seed, records).unwrap();
                    assert_eq!(a, b, "rest {rest}, kind {kind}, stride_words {per}, records {records}");
                }
            });
        }
    }

    /// Clamps `records` records of `kind`, filled by repeating `words`,
    /// `calls` times in one list: each record's first word in the list's
    /// argument buffer after (zero when it runs empty). The title's copy,
    /// one record more included, stays as written. `None` without a host
    /// GPU.
    fn clamped(kind: u32, words: &[u32], records: u32, calls: u32) -> Option<Vec<u32>> {
        clamped_under(kind, words, records, calls, None)
    }

    /// As [`clamped`], with the title's record count `count` (a word in a
    /// buffer of its own) if there is one.
    fn clamped_under(
        kind: u32,
        words: &[u32],
        records: u32,
        calls: u32,
        count: Option<u32>,
    ) -> Option<Vec<u32>> {
        let host = super::host()?;
        // Words in a record of `kind`.
        let per = match kind {
            0 => 4,
            1 => 5,
            _ => 3,
        };
        let spec = (kind, per as u32 * 4, records);
        let all = per * (records as usize + 1);
        let (buffer, data) = host.shared_buffer(all as u64 * 4).unwrap();
        let data = data.cast::<u32>();
        let list = host.new_list().unwrap();
        // SAFETY: our mapped buffer, written before the list runs and read
        // after it has; a live buffer.
        unsafe {
            for at in 0..all {
                data.add(at).write_volatile(words[at % words.len()]);
            }
            let gpu = buffer.GetGPUVirtualAddress();
            let (counter, word) = host.shared_buffer(4).unwrap();
            word.cast::<u32>().write_volatile(count.unwrap_or(0));
            let counted = count.map(|_| counter.GetGPUVirtualAddress());
            let mut out = None;
            for _ in 0..calls {
                let clamped = super::bound(&list, gpu, spec, counted).unwrap();
                super::done(&list, &clamped);
                out = Some(clamped);
            }
            list.close();
            host.execute(&[&list]);
            for at in 0..all {
                assert_eq!(data.add(at).read_volatile(), words[at % words.len()], "the title's copy as written");
            }
            let mut out_data = std::ptr::null_mut();
            out?.Map(0, None, Some(&mut out_data)).unwrap();
            let first = |i: usize| out_data.cast::<u32>().add(i * per).read_volatile();
            Some((0..records as usize).map(first).collect())
        }
    }

    /// Unfilled draw records (`>= 1 << 23` vertices/indices) and unfilled
    /// mesh dispatch records (`> 65535` groups, including `0xffffffff`) are
    /// zeroed with trailing words preserved and spend zero budget.
    #[test]
    fn unfilled_draw_and_mesh_records_run_empty_without_spending_budget() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        // 80 draw records (`stride = 32` bytes = 8 words): 76 unfilled
        // (`1 << 23` and `u32::MAX`) followed by 4 valid records.
        const COUNT: usize = 80;
        let (draw_buf, draw_ptr) = host.shared_buffer((COUNT * 32) as u64).unwrap();
        let (mesh_buf, mesh_ptr) = host.shared_buffer((COUNT * 32) as u64).unwrap();
        // SAFETY: our mapped buffers, written before execution and read after.
        unsafe {
            let draws = std::slice::from_raw_parts_mut(draw_ptr.cast::<[u32; 8]>(), COUNT);
            for (i, d) in draws.iter_mut().enumerate() {
                let k = i as u32;
                let head = match i {
                    0..=37 => [u32::MAX, u32::MAX],
                    38..=75 => [1 << 23, 4],
                    76 => [(1 << 23) - 1, 1],
                    _ => [12, 2],
                };
                *d = [head[0], head[1], 100 + k, 200 + k, 0xa000 + k, 0xb000 + k, 0xc000 + k, 0xd000 + k];
            }
            let meshes = std::slice::from_raw_parts_mut(mesh_ptr.cast::<[u32; 8]>(), COUNT);
            for (i, m) in meshes.iter_mut().enumerate() {
                let k = i as u32;
                let gx = match i {
                    0..=37 => u32::MAX,
                    38..=75 => 65_536,
                    76 => 65_535,
                    _ => 7,
                };
                *m = [gx, 9, 5, 0x1000 + k, 0x2000 + k, 0x3000 + k, 0x4000 + k, 0x5000 + k];
            }

            let list = host.new_list().unwrap();
            let out_d = super::bound(&list, draw_buf.GetGPUVirtualAddress(), (0, 32, COUNT as u32), None).unwrap();
            super::done(&list, &out_d);
            list.close();
            host.execute(&[&list]);
            let mut p = std::ptr::null_mut();
            out_d.Map(0, None, Some(&mut p)).unwrap();
            let got_d = std::slice::from_raw_parts(p.cast::<[u32; 8]>(), COUNT);
            for (i, r) in got_d.iter().enumerate() {
                let k = i as u32;
                let expected_head = match i {
                    0..=75 => [0, 0],
                    76 => [(1 << 23) - 1, 1],
                    _ => [12, 2],
                };
                assert_eq!(
                    *r,
                    [expected_head[0], expected_head[1], 100 + k, 200 + k, 0xa000 + k, 0xb000 + k, 0xc000 + k, 0xd000 + k],
                    "draw record {i}"
                );
            }
            out_d.Unmap(0, None);

            // On a fresh list, 76 unfilled `0xffffffff` mesh records before 16
            // max-cost `65535` mesh records spend nothing so all 16 run.
            let (mesh16_buf, mesh16_ptr) = host.shared_buffer(((76 + 16) * 12) as u64).unwrap();
            let m16 = std::slice::from_raw_parts_mut(mesh16_ptr.cast::<[u32; 3]>(), 76 + 16);
            for (i, m) in m16.iter_mut().enumerate() {
                *m = if i < 76 { [u32::MAX; 3] } else { [65_535, 4, 4] };
            }
            list.reset();
            let out_m16 = super::bound(&list, mesh16_buf.GetGPUVirtualAddress(), (10, 12, (76 + 16) as u32), None).unwrap();
            super::done(&list, &out_m16);
            list.close();
            host.execute(&[&list]);
            out_m16.Map(0, None, Some(&mut p)).unwrap();
            let got_m16 = std::slice::from_raw_parts(p.cast::<[u32; 3]>(), 76 + 16);
            assert!(got_m16[..76].iter().all(|&m| m == [0, 0, 0]));
            assert!(got_m16[76..].iter().all(|&m| m == [65_535, 1, 1]));
            out_m16.Unmap(0, None);

            list.reset();
            let out_m = super::bound(&list, mesh_buf.GetGPUVirtualAddress(), (10, 32, COUNT as u32), None).unwrap();
            super::done(&list, &out_m);
            list.close();
            host.execute(&[&list]);
            out_m.Map(0, None, Some(&mut p)).unwrap();
            let got_m = std::slice::from_raw_parts(p.cast::<[u32; 8]>(), COUNT);
            for (i, r) in got_m.iter().enumerate() {
                let k = i as u32;
                let head = match i {
                    0..=75 => [0, 0, 0],
                    76 => [65_535, 1, 1],
                    _ => [7, 1, 1],
                };
                assert_eq!(
                    *r,
                    [head[0], head[1], head[2], 0x1000 + k, 0x2000 + k, 0x3000 + k, 0x4000 + k, 0x5000 + k],
                    "mesh record {i}"
                );
            }
            out_m.Unmap(0, None);
        }
    }

    /// Rejected records do not wrap the budget: of 128 max-cost draws (2^26
    /// vertex instances each once clamped) two run, where a wrapping add
    /// let four; of 600 max-cost mesh dispatches 16, not 31.
    #[test]
    fn rejected_records_do_not_wrap_the_budget() {
        let Some(draws) = clamped(0, &[1 << 22, u32::MAX, 0, 0], 128, 1) else {
            return; // no host GPU here
        };
        assert_eq!(draws.iter().filter(|&&count| count != 0).count(), 2);
        let meshes = clamped(10, &[65_535, 1, 1], 600, 1).unwrap();
        assert_eq!(meshes.iter().filter(|&&groups| groups != 0).count(), 16);
    }

    /// The budget goes to the first records, in order: of 200 draws of
    /// 3 * 2^18 vertices the first 170 run (it ends inside the third tile);
    /// small records that all fit all run.
    #[test]
    fn the_budget_goes_to_the_first_records() {
        let Some(draws) = clamped(0, &[3 << 18, 1, 0, 0], 200, 1) else {
            return; // no host GPU here
        };
        assert!(draws[..170].iter().all(|&count| count != 0));
        assert!(draws[170..].iter().all(|&count| count == 0));
        let small = clamped(0, &[1], 1 << 16, 1).unwrap();
        assert!(small.iter().all(|&count| count == 1));
    }

    /// By default (`VICTORY_INDIRECT_REST=0`), a record that does not fit in
    /// the remaining budget runs empty while smaller records after it still
    /// run; with `VICTORY_INDIRECT_REST=1`, it takes what is left so later
    /// records run empty too.
    #[test]
    fn a_record_that_does_not_fit_respects_indirect_rest() {
        let mut words = [1 << 22, 12, 0, 0].repeat(3);
        words.extend([3, 1, 0, 0].repeat(5));
        let Some(draws_default) = clamped(0, &words, 8, 1) else {
            return; // no host GPU here
        };
        let ran_default: Vec<bool> = draws_default.iter().map(|&count| count != 0).collect();
        assert_eq!(
            ran_default,
            [true, true, false, true, true, true, true, true],
            "VICTORY_INDIRECT_REST=0 keeps budget for later fitting records"
        );
        let draws_rest = super::with_indirect_rest(true, || clamped(0, &words, 8, 1).unwrap());
        let ran_rest: Vec<bool> = draws_rest.iter().map(|&count| count != 0).collect();
        assert_eq!(
            ran_rest,
            [true, true, false, false, false, false, false, false],
            "VICTORY_INDIRECT_REST=1 ends the budget on the first overflow"
        );
    }

    /// Two lists clamping into their own argument buffers, run in either
    /// order: every barrier starts from the buffer's current state (the
    /// debug layer's rule; runs light19 and light21 had the title's GPU
    /// heap moved to unordered access under the input assembler), the
    /// title's copy is never transitioned, and all end common.
    #[test]
    fn argument_buffers_move_from_their_current_states() {
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (title, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: our mapped buffer; a live buffer.
        let gpu = unsafe {
            data.cast::<[u32; 3]>().write_volatile([70_000, 1, 1]);
            title.GetGPUVirtualAddress()
        };
        let key = |r: &super::ID3D12Resource| r.as_raw() as usize;
        for a_first in [true, false] {
            let lists = [host.new_list().unwrap(), host.new_list().unwrap()];
            issued();
            let recorded: Vec<_> = lists
                .iter()
                .map(|l| {
                    let out = super::bound(l, gpu, (2, 12, 1), None).unwrap();
                    super::done(l, &out);
                    l.close();
                    (out, issued())
                })
                .collect();
            let order = if a_first { [0, 1] } else { [1, 0] };
            host.execute(&[&lists[order[0]], &lists[order[1]]]);
            let mut run = issued();
            run.extend(order.iter().flat_map(|&i| recorded[i].1.clone()));
            let mut states: std::collections::HashMap<usize, u32> =
                [key(&title), key(&recorded[0].0), key(&recorded[1].0)].into_iter().map(|k| (k, 0)).collect();
            replay(&run, &mut states).unwrap();
            assert!(run.iter().all(|t| t.0 != key(&title)), "the title's copy is never transitioned");
            assert!(states.values().all(|&s| s == 0), "all common again");
            for (out, _) in &recorded {
                let mut p = std::ptr::null_mut();
                // SAFETY: a mapped buffer of ours, the lists have run.
                let clamped = unsafe {
                    out.Map(0, None, Some(&mut p)).unwrap();
                    p.cast::<[u32; 3]>().read_volatile()
                };
                assert_eq!(clamped, [70_000.min(super::widest()), 1, 1]);
            }
            // SAFETY: as above.
            assert_eq!(unsafe { data.cast::<[u32; 3]>().read_volatile() }, [70_000, 1, 1], "the title's copy as written");
        }
    }

    /// The budget is the list's: a second execution of the same records
    /// finds it spent, and they all run empty.
    #[test]
    fn a_spent_budget_stays_spent() {
        let Some(twice) = clamped(1, &[1 << 22, u32::MAX, 0, 0, 0], 64, 2) else {
            return; // no host GPU here
        };
        assert!(twice.iter().all(|&count| count == 0));
    }

    /// The walk stops at the count: of 128 max-cost draws under a count of 3
    /// the first two run and the third runs empty, as they do without one,
    /// and the others are never read (the argument buffer's fresh words stay
    /// zero). A count above the records changes nothing; a count of zero
    /// walks none.
    #[test]
    fn the_walk_stops_at_the_count() {
        let words = [1 << 22, u32::MAX, 0, 0];
        let Some(under) = clamped_under(0, &words, 128, 1, Some(3)) else {
            return; // no host GPU here
        };
        assert_eq!(under[..3], [1 << 22, 1 << 22, 0]);
        assert!(under[3..].iter().all(|&count| count == 0));
        let all = clamped(0, &words, 128, 1).unwrap();
        assert_eq!(under[..3], all[..3]);
        let above = clamped_under(0, &words, 128, 1, Some(1 << 20)).unwrap();
        assert_eq!(above, all);
        let none = clamped_under(0, &words, 128, 1, Some(0)).unwrap();
        assert!(none.iter().all(|&count| count == 0));
    }

    /// Only the records that run are charged to the budget: after a call
    /// under a count of 1 a second finds room for that record again, where
    /// the same draws with no count spent it all on records that never ran.
    #[test]
    fn records_past_the_count_are_not_charged() {
        let words = [1 << 22, u32::MAX, 0, 0];
        let Some(twice) = clamped_under(0, &words, 128, 2, Some(1)) else {
            return; // no host GPU here
        };
        assert_eq!(twice[0], 1 << 22);
        let spent = clamped(0, &words, 128, 2).unwrap();
        assert!(spent.iter().all(|&count| count == 0));
    }

    /// The groups of one indirect dispatch of `groups` x 1 x 1 that run on
    /// the host, through the clamp when `clamped`, else straight from the
    /// record. Each group marks its own word. `None` without a host GPU.
    fn groups_that_run(groups: u32, clamped: bool) -> Option<usize> {
        use super::*;
        use crate::compute::Param;
        const MARK: &str = r#"
RWByteAddressBuffer marks : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 g : SV_GroupID) {
    marks.Store(g.x * 4, 1);
}
"#;
        let host = host()?;
        let (root, pso) =
            crate::compute::pipeline(MARK, s!("marks"), 0, &[Param::Uav { register: 0 }]).unwrap();
        let (record, words) = host.shared_buffer(12).unwrap();
        let (marks, seen) = host.shared_buffer(u64::from(groups) * 4).unwrap();
        let argument = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        };
        let desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: 12,
            NumArgumentDescs: 1,
            pArgumentDescs: &argument,
            NodeMask: 0,
        };
        let list = host.new_list().unwrap();
        // SAFETY: our mapped buffers, written before the list runs and read
        // after it has; our pipeline, signature and buffers.
        unsafe {
            let mut signature: Option<ID3D12CommandSignature> = None;
            host.device
                .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut signature)
                .unwrap();
            let signature = signature.unwrap();
            words.cast::<[u32; 3]>().write_volatile([groups, 1, 1]);
            std::ptr::write_bytes(seen, 0, groups as usize * 4);
            let arguments = if clamped {
                bound(&list, record.GetGPUVirtualAddress(), (2, 12, 1), None).unwrap()
            } else {
                list.list.ResourceBarrier(&[transition(
                    &record,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                )]);
                record.clone()
            };
            let l = &list.list;
            l.SetComputeRootSignature(&root);
            l.SetPipelineState(&pso);
            l.SetComputeRootUnorderedAccessView(0, marks.GetGPUVirtualAddress());
            l.ExecuteIndirect(&signature, 1, &arguments, 0, None, 0);
            if clamped {
                done(&list, &arguments);
            } else {
                l.ResourceBarrier(&[transition(
                    &record,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
            list.close();
            host.execute(&[&list]);
            let seen = seen.cast::<u32>();
            Some(
                (0..groups as usize)
                    .filter(|&i| seen.add(i).read_volatile() != 0)
                    .count(),
            )
        }
    }

    /// A tile pass of the title (67,440 groups, one class of its 8x4 tiles at
    /// 2560x1440) runs every group where the host runs dispatches that wide,
    /// and its first 65,535 elsewhere. The host runs the widest dispatch the
    /// clamp lets through, so a host that stops running them fails here.
    #[test]
    fn a_wide_tile_pass_runs_every_group_the_host_runs() {
        let Some(ran) = groups_that_run(67_440, true) else {
            return; // no host GPU here
        };
        let widest = super::widest();
        assert_eq!(ran, 67_440.min(widest) as usize);
        assert_eq!(
            groups_that_run(widest, false).unwrap(),
            widest as usize,
            "the host runs the widest dispatch"
        );
        assert_eq!(
            groups_that_run(widest, true).unwrap(),
            widest as usize,
            "the clamp lets it through"
        );
    }

    /// Dispatches wider than [`super::widest`] are cut to 65,535 groups in x
    /// as before, and one whose groups pass 2^20 in all runs empty, also
    /// where x times y would wrap 32 bits.
    #[test]
    fn dispatches_past_the_bounds_are_cut() {
        let widest = super::widest();
        let Some(past) = clamped(2, &[widest + 1, 1, 1], 1, 1) else {
            return; // no host GPU here
        };
        assert_eq!(past, [65_535]);
        assert_eq!(clamped(2, &[widest, 1, 1], 1, 1).unwrap(), [widest]);
        // Past 2^20 groups in all: 65,535 by 17, 2^10 by 2^10 by 2, and
        // 2^20 by 4,096, whose x * y wraps 32 bits to zero.
        assert_eq!(clamped(2, &[65_535, 17, 1], 1, 1).unwrap(), [0]);
        assert_eq!(clamped(2, &[1 << 10, 1 << 10, 2], 1, 1).unwrap(), [0]);
        assert_eq!(clamped(2, &[1 << 20, 4_096, 1], 1, 1).unwrap(), [0]);
    }
}
