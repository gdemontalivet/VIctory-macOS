//! Which title memory changed since a host copy took it (decision 0007,
//! "Copies of title memory" and "Change tracking"). Each host copy of title
//! memory (`crate::shadow`) and each texture's staging (`crate::texture`)
//! tracks its memory ([`Tracked`]): the 64 KiB pages it has not taken since
//! they last changed. Taking pages makes their blocks read-only (guards);
//! the first write to a guarded block faults into our vectored handler,
//! which gives write access back and marks the block changed in every copy
//! holding it. Writers that do not fault say so: our own and the kernel's
//! writes into title memory ([`writing`], before they write), and the
//! runtime mapping or unmapping memory ([`title_memory_remapped`]). So
//! bringing a copy up to date costs what changed since, not how much memory
//! it covers: no system call for memory nobody wrote.
//!
//! Only the runtime's GPU memory is tracked: its write-watched pools, its
//! physical-page views and their placeholders. Other memory a copy takes
//! (the C heap, where the kernel writes system call results into buffers a
//! guard would make fail) is copied whole every time.
//!
//! Blocks that cannot be made read-only (placeholders, nothing mapped)
//! read as zeros and count as taken: only a mapping changes them, which the
//! runtime reports.
//!
//! Physical pages may be mapped at several addresses (the title fills
//! pages through a pool view and maps them into its heaps), so a write at
//! one is a change at all of them: the runtime says where else memory is
//! mapped whenever it maps it ([`title_memory_aliased`]); taking a page
//! guards every address showing it, and a write at any of them marks the
//! trackers at all. (Asking the runtime at each take instead waited for
//! its lock, which the title's page mappings hold: 4 s per world frame.)
//! DirectStorage writing pages through views of the runtime's own reports
//! the addresses showing them as written.
//!
//! shortcut: a system call writing into a guarded block fails; the known
//! ones (file reads, DirectStorage) open their blocks first. Memory freed
//! and allocated again while a tracker holds it is not seen.
//!
//! Under Rosetta (Wine on macOS) memory is tracked only when asked for, with
//! the stores Rosetta cannot resume rewritten at load time: see
//! [`fault_tracking`].

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Whether title writes may fault into a tracker: the guards here and the
/// write watch of the runtime's pools (`victory_hle`'s `XMemVirtualAlloc`,
/// which Wine emulates with page protection). Every store that hits a
/// tracked page must be resumed after the fault. Rosetta cannot resume
/// some: it prints `unexpectedly need to EmulateForward on a synchronous
/// exception` and the process hangs (VIctory-macOS issue #1). On macOS 27
/// those are AVX masked stores that start in writable memory and end in a
/// protected page, and `vcvtps2ph` stores anywhere in one; the loader
/// rewrites both as plain stores whenever tracking runs under Rosetta
/// ([`stores_need_patching`]). Earlier Rosetta also stops on a plain 32-byte
/// `vmovups` across pages (macOS 26.6, #1), and #144 hung on an M1 with
/// both switches on, so under Rosetta tracking stays off unless
/// `VICTORY_FAULT_TRACKING=1` asks for it. Without tracking, copies take
/// their memory whole every time, as they do for memory that is not the
/// runtime's (tens of GB per world frame, #50). `0` turns tracking off
/// anywhere. Tests keep it on: their stores do not cross pages.
pub fn fault_tracking() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| {
        decide(
            std::env::var("VICTORY_FAULT_TRACKING").ok().as_deref(),
            cfg!(test) || !rosetta(),
        )
    });
    *ON
}

/// Whether tracking runs under Rosetta, so the stores Rosetta cannot resume
/// after a fault must be rewritten at load time (`victory_loader::masked`).
pub fn stores_need_patching() -> bool {
    fault_tracking() && rosetta()
}

/// `fault_tracking` from the variable's value and the default.
fn decide(var: Option<&str>, default: bool) -> bool {
    match var {
        Some("1") => true,
        Some("0") => false,
        _ => default,
    }
}

/// Whether pages written repeatedly stay read-write across takes (`Guards::hot`)
/// instead of being guarded again after every submission (`VICTORY_HOT_PAGES=0`
/// turns it off; on by default outside `cfg!(test)`, off in tests unless
/// overridden so guard-count assertions stay exact).
fn hot_pages_on() -> bool {
    #[cfg(test)]
    if let Some(on) = tests::HOT_PAGES.with(std::cell::Cell::get) {
        return on;
    }
    static ON: LazyLock<bool> = LazyLock::new(|| {
        decide_hot_pages(
            std::env::var("VICTORY_HOT_PAGES").ok().as_deref(),
            cfg!(test),
        )
    });
    *ON
}

fn decide_hot_pages(var: Option<&str>, is_test: bool) -> bool {
    if is_test {
        decide(var, false)
    } else {
        var != Some("0")
    }
}

/// How long a page opened by a write fault stays read-write across takes
/// before being re-guarded (`VICTORY_GUARD_REST=<microseconds>`, default
/// `3000` µs outside `cfg!(test)`, `0` disables).
fn guard_rest() -> Duration {
    #[cfg(test)]
    if let Some(d) = tests::GUARD_REST.with(std::cell::Cell::get) {
        return d;
    }
    #[cfg(test)]
    {
        Duration::ZERO
    }
    #[cfg(not(test))]
    {
        static REST: OnceLock<Duration> = OnceLock::new();
        *REST.get_or_init(|| {
            let us = match std::env::var("VICTORY_GUARD_REST") {
                Ok(v) if v.trim() == "0" => 0,
                Ok(v) => v.trim().parse::<u64>().unwrap_or(3_000),
                Err(_) => 3_000,
            };
            Duration::from_micros(us)
        })
    }
}

/// Whether a write fault opens only the faulting view and leaves alias views
/// read-only while marking their trackers changed (`VICTORY_OPEN_ONE_VIEW=0`
/// opens every alias view as well).
fn open_one_view() -> bool {
    #[cfg(test)]
    if let Some(one) = tests::OPEN_ONE_VIEW.with(std::cell::Cell::get) {
        return one;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_OPEN_ONE_VIEW").as_deref() != Ok("0"))
}

/// Whether subsequent takes on write-watched pools copy only the 4 KiB kernel
/// subpages reported by `GetWriteWatch` instead of whole 64 KiB console pages
/// (`VICTORY_PAGE_PARTS=1` or `=0` overrides; off by default under Rosetta/Wine,
/// where `GetWriteWatch(WRITE_WATCH_FLAG_RESET)` re-protects 4 KiB pages under
/// `virtual_mutex`, and off in `cfg!(test)` unless `with_page_parts`).
fn page_parts_on() -> bool {
    #[cfg(test)]
    if let Some(on) = tests::PAGE_PARTS.with(std::cell::Cell::get) {
        return on;
    }
    static ON: LazyLock<bool> = LazyLock::new(|| {
        decide_page_parts(
            std::env::var("VICTORY_PAGE_PARTS").ok().as_deref(),
            cfg!(test),
            rosetta(),
        )
    });
    *ON
}

fn decide_page_parts(var: Option<&str>, is_test: bool, is_rosetta: bool) -> bool {
    decide(var, !is_test && !is_rosetta)
}

/// Whether GPU shader copy-back writes directly through pinned `xmem` section
/// views without opening title page guards (`VICTORY_BACK_OWN_VIEW=0` disables;
/// on by default).
fn back_own_view_on() -> bool {
    #[cfg(test)]
    if let Some(on) = tests::BACK_OWN_VIEW.with(std::cell::Cell::get) {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_BACK_OWN_VIEW").as_deref() != Ok("0"))
}

/// Set once the runtime sees a title request involving `PAGE_GUARD`.
static TITLE_PAGE_GUARD_SEEN: AtomicBool = AtomicBool::new(false);

/// Notes that the title requested `PAGE_GUARD` protection, enabling per-page
/// `VirtualQuery` inspection before `VirtualProtect` so title guard subpages
/// are never stripped.
pub fn note_page_guard() {
    TITLE_PAGE_GUARD_SEEN.store(true, Ordering::Release);
}

/// Whether `protect_inner` queries non-write-watched pages with `VirtualQuery`
/// to preserve title `PAGE_GUARD` subpages (`VICTORY_CHECK_PAGE_GUARD=1` forces
/// it on; enabled by default in `cfg!(test)` or after [`note_page_guard`]).
fn check_page_guard_on() -> bool {
    #[cfg(test)]
    if let Some(on) = tests::CHECK_PAGE_GUARD.with(std::cell::Cell::get) {
        return on;
    }
    if TITLE_PAGE_GUARD_SEEN.load(Ordering::Relaxed) {
        return true;
    }
    static ON: LazyLock<bool> = LazyLock::new(|| {
        decide_check_page_guard(
            std::env::var("VICTORY_CHECK_PAGE_GUARD").ok().as_deref(),
            cfg!(test),
        )
    });
    *ON
}

fn decide_check_page_guard(var: Option<&str>, is_test: bool) -> bool {
    decide(var, is_test)
}

/// Minimum and maximum frames a hot page stays read-write before one re-sample
/// guard checks whether the title still writes it (staggered by page so pages
/// promoted together do not all re-guard on the same frame).
const HOT_FRAMES_MIN: u64 = 8;
const HOT_FRAMES_MAX: u64 = 15;

/// Frame window within which a second write fault (or a re-sample fault after
/// cooling) promotes a page and its aliases into [`Guards::hot`], covering
/// 30 fps / multi-frame animated textures as well as every-frame writers.
const HOT_WINDOW_FRAMES: u64 = 4;

/// Frames a page touched or armed for GPU shader writes stays out of
/// [`Guards::hot`], keeping exact fault detection for shader copy-back.
const GPU_COLD_FRAMES: u64 = 64;

fn hot_span(group_root: usize) -> u64 {
    let span = HOT_FRAMES_MAX - HOT_FRAMES_MIN + 1;
    HOT_FRAMES_MIN + ((group_root as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 61) % span
}

fn monotonic_ns() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH
        .get_or_init(Instant::now)
        .elapsed()
        .as_nanos()
        .min(u128::from(u64::MAX - 1)) as u64
}

/// Whether x86-64 code here runs under Rosetta: Wine on macOS, whose
/// x86-64 Wine only runs that way on the Apple silicon Macs D3DMetal needs.
pub fn rosetta() -> bool {
    use std::ffi::{CStr, c_char};
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    type HostVersion = unsafe extern "C" fn(*mut *const c_char, *mut *const c_char);
    // SAFETY: ntdll is always loaded and both names end with a NUL. Wine's
    // `wine_get_host_version` (cdecl, two out pointers) sets them to static
    // strings: the host's `uname` system name and release.
    unsafe {
        let Some(f) = GetProcAddress(
            GetModuleHandleA(c"ntdll.dll".as_ptr().cast()),
            c"wine_get_host_version".as_ptr().cast(),
        ) else {
            return false;
        };
        let f: HostVersion = std::mem::transmute(f);
        let (mut sysname, mut release) = (std::ptr::null(), std::ptr::null());
        f(&mut sysname, &mut release);
        !sysname.is_null() && CStr::from_ptr(sysname) == c"Darwin"
    }
}

/// The unit tracked and copied: 64 KiB, the console's page. Finer units
/// made the checks of the world's big buffers cost more than the copies
/// they saved.
pub(crate) const PAGE: usize = 64 << 10;

/// Host kernel write-watch granularity (4 KiB, 16 parts per [`PAGE`]).
const KERNEL_PAGE: usize = 4096;
const PARTS: usize = PAGE / KERNEL_PAGE;
const WHOLE: u16 = u16::MAX;

/// Shared 4 KiB subpage write-watch journal so resetting kernel write-watch
/// bits on one tracker never hides subpage writes from another tracker sharing
/// the same 64 KiB pool page.
#[derive(Default)]
struct Subpages {
    seq: u64,
    written: std::collections::HashMap<usize, u64>,
    parts: std::collections::HashMap<usize, [u64; PARTS]>,
}

impl Subpages {
    fn note(&mut self, page: usize, mask: u16, now: u64) {
        if mask == 0 {
            return;
        }
        let before = self.written.insert(page, now).unwrap_or(0);
        if mask == WHOLE {
            self.parts.remove(&page);
            return;
        }
        let all = self.parts.entry(page).or_insert([before; PARTS]);
        for (k, slot) in all.iter_mut().enumerate() {
            if (mask >> k) & 1 != 0 {
                *slot = now;
            }
        }
    }

    fn parts_since(&self, page: usize, taken: u64) -> u16 {
        if taken == 0 {
            return WHOLE;
        }
        let Some(&last) = self.written.get(&page) else {
            return 0;
        };
        if last <= taken {
            return 0;
        }
        match self.parts.get(&page) {
            Some(all) => all
                .iter()
                .enumerate()
                .filter(|&(_, &at)| at > taken)
                .fold(0u16, |m, (k, _)| m | (1u16 << k)),
            None => WHOLE,
        }
    }
}

static SUBPAGES: LazyLock<Mutex<Subpages>> = LazyLock::new(|| Mutex::new(Subpages::default()));

/// Harvests `GetWriteWatch(WRITE_WATCH_FLAG_RESET)` for sorted `[first, last)`
/// write-watched page runs into [`SUBPAGES`] and returns the new sequence number.
fn harvest_watched_runs(runs: &[(usize, usize)]) -> u64 {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    thread_local! {
        static FOUND: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let mut sub = SUBPAGES.lock().unwrap_or_else(|e| e.into_inner());
    sub.seq += 1;
    let now = sub.seq;
    if runs.is_empty() {
        return now;
    }
    #[cfg(feature = "diagnostics")]
    let _timed = Timed(Instant::now(), crate::copystat::C::KernelUs);
    FOUND.with_borrow_mut(|found| {
        let mut harvest_span = |sub: &mut Subpages, first: usize, last: usize| -> bool {
            let max_addrs = (last - first) * PARTS + 1;
            if found.len() < max_addrs {
                found.resize(max_addrs, 0);
            }
            let mut count = max_addrs;
            let mut gran = 0u32;
            // SAFETY: `found` has room for `count` addresses.
            let rc = unsafe {
                GetWriteWatch(
                    WRITE_WATCH_FLAG_RESET,
                    (first * PAGE) as *const _,
                    (last - first) * PAGE,
                    found.as_mut_ptr().cast(),
                    &mut count,
                    &mut gran,
                )
            };
            if rc != 0 {
                return false;
            }
            let mut page_masks: Vec<(usize, u16)> = Vec::new();
            for &at in &found[..count] {
                let (page, bit) = (at / PAGE, 1u16 << ((at % PAGE) / KERNEL_PAGE));
                match page_masks.last_mut() {
                    Some((last_p, m)) if *last_p == page => *m |= bit,
                    _ => page_masks.push((page, bit)),
                }
            }
            for (page, mask) in page_masks {
                sub.note(page, mask, now);
            }
            true
        };

        for &(first, last) in runs {
            if !harvest_span(&mut sub, first, last) {
                for page in first..last {
                    if !harvest_span(&mut sub, page, page + 1) {
                        sub.note(page, WHOLE, now);
                    }
                }
            }
        }
    });
    now
}

fn note_subpages_bytes(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    let (first_k, last_k) = (at / KERNEL_PAGE, end.div_ceil(KERNEL_PAGE));
    let mut sub = SUBPAGES.lock().unwrap_or_else(|e| e.into_inner());
    sub.seq += 1;
    let now = sub.seq;
    let mut masks: Vec<(usize, u16)> = Vec::new();
    for kp in first_k..last_k {
        let (page, bit) = (kp / PARTS, 1u16 << (kp % PARTS));
        match masks.last_mut() {
            Some((last_p, m)) if *last_p == page => *m |= bit,
            _ => masks.push((page, bit)),
        }
    }
    for (page, mask) in masks {
        sub.note(page, mask, now);
    }
}

fn note_subpages_whole(first: usize, last: usize) {
    if first >= last {
        return;
    }
    let mut sub = SUBPAGES.lock().unwrap_or_else(|e| e.into_inner());
    sub.seq += 1;
    let now = sub.seq;
    for page in first..last {
        sub.note(page, WHOLE, now);
    }
}

/// Per-tracker page state under [`Changed`]'s lock. The three dirty sets are
/// disjoint, and [`DIRTY`] holds `1` while any of them is non-empty.
#[derive(Default)]
struct Dirty {
    /// Pages changed since last taken (at creation, by a write fault or
    /// reported write, by [`retake`], by hot/rest expiry, or while `busy`).
    pages: BTreeSet<usize>,
    /// Pages already taken while kept open (`Guards::hot` or
    /// `Guards::resting`): left `PAGE_READWRITE`, so every take returns them
    /// again without `VirtualProtect` or taking [`GUARDS`].
    hot: BTreeSet<usize>,
    /// Pages [`Tracked::arm`] or [`Tracked::unarm`] cooled and guarded after
    /// this tracker had already taken them as open: still due for any pending
    /// take, while `pages` stays clear so a title write after `arm` snapshots
    /// pre-write bytes and unmarked shader copy-back sees no false title write.
    rearmed: BTreeSet<usize>,
    /// Last [`Subpages::seq`] at which this tracker took each whole 64 KiB page.
    taken_seq: BTreeMap<usize, u64>,
}

impl Dirty {
    fn is_empty(&self) -> bool {
        self.pages.is_empty() && self.hot.is_empty() && self.rearmed.is_empty()
    }
}

/// The pages a tracker has not taken since they changed, what it saves of
/// pages it armed, its bit in [`DIRTY`] (none: always taken as dirty), and
/// whether it is a host buffer copy's (the only trackers [`retake`] marks).
struct Changed(Mutex<Dirty>, Mutex<Saved>, Option<u32>, bool);

/// A bit per tracker, set while it holds changed pages: set and cleared
/// under the tracker's lock, read without one ([`Tracked::clean`]). Dense,
/// so that a submission asking about every copy its heaps reach (~14,000)
/// reads a few KB instead of a cache line per copy.
static DIRTY: [AtomicU64; DIRTY_WORDS] = [const { AtomicU64::new(0) }; DIRTY_WORDS];
const DIRTY_WORDS: usize = 1 << 14;

/// Bits of [`DIRTY`] given back, and the first never handed out.
static FREE_BITS: Mutex<(Vec<u32>, u32)> = Mutex::new((Vec::new(), 0));

fn dirty_bit() -> Option<u32> {
    let mut free = FREE_BITS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(bit) = free.0.pop() {
        return Some(bit);
    }
    let bit = free.1;
    (bit < (DIRTY_WORDS * 64) as u32).then(|| {
        free.1 += 1;
        bit
    })
}

fn set_dirty(bit: Option<u32>, dirty: bool) {
    if let Some(bit) = bit {
        let (word, mask) = (&DIRTY[bit as usize / 64], 1u64 << (bit % 64));
        if dirty {
            word.fetch_or(mask, Ordering::Release);
        } else {
            word.fetch_and(!mask, Ordering::Release);
        }
    }
}

/// Whether the tracker with bit `bit` holds no changed page (see
/// [`Tracked::clean`]).
pub(crate) fn clean_bit(bit: Option<u32>) -> bool {
    bit.is_some_and(|bit| DIRTY[bit as usize / 64].load(Ordering::Acquire) & 1 << (bit % 64) == 0)
}

impl Drop for Changed {
    fn drop(&mut self) {
        if let Some(bit) = self.2 {
            set_dirty(Some(bit), false);
            FREE_BITS
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .0
                .push(bit);
        }
    }
}

/// Pages (page numbers) whose bytes a tracker keeps when they are first
/// written after it took them, and those bytes: while shaders may be
/// writing its copy of them, what it held before (`crate::shadow`'s
/// copy-back of marked pages), with what the copy took of the page since
/// laid over them ([`Tracked::refreshed`]).
#[derive(Default)]
struct Saved {
    /// The armed pages as runs `[first, last)`, sorted, apart and not
    /// touching (runs armed that meet merge).
    armed: Vec<(usize, usize)>,
    /// Pages of those runs that are not armed any more.
    unarmed: BTreeSet<usize>,
    bytes: std::collections::HashMap<usize, Box<[u8]>>,
}

impl Saved {
    /// Whether page `page` is armed.
    fn armed(&self, page: usize) -> bool {
        // The first run ending after `page` is the only one that can hold it.
        let i = self.armed.partition_point(|&(_, last)| last <= page);
        self.armed.get(i).is_some_and(|&(first, _)| first <= page) && !self.unarmed.contains(&page)
    }

    /// Arms pages `[first, last)`.
    fn arm(&mut self, first: usize, last: usize) {
        if first >= last {
            return;
        }
        if !self.unarmed.is_empty() {
            let pages: Vec<usize> = self.unarmed.range(first..last).copied().collect();
            for page in pages {
                self.unarmed.remove(&page);
            }
        }
        // Runs `[lo, hi)` overlap or touch the new one: it replaces them.
        let lo = self.armed.partition_point(|&(_, b)| b < first);
        let hi = self.armed.partition_point(|&(a, _)| a <= last);
        if lo + 1 == hi && self.armed[lo].0 <= first && last <= self.armed[lo].1 {
            return;
        }
        let run = if lo < hi {
            (self.armed[lo].0.min(first), self.armed[hi - 1].1.max(last))
        } else {
            (first, last)
        };
        self.armed.splice(lo..hi, [run]);
    }
}

impl Changed {
    /// `pages` were taken: anything saved of them is stale.
    fn taken_many(&self, pages: &[usize]) {
        if pages.is_empty() {
            return;
        }
        let mut saved = self.1.lock().unwrap_or_else(|e| e.into_inner());
        if !saved.bytes.is_empty() {
            for &page in pages {
                saved.bytes.remove(&page);
            }
        }
    }

    /// Page `page` is about to change for the first time since it was
    /// taken: its bytes now, if armed for it.
    fn save(&self, page: usize) {
        let mut saved = self.1.lock().unwrap_or_else(|e| e.into_inner());
        if !saved.armed(page) || saved.bytes.contains_key(&page) {
            return;
        }
        let mut bytes = vec![0u8; PAGE].into_boxed_slice();
        if crate::host::read_into((page * PAGE) as *const u8, &mut bytes) {
            saved.bytes.insert(page, bytes);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::SavedPages, 1);
        }
    }
}

/// Byte runs `[start, end)` in order.
pub(crate) type Runs = Vec<(usize, usize)>;

/// Title memory a host copy or texture staging takes: bytes `[start, end)`.
pub(crate) struct Tracked {
    start: usize,
    end: usize,
    changed: Arc<Changed>,
    /// Parts of the range on pages that are not the runtime's memory (in
    /// order): not tracked, copied whole at each use ([`Tracked::untracked`]).
    untracked: Vec<(usize, usize)>,
    /// Coalesced `[first_page, last_page)` runs of write-watched pool pages in
    /// `[start, end)`, eligible for 4 KiB subpage cutting (`VICTORY_PAGE_PARTS`).
    watched: Vec<(usize, usize)>,
}

struct Guards {
    /// Blocks (pages) we made read-only.
    protected: BTreeSet<usize>,
    /// Blocks currently being transitioned to `PAGE_READONLY` outside
    /// [`GUARDS`] (`page -> (op_id, cancelled)`); marked `cancelled` by
    /// [`changed`] if a concurrent fault, kernel write, or remap arrives
    /// before `VirtualProtect` finishes, and removed only by `protect_inner`
    /// once any needed reopen completes.
    guarding: BTreeMap<usize, (u64, bool)>,
    next_guard_op: u64,
    /// Page runs `[first, last)` currently being transitioned to
    /// `PAGE_READWRITE` outside [`GUARDS`].
    opening: Vec<(usize, usize)>,
    /// The trackers holding each page.
    holders: BTreeMap<usize, Vec<Arc<Changed>>>,
    /// Pages `[first, last)` of writes the guards do not see that are under
    /// way ([`Writing`]): not guarded until they end, or the write fails.
    busy: Vec<(usize, usize)>,
    /// Pages showing the same physical pages elsewhere, both ways.
    alias_of: BTreeMap<usize, Vec<usize>>,
    /// Live `xmem` section views mapped by the runtime (`start_page -> end_page`).
    xmem_views: BTreeMap<usize, usize>,
    /// Pages kept read-write across takes until the given [`Guards::frame`].
    hot: BTreeMap<usize, u64>,
    /// Pages kept read-write briefly after a write fault until the given
    /// monotonic nanosecond deadline ([`guard_rest`]).
    resting: BTreeMap<usize, u64>,
    /// Write faults per page in the current frame.
    faults: BTreeMap<usize, u32>,
    /// Last [`Guards::frame`] in which each page (or an alias of it) faulted.
    last_fault_frame: BTreeMap<usize, u64>,
    /// [`Guards::frame`] at which a hot lease on each page recently expired,
    /// so a single re-sample fault within [`HOT_WINDOW_FRAMES`] re-promotes it.
    cooled_frame: BTreeMap<usize, u64>,
    /// Pages (and their aliases) touched by GPU shader writes or `arm`,
    /// excluded from `hot` until the given [`Guards::frame`].
    gpu_until: BTreeMap<usize, u64>,
    /// Logical frame counter advanced once per presented frame (starts at 1 so
    /// frame 0 is never confused with "never seen").
    frame: u64,
    /// Last `crate::queue::presented()` seen.
    last_presented: u64,
    /// Tracked pages written this frame and the frame before (diagnostics).
    #[cfg(feature = "diagnostics")]
    written: BTreeSet<usize>,
    #[cfg(feature = "diagnostics")]
    written_before: BTreeSet<usize>,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BTreeSet::new(),
    guarding: BTreeMap::new(),
    next_guard_op: 1,
    opening: Vec::new(),
    holders: BTreeMap::new(),
    busy: Vec::new(),
    alias_of: BTreeMap::new(),
    xmem_views: BTreeMap::new(),
    hot: BTreeMap::new(),
    resting: BTreeMap::new(),
    faults: BTreeMap::new(),
    last_fault_frame: BTreeMap::new(),
    cooled_frame: BTreeMap::new(),
    gpu_until: BTreeMap::new(),
    frame: 1,
    last_presented: 0,
    #[cfg(feature = "diagnostics")]
    written: BTreeSet::new(),
    #[cfg(feature = "diagnostics")]
    written_before: BTreeSet::new(),
});

static OPENING_CV: Condvar = Condvar::new();
static GUARDING_CV: Condvar = Condvar::new();

/// Serializes `VirtualQuery`/`VirtualProtect` batches in [`protect_inner`]
/// outside [`GUARDS`] so parallel refresh workers do not stampede Wine's
/// process-global `virtual_mutex` while title fault handlers stay lock-free of
/// `VirtualProtect`. Never acquired while holding [`GUARDS`].
static PROTECT_SYSCALL: Mutex<()> = Mutex::new(());

/// Number of pages currently in [`Guards::hot`], for lock-free fast paths.
static HOT_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Number of pages currently in [`Guards::resting`].
static RESTING_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Earliest monotonic nanosecond deadline in [`Guards::resting`] (`u64::MAX` when empty).
static NEXT_REST_EXPIRY_NS: AtomicU64 = AtomicU64::new(u64::MAX);
/// Mirrors [`Guards::last_presented`] so [`sync_leases`] costs one relaxed load
/// on submissions within the same frame.
static LAST_PRESENTED: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// Whether this thread holds the guard lock. A guarded block it faults
    /// on then is one the lock's own work touched (#50: an allocation made
    /// under the lock landed in a guarded block of the host heap, which
    /// title XMem blocks share), and taking the lock again would hang it.
    static HOLDING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Blocks opened by faults taken under the guard lock, for its next holder
/// to mark changed (0: free slot). Fixed: the handler must not allocate.
static OPENED_UNDER_LOCK: [std::sync::atomic::AtomicUsize; 64] =
    [const { std::sync::atomic::AtomicUsize::new(0) }; 64];

/// Whether any block waits in them: the next holder of the lock scans
/// [`OPENED_UNDER_LOCK`] only when one was opened, not on every take. Only
/// touched while the lock is held, so a clear scan leaves no block behind.
static OPENED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Records `page` in [`OPENED_UNDER_LOCK`] for the next holder of [`GUARDS`]
/// to mark changed. Lock-free and allocation-free.
fn note_opened_under_lock(page: usize) {
    use std::sync::atomic::Ordering::{AcqRel, Relaxed, Release};
    if OPENED_UNDER_LOCK
        .iter()
        .any(|s| s.compare_exchange(0, page, AcqRel, Relaxed).is_ok())
    {
        OPENED.store(true, Release);
    } else {
        static LOST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        LOST.fetch_add(1, Relaxed);
    }
}

/// A fault on the block `page` by the thread holding the guard lock: write
/// access back now, marked changed by the lock's next holder.
fn open_under_lock(page: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let mut old = 0;
    // SAFETY: a block of this process; fails harmlessly if it is gone.
    unsafe { VirtualProtect((page * PAGE) as *const _, PAGE, PAGE_READWRITE, &mut old) };
    note_opened_under_lock(page);
}

/// The guard lock, held while this lives. With diagnostics, timed for the
/// line that took it ([`held_times`]).
struct Held {
    guard: Option<std::sync::MutexGuard<'static, Guards>>,
    #[cfg(any(test, feature = "diagnostics"))]
    since: Instant,
    #[cfg(feature = "diagnostics")]
    line: u32,
}

#[cfg(test)]
static TEST_HOLD_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(test)]
static TEST_WAIT_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(test)]
thread_local! {
    static THREAD_HOLD_NS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[track_caller]
fn guards() -> Held {
    #[cfg(test)]
    let wait_started = Instant::now();
    let guard = GUARDS.lock().unwrap_or_else(|e| e.into_inner());
    #[cfg(test)]
    TEST_WAIT_NS.fetch_add(wait_started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    HOLDING.with(|h| h.set(true));
    let mut held = Held {
        guard: Some(guard),
        #[cfg(any(test, feature = "diagnostics"))]
        since: Instant::now(),
        #[cfg(feature = "diagnostics")]
        line: std::panic::Location::caller().line(),
    };
    held.drain_opened_under_lock();
    held
}

impl Held {
    fn record_hold_segment(&mut self) {
        #[cfg(test)]
        {
            let ns = self.since.elapsed().as_nanos() as u64;
            TEST_HOLD_NS.fetch_add(ns, Ordering::Relaxed);
            THREAD_HOLD_NS.with(|c| c.set(c.get().saturating_add(ns)));
        }
        #[cfg(feature = "diagnostics")]
        {
            use std::sync::atomic::Ordering::Relaxed;
            let us = self.since.elapsed().as_micros() as u64;
            let line = u64::from(self.line);
            if let Some(slot) = HELD.iter().find(|s| {
                s[0].load(Relaxed) == line
                    || s[0].compare_exchange(0, line, Relaxed, Relaxed).is_ok()
            }) {
                slot[1].fetch_add(us, Relaxed);
                slot[2].fetch_max(us, Relaxed);
            }
        }
    }

    fn drain_opened_under_lock(&mut self) {
        use std::sync::atomic::Ordering::AcqRel;
        while OPENED.swap(false, AcqRel) {
            for slot in &OPENED_UNDER_LOCK {
                let page = slot.swap(0, AcqRel);
                if page != 0 {
                    changed(self, page, page + 1, false);
                }
            }
        }
    }

    /// Releases [`GUARDS`] for the duration of `f` (used around `VirtualProtect`
    /// and `VirtualQuery` kernel calls) and re-acquires it afterwards.
    fn unlock_for<R>(&mut self, f: impl FnOnce() -> R) -> R {
        self.record_hold_segment();
        drop(self.guard.take());
        HOLDING.with(|h| h.set(false));
        let out = f();
        #[cfg(test)]
        let wait_started = Instant::now();
        let guard = GUARDS.lock().unwrap_or_else(|e| e.into_inner());
        #[cfg(test)]
        TEST_WAIT_NS.fetch_add(wait_started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        HOLDING.with(|h| h.set(true));
        self.guard = Some(guard);
        #[cfg(any(test, feature = "diagnostics"))]
        {
            self.since = Instant::now();
        }
        self.drain_opened_under_lock();
        out
    }

    /// Waits on `cv`, properly clearing [`HOLDING`] and hold-time accounting
    /// while [`GUARDS`] is released.
    fn wait(&mut self, cv: &'static Condvar) {
        self.record_hold_segment();
        let guard = self.guard.take().expect("held until dropped");
        HOLDING.with(|h| h.set(false));
        #[cfg(test)]
        let wait_started = Instant::now();
        let guard = cv.wait(guard).unwrap_or_else(|e| e.into_inner());
        #[cfg(test)]
        TEST_WAIT_NS.fetch_add(wait_started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        HOLDING.with(|h| h.set(true));
        self.guard = Some(guard);
        #[cfg(any(test, feature = "diagnostics"))]
        {
            self.since = Instant::now();
        }
        self.drain_opened_under_lock();
    }
}

impl std::ops::Deref for Held {
    type Target = Guards;
    fn deref(&self) -> &Guards {
        self.guard.as_ref().expect("held until dropped")
    }
}

impl std::ops::DerefMut for Held {
    fn deref_mut(&mut self) -> &mut Guards {
        self.guard.as_mut().expect("held until dropped")
    }
}

/// Per line taking the guard lock: the line, microseconds held in all, the
/// longest hold (no allocation: the fault handler takes the lock).
#[cfg(feature = "diagnostics")]
static HELD: [[std::sync::atomic::AtomicU64; 3]; 16] =
    [const { [const { std::sync::atomic::AtomicU64::new(0) }; 3] }; 16];

impl Drop for Held {
    fn drop(&mut self) {
        if self.guard.is_some() {
            self.record_hold_segment();
            drop(self.guard.take());
            HOLDING.with(|h| h.set(false));
        }
    }
}

/// Since the last call, per line of this file taking the guard lock: (line,
/// milliseconds held, longest hold in milliseconds), most held first.
#[cfg(feature = "diagnostics")]
pub(crate) fn held_times() -> Vec<(u64, f64, f64)> {
    use std::sync::atomic::Ordering::Relaxed;
    let mut out: Vec<(u64, f64, f64)> = HELD
        .iter()
        .filter(|s| s[0].load(Relaxed) != 0)
        .map(|s| {
            (
                s[0].load(Relaxed),
                s[1].swap(0, Relaxed) as f64 / 1e3,
                s[2].swap(0, Relaxed) as f64 / 1e3,
            )
        })
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

/// (page, other page) for the pages (page numbers) showing the same memory
/// as `pages` elsewhere.
fn aliases(g: &Guards, pages: &[usize]) -> Vec<(usize, usize)> {
    pages
        .iter()
        .flat_map(|&p| {
            g.alias_of
                .get(&p)
                .into_iter()
                .flatten()
                .map(move |&a| (p, a))
        })
        .collect()
}

/// If `page` lies in one of the sorted coalesced write-watched pool runs
/// `watched`, returns that run's exclusive end page.
fn watched_run_end(watched: &[(usize, usize)], page: usize) -> Option<usize> {
    let i = watched.partition_point(|&(_, last)| last <= page);
    watched
        .get(i)
        .filter(|&&(first, _)| first <= page)
        .map(|&(_, last)| last)
}

/// Tracks `[start, start + len)` if it is the runtime's GPU memory (every
/// page changed at first); `None` for other memory, and with `views` false
/// for anything but its write-watched pools.
pub(crate) fn track(start: usize, len: usize, views: bool) -> Option<Tracked> {
    track_as(start, len, views, false)
}

/// [`track`] for a host buffer copy: what shaders write lands in it, so a
/// [`retake`] of its memory marks it.
pub(crate) fn track_copy(start: usize, len: usize, views: bool) -> Option<Tracked> {
    track_as(start, len, views, true)
}

fn track_as(start: usize, len: usize, views: bool, copy: bool) -> Option<Tracked> {
    if len == 0 || !fault_tracking() {
        return None;
    }
    let end = start + len;
    let (pages, watched) = runtime_pages(start, end, views);
    if pages.is_empty() {
        return None;
    }
    install();
    // The rest of the range, as byte runs.
    let mut untracked: Vec<(usize, usize)> = Vec::new();
    let mut at = start;
    for &page in pages.iter().chain([&usize::MAX]) {
        let (a, b) = (
            (page.saturating_mul(PAGE)).max(start).min(end),
            (page.saturating_add(1).saturating_mul(PAGE)).min(end),
        );
        if at < a {
            untracked.push((at, a));
        }
        at = at.max(b);
    }
    // A tracker with untracked parts is never clean: no bit.
    let bit = if untracked.is_empty() {
        dirty_bit()
    } else {
        None
    };
    set_dirty(bit, true);
    let dirty = Dirty {
        pages: pages.iter().copied().collect(),
        hot: BTreeSet::new(),
        rearmed: BTreeSet::new(),
        taken_seq: BTreeMap::new(),
    };
    let changed = Arc::new(Changed(Mutex::new(dirty), Mutex::default(), bit, copy));
    let mut g = guards();
    for &page in &pages {
        g.holders.entry(page).or_default().push(changed.clone());
    }
    drop(g);
    Some(Tracked {
        start,
        end,
        changed,
        untracked,
        watched,
    })
}

/// Whether the page holding `at` is in a write-watched pool (one lookup).
fn write_watched(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    let page = at & !(4096 - 1);
    let (mut found, mut count, mut granularity) = (0usize, 1usize, 0u32);
    // SAFETY: room for one answer; no reset, so nothing changes.
    unsafe {
        GetWriteWatch(
            0,
            page as *const _,
            4096,
            (&raw mut found).cast(),
            &mut count,
            &mut granularity,
        ) == 0
    }
}

/// The pages of `[start, end)` that are the runtime's GPU memory: a
/// write-watched pool, or (with `views`) anything but plain committed memory
/// and images (views, placeholders, nothing mapped yet), plus coalesced
/// `[first, last)` runs of write-watched pool pages.
fn runtime_pages(start: usize, end: usize, views: bool) -> (Vec<usize>, Vec<(usize, usize)>) {
    use windows_sys::Win32::System::Memory::{
        GetWriteWatch, MEM_COMMIT, MEM_IMAGE, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    let watched = |at: usize, len: usize| {
        let (mut found, mut count, mut granularity) = (0usize, 1usize, 0u32);
        // SAFETY: room for one answer; no reset, so nothing changes.
        unsafe {
            GetWriteWatch(
                0,
                at as *const _,
                len,
                (&raw mut found).cast(),
                &mut count,
                &mut granularity,
            ) == 0
        }
    };
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    if watched(start & !(4096 - 1), end - (start & !(4096 - 1))) {
        return ((first..last).collect(), vec![(first, last)]);
    }
    let (mut runtime, mut other, mut watched_pages) = (
        std::collections::BTreeSet::new(),
        std::collections::BTreeSet::new(),
        std::collections::BTreeSet::new(),
    );
    let mut at = start & !(4096 - 1);
    while at < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0 {
            break;
        }
        let next = (info.BaseAddress as usize + info.RegionSize)
            .max(at + 1)
            .min(end);
        let is_watched = watched(at, next - at);
        if is_watched {
            watched_pages.extend(at / PAGE..next.div_ceil(PAGE));
        }
        let ours = is_watched
            || views
                && info.Type != MEM_IMAGE
                && !(info.State == MEM_COMMIT && info.Type == MEM_PRIVATE);
        (if ours { &mut runtime } else { &mut other }).extend(at / PAGE..next.div_ceil(PAGE));
        at = next;
    }
    let pages: Vec<usize> = runtime.difference(&other).copied().collect();
    let mut watched_runs: Vec<(usize, usize)> = Vec::new();
    for &p in &pages {
        if watched_pages.contains(&p) {
            match watched_runs.last_mut() {
                Some((_, last_p)) if *last_p == p => *last_p = p + 1,
                _ => watched_runs.push((p, p + 1)),
            }
        }
    }
    (pages, watched_runs)
}

impl Drop for Tracked {
    fn drop(&mut self) {
        let mut g = guards();
        let mut free = Vec::new();
        for page in self.start / PAGE..self.end.div_ceil(PAGE) {
            if let Some(holders) = g.holders.get_mut(&page) {
                holders.retain(|c| !Arc::ptr_eq(c, &self.changed));
                if holders.is_empty() {
                    g.holders.remove(&page);
                    free.push(page);
                }
            }
        }
        let mut to_open = Vec::new();
        let mut hot_changed = false;
        let mut rest_changed = false;
        for page in free {
            let watched_elsewhere = g
                .alias_of
                .get(&page)
                .is_some_and(|o| o.iter().any(|p| g.holders.contains_key(p)));
            if !watched_elsewhere {
                hot_changed |= g.hot.remove(&page).is_some();
                rest_changed |= g.resting.remove(&page).is_some();
                g.faults.remove(&page);
                g.last_fault_frame.remove(&page);
                g.cooled_frame.remove(&page);
                g.gpu_until.remove(&page);
                if g.protected.remove(&page) {
                    to_open.push(page);
                }
                if let Some((_, cancelled)) = g.guarding.get_mut(&page) {
                    *cancelled = true;
                }
            }
        }
        if hot_changed {
            HOT_COUNT.store(g.hot.len(), Ordering::Release);
        }
        if rest_changed {
            sync_resting_atomics(&g);
        }
        open_sorted_pages(&mut g, &to_open);
    }
}

/// Coalesces sorted `pages` (already removed from `g.protected`/`g.guarding`)
/// into contiguous runs and calls `VirtualProtect(..., PAGE_READWRITE)`
/// outside [`GUARDS`], holding [`Guards::opening`] across the syscalls so
/// concurrent takes and faults serialize cleanly on only those pages.
fn open_sorted_pages(g: &mut Held, pages: &[usize]) {
    if pages.is_empty() {
        return;
    }
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < pages.len() {
        let mut j = i + 1;
        while j < pages.len() && pages[j] == pages[j - 1] + 1 {
            j += 1;
        }
        runs.push((pages[i], pages[j - 1] + 1));
        i = j;
    }
    g.opening.extend(runs.iter().copied());
    let opened = g.unlock_for(|| {
        use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
        let mut count = 0u64;
        for &(first, last) in &runs {
            let mut old = 0;
            if last - first == 1 {
                // SAFETY: a block we made read-only (fails harmlessly if gone).
                unsafe {
                    VirtualProtect((first * PAGE) as *const _, PAGE, PAGE_READWRITE, &mut old)
                };
                count += 1;
                continue;
            }
            // SAFETY: blocks we made read-only.
            if unsafe {
                VirtualProtect(
                    (first * PAGE) as *const _,
                    (last - first) * PAGE,
                    PAGE_READWRITE,
                    &mut old,
                )
            } != 0
            {
                count += (last - first) as u64;
            } else {
                for page in first..last {
                    // SAFETY: as above.
                    unsafe {
                        VirtualProtect((page * PAGE) as *const _, PAGE, PAGE_READWRITE, &mut old)
                    };
                    count += 1;
                }
            }
        }
        count
    });
    for run in &runs {
        if let Some(pos) = g.opening.iter().position(|r| r == run) {
            g.opening.swap_remove(pos);
        }
    }
    OPENING_CV.notify_all();
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::OpenPages, opened);
    #[cfg(not(feature = "diagnostics"))]
    let _ = opened;
}

/// `(page, prev_seq, explicit_dirty)` entry passed to [`Tracked::build_copy_runs`].
type CopyEntry = (usize, u64, bool);

impl Tracked {
    fn is_watched_page(&self, page: usize) -> bool {
        let i = self.watched.partition_point(|&(_, last)| last <= page);
        self.watched.get(i).is_some_and(|&(first, _)| first <= page)
    }

    fn watched_overlap(&self, first: usize, last: usize) -> Vec<(usize, usize)> {
        self.watched
            .iter()
            .map(|&(a, b)| (a.max(first), b.min(last)))
            .filter(|&(a, b)| a < b)
            .collect()
    }

    /// Expands `entries` (`(page, prev_seq, explicit_dirty)`) into byte runs,
    /// cutting write-watched pages to their changed 4 KiB subpages when
    /// `use_parts` is active.
    fn build_copy_runs(
        &self,
        entries: &[CopyEntry],
        unmapped: &[usize],
        from: usize,
        to: usize,
        use_parts: bool,
    ) -> (Runs, Runs) {
        let (mut copy, mut zero): (Runs, Runs) = (Vec::new(), Vec::new());
        let mut unmapped_it = unmapped.iter().copied().peekable();
        let sub = use_parts.then(|| SUBPAGES.lock().unwrap_or_else(|e| e.into_inner()));
        for &(p, prev_seq, explicit_dirty) in entries {
            while unmapped_it.next_if(|&u| u < p).is_some() {}
            if unmapped_it.next_if_eq(&p).is_some() {
                let (a, b) = ((p * PAGE).max(from), ((p + 1) * PAGE).min(to));
                match zero.last_mut() {
                    Some((_, end)) if *end == a => *end = b,
                    _ => zero.push((a, b)),
                }
                continue;
            }
            if let Some(sub) = &sub
                && self.is_watched_page(p)
            {
                let mut mask = sub.parts_since(p, prev_seq);
                if mask == 0 && explicit_dirty {
                    mask = WHOLE;
                }
                while mask != 0 {
                    let first_k = mask.trailing_zeros() as usize;
                    let past_k = first_k + (!(mask >> first_k)).trailing_zeros() as usize;
                    mask &= if past_k >= PARTS {
                        0
                    } else {
                        u16::MAX << past_k
                    };
                    let (a, b) = (
                        (p * PAGE + first_k * KERNEL_PAGE).max(from),
                        (p * PAGE + past_k.min(PARTS) * KERNEL_PAGE).min(to),
                    );
                    if a >= b {
                        continue;
                    }
                    match copy.last_mut() {
                        Some((_, end)) if *end == a => *end = b,
                        _ => copy.push((a, b)),
                    }
                }
                continue;
            }
            let (a, b) = ((p * PAGE).max(from), ((p + 1) * PAGE).min(to));
            match copy.last_mut() {
                Some((_, end)) if *end == a => *end = b,
                _ => copy.push((a, b)),
            }
        }
        (copy, zero)
    }

    /// Takes the changed pages of `[from, to)`: guards them (writes from
    /// now on mark them again) and returns the byte ranges to copy, cut to
    /// `[from, to)`, in order. A page whose part in the tracked memory lies
    /// only partly in `[from, to)` is copied in part and stays changed.
    pub(crate) fn take(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let (copy, zero) = self.take_split(from, to);
        let mut runs: Vec<(usize, usize)> = copy.into_iter().chain(zero).collect();
        runs.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
        for (a, b) in runs {
            match merged.last_mut() {
                Some((_, end)) if *end == a => *end = b,
                _ => merged.push((a, b)),
            }
        }
        merged
    }

    /// [`Tracked::take`], the runs on pages with nothing mapped apart: they
    /// read as zeros, with no copy to try.
    pub(crate) fn take_split(&self, from: usize, to: usize) -> (Runs, Runs) {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from >= to {
            return (Vec::new(), Vec::new());
        }
        sync_leases();
        if self.clean() {
            return (Vec::new(), Vec::new());
        }
        let (first, last) = (from / PAGE, to.div_ceil(PAGE));
        let is_whole =
            |p: usize| (p * PAGE).max(self.start) >= from && ((p + 1) * PAGE).min(self.end) <= to;
        let use_parts = page_parts_on() && !self.watched.is_empty();

        // Fast path: when every changed page in `[first, last)` is already kept
        // open (`d.hot`, via hot-page or guard-rest lease) or `rearmed`, no
        // `VirtualProtect` or `GUARDS` lock is needed.
        let fast_entries: Option<(Vec<CopyEntry>, Vec<usize>)> = {
            #[cfg(feature = "diagnostics")]
            let waited = Timed(Instant::now(), crate::copystat::C::GuardWaitUs);
            let mut d = self.changed.0.lock().unwrap_or_else(|e| e.into_inner());
            #[cfg(feature = "diagnostics")]
            drop(waited);
            if d.pages.range(first..last).next().is_none() {
                let mut pages: Vec<usize> = d.hot.range(first..last).copied().collect();
                let rearmed: Vec<usize> = d.rearmed.range(first..last).copied().collect();
                let mut taken_rearmed = Vec::new();
                if !rearmed.is_empty() {
                    for &p in &rearmed {
                        if is_whole(p) {
                            d.rearmed.remove(&p);
                            taken_rearmed.push(p);
                        }
                    }
                    set_dirty(self.changed.2, !d.is_empty());
                    pages.extend(rearmed.iter().copied());
                    pages.sort_unstable();
                    pages.dedup();
                }
                let entries: Vec<(usize, u64, bool)> = pages
                    .iter()
                    .map(|&p| {
                        let prev = if use_parts {
                            d.taken_seq.get(&p).copied().unwrap_or(0)
                        } else {
                            0
                        };
                        let explicit = rearmed.binary_search(&p).is_ok();
                        (p, prev, explicit)
                    })
                    .collect();
                let whole_pages: Vec<usize> = pages.into_iter().filter(|&p| is_whole(p)).collect();
                drop(d);
                if !taken_rearmed.is_empty() {
                    self.changed.taken_many(&taken_rearmed);
                }
                Some((entries, whole_pages))
            } else {
                None
            }
        };
        if let Some((entries, whole_pages)) = fast_entries {
            if use_parts && !entries.is_empty() {
                let overlap = self.watched_overlap(first, last);
                let now_seq = harvest_watched_runs(&overlap);
                if !whole_pages.is_empty() {
                    let mut d = self.changed.0.lock().unwrap_or_else(|e| e.into_inner());
                    for p in whole_pages {
                        if self.is_watched_page(p) {
                            d.taken_seq.insert(p, now_seq);
                        }
                    }
                }
            }
            return self.build_copy_runs(&entries, &[], from, to, use_parts);
        }

        // Every address showing the same pages is guarded too (unless kept open).
        #[cfg(feature = "diagnostics")]
        let waited = Timed(Instant::now(), crate::copystat::C::GuardWaitUs);
        let mut g = guards();
        let d = self.changed.0.lock().unwrap_or_else(|e| e.into_inner());
        #[cfg(feature = "diagnostics")]
        drop(waited);
        let mut pages: Vec<usize> = d
            .pages
            .range(first..last)
            .copied()
            .chain(d.hot.range(first..last).copied())
            .chain(d.rearmed.range(first..last).copied())
            .collect();
        pages.sort_unstable();
        pages.dedup();
        if pages.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let mut rearmed_whole = Vec::new();
        let mut candidate_whole = Vec::new();
        let mut entries: Vec<(usize, u64, bool)> = Vec::with_capacity(pages.len());
        for &p in &pages {
            let in_rearmed = d.rearmed.contains(&p);
            let explicit = in_rearmed || d.pages.contains(&p);
            let prev = if use_parts {
                d.taken_seq.get(&p).copied().unwrap_or(0)
            } else {
                0
            };
            entries.push((p, prev, explicit));
            if is_whole(p) {
                if in_rearmed {
                    rearmed_whole.push(p);
                } else {
                    candidate_whole.push(p);
                }
            }
        }
        drop(d);

        let now_ns = monotonic_ns();
        let (mut hot_whole, mut cold_whole) = (Vec::new(), Vec::new());
        for p in candidate_whole {
            if is_keep_open_eligible(&g, p, now_ns) {
                hot_whole.push(p);
            } else {
                cold_whole.push(p);
            }
        }
        let mut unmapped = Vec::new();
        let mut taken_whole_watched = Vec::new();
        let guarded = if cold_whole.is_empty() {
            Vec::new()
        } else {
            #[cfg(feature = "diagnostics")]
            let asked = Timed(Instant::now(), crate::copystat::C::AliasUs);
            let others = aliases(&g, &cold_whole);
            #[cfg(feature = "diagnostics")]
            drop(asked);
            protect_with_aliases(
                &mut g,
                &cold_whole,
                &others,
                &self.watched,
                Some(&mut unmapped),
                |taken| {
                    let mut d = self.changed.0.lock().unwrap_or_else(|e| e.into_inner());
                    for &page in taken {
                        d.pages.remove(&page);
                        d.hot.remove(&page);
                        d.rearmed.remove(&page);
                        if use_parts && self.is_watched_page(page) {
                            taken_whole_watched.push(page);
                        }
                    }
                    set_dirty(self.changed.2, !d.is_empty());
                    drop(d);
                    self.changed.taken_many(taken);
                },
            )
        };
        let _ = &guarded;

        let mut d = self.changed.0.lock().unwrap_or_else(|e| e.into_inner());
        for &page in &hot_whole {
            // Only move `page` to `d.hot` if it is still eligible (a concurrent
            // fault/arm while `protect_noting` unlocked `g` may have cooled it).
            if is_keep_open_eligible(&g, page, now_ns) {
                d.pages.remove(&page);
                d.rearmed.remove(&page);
                d.hot.insert(page);
                if use_parts && self.is_watched_page(page) {
                    taken_whole_watched.push(page);
                }
            }
        }
        for &page in &rearmed_whole {
            d.rearmed.remove(&page);
            if use_parts && self.is_watched_page(page) {
                taken_whole_watched.push(page);
            }
        }
        set_dirty(self.changed.2, !d.is_empty());
        drop(d);
        self.changed.taken_many(&hot_whole);
        self.changed.taken_many(&rearmed_whole);
        drop(g);

        if use_parts {
            // Harvest AFTER `protect_noting` guarded `cold_whole`, so any pre-guard
            // write is captured in `SUBPAGES` and any post-guard write faults `on_write`.
            let overlap = self.watched_overlap(first, last);
            let now_seq = harvest_watched_runs(&overlap);
            if !taken_whole_watched.is_empty() {
                let mut d = self.changed.0.lock().unwrap_or_else(|e| e.into_inner());
                for page in taken_whole_watched {
                    if !d.pages.contains(&page) {
                        d.taken_seq.insert(page, now_seq);
                    }
                }
            }
        }

        self.build_copy_runs(&entries, &unmapped, from, to, use_parts)
    }

    /// Whether page `page` (a page number) may have been written by the title
    /// since it was taken — including pages left writable while hot/resting
    /// (`hot`) or cooled after being open without a fresh take (`rearmed`).
    pub(crate) fn changed(&self, page: usize) -> bool {
        let d = self.changed.0.lock().unwrap_or_else(|e| e.into_inner());
        d.pages.contains(&page) || d.hot.contains(&page) || d.rearmed.contains(&page)
    }

    /// The caller made pages (page numbers) of its copy equal to title
    /// memory itself (what shaders wrote went back): taken and guarded.
    /// Untracked pages (never guarded) are skipped.
    pub(crate) fn taken(&self, pages: &[usize]) {
        let mut sorted: Vec<usize> = pages
            .iter()
            .copied()
            .filter(|&p| !self.is_untracked(p))
            .collect();
        if sorted.is_empty() {
            return;
        }
        sorted.sort_unstable();
        sorted.dedup();
        let mut g = guards();
        if hot_pages_on()
            || !guard_rest().is_zero()
            || HOT_COUNT.load(Ordering::Acquire) != 0
            || RESTING_COUNT.load(Ordering::Acquire) != 0
        {
            cool_for_gpu(&mut g, &sorted, None);
        }
        let others = aliases(&g, &sorted);
        let protected = protect_with_aliases(
            &mut g,
            &sorted,
            &others,
            &self.watched,
            None,
            |taken| {
                let mut d = self.changed.0.lock().unwrap_or_else(|e| e.into_inner());
                for &page in taken {
                    d.pages.remove(&page);
                    d.hot.remove(&page);
                    d.rearmed.remove(&page);
                }
                set_dirty(self.changed.2, !d.is_empty());
                drop(d);
                self.changed.taken_many(taken);
            },
        );
        drop(g);
        if page_parts_on() && !self.watched.is_empty() && !protected.is_empty() {
            let mut watched_runs: Vec<(usize, usize)> = Vec::new();
            for &p in &protected {
                if self.is_watched_page(p) {
                    match watched_runs.last_mut() {
                        Some((_, end)) if *end == p => *end = p + 1,
                        _ => watched_runs.push((p, p + 1)),
                    }
                }
            }
            if !watched_runs.is_empty() {
                let now_seq = harvest_watched_runs(&watched_runs);
                let mut d = self.changed.0.lock().unwrap_or_else(|e| e.into_inner());
                for &(a, b) in &watched_runs {
                    for page in a..b {
                        if !d.pages.contains(&page) {
                            d.taken_seq.insert(page, now_seq);
                        }
                    }
                }
            }
        }
    }

    /// Whether any byte of page `page` (a page number) lies in the tracker's
    /// untracked parts (never guarded, copied whole at each use).
    pub(crate) fn is_untracked(&self, page: usize) -> bool {
        let (a, b) = (
            page.saturating_mul(PAGE),
            page.saturating_add(1).saturating_mul(PAGE),
        );
        self.untracked.iter().any(|&(u, v)| u < b && a < v)
    }

    /// Whether no page changed since it was taken: then [`Tracked::take`]
    /// finds nothing, and asking costs no lock.
    pub(crate) fn clean(&self) -> bool {
        clean_bit(self.changed.2)
    }

    /// The tracker's bit for [`clean_bit`], for callers that ask about
    /// many trackers at once without touching each (none: never clean).
    pub(crate) fn bit(&self) -> Option<u32> {
        self.changed.2
    }

    /// The parts of `[from, to)` that are not tracked (see
    /// [`Tracked::untracked`] field): to be copied whole at each use.
    pub(crate) fn untracked(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        self.untracked
            .iter()
            .map(|&(a, b)| (a.max(from), b.min(to)))
            .filter(|&(a, b)| a < b)
            .collect()
    }

    /// Shaders may write the copy's bytes `[from, to)` until [`Tracked::disarm`]:
    /// pages there written next after they were taken have their bytes kept
    /// first ([`Tracked::saved`]). Returns whether any hot/resting pages in
    /// `[from, to)` were cooled and re-guarded so the caller can pull pre-arm
    /// silent CPU writes into its shadow copy before GPU execution.
    pub(crate) fn arm(&self, from: usize, to: usize) -> bool {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from >= to {
            return false;
        }
        let (first, last) = (from / PAGE, to.div_ceil(PAGE));
        self.changed
            .1
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .arm(first, last);
        if HOT_COUNT.load(Ordering::Acquire) != 0 || RESTING_COUNT.load(Ordering::Acquire) != 0 {
            let hot_pages: Vec<usize> = self
                .changed
                .0
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .hot
                .range(first..last)
                .copied()
                .collect();
            if !hot_pages.is_empty() {
                let mut g = guards();
                cool_for_gpu(&mut g, &hot_pages, Some((&self.changed, first, last)));
                return true;
            }
        }
        false
    }

    /// Page `page` (a page number) needs no saving any more: its copy-back
    /// is under way. If `page` was left writable (hot or resting), cool it
    /// into [`Dirty::pages`] so copy-back keeps any silent CPU writes while
    /// future takes guard it again.
    pub(crate) fn unarm(&self, page: usize) {
        {
            let mut saved = self.changed.1.lock().unwrap_or_else(|e| e.into_inner());
            if saved.armed(page) {
                saved.unarmed.insert(page);
            }
        }
        if (HOT_COUNT.load(Ordering::Acquire) != 0 || RESTING_COUNT.load(Ordering::Acquire) != 0)
            && self
                .changed
                .0
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .hot
                .contains(&page)
        {
            let mut g = guards();
            cool_for_gpu(&mut g, &[page], None);
        }
    }

    /// No shader writes the copy any more: nothing is armed or kept.
    pub(crate) fn disarm(&self) {
        let mut saved = self.changed.1.lock().unwrap_or_else(|e| e.into_inner());
        saved.armed.clear();
        saved.unarmed.clear();
        saved.bytes.clear();
    }

    /// The bytes page `page` (a page number) held when it was first written
    /// since it was taken, if it was armed then (and so changed), with what
    /// the copy took of it since: the copy's bytes before the submission.
    pub(crate) fn saved(&self, page: usize) -> Option<Box<[u8]>> {
        self.changed
            .1
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .bytes
            .remove(&page)
    }

    /// The copy took `bytes` at title address `from`: any saved bytes of pages
    /// it covers take them too, so marked copy-back neither takes a partial
    /// refresh for the GPU's nor drops a shader write back to the pre-fault byte.
    pub(crate) fn refreshed(&self, from: usize, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let mut saved = self.changed.1.lock().unwrap_or_else(|e| e.into_inner());
        if saved.bytes.is_empty() {
            return;
        }
        let to = from + bytes.len();
        for page in from / PAGE..to.div_ceil(PAGE) {
            if let Some(before) = saved.bytes.get_mut(&page) {
                let (a, b) = (from.max(page * PAGE), to.min((page + 1) * PAGE));
                before[a - page * PAGE..b - page * PAGE]
                    .copy_from_slice(&bytes[a - from..b - from]);
            }
        }
    }
}

/// Makes sorted `pages` read-only where not already (in runs, with [`GUARDS`]
/// released across kernel calls), and returns sorted pages now guarded or with
/// nothing mapped; mapped pages that could not be guarded or were concurrently
/// faulted/opened are left out.
fn protect(g: &mut Held, pages: &[usize]) -> Vec<usize> {
    protect_with_aliases(g, pages, &[], &[], None, |_| {})
}

/// Guards sorted `pages` AND all alias pages in `pairs` (`(page, alias_page)`)
/// in a single unlocked `PROTECT_SYSCALL` batch, avoiding two separate `GUARDS`
/// / `PROTECT_SYSCALL` round trips per take. Commits only those `page`s in
/// `pages` that are now guarded (or unmapped) AND whose aliases in `pairs` are
/// all guarded.
fn protect_with_aliases(
    g: &mut Held,
    pages: &[usize],
    pairs: &[(usize, usize)],
    watched: &[(usize, usize)],
    mut unmapped_pages: Option<&mut Vec<usize>>,
    on_commit: impl FnOnce(&[usize]),
) -> Vec<usize> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualProtect, VirtualQuery,
    };
    if pages.is_empty() {
        return Vec::new();
    }
    #[cfg(feature = "diagnostics")]
    let _timed = Timed(Instant::now(), crate::copystat::C::ProtectUs);

    let mut alias_pages: Vec<usize> = pairs.iter().map(|&(_, a)| a).collect();
    alias_pages.sort_unstable();
    alias_pages.dedup();

    // Every unique page across `pages ∪ alias_pages` that is not already
    // protected, busy, opening, or mid-guarding becomes one candidate in the
    // single unlocked syscall batch.
    let mut eligible_pages = Vec::with_capacity(pages.len());
    // `(page, known_region_end, known_view)`
    let mut candidates: Vec<(usize, Option<usize>, bool)> = Vec::new();
    let op_id = g.next_guard_op;
    g.next_guard_op = g.next_guard_op.wrapping_add(1).max(1);

    let mut enqueue = |g: &mut Guards, page: usize, is_alias: bool| -> bool {
        if is_busy(g, page) || is_opening(g, page) {
            return false;
        }
        if g.protected.contains(&page) {
            return true;
        }
        if g.guarding.contains_key(&page) {
            return false;
        }
        g.guarding.insert(page, (op_id, false));
        let end = xmem_view_end(g, page).or_else(|| {
            if is_alias {
                None
            } else {
                watched_run_end(watched, page)
            }
        });
        candidates.push((page, end, is_alias));
        true
    };

    for &page in pages {
        if enqueue(g, page, false) {
            eligible_pages.push(page);
        }
    }
    for &alias in &alias_pages {
        if pages.binary_search(&alias).is_err() {
            let _ = enqueue(g, alias, true);
        }
    }

    if candidates.is_empty() {
        let mut committed: Vec<usize> = eligible_pages
            .into_iter()
            .filter(|&p| {
                g.protected.contains(&p)
                    && pairs
                        .iter()
                        .filter(|&&(src, _)| src == p)
                        .all(|&(_, a)| g.protected.contains(&a))
            })
            .collect();
        committed.sort_unstable();
        committed.dedup();
        on_commit(&committed);
        return committed;
    }

    candidates.sort_unstable_by_key(|&(p, _, _)| p);

    let (succeeded, unmapped_found, calls, protected_pages) = g.unlock_for(|| {
        let _syscall = PROTECT_SYSCALL.lock().unwrap_or_else(|e| e.into_inner());
        let set = |from: usize, to: usize| {
            #[cfg(test)]
            tests::PROTECTS.with(|n| n.set(n.get() + 1));
            let mut old = 0;
            // SAFETY: runtime GPU memory; only its protection changes.
            unsafe {
                VirtualProtect(
                    (from * PAGE) as *const _,
                    (to - from) * PAGE,
                    PAGE_READONLY,
                    &mut old,
                ) != 0
            }
        };
        let unmapped = |page: usize| {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            let n =
                unsafe { VirtualQuery((page * PAGE) as *const _, &mut info, size_of_val(&info)) };
            n == 0 || info.State != MEM_COMMIT
        };

        let check_guard = check_page_guard_on();
        let mut allowed = RuntimeRegions::default();
        let mut accepted: Vec<(usize, usize)> = Vec::with_capacity(candidates.len());
        let mut hole_pages = Vec::new();
        for &(p, known_end, known_views) in &candidates {
            if !check_guard {
                if let Some(end) = known_end {
                    accepted.push((p, end));
                    continue;
                }
                if known_views {
                    accepted.push((p, usize::MAX));
                    continue;
                }
            }
            match allowed.classify(p, known_views) {
                PageClass::Alike(end) => accepted.push((p, end)),
                PageClass::Unmapped => hole_pages.push(p),
                PageClass::Skip => {}
            }
        }

        let mut ok_pages = Vec::with_capacity(accepted.len());
        let (mut calls, mut protected_pages) = (0u64, 0u64);
        let mut i = 0;
        while i < accepted.len() {
            let (first_page, region_end) = accepted[i];
            while i + 1 < accepted.len()
                && accepted[i + 1].0 == accepted[i].0 + 1
                && accepted[i + 1].0 < region_end
            {
                i += 1;
            }
            let last_page = accepted[i].0 + 1;
            i += 1;
            calls += 1;
            protected_pages += (last_page - first_page) as u64;
            if set(first_page, last_page) {
                ok_pages.extend(first_page..last_page);
                continue;
            }
            for page in first_page..last_page {
                calls += 1;
                protected_pages += 1;
                if set(page, page + 1) {
                    ok_pages.push(page);
                } else if unmapped(page) {
                    hole_pages.push(page);
                }
            }
        }
        (ok_pages, hole_pages, calls, protected_pages)
    });

    #[cfg(feature = "diagnostics")]
    {
        crate::copystat::add(crate::copystat::C::ProtectCalls, calls);
        crate::copystat::add(crate::copystat::C::ProtectPages, protected_pages);
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = (calls, protected_pages);

    // Reconcile under `GUARDS`: any candidate marked `cancelled` in
    // `g.guarding` (or marked `busy`/`opening`) while unlocked was concurrently
    // faulted/opened or remapped, so it must stay dirty and be reopened to
    // `PAGE_READWRITE` before `GUARDING_CV` wakes waiters.
    let mut cancelled_set = BTreeSet::new();
    for &(page, _, _) in &candidates {
        if let Some(&(id, cancelled)) = g.guarding.get(&page)
            && id == op_id
        {
            g.guarding.remove(&page);
            if cancelled || is_busy(g, page) || is_opening(g, page) {
                cancelled_set.insert(page);
            }
        } else {
            cancelled_set.insert(page);
        }
    }

    let mut reopen_cancelled = Vec::new();
    for page in succeeded {
        if cancelled_set.contains(&page) {
            reopen_cancelled.push(page);
        } else {
            g.protected.insert(page);
        }
    }
    let mut unmapped_ok = BTreeSet::new();
    for page in unmapped_found {
        if !cancelled_set.contains(&page) {
            unmapped_ok.insert(page);
        }
    }

    // Check `g.protected` AFTER re-acquiring `GUARDS` so any page (or alias)
    // that was already protected before `unlock_for` and concurrently faulted
    // while unlocked is excluded from `taken` and stays dirty.
    let mut taken = Vec::with_capacity(eligible_pages.len());
    for p in eligible_pages {
        let is_unmapped = unmapped_ok.contains(&p);
        if !is_unmapped && !g.protected.contains(&p) {
            continue;
        }
        let aliases_ok = pairs
            .iter()
            .filter(|&&(src, _)| src == p)
            .all(|&(_, a)| g.protected.contains(&a) || unmapped_ok.contains(&a));
        if !aliases_ok {
            continue;
        }
        taken.push(p);
        if is_unmapped && let Some(out) = unmapped_pages.as_deref_mut() {
            out.push(p);
        }
    }

    taken.sort_unstable();
    taken.dedup();
    if let Some(out) = unmapped_pages {
        out.sort_unstable();
        out.dedup();
    }
    // Commit tracker state atomically under `GUARDS` before `open_sorted_pages`
    // unlocks `GUARDS` to reopen cancelled pages: otherwise a newly guarded page
    // in `taken` could fault and reopen while unlocked, only for the caller to
    // clear its dirty flag after `protect_with_aliases` returns.
    on_commit(&taken);

    if !reopen_cancelled.is_empty() {
        open_sorted_pages(g, &reopen_cancelled);
    }
    GUARDING_CV.notify_all();
    taken
}

/// Classification of full 64 KiB pages inside a single `VirtualQuery` region.
#[derive(Clone, Copy)]
enum PageClass {
    /// Committed runtime memory that is `PAGE_READONLY` or `PAGE_READWRITE`
    /// throughout `[page, end)` within one region, safe for one `VirtualProtect`.
    Alike(usize),
    /// Uncommitted (`MEM_RESERVE` / `MEM_FREE`) hole.
    Unmapped,
    /// Foreign `MEM_PRIVATE` memory or runtime memory whose 64 KiB span holds
    /// `PAGE_GUARD` (or another non-plain protection): leave untouched.
    Skip,
}

/// Which pages are guardable runtime GPU memory, asked once per `VirtualQuery` region.
#[derive(Default)]
struct RuntimeRegions {
    last: Option<(usize, usize, PageClass)>,
}

impl RuntimeRegions {
    fn classify(&mut self, page: usize, known_views: bool) -> PageClass {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, PAGE_READONLY, PAGE_READWRITE,
            VirtualQuery,
        };
        if let Some((a, b, kind)) = self.last
            && (a..b).contains(&page)
        {
            return kind;
        }
        // A pool's write watch answers with one lookup. `VirtualQuery` walks
        // Wine's page table to the region's end (~2 ms per call in big pools).
        if !known_views && write_watched(page * PAGE) {
            return PageClass::Alike(usize::MAX);
        }
        #[cfg(test)]
        tests::QUERIES.with(|n| n.set(n.get() + 1));
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery((page * PAGE) as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return PageClass::Unmapped;
        }
        let base = info.BaseAddress as usize;
        let reg_end = base.saturating_add(info.RegionSize);
        let a = base.div_ceil(PAGE).min(page);
        let full_end = reg_end / PAGE;
        if full_end <= page {
            // Protection or commit state changes mid-64 KiB page (such as a
            // title `PAGE_GUARD` subpage inside `page`): leave the whole 64 KiB
            // page as it is so `VirtualProtect` never strips the title's guard.
            return PageClass::Skip;
        }
        let kind = if info.State != MEM_COMMIT {
            PageClass::Unmapped
        } else if !known_views && info.Type == MEM_PRIVATE {
            PageClass::Skip
        } else if matches!(info.Protect, PAGE_READONLY | PAGE_READWRITE) {
            PageClass::Alike(full_end)
        } else {
            PageClass::Skip
        };
        self.last = Some((a, full_end, kind));
        kind
    }
}

fn xmem_view_end(g: &Guards, page: usize) -> Option<usize> {
    g.xmem_views
        .range(..=page)
        .next_back()
        .filter(|&(_, &end)| page < end)
        .map(|(_, &end)| end)
}

fn remove_xmem_views(g: &mut Guards, first: usize, last: usize) {
    if first >= last || g.xmem_views.is_empty() {
        return;
    }
    let start_key = g
        .xmem_views
        .range(..=first)
        .next_back()
        .filter(|&(_, &end)| end > first)
        .map(|(&s, _)| s)
        .unwrap_or(first);
    let hit: Vec<(usize, usize)> = g
        .xmem_views
        .range(start_key..last)
        .map(|(&s, &e)| (s, e))
        .collect();
    for (s, e) in hit {
        g.xmem_views.remove(&s);
        if s < first {
            g.xmem_views.insert(s, first);
        }
        if e > last {
            g.xmem_views.insert(last, e);
        }
    }
}

fn is_busy(g: &Guards, page: usize) -> bool {
    g.busy.iter().any(|&(a, b)| (a..b).contains(&page))
}

fn is_opening(g: &Guards, page: usize) -> bool {
    g.opening.iter().any(|&(a, b)| (a..b).contains(&page))
}

fn overlaps_opening(g: &Guards, first: usize, last: usize) -> bool {
    first < last && g.opening.iter().any(|&(a, b)| a < last && first < b)
}

fn aliases_opening(g: &Guards, first: usize, last: usize) -> bool {
    !open_one_view()
        && g.alias_of
            .range(first..last)
            .flat_map(|(_, o)| o.iter().copied())
            .any(|a| is_opening(g, a))
}

fn aliases_guarding(g: &Guards, first: usize, last: usize) -> bool {
    !open_one_view()
        && g.alias_of
            .range(first..last)
            .flat_map(|(_, o)| o.iter().copied())
            .any(|a| g.guarding.contains_key(&a))
}

fn is_page_armed(g: &Guards, page: usize) -> bool {
    g.holders.get(&page).is_some_and(|holders| {
        holders
            .iter()
            .any(|c| c.1.lock().unwrap_or_else(|e| e.into_inner()).armed(page))
    })
}

fn is_group_armed(g: &Guards, group: &[usize]) -> bool {
    group.iter().any(|&p| is_page_armed(g, p))
}

fn is_keep_open_eligible(g: &Guards, page: usize, now_ns: u64) -> bool {
    let has_lease = g.hot.contains_key(&page)
        || g.resting
            .get(&page)
            .is_some_and(|&until_ns| now_ns < until_ns);
    if !has_lease || is_busy(g, page) || is_opening(g, page) || is_page_armed(g, page) {
        return false;
    }
    g.alias_of.get(&page).is_none_or(|aliases| {
        aliases
            .iter()
            .all(|&a| !is_busy(g, a) && !is_opening(g, a) && !is_page_armed(g, a))
    })
}

fn sync_resting_atomics(g: &Guards) {
    RESTING_COUNT.store(g.resting.len(), Ordering::Release);
    let next = g.resting.values().copied().min().unwrap_or(u64::MAX);
    NEXT_REST_EXPIRY_NS.store(next, Ordering::Release);
}

/// Excludes `pages` and their aliases from [`Guards::hot`] (for
/// [`GPU_COLD_FRAMES`] frames) and clears any active [`Guards::resting`] lease.
/// If `rearm` names a tracker and range, open pages it had already taken move
/// to [`Dirty::rearmed`] and are guarded at once (along with their aliases);
/// other holders move them to [`Dirty::pages`].
fn cool_for_gpu(g: &mut Held, pages: &[usize], rearm: Option<(&Arc<Changed>, usize, usize)>) {
    if pages.is_empty() {
        return;
    }
    let until = g.frame + GPU_COLD_FRAMES;
    let mut group: Vec<usize> = pages.to_vec();
    for &p in pages {
        if let Some(aliases) = g.alias_of.get(&p) {
            group.extend(aliases.iter().copied());
        }
    }
    group.sort_unstable();
    group.dedup();
    let mut cooled = Vec::new();
    let mut hot_changed = false;
    let mut rest_changed = false;
    for &p in &group {
        g.gpu_until.insert(p, until);
        let was_hot = g.hot.remove(&p).is_some();
        let was_rest = g.resting.remove(&p).is_some();
        hot_changed |= was_hot;
        rest_changed |= was_rest;
        if was_hot || was_rest {
            cooled.push(p);
        }
    }
    if hot_changed {
        HOT_COUNT.store(g.hot.len(), Ordering::Release);
    }
    if rest_changed {
        sync_resting_atomics(g);
    }
    if cooled.is_empty() {
        return;
    }
    let mut guard_now = Vec::new();
    for &p in &cooled {
        if let Some(holders) = g.holders.get(&p) {
            for c in holders {
                let mut d = c.0.lock().unwrap_or_else(|e| e.into_inner());
                if d.hot.remove(&p) {
                    if let Some((owner, first, last)) = rearm
                        && Arc::ptr_eq(c, owner)
                        && (first..last).contains(&p)
                    {
                        d.rearmed.insert(p);
                        guard_now.push(p);
                    } else {
                        d.pages.insert(p);
                    }
                    set_dirty(c.2, true);
                }
            }
        }
    }
    if !guard_now.is_empty() {
        guard_now.sort_unstable();
        guard_now.dedup();
        let others = aliases(g, &guard_now);
        let guarded = protect_with_aliases(g, &guard_now, &others, &[], None, |_| {});
        if guarded.len() < guard_now.len()
            && let Some((owner, _, _)) = rearm
        {
            let mut d = owner.0.lock().unwrap_or_else(|e| e.into_inner());
            for &p in &guard_now {
                if guarded.binary_search(&p).is_err() && d.rearmed.remove(&p) {
                    d.pages.insert(p);
                }
            }
        }
    }
}

/// Records a write fault at `fault_page`: promotes `fault_page` (and its
/// aliases) into [`Guards::hot`] on repeated faults, or into
/// [`Guards::resting`] for [`guard_rest`] on initial faults.
fn note_fault(g: &mut Guards, fault_page: usize) {
    sync_leases_under_lock(g);
    let hot_on = hot_pages_on();
    let rest = guard_rest();
    if !hot_on && rest.is_zero() {
        return;
    }
    let single = [fault_page];
    let mut aliased_buf = Vec::new();
    let group: &[usize] = if let Some(aliases) = g.alias_of.get(&fault_page) {
        aliased_buf.reserve(1 + aliases.len());
        aliased_buf.push(fault_page);
        aliased_buf.extend(aliases.iter().copied());
        aliased_buf.sort_unstable();
        aliased_buf.dedup();
        &aliased_buf
    } else {
        &single
    };
    let frame = g.frame;
    let mut max_faults = 0;
    let mut seen_recently = false;
    for &p in group {
        let count = g.faults.entry(p).or_insert(0);
        *count += 1;
        max_faults = max_faults.max(*count);
        if let Some(&prev) = g.last_fault_frame.get(&p)
            && prev < frame
            && frame - prev <= HOT_WINDOW_FRAMES
        {
            seen_recently = true;
        }
        if let Some(&cooled) = g.cooled_frame.get(&p)
            && frame.saturating_sub(cooled) <= HOT_WINDOW_FRAMES
        {
            seen_recently = true;
        }
        g.last_fault_frame.insert(p, frame);
    }
    if group.iter().any(|&p| is_busy(g, p) || is_opening(g, p)) || is_group_armed(g, group) {
        return;
    }
    let gpu_cold = group
        .iter()
        .any(|&p| g.gpu_until.get(&p).is_some_and(|&u| frame < u));
    if hot_on && !gpu_cold && (max_faults >= 2 || seen_recently) {
        let until = frame + hot_span(group[0]);
        let mut rest_removed = false;
        for &p in group {
            g.hot.insert(p, until);
            g.cooled_frame.remove(&p);
            rest_removed |= g.resting.remove(&p).is_some();
        }
        HOT_COUNT.store(g.hot.len(), Ordering::Release);
        if rest_removed {
            sync_resting_atomics(g);
        }
        return;
    }
    if !rest.is_zero() {
        let until_ns =
            monotonic_ns().saturating_add(rest.as_nanos().min(u128::from(u64::MAX / 2)) as u64);
        let mut added = false;
        for &p in group {
            if !g.hot.contains_key(&p) {
                g.resting.insert(p, until_ns);
                added = true;
            }
        }
        if added {
            sync_resting_atomics(g);
        }
    }
}

fn sync_leases() {
    let now = crate::queue::presented();
    let rest_due = RESTING_COUNT.load(Ordering::Acquire) != 0
        && monotonic_ns() >= NEXT_REST_EXPIRY_NS.load(Ordering::Acquire);
    if now != LAST_PRESENTED.load(Ordering::Relaxed) || rest_due {
        let mut g = guards();
        sync_leases_under_lock(&mut g);
    }
}

fn sync_leases_under_lock(g: &mut Guards) {
    let now = crate::queue::presented();
    if now != g.last_presented {
        g.last_presented = now;
        LAST_PRESENTED.store(now, Ordering::Relaxed);
        advance_frame(g);
    }
    if !g.resting.is_empty() {
        let now_ns = monotonic_ns();
        if now_ns >= NEXT_REST_EXPIRY_NS.load(Ordering::Relaxed) {
            expire_resting(g, now_ns);
        }
    }
}

fn expire_resting(g: &mut Guards, now_ns: u64) {
    let mut expired = Vec::new();
    g.resting.retain(|&page, &mut until_ns| {
        if now_ns >= until_ns {
            expired.push(page);
            false
        } else {
            true
        }
    });
    if expired.is_empty() {
        return;
    }
    sync_resting_atomics(g);
    for page in expired {
        if g.hot.contains_key(&page) {
            continue;
        }
        if let Some(holders) = g.holders.get(&page) {
            for c in holders {
                let mut d = c.0.lock().unwrap_or_else(|e| e.into_inner());
                if d.hot.remove(&page) {
                    d.pages.insert(page);
                    set_dirty(c.2, true);
                }
            }
        }
    }
}

fn advance_frame(g: &mut Guards) {
    g.frame += 1;
    let frame = g.frame;
    #[cfg(feature = "diagnostics")]
    {
        let diag_written = std::mem::take(&mut g.written);
        let again = diag_written.intersection(&g.written_before).count();
        crate::copystat::add(crate::copystat::C::WrittenPages, diag_written.len() as u64);
        crate::copystat::add(crate::copystat::C::WrittenAgain, again as u64);
        g.written_before = diag_written;
    }
    g.faults.clear();
    g.last_fault_frame
        .retain(|_, &mut prev| frame.saturating_sub(prev) <= HOT_WINDOW_FRAMES);
    g.cooled_frame
        .retain(|_, &mut cooled| frame.saturating_sub(cooled) <= HOT_WINDOW_FRAMES);
    g.gpu_until.retain(|_, &mut until| frame < until);

    let mut expired = Vec::new();
    g.hot.retain(|&page, &mut until| {
        if frame >= until {
            expired.push(page);
            false
        } else {
            true
        }
    });
    if !expired.is_empty() {
        HOT_COUNT.store(g.hot.len(), Ordering::Release);
        for page in expired {
            g.cooled_frame.insert(page, frame);
            if g.resting.contains_key(&page) {
                continue;
            }
            if let Some(holders) = g.holders.get(&page) {
                for c in holders {
                    let mut d = c.0.lock().unwrap_or_else(|e| e.into_inner());
                    if d.hot.remove(&page) {
                        d.pages.insert(page);
                        set_dirty(c.2, true);
                    }
                }
            }
        }
    }
}

/// Pages `[first, last)` changed, and so the pages showing the same memory
/// elsewhere: gives write access back where we guarded them (releasing
/// [`GUARDS`] across `VirtualProtect`) and marks them in every tracker holding
/// them. True if one of `[first, last)` was guarded (or mid-guarding).
fn changed(g: &mut Held, first: usize, last: usize, save: bool) -> bool {
    changed_inner(g, first, last, save, true)
}

/// Like [`changed`], with `open_self = false` when the OS mapping for
/// `[first, last)` was already unmapped or freshly mapped read-write so no
/// `VirtualProtect(PAGE_READWRITE)` syscall is needed on `[first, last)`.
fn changed_inner(g: &mut Held, first: usize, last: usize, save: bool, open_self: bool) -> bool {
    let mut to_open: Vec<usize> = g.protected.range(first..last).copied().collect();
    let mut cancelled_guarding = false;
    for (_, (_, cancelled)) in g.guarding.range_mut(first..last) {
        *cancelled = true;
        cancelled_guarding = true;
    }
    let guarded = !to_open.is_empty() || cancelled_guarding;
    for &p in &to_open {
        g.protected.remove(&p);
    }
    if !open_self {
        to_open.clear();
    }

    let others: Vec<usize> = g
        .alias_of
        .range(first..last)
        .flat_map(|(_, o)| o.iter().copied())
        .collect();
    #[cfg(feature = "diagnostics")]
    if !others.is_empty() {
        let held = others.iter().filter(|p| g.holders.contains_key(p)).count();
        crate::copystat::add(crate::copystat::C::AliasMarks, held as u64);
    }

    // Every page's bytes are saved before any is opened: while the page is
    // still `PAGE_READONLY` in the OS, no concurrent CPU thread can write it.
    mark_changed(g, first, last, save);
    let one_view = open_one_view();
    for &page in &others {
        mark_changed(g, page, page + 1, save);
        if !one_view && !(first..last).contains(&page) {
            if g.protected.remove(&page) {
                to_open.push(page);
            }
            if let Some((_, cancelled)) = g.guarding.get_mut(&page) {
                *cancelled = true;
            }
        }
    }
    if !one_view && to_open.len() > 1 {
        to_open.sort_unstable();
        to_open.dedup();
    }
    open_sorted_pages(g, &to_open);
    guarded
}

/// Marks pages `[first, last)` changed in every tracker holding them,
/// leaving their protection as it is. With `save`, before the write lands:
/// trackers armed for a page they had taken keep its bytes.
fn mark_changed(g: &mut Guards, first: usize, last: usize, save: bool) {
    for (&page, holders) in g.holders.range(first..last) {
        for c in holders {
            let first_write = {
                let mut d = c.0.lock().unwrap_or_else(|e| e.into_inner());
                let was_open = d.hot.remove(&page) | d.rearmed.remove(&page);
                let first_write = d.pages.insert(page) && !was_open;
                set_dirty(c.2, true);
                first_write
            };
            if first_write && save {
                c.save(page);
            }
        }
        #[cfg(feature = "diagnostics")]
        g.written.insert(page);
    }
}

/// At each presented frame: expires cooled hot-page leases and (with
/// diagnostics) counts the tracked pages written in the frame and those
/// written the frame before too.
#[cfg(any(test, feature = "diagnostics"))]
pub(crate) fn frame_done() {
    let mut g = guards();
    let now = crate::queue::presented();
    if !cfg!(test) && now != 0 && g.last_presented == now {
        return;
    }
    g.last_presented = now;
    LAST_PRESENTED.store(now, Ordering::Relaxed);
    advance_frame(&mut g);
}

/// The memory runtime pins writable section views covering `[at, at + len)`
/// for the duration of the closure, passing contiguous segments in address
/// order. False means no bytes were touched.
type CopyBackView = fn(usize, usize, &mut dyn FnMut(&mut [&mut [u8]])) -> bool;
static COPY_BACK_VIEW: OnceLock<CopyBackView> = OnceLock::new();

pub fn use_copy_back_view(view: CopyBackView) {
    let _ = COPY_BACK_VIEW.set(view);
}

/// Saves armed snapshots and notifies every tracker before changing the shared
/// section bytes through unguarded views, keeping title page guards intact.
fn write_through_view(
    at: usize,
    slices: &mut [&mut [u8]],
    copy: &mut dyn FnMut(&mut [&mut [u8]]),
) {
    let len: usize = slices.iter().map(|s| s.len()).sum();
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    {
        let mut g = guards();
        mark_changed(&mut g, first, last, true);
        let others: Vec<usize> = g
            .alias_of
            .range(first..last)
            .flat_map(|(_, pages)| pages.iter().copied())
            .collect();
        for page in others {
            mark_changed(&mut g, page, page + 1, true);
        }
        copy(slices);
    }
    note_subpages_bytes(at, len);
}

/// Writes `src` into title memory at `at` through unguarded `xmem` section
/// views, keeping title page guards intact across single- and multi-view spans.
pub(crate) fn copy_back_bytes(at: usize, src: &[u8]) -> bool {
    if !back_own_view_on() || src.is_empty() {
        return false;
    }
    let stored = COPY_BACK_VIEW.get().is_some_and(|view| {
        view(at, src.len(), &mut |slices| {
            write_through_view(at, slices, &mut |dsts| {
                let mut off = 0;
                for dst in dsts.iter_mut() {
                    let n = dst.len();
                    dst.copy_from_slice(&src[off..off + n]);
                    off += n;
                }
            });
        })
    });
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(
            if stored {
                C::OwnBackBytes
            } else {
                C::OwnBackMissBytes
            },
            src.len() as u64,
        );
    }
    stored
}

/// Reads `[at, at + len)` directly from the runtime's pinned `xmem` section
/// views in userspace without kernel `ReadProcessMemory` calls or touching
/// title page guards; invokes `read(byte_offset, chunk)` for each segment.
pub(crate) fn read_own_view_chunks(
    at: usize,
    len: usize,
    mut read: impl FnMut(usize, &[u8]),
) -> bool {
    if !back_own_view_on() || len == 0 {
        return false;
    }
    COPY_BACK_VIEW.get().is_some_and(|view| {
        view(at, len, &mut |slices| {
            let mut off = 0;
            for src in slices.iter() {
                read(off, src);
                off += src.len();
            }
        })
    })
}

/// A write into title memory the guards do not see, under way: its pages
/// stay open until dropped, which marks them changed once more.
#[must_use = "the write must happen while this lives"]
pub struct Writing {
    at: usize,
    len: usize,
    first: usize,
    last: usize,
}

impl Drop for Writing {
    fn drop(&mut self) {
        // An unmapping with tracking off: nothing was noted.
        if self.first == self.last {
            return;
        }
        {
            let mut g = guards();
            if let Some(i) = g.busy.iter().position(|&r| r == (self.first, self.last)) {
                g.busy.swap_remove(i);
            }
            changed(&mut g, self.first, self.last, false);
        }
        note_subpages_bytes(self.at, self.len);
    }
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler) or another write the
/// guards do not see: opens its guarded blocks, marks them changed and
/// keeps them open while the result lives.
pub fn writing(at: usize, len: usize) -> Writing {
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    {
        let mut g = guards();
        g.busy.push((first, last));
        loop {
            changed(&mut g, first, last, true);
            if g.guarding.range(first..last).next().is_some()
                || aliases_guarding(&g, first, last)
            {
                g.wait(&GUARDING_CV);
                continue;
            }
            if overlaps_opening(&g, first, last) || aliases_opening(&g, first, last) {
                g.wait(&OPENING_CV);
                continue;
            }
            break;
        }
    }
    note_subpages_bytes(at, len);
    Writing {
        at,
        len,
        first,
        last,
    }
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new, and read-write.
pub fn title_memory_remapped(at: usize, len: usize) {
    let mut g = guards();
    title_memory_remapped_under(&mut g, at, len, false, true);
    drop(g);
    note_subpages_whole(at / PAGE, at.saturating_add(len).div_ceil(PAGE));
}

/// The runtime already unmapped `[at, at + len)` back to a placeholder: drops
/// guard state and notifies trackers without issuing a redundant
/// `VirtualProtect(PAGE_READWRITE)` on the unmapped range.
pub fn title_memory_unmapped(at: usize, len: usize) {
    let mut g = guards();
    title_memory_remapped_under(&mut g, at, len, false, false);
    drop(g);
    note_subpages_whole(at / PAGE, at.saturating_add(len).div_ceil(PAGE));
}

/// The runtime mapped a live `xmem` section view over `[at, at + len)` (fresh
/// `PAGE_READWRITE`): notifies trackers and records the view span without
/// issuing a redundant `VirtualProtect(PAGE_READWRITE)` on `[at, at + len)`.
pub fn title_memory_mapped(at: usize, len: usize) {
    let mut g = guards();
    title_memory_remapped_under(&mut g, at, len, true, false);
    drop(g);
    note_subpages_whole(at / PAGE, at.saturating_add(len).div_ceil(PAGE));
}

/// Combined [`title_memory_mapped`] + batched [`title_memory_aliased`] for all
/// aliased pages of a newly mapped `xmem` view in one `GUARDS` acquisition and
/// one coalesced `protect` batch.
pub fn title_memory_mapped_aliased(base: usize, len: usize, aliased: &[(usize, Vec<usize>)]) {
    let mut g = guards();
    title_memory_remapped_under(&mut g, base, len, true, false);
    if !aliased.is_empty() {
        register_aliases_under(&mut g, aliased);
    }
    drop(g);
    note_subpages_whole(base / PAGE, base.saturating_add(len).div_ceil(PAGE));
}

/// Before title memory `[at, at + len)` is unmapped or freed (`len` 0: the
/// whole allocation at `at`, as `VirtualFree` with `MEM_RELEASE` frees):
/// as [`title_memory_remapped`], and its pages stay unguarded until the
/// result is dropped, after the memory went.
///
/// A page guarded while it goes stays in the guard set after it went.
/// Memory allocated at that address later (the title's pools are freed
/// and allocated again; nothing tells the guards of an allocation) then
/// counts as guarded already and is never made read-only: its writes go
/// unseen, and copies of it keep stale bytes. So the runtime calls this
/// around its unmappings and the title's frees, and a tracker taking the
/// pages meanwhile leaves them changed, as during a kernel write.
pub fn title_memory_unmapping(at: usize, len: usize) -> Writing {
    if !fault_tracking() || at == 0 {
        return Writing {
            at: 0,
            len: 0,
            first: 0,
            last: 0,
        };
    }
    let len = if len == 0 { allocation_len(at) } else { len };
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    let mut g = guards();
    g.busy.push((first, last));
    title_memory_remapped_under(&mut g, at, len, false, true);
    drop(g);
    Writing {
        at,
        len,
        first,
        last,
    }
}

/// The bytes of the allocation that starts at `at`: its regions while they
/// share that allocation base (guarded and open pages are regions of their
/// own). 0 if nothing is allocated there.
fn allocation_len(at: usize) -> usize {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut len = 0usize;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery((at + len) as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.AllocationBase as usize != at || info.RegionSize == 0 {
            return len;
        }
        len += info.RegionSize;
    }
}

fn title_memory_remapped_under(
    g: &mut Held,
    at: usize,
    len: usize,
    mapped: bool,
    open_self: bool,
) {
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    let mut reset: Vec<usize> = g
        .hot
        .range(first..last)
        .map(|(&p, _)| p)
        .chain(g.resting.range(first..last).map(|(&p, _)| p))
        .collect();
    for (_, aliases) in g.alias_of.range(first..last) {
        for &a in aliases {
            if g.hot.contains_key(&a) || g.resting.contains_key(&a) {
                reset.push(a);
            }
        }
    }
    if !reset.is_empty() {
        reset.sort_unstable();
        reset.dedup();
        let mut hot_changed = false;
        let mut rest_changed = false;
        for &p in &reset {
            hot_changed |= g.hot.remove(&p).is_some();
            rest_changed |= g.resting.remove(&p).is_some();
            if let Some(holders) = g.holders.get(&p) {
                for c in holders {
                    let mut d = c.0.lock().unwrap_or_else(|e| e.into_inner());
                    let had = d.hot.remove(&p) | d.rearmed.remove(&p);
                    if had {
                        d.pages.insert(p);
                        set_dirty(c.2, true);
                    }
                }
            }
        }
        if hot_changed {
            HOT_COUNT.store(g.hot.len(), Ordering::Release);
        }
        if rest_changed {
            sync_resting_atomics(g);
        }
    }
    let clear_keys: Vec<usize> = g
        .faults
        .range(first..last)
        .map(|(&p, _)| p)
        .chain(g.last_fault_frame.range(first..last).map(|(&p, _)| p))
        .chain(g.cooled_frame.range(first..last).map(|(&p, _)| p))
        .chain(g.gpu_until.range(first..last).map(|(&p, _)| p))
        .collect();
    for p in clear_keys {
        g.faults.remove(&p);
        g.last_fault_frame.remove(&p);
        g.cooled_frame.remove(&p);
        g.gpu_until.remove(&p);
    }
    changed_inner(g, first, last, false, open_self);
    let stale: Vec<usize> = g.alias_of.range(first..last).map(|(&p, _)| p).collect();
    for page in stale {
        for other in g.alias_of.remove(&page).unwrap_or_default() {
            if let Some(links) = g.alias_of.get_mut(&other) {
                links.retain(|&p| p != page);
                if links.is_empty() {
                    g.alias_of.remove(&other);
                }
            }
        }
    }
    remove_xmem_views(g, first, last);
    if mapped && first < last {
        g.xmem_views.insert(first, last);
    }
}

/// Reports GPU shader writes into aliased title memory `[at, at + len)` so
/// any other trackers over those pages (such as textures' staging) know they
/// changed and re-upload on next use.
pub(crate) fn notify_written(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    {
        let mut g = guards();
        if g.holders.range(first..last).next().is_none()
            && g.alias_of.range(first..last).next().is_none()
        {
            return;
        }
        changed(&mut g, first, last, false);
    }
    note_subpages_bytes(at, len);
}

/// What shaders wrote into host copies of `[at, at + len)` is dropped:
/// every host buffer copy holding the memory takes it from title memory
/// again at its next refresh.
pub(crate) fn retake(at: usize, len: usize) {
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    let others: Vec<usize> = {
        let g = guards();
        let others: Vec<usize> = g
            .alias_of
            .range(first..last)
            .flat_map(|(_, o)| o.iter().copied())
            .collect();
        mark_copies(&g, first, last);
        for &page in &others {
            mark_copies(&g, page, page + 1);
        }
        others
    };
    note_subpages_bytes(at, len);
    for page in others {
        note_subpages_whole(page, page + 1);
    }
}

/// Pages `[first, last)` count as changed in every host buffer copy's
/// tracker holding them.
fn mark_copies(g: &Guards, first: usize, last: usize) {
    for (&page, holders) in g.holders.range(first..last) {
        for c in holders.iter().filter(|c| c.3) {
            let mut d = c.0.lock().unwrap_or_else(|e| e.into_inner());
            d.hot.remove(&page);
            d.rearmed.remove(&page);
            d.pages.insert(page);
            set_dirty(c.2, true);
        }
    }
}

/// The runtime mapped the page at `at` where `others` show it already: the
/// pages are linked both ways, and if we guard it there, it is guarded here
/// too.
pub fn title_memory_aliased(at: usize, others: &[usize]) {
    if others.is_empty() {
        return;
    }
    let mut g = guards();
    register_aliases_under(&mut g, &[(at, others.to_vec())]);
}

fn register_aliases_under(g: &mut Held, aliased: &[(usize, Vec<usize>)]) {
    let other_guarded = |g: &Guards, o: usize| {
        g.protected.contains(&(o / PAGE))
            || g.guarding.get(&(o / PAGE)).is_some_and(|&(_, c)| !c)
    };
    let mut hot_changed = false;
    let mut rest_changed = false;
    let mut to_guard: Vec<usize> = Vec::new();
    let mut busy_mark: Vec<usize> = Vec::new();

    for (at, others) in aliased {
        if others.is_empty() {
            continue;
        }
        let page = at / PAGE;
        for &other in others {
            for (x, y) in [(other / PAGE, page), (page, other / PAGE)] {
                let links = g.alias_of.entry(x).or_default();
                if !links.contains(&y) {
                    links.push(y);
                }
            }
        }
        if let Some(until) = others
            .iter()
            .filter_map(|&o| g.gpu_until.get(&(o / PAGE)).copied())
            .max()
        {
            g.gpu_until.insert(page, until);
        }
        if is_busy(g, page) || is_opening(g, page) {
            if others.iter().any(|&o| other_guarded(g, o)) {
                busy_mark.push(page);
            }
            continue;
        }
        if let Some(until) = others
            .iter()
            .filter_map(|&o| g.hot.get(&(o / PAGE)).copied())
            .max()
        {
            g.hot.insert(page, until);
            hot_changed = true;
        } else if let Some(until_ns) = others
            .iter()
            .filter_map(|&o| g.resting.get(&(o / PAGE)).copied())
            .max()
        {
            g.resting.insert(page, until_ns);
            rest_changed = true;
        } else if others.iter().any(|&o| other_guarded(g, o)) {
            to_guard.push(page);
        }
    }
    if hot_changed {
        HOT_COUNT.store(g.hot.len(), Ordering::Release);
    }
    if rest_changed {
        sync_resting_atomics(g);
    }
    for page in busy_mark {
        changed(g, page, page + 1, false);
    }
    if !to_guard.is_empty() {
        to_guard.sort_unstable();
        to_guard.dedup();
        let guarded = protect(g, &to_guard);
        for page in to_guard {
            if guarded.binary_search(&page).is_err() {
                changed(g, page, page + 1, false);
            }
        }
    }
}

/// Whether a page of `[at, at + len)` is shown elsewhere too.
pub(crate) fn aliased(at: usize, len: usize) -> bool {
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    guards().alias_of.range(first..last).next().is_some()
}

/// Handles a write access violation at `at`. Returns `true` when execution
/// should resume (`EXCEPTION_CONTINUE_EXECUTION`), `false` when the fault does
/// not belong to our guards (`EXCEPTION_CONTINUE_SEARCH`).
fn handle_write_fault(at: usize) -> bool {
    let page = at / PAGE;
    #[cfg(feature = "diagnostics")]
    let _timed = Timed(Instant::now(), crate::copystat::C::HandlerUs);
    if HOLDING.with(|h| h.get()) {
        if read_only(at) {
            open_under_lock(page);
            return true;
        }
        // Another thread's `open_sorted_pages` can concurrently restore
        // `PAGE_READWRITE` before `read_only(at)` runs.
        if writable(at) {
            note_opened_under_lock(page);
            return true;
        }
        return false;
    }
    let mut g = guards();
    let mut waited_opening = false;
    let mut cancelled_guard = false;
    while is_opening(&g, page) || g.guarding.contains_key(&page) {
        if is_opening(&g, page) {
            waited_opening = true;
            g.wait(&OPENING_CV);
        } else {
            mark_changed(&mut g, page, page + 1, true);
            let others: Vec<usize> = g
                .alias_of
                .get(&page)
                .into_iter()
                .flatten()
                .copied()
                .collect();
            for alias in others {
                mark_changed(&mut g, alias, alias + 1, true);
            }
            if let Some((_, cancelled)) = g.guarding.get_mut(&page) {
                *cancelled = true;
                cancelled_guard = true;
            }
            g.wait(&GUARDING_CV);
        }
    }
    if changed(&mut g, page, page + 1, true) || cancelled_guard {
        note_fault(&mut g, page);
        drop(g);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardFaults, 1);
        return true;
    }
    // Check `writable(at)` while still holding `GUARDS` so a concurrent
    // `protect_inner` cannot flip `page` back to `PAGE_READONLY` between
    // dropping `GUARDS` and querying the page's protection, which would
    // otherwise turn a stale fault on an already-opened guarded page into an
    // unhandled access violation.
    let handled = waited_opening || writable(at);
    drop(g);
    handled
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        if handle_write_fault(r.ExceptionInformation[1]) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed memory that is read-only, as our guards make it.
fn read_only(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_READONLY, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0
        && info.State == MEM_COMMIT
        && info.Protect & PAGE_GUARD == 0
        && info.Protect & 0xff == PAGE_READONLY
}

/// Whether `at` is committed read-write memory now.
pub(crate) fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READWRITE, PAGE_GUARD, PAGE_READWRITE,
        VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0
        && info.State == MEM_COMMIT
        && info.Protect & PAGE_GUARD == 0
        && matches!(info.Protect & 0xff, PAGE_READWRITE | PAGE_EXECUTE_READWRITE)
}

/// Adds the time since it was made to a counter when dropped (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) struct Timed(pub(crate) Instant, pub(crate) crate::copystat::C);

#[cfg(feature = "diagnostics")]
impl Drop for Timed {
    fn drop(&mut self) {
        let c = std::mem::replace(&mut self.1, crate::copystat::C::Submissions);
        crate::copystat::add(c, self.0.elapsed().as_micros() as u64);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn a_fault_under_the_guard_lock_does_not_hang() {
        use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
        install();
        // SAFETY: a fresh block of our own.
        let block = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                PAGE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        }
        .cast::<u8>();
        assert!(!block.is_null());
        let mut old = 0;
        // SAFETY: as above.
        let made_ro = unsafe { VirtualProtect(block.cast(), PAGE, PAGE_READONLY, &mut old) };
        assert_ne!(made_ro, 0);
        {
            let _g = guards();
            // SAFETY: inside the block; the handler gives write access back.
            unsafe { block.add(100).write_volatile(7) };
        }
        // SAFETY: as above.
        assert_eq!(unsafe { block.add(100).read_volatile() }, 7);
        assert!(writable(block as usize));
        drop(guards());
        assert!(
            OPENED_UNDER_LOCK
                .iter()
                .all(|s| s.load(std::sync::atomic::Ordering::Relaxed) != block as usize / PAGE)
        );
        // SAFETY: allocated above.
        assert_ne!(unsafe { VirtualFree(block.cast(), 0, MEM_RELEASE) }, 0);
    }

    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    pub(crate) const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn fault_tracking_follows_the_variable() {
        assert!(decide(Some("1"), false));
        assert!(!decide(Some("0"), true));
        assert!(decide(None, true));
        assert!(!decide(None, false));
        assert!(!decide(Some("yes"), false));
        assert!(fault_tracking() || std::env::var("VICTORY_FAULT_TRACKING").as_deref() == Ok("0"));
    }

    #[test]
    fn tracking_is_opt_in_under_rosetta() {
        assert!(decide(Some("1"), false));
        assert!(!decide(Some("0"), true));
        assert!(decide(None, true));
        assert!(!decide(None, false));
        assert!(!decide(Some("2"), false));
        assert_eq!(stores_need_patching(), fault_tracking() && rosetta());
    }

    /// A write-watched allocation of `pages` pages (as the runtime's pools),
    /// on a 64 KiB boundary (the allocation granularity).
    pub(crate) fn pool(pages: usize) -> usize {
        // SAFETY: a fresh allocation, freed by the test.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                pages * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert!(p != 0 && p.is_multiple_of(PAGE));
        p
    }

    /// Frees a test's allocation at `p`, unless the guards still guard or
    /// hold a page of it: then it stays allocated, as the other tests'
    /// pools do.
    ///
    /// A copy that a batch, a page table or a widened view still holds
    /// keeps its tracker after the test drops its own handle. Freed under
    /// it, the pages stayed in `protected`, and a test running at the same
    /// time that was given the same address got them unguarded:
    /// [`protect`] skips a page it has guarded already, so the new
    /// allocation stayed writable and its writes went unseen (#31: a guard
    /// test failed in 7 of 150 full runs on an M5 Pro, whichever drew the
    /// address; 2 of 300 with this). The title's own memory does not go
    /// this way: the runtime tells the guards when it unmaps it
    /// ([`title_memory_remapped`]).
    pub(crate) fn free(p: usize) {
        if still_watched(p) {
            return;
        }
        // SAFETY: an allocation of ours.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// Whether the guards guard or hold a page of the allocation at `p`.
    fn still_watched(p: usize) -> bool {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // The allocation's length: its regions, while they start from `p`
        // (guarded and open pages are regions of their own).
        let mut len = 0;
        loop {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            let n = unsafe { VirtualQuery((p + len) as *const _, &mut info, size_of_val(&info)) };
            if n == 0 || info.AllocationBase as usize != p || info.RegionSize == 0 {
                break;
            }
            len += info.RegionSize;
        }
        let (first, last) = (p / PAGE, (p + len).div_ceil(PAGE));
        let g = guards();
        g.protected.range(first..last).next().is_some()
            || g.guarding.range(first..last).next().is_some()
            || overlaps_opening(&g, first, last)
            || g.holders.range(first..last).next().is_some()
    }

    /// Everything is changed at first; after taking, only what is written
    /// (by faults, or reported) is, and every tracker of a page sees it.
    /// A tracked range that starts in reserved memory and runs into plain
    /// committed memory (the host's, as the process heap was in #50) never
    /// makes that memory read-only; it is left out of the tracker instead
    /// (copied whole at each use, so the tracker is never clean).
    #[test]
    fn plain_committed_memory_is_never_guarded() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_READWRITE,
            VirtualAlloc, VirtualFree, VirtualQuery,
        };
        // SAFETY: plain reservation and commit of our own memory.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE, PAGE_READWRITE) }
            as usize;
        assert_ne!(base, 0);
        // SAFETY: inside our reservation.
        let committed = unsafe {
            VirtualAlloc(
                (base + 2 * PAGE) as *const _,
                2 * PAGE,
                MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        assert!(!committed.is_null());
        let t = track(base, 4 * PAGE, true).expect("reserved memory counts as runtime memory");
        assert!(!t.is_untracked(base / PAGE) && !t.is_untracked(base / PAGE + 1));
        assert!(t.is_untracked(base / PAGE + 2) && t.is_untracked(base / PAGE + 3));
        let _ = t.take(base, base + 4 * PAGE);
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery((base + 2 * PAGE) as *const _, &mut info, size_of_val(&info)) };
        assert_eq!(info.Protect, PAGE_READWRITE, "still writable");
        assert!(
            !t.changed(base / PAGE + 2) && !t.changed(base / PAGE + 3),
            "not tracked"
        );
        assert_eq!(
            t.untracked(base, base + 4 * PAGE),
            [(base + 2 * PAGE, base + 4 * PAGE)],
            "copied whole"
        );
        assert!(!t.clean(), "never clean");
        // SAFETY: writable, so no fault.
        unsafe { ((base + 2 * PAGE + 8) as *mut u8).write_volatile(1) };
        drop(t);
        // SAFETY: our reservation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn writes_mark_every_tracker_of_their_page() {
        let p = pool(4);
        let a = track(p, 4 * PAGE, true).expect("pool memory is tracked");
        let b = track(p + PAGE, 2 * PAGE, true).expect("tracked");
        assert!(!a.clean() && !b.clean(), "changed at first");
        assert_eq!(a.take(p, p + 4 * PAGE), [(p, p + 4 * PAGE)]);
        assert!(a.clean() && !b.clean());
        assert_eq!(b.take(p, p + 4 * PAGE), [(p + PAGE, p + 3 * PAGE)]);
        assert!(a.take(p, p + 4 * PAGE).is_empty() && b.take(p, p + 4 * PAGE).is_empty());
        assert!(a.clean() && b.clean());
        // SAFETY: committed pages of ours, now read-only: the handler opens
        // the block; the second write needs no fault.
        unsafe {
            ((p + PAGE + 8) as *mut u8).write_volatile(1);
            ((p + PAGE + 9) as *mut u8).write_volatile(2);
        }
        assert!(!a.clean() && !b.clean(), "a write is seen by every tracker");
        assert_eq!(a.take(p, p + 4 * PAGE), [(p + PAGE, p + 2 * PAGE)]);
        assert!(a.clean());
        assert!(b.changed(p / PAGE + 1));
        assert_eq!(b.take(p, p + 4 * PAGE), [(p + PAGE, p + 2 * PAGE)]);
        // SAFETY: as above: a fault again, for both.
        unsafe { ((p + PAGE + 10) as *mut u8).write_volatile(3) };
        assert!(a.changed(p / PAGE + 1) && b.changed(p / PAGE + 1));
        assert_eq!(a.take(p, p + 4 * PAGE), [(p + PAGE, p + 2 * PAGE)]);
        drop(writing(p + 3 * PAGE + 5, 1));
        assert_eq!(a.take(p, p + 4 * PAGE), [(p + 3 * PAGE, p + 4 * PAGE)]);
        title_memory_remapped(p, PAGE);
        assert!(a.changed(p / PAGE) && !a.changed(p / PAGE + 1));
        drop((a, b));
        free(p);
    }

    #[test]
    fn a_range_over_foreign_memory_guards_only_the_runtimes() {
        use windows_sys::Win32::System::Memory::{
            MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualQuery,
        };
        // This test tracks a mostly free address range. In the parallel suite,
        // other tests can map their views in its holes even after both of our
        // allocations succeed. Keep those views out of this address space.
        const CHILD: &str = "VICTORY_TEST_FOREIGN_WINDOW_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let name = "watch::tests::a_range_over_foreign_memory_guards_only_the_runtimes";
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", name, "--nocapture"])
                .env(CHILD, "1")
                .output()
                .expect("run foreign-window test in a child");
            let stdout = String::from_utf8_lossy(&result.stdout);
            assert!(
                result.status.success() && stdout.contains("1 passed"),
                "foreign-window child {}: {stdout}{}",
                result.status,
                String::from_utf8_lossy(&result.stderr)
            );
            return;
        }
        const WINDOW: usize = 4 << 20;
        // In a free 4 MiB-aligned window: plain committed memory (as the
        // process heap's) 1 MiB in, a pool 2 MiB in.
        let (base, foreign, pool) = (0..8)
            .find_map(|_| {
                // SAFETY: a fresh reservation, released at once; then
                // allocations in the window it left free, freed below.
                unsafe {
                    let p = VirtualAlloc(std::ptr::null(), 2 * WINDOW, MEM_RESERVE, PAGE_READWRITE)
                        as usize;
                    VirtualFree(p as *mut _, 0, MEM_RELEASE);
                    let base = p.next_multiple_of(WINDOW);
                    let at = |offset: usize, kind: u32| {
                        VirtualAlloc(
                            (base + offset) as *const _,
                            PAGE,
                            MEM_RESERVE | MEM_COMMIT | kind,
                            PAGE_READWRITE,
                        ) as usize
                    };
                    let (foreign, pool) = (at(1 << 20, 0), at(2 << 20, MEM_WRITE_WATCH));
                    if p != 0 && foreign == base + (1 << 20) && pool == base + (2 << 20) {
                        return Some((base, foreign, pool));
                    }
                    [foreign, pool]
                        .into_iter()
                        .filter(|&q| q != 0)
                        .for_each(free);
                    None
                }
            })
            .expect("a free window");
        let t = track(base, WINDOW, true).expect("the pool is tracked");
        let taken = t.take(base, base + WINDOW);
        assert!(
            taken.iter().any(|&(a, b)| a <= pool && pool + PAGE <= b),
            "{taken:x?}"
        );
        assert!(
            !taken
                .iter()
                .any(|&(a, b)| a < foreign + PAGE && foreign < b),
            "{taken:x?}"
        );
        let protection = |p: usize| {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
            info.Protect
        };
        assert_eq!(protection(pool), PAGE_READONLY, "the pool is guarded");
        assert_eq!(
            protection(foreign),
            PAGE_READWRITE,
            "the other allocation is not"
        );
        drop(t);
        free(foreign);
        free(pool);
    }

    #[test]
    fn a_new_tracker_starts_dirty_on_a_bit_given_back() {
        let p = pool(1);
        let a = track(p, PAGE, true).expect("tracked");
        assert_eq!(a.take(p, p + PAGE).len(), 1);
        assert!(a.clean() && clean_bit(a.bit()));
        drop(a);
        let b = track(p, PAGE, true).expect("tracked");
        assert!(!b.clean() && !clean_bit(b.bit()));
        assert!(!clean_bit(None), "no bit: never clean");
        drop(b);
        free(p);
    }

    #[test]
    fn unmapped_pages_are_taken_apart() {
        // SAFETY: a fresh write-watched reservation of two pages, the first
        // committed; freed below.
        let p = unsafe {
            let p = VirtualAlloc(
                std::ptr::null(),
                2 * PAGE,
                MEM_RESERVE | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            ) as usize;
            assert_ne!(p, 0);
            assert_ne!(
                VirtualAlloc(p as *const _, PAGE, MEM_COMMIT, PAGE_READWRITE) as usize,
                0
            );
            p
        };
        let t = track(p, 2 * PAGE, true).expect("tracked");
        assert_eq!(
            t.take_split(p, p + 2 * PAGE),
            (vec![(p, p + PAGE)], vec![(p + PAGE, p + 2 * PAGE)])
        );
        assert!(t.take(p, p + 2 * PAGE).is_empty(), "both taken");
        drop(t);
        free(p);
    }

    #[test]
    fn partial_pages_stay_changed() {
        let p = pool(3);
        let t = track(p + 100, 2 * PAGE, true).expect("tracked");
        assert_eq!(t.take(p + 100, p + PAGE + 10), [(p + 100, p + PAGE + 10)]);
        assert!(!t.changed(p / PAGE) && t.changed(p / PAGE + 1));
        assert!(!t.clean(), "a page taken in part stays changed");
        assert_eq!(t.take(p, p + 3 * PAGE), [(p + PAGE, p + 2 * PAGE + 100)]);
        assert!(t.take(p, p + 3 * PAGE).is_empty());
        assert!(t.clean());
        drop(t);
        free(p);
    }

    #[test]
    fn armed_ranges_keep_their_pages_bytes_from_before_the_first_write() {
        let p = pool(4);
        let t = track(p, 4 * PAGE, true).expect("pool memory is tracked");
        let all = || t.take(p, p + 4 * PAGE);
        // SAFETY: (every write below) a byte of our pool.
        let write = |page: usize, value: u8| unsafe {
            ((p + page * PAGE + 8) as *mut u8).write_volatile(value)
        };
        let saved = |page: usize| t.saved(p / PAGE + page).map(|bytes| bytes[8]);
        all();
        write(0, 1);
        assert_eq!(saved(0), None, "nothing armed yet");
        all();
        t.arm(p + 100, p + 2 * PAGE + 1);
        t.arm(p + PAGE, p + 2 * PAGE);
        t.arm(p + PAGE + 5, p + 2 * PAGE + 1);
        t.unarm(p / PAGE + 1);
        for page in 0..4 {
            write(page, 2);
        }
        assert_eq!(
            [saved(0), saved(1), saved(2), saved(3)],
            [Some(1), None, Some(0), None]
        );
        assert_eq!(saved(0), None, "handed out once");
        all();
        t.arm(p + PAGE, p + 2 * PAGE);
        write(1, 3);
        write(1, 4);
        assert_eq!(saved(1), Some(2), "from before the first write");
        t.disarm();
        all();
        write(2, 5);
        assert_eq!(saved(2), None, "disarmed");
        drop(t);
        free(p);
    }

    #[test]
    fn adjacent_and_overlapping_armed_runs_coalesce() {
        let mut s = Saved::default();
        for k in 0..200 {
            s.arm(2 * k, 2 * k + 2);
        }
        assert_eq!(s.armed, [(0, 400)]);
        let mut s = Saved::default();
        for k in (0..200).rev() {
            s.arm(2 * k, 2 * k + 2);
        }
        assert_eq!(s.armed, [(0, 400)]);
        let mut s = Saved::default();
        s.arm(10, 20);
        s.arm(15, 25);
        assert_eq!(s.armed, [(10, 25)]);
        s.arm(12, 14);
        assert_eq!(s.armed, [(10, 25)], "nested");
        s.arm(40, 50);
        s.arm(0, 5);
        assert_eq!(s.armed, [(0, 5), (10, 25), (40, 50)], "disjoint, sorted");
        s.arm(5, 10);
        assert_eq!(s.armed, [(0, 25), (40, 50)], "touching both sides");
        s.arm(30, 35);
        s.arm(20, 45);
        assert_eq!(s.armed, [(0, 50)], "spanning several");
        s.arm(60, 70);
        s.arm(55, 55);
        assert_eq!(s.armed, [(0, 50), (60, 70)], "an empty run arms nothing");
        assert!(s.armed(0) && s.armed(49) && !s.armed(50) && !s.armed(59));
        assert!(s.armed(60) && s.armed(69) && !s.armed(70) && !s.armed(1000));
        s.unarmed.insert(45);
        s.unarmed.insert(65);
        s.arm(50, 60);
        assert_eq!(s.armed, [(0, 70)]);
        assert!(!s.armed(45) && !s.armed(65) && s.armed(55));
        s.arm(44, 46);
        assert!(s.armed(45) && !s.armed(65));
        s.arm(0, 70);
        assert!(s.armed(65) && s.unarmed.is_empty());
    }

    #[test]
    fn heap_memory_is_not_tracked_and_placeholders_wait_for_mappings() {
        let plain = vec![0u8; 2 * PAGE];
        assert!(track(plain.as_ptr() as usize, plain.len(), true).is_none());
        // SAFETY: a reservation of ours, freed below.
        let r = unsafe { VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) }
            as usize;
        assert_ne!(r, 0);
        let t = track(r, 2 * PAGE, true).expect("reserved memory is tracked");
        assert_eq!(t.take(r, r + 2 * PAGE).len(), 1, "copied once (as zeros)");
        assert!(t.take(r, r + 2 * PAGE).is_empty());
        title_memory_remapped(r, PAGE);
        assert_eq!(t.take(r, r + 2 * PAGE), [(r, r + PAGE)]);
        drop(t);
        // SAFETY: our reservation.
        unsafe { VirtualFree(r as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * PAGE) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * PAGE).Value as usize
        };
        assert_ne!(view, 0);
        assert!(
            track(view, 2 * PAGE, false).is_none(),
            "views only where asked"
        );
        let t = track(view, 2 * PAGE, true).expect("views are tracked");
        assert_eq!(t.take(view, view + 2 * PAGE), [(view, view + 2 * PAGE)]);
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        let write = writing(view + 100, 13);
        assert_eq!(t.take(view, view + 2 * PAGE), [(view, view + PAGE)]);
        assert!(read(&mut file));
        drop(write);
        // SAFETY: the bytes just read.
        let read_bytes = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(read_bytes, b"from the disk");
        assert_eq!(t.take(view, view + 2 * PAGE), [(view, view + PAGE)]);
        drop(t);
        // As the runtime does before it unmaps: the guards let go of the
        // view's pages. Dropping the tracker opens a page only when nothing
        // else holds it; otherwise the pages stayed in the guard set after
        // the unmap, and memory a test running at the same time was given
        // there counted as guarded already, so its writes went unseen (#31).
        title_memory_remapped(view, 2 * PAGE);
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
    }

    pub(crate) fn twin_views(pages: usize) -> (usize, usize) {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        let len = pages * PAGE;
        // SAFETY: a pagefile section mapped read-write twice.
        let (a, b) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                len as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (
                MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize,
                MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize,
            )
        };
        assert!(a != 0 && b != 0);
        for k in (0..len).step_by(PAGE) {
            title_memory_aliased(b + k, &[a + k]);
        }
        (a, b)
    }

    thread_local! {
        /// `VirtualProtect` calls this thread made in `protect`.
        pub(super) static PROTECTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        /// `VirtualQuery` calls this thread made in `RuntimeRegions::classify`.
        pub(super) static QUERIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        pub(super) static PAGE_PARTS: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
        pub(super) static HOT_PAGES: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
        pub(super) static GUARD_REST: std::cell::Cell<Option<Duration>> = const { std::cell::Cell::new(None) };
        pub(super) static OPEN_ONE_VIEW: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
        pub(super) static BACK_OWN_VIEW: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
        pub(super) static CHECK_PAGE_GUARD: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    }

    fn with_check_page_guard<R>(on: bool, f: impl FnOnce() -> R) -> R {
        struct Reset(Option<bool>);
        impl Drop for Reset {
            fn drop(&mut self) {
                CHECK_PAGE_GUARD.with(|c| c.set(self.0));
            }
        }
        let _reset = Reset(CHECK_PAGE_GUARD.with(|c| c.replace(Some(on))));
        f()
    }

    pub(crate) fn with_page_parts<R>(on: bool, f: impl FnOnce() -> R) -> R {
        struct Reset(Option<bool>);
        impl Drop for Reset {
            fn drop(&mut self) {
                PAGE_PARTS.with(|c| c.set(self.0));
            }
        }
        let _reset = Reset(PAGE_PARTS.with(|c| c.replace(Some(on))));
        f()
    }

    pub(crate) fn with_hot_pages<R>(on: bool, f: impl FnOnce() -> R) -> R {
        struct Reset(Option<bool>);
        impl Drop for Reset {
            fn drop(&mut self) {
                HOT_PAGES.with(|h| h.set(self.0));
            }
        }
        let _reset = Reset(HOT_PAGES.with(|h| h.replace(Some(on))));
        f()
    }

    pub(crate) fn with_guard_rest<R>(rest: Duration, f: impl FnOnce() -> R) -> R {
        struct Reset(Option<Duration>);
        impl Drop for Reset {
            fn drop(&mut self) {
                GUARD_REST.with(|r| r.set(self.0));
            }
        }
        let _reset = Reset(GUARD_REST.with(|r| r.replace(Some(rest))));
        f()
    }

    fn with_open_one_view<R>(one: bool, f: impl FnOnce() -> R) -> R {
        struct Reset(Option<bool>);
        impl Drop for Reset {
            fn drop(&mut self) {
                OPEN_ONE_VIEW.with(|o| o.set(self.0));
            }
        }
        let _reset = Reset(OPEN_ONE_VIEW.with(|o| o.replace(Some(one))));
        f()
    }

    fn views_side_by_side(single: usize) -> (usize, usize) {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let len = 2 * single * PAGE;
        // SAFETY: a pagefile section and a placeholder of our own.
        unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                len as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let base = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(base, 0);
            let map = |at: usize, offset: usize, n: usize| {
                VirtualFree(
                    at as *mut _,
                    n * PAGE,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                let view = MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    at as *const _,
                    offset as u64,
                    n * PAGE,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                );
                assert_eq!(view.Value as usize, at);
            };
            for k in 0..single {
                map(base + k * PAGE, k * PAGE, 1);
            }
            let long = base + single * PAGE;
            map(long, single * PAGE, single);
            (base, long)
        }
    }

    #[test]
    fn takes_guard_one_view_per_call() {
        let calls = || PROTECTS.with(std::cell::Cell::get);
        let (base, _) = views_side_by_side(4);
        let t = track(base, 8 * PAGE, true).expect("views are tracked");
        let before = calls();
        assert_eq!(t.take(base, base + 8 * PAGE), [(base, base + 8 * PAGE)]);
        assert_eq!(calls() - before, 5);
        assert!((0..8).all(|k| read_only(base + k * PAGE)));
        drop(t);
        let (base, long) = views_side_by_side(4);
        for (at, want) in [(base, 4), (long, 1)] {
            let t = track(at, 4 * PAGE, true).expect("views are tracked");
            let before = calls();
            assert_eq!(t.take(at, at + 4 * PAGE), [(at, at + 4 * PAGE)]);
            assert_eq!(calls() - before, want, "calls for the views at {at:#x}");
            for k in 0..4 {
                assert!(read_only(at + k * PAGE), "page {k} at {at:#x} guarded");
            }
            // SAFETY: a page of the views; the handler opens it.
            unsafe { ((at + 2 * PAGE + 8) as *mut u8).write_volatile(1) };
            assert!(t.changed(at / PAGE + 2));
            assert!(!t.changed(at / PAGE + 1));
        }
    }

    #[test]
    fn writes_through_another_mapping_mark_the_tracker() {
        let (a, b) = twin_views(2);
        let t = track(a, 2 * PAGE, true).expect("views are tracked");
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + 2 * PAGE)]);
        // SAFETY: the second view of the pages, guarded by the take.
        unsafe { ((b + PAGE + 4) as *mut u32).write_volatile(7) };
        assert!(
            read_only(a + PAGE),
            "marked, still guarded: no call to guard it again"
        );
        assert_eq!(t.take(a, a + 2 * PAGE), [(a + PAGE, a + 2 * PAGE)]);
        // SAFETY: the tracked view, which shows the write.
        assert_eq!(unsafe { ((a + PAGE + 4) as *const u32).read_volatile() }, 7);
        drop(writing(b, 8));
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + PAGE)]);
        assert!(t.take(a, a + 2 * PAGE).is_empty());
        let (c, _) = twin_views(1);
        title_memory_aliased(c, &[a]);
        // SAFETY: a view of ours.
        unsafe { (c as *mut u32).write_volatile(1) };
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + PAGE)]);
    }

    #[test]
    fn aliased_pages_are_found_by_range() {
        let (a, b) = twin_views(3);
        assert!(
            aliased(a, 1) && aliased(b + 2 * PAGE + 5, 10),
            "either view, any byte"
        );
        assert!(aliased(a + PAGE - 1, 2), "a range that reaches a page");
        assert!(!aliased(a, 0), "no bytes");
        let plain = pool(1);
        assert!(!aliased(plain, PAGE));
        free(plain);
        title_memory_remapped(b + PAGE, PAGE);
        assert!(!aliased(a + PAGE, PAGE) && !aliased(b + PAGE, PAGE));
        assert!(aliased(a, PAGE) && aliased(b + 2 * PAGE, PAGE));
        assert!(aliased(a, usize::MAX), "no overflow");
    }

    #[test]
    fn a_tracker_dropped_on_one_mapping_leaves_the_other_guarded() {
        let (a, b) = twin_views(1);
        let t = track(a, PAGE, true).expect("views are tracked");
        let u = track(b, PAGE, true).expect("views are tracked");
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert_eq!(u.take(b, b + PAGE), [(b, b + PAGE)]);
        drop(u);
        // SAFETY: the second view, guarded for `t`'s page too.
        unsafe { ((b + 4) as *mut u32).write_volatile(7) };
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
    }

    #[test]
    fn a_retake_marks_the_copies_and_keeps_the_guard() {
        use windows_sys::Win32::System::Memory::{
            MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualQuery,
        };
        let protection = |at: usize| {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
            info.Protect
        };
        let (a, b) = twin_views(2);
        let t = track_copy(a, 2 * PAGE, true).expect("views are tracked");
        let u = track_copy(b, 2 * PAGE, true).expect("views are tracked");
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + 2 * PAGE)]);
        assert_eq!(u.take(b, b + 2 * PAGE), [(b, b + 2 * PAGE)]);
        assert!(t.clean() && u.clean());
        retake(a + PAGE + 16, 32);
        assert!(!t.clean() && !u.clean());
        assert!(t.changed((a + PAGE) / PAGE) && u.changed((b + PAGE) / PAGE));
        assert!(!t.changed(a / PAGE));
        assert_eq!(protection(a + PAGE), PAGE_READONLY);
        assert_eq!(protection(b + PAGE), PAGE_READONLY);
        assert_eq!(t.take(a, a + 2 * PAGE), [(a + PAGE, a + 2 * PAGE)]);
        assert_eq!(u.take(b, b + 2 * PAGE), [(b + PAGE, b + 2 * PAGE)]);
        assert!(t.clean() && u.clean());
        assert_eq!(protection(a + PAGE), PAGE_READONLY);
        // SAFETY: the second view, guarded all along.
        unsafe { ((b + PAGE + 8) as *mut u32).write_volatile(3) };
        assert_eq!(t.take(a, a + 2 * PAGE), [(a + PAGE, a + 2 * PAGE)]);
        assert_eq!(u.take(b, b + 2 * PAGE), [(b + PAGE, b + 2 * PAGE)]);
        retake(a, 2 * PAGE);
        drop(u);
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + 2 * PAGE)]);
        assert!(t.clean());
    }

    #[test]
    fn a_retake_marks_buffer_copies_alone() {
        let p = pool(2);
        let copy = track_copy(p, 2 * PAGE, true).expect("pool memory is tracked");
        let texture = track(p, 2 * PAGE, true).expect("pool memory is tracked");
        assert_eq!(copy.take(p, p + 2 * PAGE), [(p, p + 2 * PAGE)]);
        assert_eq!(texture.take(p, p + 2 * PAGE), [(p, p + 2 * PAGE)]);
        retake(p + PAGE, 16);
        assert!(copy.changed(p / PAGE + 1), "the copy takes it again");
        assert!(
            !texture.changed(p / PAGE + 1) && texture.clean(),
            "the texture's tracker is left alone"
        );
        assert_eq!(copy.take(p, p + 2 * PAGE), [(p + PAGE, p + 2 * PAGE)]);
        // SAFETY: a page of our pool, guarded: the handler opens it.
        unsafe { ((p + PAGE + 8) as *mut u8).write_volatile(1) };
        assert!(
            copy.changed(p / PAGE + 1) && texture.changed(p / PAGE + 1),
            "a title write marks both"
        );
        drop((copy, texture));
        free(p);
    }

    #[test]
    fn a_busy_alias_leaves_the_page_changed_and_keeps_saved_bytes() {
        let (a, b) = twin_views(1);
        let t = track(a, PAGE, true).expect("views are tracked");
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        t.arm(a, a + PAGE);
        let write = writing(b, 16);
        assert!(t.changed(a / PAGE));
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert!(t.changed(a / PAGE), "take must leave the page changed");
        t.taken(&[a / PAGE]);
        assert!(t.changed(a / PAGE), "taken must leave the page changed");
        // SAFETY: the second view of the page, kept open by `write`.
        unsafe { (b as *mut u32).write_volatile(0x1234) };
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        drop(write);
        let saved = t.saved(a / PAGE).expect("saved bytes survive");
        assert_eq!(&saved[..4], &[0, 0, 0, 0]);
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert!(t.take(a, a + PAGE).is_empty(), "taken once the write ended");
        let (c, _) = twin_views(1);
        let busy_c = writing(c, 16);
        title_memory_aliased(c, &[a]);
        assert!(
            t.changed(a / PAGE),
            "aliasing a busy page marks the tracked one"
        );
        drop(busy_c);
    }

    /// Memory given up while a tracker holds it guarded (a pool page
    /// decommitted here; the title frees and allocates its pools again,
    /// which no one tells the guards of) is guarded when it comes back: the
    /// guard goes with it, and a take while it goes leaves it unguarded. A
    /// guard set that still named the page would skip it as guarded already,
    /// and the new memory's writes would go unseen.
    #[test]
    fn memory_given_up_under_a_guard_is_guarded_when_it_comes_back() {
        use windows_sys::Win32::System::Memory::MEM_DECOMMIT;
        let p = pool(2);
        let page = p + PAGE;
        let t = track(p, 2 * PAGE, true).expect("pool memory is tracked");
        assert_eq!(t.take(p, p + 2 * PAGE), [(p, p + 2 * PAGE)]);
        assert!(read_only(page));
        let going = title_memory_unmapping(page, PAGE);
        assert!(writable(page), "given up: no guard while it goes");
        assert_eq!(
            t.take(p, p + 2 * PAGE),
            [(page, p + 2 * PAGE)],
            "changed, and taken"
        );
        assert!(writable(page), "a take while it goes guards nothing");
        // SAFETY: the second page of our pool; the reservation stays.
        let freed = unsafe { VirtualFree(page as *mut _, PAGE, MEM_DECOMMIT) };
        assert_ne!(freed, 0);
        drop(going);
        // SAFETY: the same page of our reservation, committed again.
        let back =
            unsafe { VirtualAlloc(page as *const _, PAGE, MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_eq!(back, page);
        assert_eq!(t.take(p, p + 2 * PAGE), [(page, p + 2 * PAGE)]);
        assert!(read_only(page), "the memory that came back is guarded");
        // SAFETY: a byte of the page, guarded: the handler opens it.
        unsafe { ((page + 8) as *mut u8).write_volatile(1) };
        assert!(t.changed(page / PAGE), "and its writes are seen");
        drop(t);
        free(p);
    }

    /// With length 0, the whole allocation at the address goes (as
    /// `VirtualFree` with `MEM_RELEASE` frees it).
    #[test]
    fn unmapping_with_no_length_takes_the_whole_allocation() {
        let p = pool(3);
        let t = track(p, 3 * PAGE, true).expect("pool memory is tracked");
        assert_eq!(t.take(p, p + 3 * PAGE), [(p, p + 3 * PAGE)]);
        let going = title_memory_unmapping(p, 0);
        assert!((0..3).all(|k| writable(p + k * PAGE)));
        assert_eq!(t.take(p, p + 3 * PAGE), [(p, p + 3 * PAGE)]);
        drop(going);
        drop(t);
        free(p);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        drop(writing(top, PAGE));
        title_memory_remapped(top, usize::MAX);
    }

    #[test]
    fn hot_pages_follows_the_variable() {
        assert!(!decide_hot_pages(None, true), "off by default in tests");
        assert!(decide_hot_pages(None, false), "on by default outside tests");
        assert!(decide_hot_pages(Some("1"), true));
        assert!(decide_hot_pages(Some("1"), false));
        assert!(!decide_hot_pages(Some("0"), true));
        assert!(!decide_hot_pages(Some("0"), false));
        assert!(!decide_hot_pages(Some("yes"), true));
        assert!(decide_hot_pages(Some("yes"), false));
        with_hot_pages(true, || assert!(hot_pages_on()));
        with_hot_pages(false, || assert!(!hot_pages_on()));
    }

    #[test]
    fn hot_pages_skip_reguard_and_cool_down_after_idle_frames() {
        let p = pool(2);
        with_hot_pages(true, || {
            let a = track_copy(p, 2 * PAGE, true).expect("pool memory is tracked");
            let b = track(p, PAGE, true).expect("pool memory is tracked");
            assert_eq!(a.take(p, p + 2 * PAGE), [(p, p + 2 * PAGE)]);
            assert_eq!(b.take(p, p + PAGE), [(p, p + PAGE)]);
            assert!(read_only(p) && read_only(p + PAGE));
            assert!(a.clean() && b.clean());

            // First fault in a frame leaves the page guarded after the next take.
            // SAFETY: committed pool page; the handler opens it.
            unsafe { (p as *mut u8).write_volatile(1) };
            assert_eq!(a.take(p, p + 2 * PAGE), [(p, p + PAGE)]);
            assert_eq!(b.take(p, p + PAGE), [(p, p + PAGE)]);

            // Second fault in the same frame promotes the page to hot.
            // SAFETY: as above.
            unsafe { (p as *mut u8).write_volatile(2) };
            let before = PROTECTS.with(std::cell::Cell::get);
            assert_eq!(a.take(p, p + 2 * PAGE), [(p, p + PAGE)]);
            assert_eq!(b.take(p, p + PAGE), [(p, p + PAGE)]);
            assert!(writable(p), "stays read-write while hot");
            assert!(!a.clean() && !b.clean() && !clean_bit(a.bit()) && !clean_bit(b.bit()));

            for step in 3..23u8 {
                // SAFETY: writable hot page, no fault needed.
                unsafe { (p as *mut u8).write_volatile(step) };
                assert!(!a.clean() && !b.clean());
                assert_eq!(a.take(p, p + 2 * PAGE), [(p, p + PAGE)]);
                assert_eq!(b.take(p, p + PAGE), [(p, p + PAGE)]);
                assert!(writable(p));
            }
            assert_eq!(
                PROTECTS.with(std::cell::Cell::get) - before,
                0,
                "zero VirtualProtect calls across repeated hot takes"
            );

            // Consecutive-frame fault promotes on second frame's first fault.
            // SAFETY: second pool page, guarded.
            unsafe { ((p + PAGE) as *mut u8).write_volatile(10) };
            assert_eq!(a.take(p, p + 2 * PAGE), [(p, p + 2 * PAGE)]);
            assert!(read_only(p + PAGE));
            frame_done();
            // SAFETY: second pool page, guarded; `last_fault_frame` promotes it.
            unsafe { ((p + PAGE) as *mut u8).write_volatile(11) };
            assert_eq!(a.take(p, p + 2 * PAGE), [(p, p + 2 * PAGE)]);
            assert!(writable(p + PAGE), "promoted across frames");

            for _ in 0..=HOT_FRAMES_MAX {
                frame_done();
            }
            assert_eq!(a.take(p, p + 2 * PAGE), [(p, p + 2 * PAGE)]);
            assert_eq!(b.take(p, p + PAGE), [(p, p + PAGE)]);
            assert!(read_only(p) && read_only(p + PAGE));
            assert!(a.clean() && b.clean() && clean_bit(a.bit()) && clean_bit(b.bit()));
            assert!(a.take(p, p + 2 * PAGE).is_empty() && b.take(p, p + PAGE).is_empty());
            drop((a, b));
        });
        free(p);
    }

    #[test]
    fn arming_a_hot_page_reguards_immediately_and_saves_bytes() {
        let p = pool(1);
        with_hot_pages(true, || {
            let t = track_copy(p, PAGE, true).expect("tracked");
            assert_eq!(t.take(p, p + PAGE), [(p, p + PAGE)]);
            for v in [1u8, 2] {
                // SAFETY: pool page of ours.
                unsafe { (p as *mut u8).write_volatile(v) };
                assert_eq!(t.take(p, p + PAGE), [(p, p + PAGE)]);
            }
            assert!(writable(p) && t.changed(p / PAGE) && !t.clean());
            // SAFETY: silent write while hot.
            unsafe { ((p + 24) as *mut u8).write_volatile(0x5a) };

            assert!(t.arm(p, p + PAGE), "arm reports cooled hot page");
            assert!(read_only(p), "arm immediately guards a hot page");
            assert!(
                t.changed(p / PAGE),
                "rearmed page stays changed until the shadow pulls pre-arm bytes"
            );
            assert_eq!(t.take(p, p + PAGE), [(p, p + PAGE)]);
            assert!(!t.changed(p / PAGE) && read_only(p));
            // SAFETY: guarded again; handler snapshots 0x5a before 0xa5 lands.
            unsafe { ((p + 24) as *mut u8).write_volatile(0xa5) };
            // SAFETY: pool byte just written.
            assert_eq!(unsafe { ((p + 24) as *const u8).read_volatile() }, 0xa5);
            assert!(t.changed(p / PAGE));
            assert_eq!(t.saved(p / PAGE).expect("pre-write bytes saved")[24], 0x5a);
            t.disarm();
            drop(t);

            let (va, vb) = twin_views(1);
            let t = track_copy(vb, PAGE, true).expect("views are tracked");
            assert_eq!(t.take(vb, vb + PAGE), [(vb, vb + PAGE)]);
            for v in [1u8, 2] {
                // SAFETY: twin view of ours.
                unsafe { ((va + 12) as *mut u8).write_volatile(v) };
                assert_eq!(t.take(vb, vb + PAGE), [(vb, vb + PAGE)]);
            }
            assert!(writable(va));
            // SAFETY: silent write through alias while hot.
            unsafe { ((va + 12) as *mut u8).write_volatile(0x33) };

            assert!(t.arm(vb, vb + PAGE), "arm reports cooled aliased hot page");
            assert!(read_only(va) && read_only(vb), "alias guarded on arm too");
            assert_eq!(t.take(vb, vb + PAGE), [(vb, vb + PAGE)]);
            // SAFETY: guarded alias; faults and saves `vb` before 0x77 lands.
            unsafe { ((va + 12) as *mut u8).write_volatile(0x77) };
            assert_eq!(t.saved(vb / PAGE).expect("saved via alias")[12], 0x33);
            // SAFETY: shared section reflects the write.
            assert_eq!(unsafe { ((vb + 12) as *const u8).read_volatile() }, 0x77);
            t.disarm();
        });
        free(p);
    }

    #[test]
    fn remapping_clears_hot_page_state() {
        let p = pool(1);
        with_hot_pages(true, || {
            let t = track_copy(p, PAGE, true).expect("tracked");
            assert_eq!(t.take(p, p + PAGE), [(p, p + PAGE)]);
            for v in [1u8, 2] {
                // SAFETY: pool page of ours.
                unsafe { (p as *mut u8).write_volatile(v) };
                assert_eq!(t.take(p, p + PAGE), [(p, p + PAGE)]);
            }
            assert!(writable(p) && !t.clean());

            title_memory_remapped(p, PAGE);
            assert!(t.changed(p / PAGE));
            assert!(!guards().hot.contains_key(&(p / PAGE)));
            assert_eq!(t.take(p, p + PAGE), [(p, p + PAGE)]);
            assert!(read_only(p) && t.clean());

            // SAFETY: guarded pool page.
            unsafe { (p as *mut u8).write_volatile(3) };
            assert_eq!(t.take(p, p + PAGE), [(p, p + PAGE)]);
            assert!(read_only(p) && t.clean());
            drop(t);
        });
        free(p);
    }

    #[test]
    fn gpu_copy_back_cools_hot_page_and_preserves_title_writes() {
        let p = pool(1);
        with_hot_pages(true, || {
            let t = track_copy(p, PAGE, true).expect("tracked");
            assert_eq!(t.take(p, p + PAGE), [(p, p + PAGE)]);
            for v in [1u8, 2] {
                // SAFETY: pool page of ours.
                unsafe { (p as *mut u8).write_volatile(v) };
                assert_eq!(t.take(p, p + PAGE), [(p, p + PAGE)]);
            }
            assert!(writable(p) && t.changed(p / PAGE) && !t.clean());

            let page = p / PAGE;
            t.unarm(page);
            assert!(
                t.changed(page),
                "an unguarded hot page may hold silent CPU writes, so copy_back_marked must keep title bytes"
            );
            t.taken(&[page]);
            assert!(read_only(p) && t.clean());

            for v in [3u8, 4, 5] {
                // SAFETY: guarded pool page.
                unsafe { (p as *mut u8).write_volatile(v) };
                assert_eq!(t.take(p, p + PAGE), [(p, p + PAGE)]);
                assert!(read_only(p) && t.clean());
            }
            drop(t);
        });
        free(p);
    }

    #[test]
    fn guard_rest_keeps_recently_faulted_page_open_and_reguards_after_expiry() {
        let p = pool(1);
        with_hot_pages(false, || {
            let t = track_copy(p, PAGE, true).expect("tracked");
            assert_eq!(t.take(p, p + PAGE), [(p, p + PAGE)]);
            assert!(read_only(p) && t.clean());

            with_guard_rest(Duration::from_millis(80), || {
                // First fault opens `p` and starts the rest cooldown.
                // SAFETY: pool page of ours.
                unsafe { (p as *mut u8).write_volatile(0x11) };
                let before = PROTECTS.with(std::cell::Cell::get);
                assert_eq!(t.take(p, p + PAGE), [(p, p + PAGE)]);
                assert!(writable(p), "page rests open across immediate takes");
                assert!(
                    t.changed(p / PAGE),
                    "writable resting page counts as changed so copy-back never clobbers silent CPU writes"
                );

                // Silent writes during the rest window keep being returned by `take`
                // with zero `VirtualProtect` calls.
                // SAFETY: writable resting page.
                unsafe { (p as *mut u8).write_volatile(0x22) };
                assert_eq!(t.take(p, p + PAGE), [(p, p + PAGE)]);
                assert_eq!(PROTECTS.with(std::cell::Cell::get) - before, 0);

                std::thread::sleep(Duration::from_millis(95));
                // First take after expiry copies final bytes and re-guards `p`.
                assert_eq!(t.take(p, p + PAGE), [(p, p + PAGE)]);
                assert!(read_only(p) && t.clean());
                assert!(t.take(p, p + PAGE).is_empty());
            });
            drop(t);
        });
        free(p);
    }

    #[test]
    fn open_one_view_switch_controls_alias_opening() {
        let (a, b) = twin_views(1);
        let ta = track(a, PAGE, true).unwrap();
        let tb = track(b, PAGE, true).unwrap();

        with_open_one_view(true, || {
            assert_eq!(ta.take(a, a + PAGE), [(a, a + PAGE)]);
            assert_eq!(tb.take(b, b + PAGE), [(b, b + PAGE)]);
            assert!(read_only(a) && read_only(b));
            // SAFETY: write through `b` opens `b` alone, leaving `a` read-only
            // while marking `ta` changed.
            unsafe { (b as *mut u32).write_volatile(0x55) };
            assert!(writable(b) && read_only(a));
            assert!(ta.changed(a / PAGE) && tb.changed(b / PAGE));
        });

        with_open_one_view(false, || {
            assert_eq!(ta.take(a, a + PAGE), [(a, a + PAGE)]);
            assert_eq!(tb.take(b, b + PAGE), [(b, b + PAGE)]);
            assert!(read_only(a) && read_only(b));
            // SAFETY: with `VICTORY_OPEN_ONE_VIEW=0`, writing `b` opens `a` too.
            unsafe { (b as *mut u32).write_volatile(0x66) };
            assert!(writable(b) && writable(a));
            assert!(ta.changed(a / PAGE) && tb.changed(b / PAGE));
        });
    }

    /// Regression test for the animated-texture / aliased-view storm
    /// (`#50` / `watch.rs` 3.36 s/frame stall): a texture tracker `tp` watches
    /// tiled-heap pages `P_0..P_N` while the CPU writes every frame (or every
    /// other frame) through aliased single-page section views `A_0..A_N`
    /// (`title_memory_aliased`). After warmup, `A_0..A_N` stay `PAGE_READWRITE`
    /// and `tp.take_split` performs **0 `VirtualProtect` calls** and **0
    /// `GUARDS` lock acquisitions** while reporting `P`'s full range as changed
    /// until `A` cools down.
    #[test]
    fn repeated_writes_through_aliased_views_promote_hot_with_zero_protects() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        const N: usize = 8;
        // Map `N` single-page views `A_0..A_N` and one contiguous `N`-page view
        // `P` over the SAME physical section offsets `0..N * PAGE`.
        // SAFETY: fresh section and placeholder of our own.
        let (a_views, p_contig) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (N * PAGE) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let base = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                2 * N * PAGE,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(base, 0);
            let map = |at: usize, offset: usize, pages: usize| {
                VirtualFree(
                    at as *mut _,
                    pages * PAGE,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                let view = MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    at as *const _,
                    offset as u64,
                    pages * PAGE,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                );
                assert_eq!(view.Value as usize, at);
            };
            for k in 0..N {
                map(base + k * PAGE, k * PAGE, 1);
            }
            let p_contig = base + N * PAGE;
            map(p_contig, 0, N);
            (base, p_contig)
        };
        for k in 0..N {
            title_memory_aliased(p_contig + k * PAGE, &[a_views + k * PAGE]);
        }
        with_hot_pages(true, || {
            with_guard_rest(Duration::ZERO, || {
                let tp = track(p_contig, N * PAGE, true).expect("views tracked");
                assert_eq!(
                    tp.take_split(p_contig, p_contig + N * PAGE),
                    (vec![(p_contig, p_contig + N * PAGE)], Vec::new())
                );
                assert!(
                    (0..N).all(|k| read_only(a_views + k * PAGE) && read_only(p_contig + k * PAGE))
                );
                assert!(tp.clean());

                // Frame 1: CPU writes through all N aliased single-page views `A`.
                for k in 0..N {
                    // SAFETY: committed section view `A_k`; handler opens it.
                    unsafe {
                        ((a_views + k * PAGE + 16) as *mut u32).write_volatile(100 + k as u32)
                    };
                }
                assert!(!tp.clean());
                assert_eq!(
                    tp.take_split(p_contig, p_contig + N * PAGE),
                    (vec![(p_contig, p_contig + N * PAGE)], Vec::new())
                );
                // Even if an animated texture skips a frame (30 fps on 60 fps),
                // the second write within `HOT_WINDOW_FRAMES` promotes `[A_k, P_k]`.
                frame_done();
                frame_done();
                for k in 0..N {
                    // SAFETY: committed section view `A_k`.
                    unsafe {
                        ((a_views + k * PAGE + 16) as *mut u32).write_volatile(200 + k as u32)
                    };
                }

                let before_protects = PROTECTS.with(std::cell::Cell::get);
                assert_eq!(
                    tp.take_split(p_contig, p_contig + N * PAGE),
                    (vec![(p_contig, p_contig + N * PAGE)], Vec::new())
                );
                assert_eq!(
                    PROTECTS.with(std::cell::Cell::get) - before_protects,
                    0,
                    "promotion take must not re-protect aliased views A"
                );
                assert!((0..N).all(|k| writable(a_views + k * PAGE)));

                // Steady-state frames: zero `VirtualProtect` calls and zero `GUARDS` holds
                // inside `take_split`!
                let mut take_hold_ns = 0u64;
                for step in 0..6u32 {
                    for k in 0..N {
                        // SAFETY: `A_k` stays writable while hot.
                        unsafe {
                            ((a_views + k * PAGE + 16) as *mut u32)
                                .write_volatile(1000 + step * 32 + k as u32)
                        };
                    }
                    assert!(!tp.clean());
                    THREAD_HOLD_NS.with(|c| c.set(0));
                    assert_eq!(
                        tp.take_split(p_contig, p_contig + N * PAGE),
                        (vec![(p_contig, p_contig + N * PAGE)], Vec::new())
                    );
                    take_hold_ns += THREAD_HOLD_NS.with(std::cell::Cell::get);
                    for k in 0..N {
                        // SAFETY: shared section backing `P_k` reflects `A_k`'s write.
                        let seen =
                            unsafe { ((p_contig + k * PAGE + 16) as *const u32).read_volatile() };
                        assert_eq!(seen, 1000 + step * 32 + k as u32);
                    }
                    frame_done();
                }
                assert_eq!(
                    PROTECTS.with(std::cell::Cell::get) - before_protects,
                    0,
                    "zero VirtualProtect calls across steady-state aliased frames"
                );
                assert_eq!(
                    take_hold_ns, 0,
                    "fast path takes zero GUARDS lock time while hot"
                );

                // After idle cooldown, `A` and `P` are guarded again and `tp` goes clean.
                for _ in 0..=HOT_FRAMES_MAX {
                    frame_done();
                }
                assert_eq!(
                    tp.take_split(p_contig, p_contig + N * PAGE),
                    (vec![(p_contig, p_contig + N * PAGE)], Vec::new())
                );
                assert!(
                    (0..N).all(|k| read_only(a_views + k * PAGE) && read_only(p_contig + k * PAGE))
                );
                assert!(tp.clean());
                assert_eq!(
                    tp.take_split(p_contig, p_contig + N * PAGE),
                    (Vec::new(), Vec::new())
                );
            });
        });
    }

    #[test]
    fn own_view_keeps_guards_and_saves_both_title_aliases() {
        let (a, b) = twin_views(1);
        let first = track(a, PAGE, true).unwrap();
        let other = track(b, PAGE, true).unwrap();
        for (tracker, base) in [(&first, a), (&other, b)] {
            tracker.take(base, base + PAGE);
            tracker.arm(base, base + PAGE);
        }
        let mut dest = [0u8; 1];
        let mut slices = [&mut dest[..]];
        write_through_view(a + 17, &mut slices, &mut |out| out[0][0] = 5);
        assert_eq!(dest, [5]);
        for (tracker, base) in [(&first, a), (&other, b)] {
            assert!(read_only(base), "title page guard stays read-only");
            assert!(tracker.changed(base / PAGE));
            assert_eq!(tracker.saved(base / PAGE).unwrap()[17], 0);
        }
    }

    #[test]
    fn subpage_journal_tracks_parts_since_each_take() {
        assert!(!decide_page_parts(None, true, false));
        assert!(!decide_page_parts(None, false, true));
        assert!(decide_page_parts(None, false, false));
        assert!(decide_page_parts(Some("1"), true, true));
        assert!(decide_page_parts(Some("1"), false, true));
        assert!(!decide_page_parts(Some("0"), false, false));

        let mut s = Subpages::default();
        assert_eq!(s.parts_since(7, 0), WHOLE);
        s.note(7, WHOLE, 3);
        s.note(7, 0b10_0001, 4);
        s.note(7, 0b10_0000, 6);
        assert_eq!(s.parts_since(7, 2), WHOLE);
        assert_eq!(s.parts_since(7, 3), 0b10_0001);
        assert_eq!(s.parts_since(7, 4), 0b10_0000);
        assert_eq!(s.parts_since(7, 5), 0b10_0000);
        assert_eq!(s.parts_since(7, 6), 0);
        s.note(7, WHOLE, 8);
        assert_eq!(s.parts_since(7, 7), WHOLE);
    }

    #[test]
    fn guarded_and_hot_pools_take_only_written_4k_subpages() {
        let p = pool(3);
        with_page_parts(true, || {
            let a = track_copy(p, 3 * PAGE, true).expect("pool tracked");
            let b = track(p + PAGE, PAGE, true).expect("second tracker on page 1");

            // First take always copies whole 64 KiB pages.
            assert_eq!(a.take(p, p + 3 * PAGE), [(p, p + 3 * PAGE)]);
            assert_eq!(b.take(p + PAGE, p + 2 * PAGE), [(p + PAGE, p + 2 * PAGE)]);
            assert!(a.clean() && b.clean());

            // Fault on subpage 1 (+4096) of guarded page 1, opening page 1, then
            // write subpage 15 (+61440) in the same open 64 KiB page!
            // SAFETY: committed pages inside `p..p + 3 * PAGE`.
            unsafe {
                ((p + PAGE + KERNEL_PAGE + 12) as *mut u32).write_volatile(0x1111);
                ((p + PAGE + 15 * KERNEL_PAGE + 20) as *mut u32).write_volatile(0x2222);
            }
            let expected = [
                (p + PAGE + KERNEL_PAGE, p + PAGE + 2 * KERNEL_PAGE),
                (p + PAGE + 15 * KERNEL_PAGE, p + 2 * PAGE),
            ];
            assert_eq!(a.take(p, p + 3 * PAGE), expected);
            // `b` shares page 1: `SUBPAGES` preserved both subpages across `a`'s reset!
            assert_eq!(b.take(p + PAGE, p + 2 * PAGE), expected);
            assert!(a.clean() && b.clean());

            // Promote page 0 to hot (`VICTORY_HOT_PAGES=1`) and verify subpage
            // precision + 0-byte idle takes on hot write-watched pages!
            with_hot_pages(true, || {
                for v in [1u32, 2] {
                    // SAFETY: inside `p`.
                    unsafe { (p as *mut u32).write_volatile(v) };
                    assert_eq!(a.take(p, p + PAGE), [(p, p + KERNEL_PAGE)]);
                }
                assert!(writable(p), "page 0 is now hot");
                let before = PROTECTS.with(std::cell::Cell::get);
                // Idle hot take: 0 subpages written -> 0 bytes to copy, 0 VirtualProtects!
                assert!(a.take(p, p + PAGE).is_empty());
                // Write subpage 7 of hot page 0 -> only 4 KiB returned, 0 VirtualProtects!
                // SAFETY: inside `p`.
                unsafe { ((p + 7 * KERNEL_PAGE + 4) as *mut u32).write_volatile(0x7777) };
                assert_eq!(
                    a.take(p, p + PAGE),
                    [(p + 7 * KERNEL_PAGE, p + 8 * KERNEL_PAGE)]
                );
                assert_eq!(PROTECTS.with(std::cell::Cell::get) - before, 0);
            });
            drop((a, b));
        });
        free(p);
    }

    /// Multi-threaded contention test & benchmark: 6 CPU writer threads fault
    /// and write across 24 tracked pages while a submission refresh thread
    /// repeatedly takes dirty ranges and copies them into a shadow mirror.
    /// Verifies zero lost writes in both modes and measures `GUARDS` lock hold
    /// + wait reduction and `VirtualProtect` savings.
    #[test]
    fn multithreaded_fault_and_take_contention_preserves_writes_and_cuts_lock_wait() {
        const THREADS: usize = 6;
        const PAGES_PER_THREAD: usize = 4;
        const TOTAL_PAGES: usize = THREADS * PAGES_PER_THREAD;
        const ROUNDS: u32 = 80;

        let measure = |optimized: bool| -> (u64, u64, usize, Duration) {
            let p = pool(TOTAL_PAGES);
            let len = TOTAL_PAGES * PAGE;
            let mut mirror = vec![0u8; len];
            let tracker = track_copy(p, len, true).expect("pool tracked");
            let rest = if optimized {
                Duration::from_millis(3)
            } else {
                Duration::ZERO
            };

            with_hot_pages(optimized, || {
                with_guard_rest(rest, || {
                    for (a, b) in tracker.take(p, p + len) {
                        // SAFETY: inside `p`.
                        let src = unsafe { std::slice::from_raw_parts(a as *const u8, b - a) };
                        mirror[a - p..b - p].copy_from_slice(src);
                    }
                });
            });

            TEST_HOLD_NS.store(0, Ordering::Relaxed);
            TEST_WAIT_NS.store(0, Ordering::Relaxed);
            PROTECTS.with(|c| c.set(0));
            let writers_left = AtomicUsize::new(THREADS);
            let wall_start = Instant::now();

            std::thread::scope(|s| {
                for tid in 0..THREADS {
                    let left = &writers_left;
                    s.spawn(move || {
                        with_hot_pages(optimized, || {
                            with_guard_rest(rest, || {
                                for r in 1..=ROUNDS {
                                    for k in 0..PAGES_PER_THREAD {
                                        let page_idx = tid * PAGES_PER_THREAD + k;
                                        let slot = p + page_idx * PAGE + (r as usize % 64) * 8;
                                        let val = ((tid as u64 + 1) << 32) | u64::from(r);
                                        // SAFETY: disjoint 64 KiB pages per thread.
                                        unsafe {
                                            (slot as *mut u64).write_volatile(val);
                                            ((p + page_idx * PAGE) as *mut u64).write_volatile(val);
                                        }
                                    }
                                    std::thread::yield_now();
                                }
                            });
                        });
                        left.fetch_sub(1, Ordering::Release);
                    });
                }

                with_hot_pages(optimized, || {
                    with_guard_rest(rest, || {
                        while writers_left.load(Ordering::Acquire) != 0 {
                            for (a, b) in tracker.take(p, p + len) {
                                // SAFETY: `p..p + len` is live and disjoint per page.
                                let src =
                                    unsafe { std::slice::from_raw_parts(a as *const u8, b - a) };
                                mirror[a - p..b - p].copy_from_slice(src);
                            }
                            std::thread::yield_now();
                        }
                    });
                });
            });

            let wall = wall_start.elapsed();
            let hold_ns = TEST_HOLD_NS.load(Ordering::Relaxed);
            let wait_ns = TEST_WAIT_NS.load(Ordering::Relaxed);
            let protects = PROTECTS.with(std::cell::Cell::get);

            // Final drain take after all writers finished: every single word
            // written across all 24 pages and 80 rounds must match `mirror`!
            with_hot_pages(optimized, || {
                with_guard_rest(rest, || {
                    for (a, b) in tracker.take(p, p + len) {
                        // SAFETY: `p..p + len` is live and quiescent.
                        let src = unsafe { std::slice::from_raw_parts(a as *const u8, b - a) };
                        mirror[a - p..b - p].copy_from_slice(src);
                    }
                });
            });
            // SAFETY: quiescent `p..p + len`.
            let truth = unsafe { std::slice::from_raw_parts(p as *const u8, len) };
            assert_eq!(
                mirror.as_slice(),
                truth,
                "zero lost writes under contention (optimized={optimized})"
            );

            drop(tracker);
            free(p);
            (hold_ns, wait_ns, protects, wall)
        };

        let (cold_hold, cold_wait, cold_protects, cold_wall) = measure(false);
        let (opt_hold, opt_wait, opt_protects, opt_wall) = measure(true);
        println!(
            "watch contention ({THREADS} writers x {PAGES_PER_THREAD} pages x {ROUNDS} rounds): \
             cold [protects={cold_protects}, hold={:.2}ms, wait={:.2}ms, wall={cold_wall:?}] -> \
             hot+rest [protects={opt_protects}, hold={:.2}ms, wait={:.2}ms, wall={opt_wall:?}]",
            cold_hold as f64 / 1e6,
            cold_wait as f64 / 1e6,
            opt_hold as f64 / 1e6,
            opt_wait as f64 / 1e6,
        );
        assert!(
            opt_protects < cold_protects || cold_protects <= TOTAL_PAGES,
            "hot+rest should not increase VirtualProtect calls ({opt_protects} vs {cold_protects})"
        );
    }

    fn protection_at(at: usize) -> u32 {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        assert_ne!(n, 0);
        info.Protect
    }

    fn title_sets_guard(at: usize) {
        use windows_sys::Win32::System::Memory::{PAGE_GUARD, VirtualProtect};
        let mut old = 0;
        // SAFETY: a 4 KiB subpage inside one of our test views.
        let ok = unsafe {
            VirtualProtect(
                at as *const _,
                KERNEL_PAGE,
                PAGE_READWRITE | PAGE_GUARD,
                &mut old,
            )
        };
        assert_ne!(ok, 0);
    }

    fn drop_twin_views(a: usize, b: usize, len: usize) {
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for v in [a, b] {
            title_memory_remapped(v, len);
            // SAFETY: view returned by `twin_views`.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: v as *mut std::ffi::c_void,
                });
            }
        }
    }

    #[test]
    fn a_page_the_title_guards_inside_a_range_keeps_its_guard() {
        use windows_sys::Win32::System::Memory::{PAGE_GUARD, PAGE_READONLY};
        const SIZE: usize = 4 * PAGE;
        let (one, two) = twin_views(4);
        let guarded = one + PAGE + KERNEL_PAGE;
        title_sets_guard(guarded);

        let t = track(one, SIZE, true).expect("tracked");
        assert_eq!(t.take(one, one + SIZE), [(one, one + SIZE)]);
        assert_eq!(
            protection_at(guarded),
            PAGE_READWRITE | PAGE_GUARD,
            "the title's guard subpage is preserved"
        );
        assert_eq!(
            protection_at(one + PAGE),
            PAGE_READWRITE,
            "its 64 KiB page is left untouched"
        );
        for block in [0, 2, 3] {
            assert_eq!(
                protection_at(one + block * PAGE),
                PAGE_READONLY,
                "block {block}"
            );
        }
        // Not guarded, so block 1 is reported as changed on every asking.
        assert_eq!(t.take(one, one + SIZE), [(one + PAGE, one + 2 * PAGE)]);
        drop(writing(one, SIZE));
        assert_eq!(
            protection_at(guarded),
            PAGE_READWRITE | PAGE_GUARD,
            "opening surrounding pages leaves the title's guard intact"
        );
        drop(t);
        drop_twin_views(one, two, SIZE);
    }

    #[test]
    fn a_page_the_title_guards_in_another_view_keeps_its_guard() {
        use windows_sys::Win32::System::Memory::{PAGE_GUARD, PAGE_READONLY};
        const SIZE: usize = 4 * PAGE;
        let (primary, other) = twin_views(4);
        let guarded = other + PAGE + KERNEL_PAGE;
        title_sets_guard(guarded);

        let t = track(primary, SIZE, true).expect("tracked");
        assert_eq!(t.take(primary, primary + SIZE), [(primary, primary + SIZE)]);
        assert_eq!(
            protection_at(guarded),
            PAGE_READWRITE | PAGE_GUARD,
            "the title's guard in the aliased view is preserved"
        );
        assert_eq!(
            protection_at(other + PAGE),
            PAGE_READWRITE,
            "the aliased 64 KiB page is left untouched"
        );
        for block in [0, 2, 3] {
            assert_eq!(
                protection_at(other + block * PAGE),
                PAGE_READONLY,
                "block {block}"
            );
        }
        // Because the other view of block 1 could not be made read-only, a write
        // there would not fault, so block 1 stays dirty and is returned again.
        assert_eq!(
            t.take(primary, primary + SIZE),
            [(primary + PAGE, primary + 2 * PAGE)]
        );
        drop(t);
        drop_twin_views(primary, other, SIZE);
    }

    /// Regression test for concurrent DirectStorage workers calling
    /// [`writing`] (`kernel_will_write`) on disjoint slices of the SAME guarded
    /// 64 KiB page: Worker 2 must wait on `OPENING_CV` / `GUARDING_CV` until
    /// Worker 1's `VirtualProtect(..., PAGE_READWRITE)` has actually completed,
    /// so its subsequent kernel `NtReadFile` (`seek_read`) never hits
    /// `ERROR_NOACCESS` (`0x800703e6`).
    #[test]
    fn concurrent_writing_on_same_guarded_page_waits_until_page_is_writable() {
        use std::io::Write;
        use std::os::windows::fs::FileExt;
        use std::sync::Barrier;
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            UnmapViewOfFile,
        };

        const WORKERS: usize = 8;
        const CHUNK: usize = 624;
        const PAGES_N: usize = 4;
        const ROUNDS: usize = 200;

        // SAFETY: fresh 4-page section mapped read-write.
        let (section, view) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (PAGES_N * PAGE) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGES_N * PAGE).Value as usize;
            assert_ne!(view, 0);
            (section as usize, view)
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dstorage_same_page.bin");
        let mut expected = vec![0u8; WORKERS * CHUNK];
        for (i, b) in expected.iter_mut().enumerate() {
            *b = ((i * 37 + 19) & 0xff) as u8;
        }
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&expected)
            .unwrap();
        let files: Vec<std::fs::File> = (0..WORKERS)
            .map(|_| std::fs::File::open(&path).unwrap())
            .collect();

        let tracker = track(view, PAGES_N * PAGE, true).expect("section view tracked");
        let start_round = Barrier::new(WORKERS + 1);
        let end_round = Barrier::new(WORKERS + 1);

        with_hot_pages(false, || {
            with_guard_rest(Duration::ZERO, || {
                std::thread::scope(|s| {
                    for (w, file) in files.iter().enumerate() {
                        let (start_round, end_round, expected) =
                            (&start_round, &end_round, &expected);
                        s.spawn(move || {
                            for round in 0..ROUNDS {
                                start_round.wait();
                                let page_base = view + (round % PAGES_N) * PAGE;
                                let dst_at = page_base + w * CHUNK;
                                let guard = writing(dst_at, CHUNK);
                                assert!(
                                    writable(page_base),
                                    "round {round} worker {w}: page must be PAGE_READWRITE before kernel read"
                                );
                                // SAFETY: `[dst_at, dst_at + CHUNK)` is disjoint across `w`
                                // and kept `PAGE_READWRITE` by `guard`.
                                let dst = unsafe {
                                    std::slice::from_raw_parts_mut(dst_at as *mut u8, CHUNK)
                                };
                                let n = file
                                    .seek_read(dst, (w * CHUNK) as u64)
                                    .unwrap_or_else(|e| {
                                        panic!(
                                            "round {round} worker {w}: kernel seek_read into guarded page failed: {e}"
                                        )
                                    });
                                assert_eq!(n, CHUNK);
                                assert_eq!(dst, &expected[w * CHUNK..(w + 1) * CHUNK]);
                                drop(guard);
                                end_round.wait();
                            }
                        });
                    }

                    for round in 0..ROUNDS {
                        let _ = tracker.take(view, view + PAGES_N * PAGE);
                        let page_base = view + (round % PAGES_N) * PAGE;
                        assert!(read_only(page_base), "round {round}: page armed PAGE_READONLY");
                        // Clear previous round's bytes would fault; leave read-only so the 8
                        // workers race `writing()` on a genuinely `PAGE_READONLY` page!
                        start_round.wait();
                        end_round.wait();
                        assert!(
                            tracker.changed(page_base / PAGE),
                            "round {round}: tracker marked changed"
                        );
                    }
                });
            });
        });

        drop(tracker);
        title_memory_remapped(view, PAGES_N * PAGE);
        // SAFETY: unmapping and closing our test section.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view as *mut _,
            });
            CloseHandle(section as _);
        }
    }

    #[test]
    fn write_watched_pool_and_mapped_views_skip_virtual_query_in_protect() {
        assert!(!decide_check_page_guard(None, false));
        assert!(decide_check_page_guard(None, true));
        assert!(decide_check_page_guard(Some("1"), false));
        assert!(!decide_check_page_guard(Some("0"), true));

        // Write-watched pools skip VirtualQuery and coalesce into one VirtualProtect
        // even when PAGE_GUARD checking is enabled.
        let p = pool(4);
        let t = track(p, 4 * PAGE, true).expect("pool tracked");
        let q_before = QUERIES.with(std::cell::Cell::get);
        let p_before = PROTECTS.with(std::cell::Cell::get);
        assert_eq!(t.take(p, p + 4 * PAGE), [(p, p + 4 * PAGE)]);
        assert_eq!(
            QUERIES.with(std::cell::Cell::get) - q_before,
            0,
            "write-watched pool protection must not call VirtualQuery"
        );
        assert_eq!(
            PROTECTS.with(std::cell::Cell::get) - p_before,
            1,
            "write-watched pool protection must coalesce into one VirtualProtect"
        );
        drop(t);
        free(p);

        // Registered xmem section views and their aliases skip VirtualQuery when
        // PAGE_GUARD checking is off (production default), while honoring view runs.
        let (va, vb) = twin_views(4);
        title_memory_mapped(va, 4 * PAGE);
        title_memory_mapped(vb, 4 * PAGE);
        for k in 0..4 {
            title_memory_aliased(vb + k * PAGE, &[va + k * PAGE]);
        }
        with_check_page_guard(false, || {
            let t = track(va, 4 * PAGE, true).expect("views tracked");
            let q_before = QUERIES.with(std::cell::Cell::get);
            let p_before = PROTECTS.with(std::cell::Cell::get);
            assert_eq!(t.take(va, va + 4 * PAGE), [(va, va + 4 * PAGE)]);
            assert_eq!(
                QUERIES.with(std::cell::Cell::get) - q_before,
                0,
                "known xmem views and aliases must not call VirtualQuery"
            );
            assert_eq!(
                PROTECTS.with(std::cell::Cell::get) - p_before,
                2,
                "one coalesced VirtualProtect for the view and one for its alias"
            );
            assert!(read_only(va) && read_only(vb));
            drop(t);
        });
        drop_twin_views(va, vb, 4 * PAGE);
    }

    #[test]
    fn stale_write_fault_after_guard_opened_is_handled_and_marks_dirty() {
        use windows_sys::Win32::System::Memory::{
            PAGE_EXECUTE_READWRITE, PAGE_GUARD, PAGE_NOACCESS, PAGE_NOCACHE, PAGE_READONLY,
            VirtualProtect,
        };
        let p = pool(1);
        let t = track(p, PAGE, true).expect("pool tracked");
        assert_eq!(t.take(p, p + PAGE), [(p, p + PAGE)]);
        assert!(t.clean());
        assert!(read_only(p));

        // Simulate another thread having opened the guard before a concurrent
        // writer's VEH handler acquired `GUARDS`: `page` is no longer in
        // `protected` and is `PAGE_READWRITE`, while the tracker is still clean.
        {
            let mut g = guards();
            assert!(g.protected.remove(&(p / PAGE)));
            open_sorted_pages(&mut g, &[p / PAGE]);
        }
        assert!(writable(p));
        assert!(t.clean());

        // Stale write fault must be handled (`EXCEPTION_CONTINUE_EXECUTION`)
        // and must mark the tracker dirty.
        assert!(
            handle_write_fault(p + 0x488),
            "stale fault on opened tracked page must resume execution"
        );
        assert!(
            t.changed(p / PAGE),
            "stale fault on tracked page must mark tracker dirty"
        );

        // Stale fault while `HOLDING == true` on an already-writable page must
        // also resume execution and queue the page for marking on next lock.
        assert_eq!(t.take(p, p + PAGE), [(p, p + PAGE)]);
        {
            let mut g = guards();
            assert!(g.protected.remove(&(p / PAGE)));
            open_sorted_pages(&mut g, &[p / PAGE]);
            assert!(
                handle_write_fault(p + 0x488),
                "stale fault under HOLDING on writable page must resume"
            );
        }
        // Next `guards()` drains `OPENED_UNDER_LOCK` and marks `t` dirty.
        drop(guards());
        assert!(t.changed(p / PAGE));

        // Negative controls: genuinely non-writable pages must stay unhandled.
        // SAFETY: fresh private pages owned by this test, freed below.
        let unmapped = unsafe { VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE, PAGE_NOACCESS) }
            as usize;
        assert_ne!(unmapped, 0);
        assert!(!handle_write_fault(unmapped + 16));
        {
            let _g = guards();
            assert!(!handle_write_fault(unmapped + 16));
        }
        // SAFETY: reservation above.
        assert_ne!(unsafe { VirtualFree(unmapped as *mut _, 0, MEM_RELEASE) }, 0);

        // SAFETY: fresh private page for protection-flag checks, freed below.
        let scratch = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                PAGE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(scratch, 0);
        let set_prot = |prot: u32| {
            let mut old = 0;
            // SAFETY: `scratch` is a live private page.
            unsafe { VirtualProtect(scratch as *const _, PAGE, prot, &mut old) != 0 }
        };
        assert!(set_prot(PAGE_NOACCESS));
        assert!(!writable(scratch) && !read_only(scratch));
        assert!(!handle_write_fault(scratch + 8));

        assert!(set_prot(PAGE_READONLY));
        assert!(read_only(scratch) && !writable(scratch));
        assert!(!handle_write_fault(scratch + 8));

        if set_prot(PAGE_READWRITE | PAGE_NOCACHE) {
            assert!(writable(scratch));
        }
        assert!(set_prot(PAGE_EXECUTE_READWRITE));
        assert!(writable(scratch));
        assert!(set_prot(PAGE_READWRITE | PAGE_GUARD));
        assert!(!writable(scratch) && !read_only(scratch));
        assert!(set_prot(PAGE_READWRITE));
        // SAFETY: private page above.
        assert_ne!(unsafe { VirtualFree(scratch as *mut _, 0, MEM_RELEASE) }, 0);

        drop(t);
        free(p);
    }

    #[test]
    fn stale_write_fault_racing_with_concurrent_reprotect_never_goes_unhandled() {
        use std::sync::atomic::AtomicBool;

        let p = pool(2);
        let t = track(p, 2 * PAGE, true).expect("pool tracked");
        let _ = t.take(p, p + 2 * PAGE);
        let done = AtomicBool::new(false);

        std::thread::scope(|s| {
            // Re-protector continuously guards both pages via `take`.
            s.spawn(|| {
                while !done.load(Ordering::Acquire) {
                    let _ = t.take(p, p + 2 * PAGE);
                    std::thread::yield_now();
                }
            });
            // Concurrent real hardware faults on page 0 + stale fault handler
            // invocations on pages 0 and 1 while re-protects race in parallel.
            let writers: Vec<_> = (0..3usize)
                .map(|worker| {
                    s.spawn(move || {
                        for iter in 0..200usize {
                            let off = (iter * 64 + worker * 8) % PAGE;
                            // SAFETY: `p` is a live 2-page pool owned by this test.
                            unsafe { ((p + off) as *mut u8).write_volatile(iter as u8) };
                            assert!(
                                handle_write_fault(p + PAGE + off),
                                "concurrent fault handler on tracked page must never return unhandled"
                            );
                        }
                    })
                })
                .collect();
            for writer in writers {
                writer.join().expect("writer thread panicked");
            }
            done.store(true, Ordering::Release);
        });

        drop(t);
        free(p);
    }

    #[test]
    fn protect_with_aliases_handles_busy_alias_and_guards_clean_siblings() {
        let (a, b) = twin_views(2);
        let t = track(a, 2 * PAGE, true).expect("views tracked");
        // Hold alias page 0 busy so page 0 cannot be committed, while page 1
        // and its alias are both guarded in the same batch and page 1 commits.
        let busy = writing(b, PAGE);
        let taken = t.take(a, a + 2 * PAGE);
        assert_eq!(taken, vec![(a, a + 2 * PAGE)]);
        assert!(t.changed(a / PAGE), "page 0 stays dirty while its alias is busy");
        assert!(!t.changed(a / PAGE + 1), "page 1 commits cleanly");
        assert!(read_only(a + PAGE) && read_only(b + PAGE));
        drop(busy);
        // Once the alias is no longer busy, page 0 and its alias both guard and commit.
        let again = t.take(a, a + 2 * PAGE);
        assert_eq!(again, vec![(a, a + PAGE)]);
        assert!(t.clean());
        assert!(read_only(a) && read_only(b));
        drop(t);
    }

    #[test]
    fn title_memory_mapped_aliased_batches_multi_page_alias_guards() {
        let (a, b) = twin_views(4);
        let t = track(a, 4 * PAGE, true).expect("views tracked");
        // Clear initial alias links from `twin_views` so we can test batched
        // `title_memory_mapped_aliased` on a freshly mapped 4-page view `b`.
        title_memory_unmapped(b, 4 * PAGE);
        assert_eq!(t.take(a, a + 4 * PAGE), vec![(a, a + 4 * PAGE)]);
        for k in 0..4 {
            assert!(read_only(a + k * PAGE));
        }

        let aliased: Vec<(usize, Vec<usize>)> = (0..4)
            .map(|k| (b + k * PAGE, vec![a + k * PAGE]))
            .collect();
        with_check_page_guard(false, || {
            PROTECTS.with(|n| n.set(0));
            title_memory_mapped_aliased(b, 4 * PAGE, &aliased);
            let calls = PROTECTS.with(|n| n.get());
            assert_eq!(
                calls, 1,
                "4-page contiguous aliased view must guard in one VirtualProtect call"
            );
        });
        for k in 0..4 {
            assert!(read_only(b + k * PAGE), "alias page {k} is guarded");
        }
        // `title_memory_unmapped` clears guard tracking without issuing an OS
        // reopen on the unmapped range.
        title_memory_unmapped(b, 4 * PAGE);
        assert!(!guards().protected.contains(&(b / PAGE)));
        drop(t);
    }

    /// `Tracked::taken` must never guard pages outside the tracker's runtime
    /// pages (such as plain committed memory inside a wider shadow window).
    #[test]
    fn taken_never_guards_untracked_pages() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualAlloc,
            VirtualFree, VirtualQuery,
        };
        // SAFETY: plain reservation and commit of our own memory.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE, PAGE_READWRITE) } as usize;
        assert_ne!(base, 0);
        // SAFETY: inside our reservation.
        assert!(!unsafe { VirtualAlloc((base + 2 * PAGE) as *const _, 2 * PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        let t = track(base, 4 * PAGE, true).expect("reserved memory counts as runtime memory");
        assert!(!t.is_untracked(base / PAGE) && !t.is_untracked(base / PAGE + 1));
        assert!(t.is_untracked(base / PAGE + 2) && t.is_untracked(base / PAGE + 3));
        let _ = t.take(base, base + 4 * PAGE);
        t.taken(&[base / PAGE + 2, base / PAGE + 3]);
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery((base + 2 * PAGE) as *const _, &mut info, size_of_val(&info)) };
        assert_eq!(info.Protect, PAGE_READWRITE, "untracked plain memory must stay writable");
        drop(t);
        // SAFETY: our reservation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }
}
