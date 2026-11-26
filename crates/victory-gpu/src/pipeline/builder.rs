//! The title's acceleration-structure builder runs whole or not at all.
//!
//! `XboxBvhBuilder.dll` builds and refits the ray-tracing structures in
//! title memory every frame through some fifty compute programs, one
//! dispatch after another. A dispatch whose host pipeline is missing is
//! dropped on its own (`State::dispatch`), and some of the builder's are
//! missing while pipelines are still being made; on a host of 32-lane waves
//! (`Host::wave32`), its 64-lane programs exchange across halves and hung
//! Wave32 GPUs (`Marvisak/VIctory#2711`). So the builder's dispatches run
//! only in frames where every pipeline it has created has its host pipeline
//! and either the host runs 64-lane waves whole or `VICTORY_BVH_BUILDER=1`
//! opts in; otherwise none of them run (the structures keep what they held).
//!
//! Its pipelines are known by where they come from: created by its code (a
//! return address in its image on the stack) or from bytecode in its image.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use super::make::{self, Pending};

/// The builder's image, `[start, end)`, once mapped.
static IMAGE: Mutex<Option<(usize, usize)>> = Mutex::new(None);

/// The builder's file name.
const FILE: &str = "XboxBvhBuilder.dll";

/// A title DLL was mapped at `base`, `size` bytes: noted if it is the
/// builder.
pub fn note_title_dll(file: &str, base: usize, size: usize) {
    if file.eq_ignore_ascii_case(FILE) {
        tracing::info!(base = format_args!("{base:#x}"), size, "BVH builder mapped");
        *IMAGE.lock().unwrap_or_else(|e| e.into_inner()) = Some((base, base + size));
    }
}

fn image() -> Option<(usize, usize)> {
    *IMAGE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether the pipeline being created is the builder's: `code` (the title's
/// bytecode pointers) lies in its image, or its code is among our callers.
#[inline(never)]
pub(crate) fn created_by_builder(code: &[*const u8]) -> bool {
    let Some((start, end)) = image() else {
        return false;
    };
    let inside = |a: usize| (start..end).contains(&a);
    if code.iter().any(|&c| inside(c as usize)) {
        return true;
    }
    callers().into_iter().any(inside)
}

/// Return addresses on the stack, ours first.
fn callers() -> Vec<usize> {
    use windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace;
    let mut frames = [std::ptr::null_mut(); 24];
    // SAFETY: room for that many frames.
    let n = unsafe {
        RtlCaptureStackBackTrace(
            1,
            frames.len() as u32,
            frames.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    frames[..usize::from(n)]
        .iter()
        .map(|&f| f as usize)
        .collect()
}

/// The builder's pipelines (alive while the title holds them).
static PIPELINES: Mutex<Vec<Weak<Pending>>> = Mutex::new(Vec::new());

/// The builder created pipeline `p`: it counts towards [`runs`], and is
/// made at once (no frame's wait budget holds it back).
pub(crate) fn note(p: &Arc<Pending>) {
    PIPELINES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(Arc::downgrade(p));
    make::begin(p);
}

/// The frame [`runs`] last decided for, and its decision.
static DECIDED: Mutex<(u64, bool)> = Mutex::new((u64::MAX, true));

/// Whether `VICTORY_BVH_BUILDER=1` opts in to running the builder on 32-lane
/// wave hosts.
fn allow_halves() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_BVH_BUILDER"))
}

/// Whether the builder's dispatches run in the current frame: decided once
/// per frame, so that a frame never runs part of the builder.
pub(crate) fn runs() -> bool {
    static SAID: AtomicBool = AtomicBool::new(false);
    let frame = crate::frame::current();
    let mut decided = DECIDED.lock().unwrap_or_else(|e| e.into_inner());
    if decided.0 == frame {
        return decided.1;
    }
    let states: Vec<Option<bool>> = {
        let mut pipelines = PIPELINES.lock().unwrap_or_else(|e| e.into_inner());
        pipelines.retain(|p| p.strong_count() > 0);
        pipelines
            .iter()
            .filter_map(Weak::upgrade)
            .map(|p| make::state(&p))
            .collect()
    };
    let halves = crate::host::host().is_some_and(crate::host::Host::wave32);
    let open = opens(&states, halves, allow_halves());
    if open != decided.1 || !SAID.swap(true, Ordering::Relaxed) {
        let (made, refused) = (
            states.iter().filter(|s| **s == Some(true)).count(),
            states.iter().filter(|s| **s == Some(false)).count(),
        );
        tracing::info!(
            frame,
            pipelines = states.len(),
            made,
            refused,
            "BVH builder {}",
            if open {
                "runs"
            } else if halves && !allow_halves() {
                "held back: 64-lane programs run as 32-lane halves here (VICTORY_BVH_BUILDER=1 overrides)"
            } else {
                "held back: not all its pipelines are made"
            }
        );
    }
    *decided = (frame, open);
    open
}

/// Whether every pipeline is made (`Some(true)`); refused ones are
/// `Some(false)`, those still being made `None`.
fn all_made(states: &[Option<bool>]) -> bool {
    states.iter().all(|s| *s == Some(true))
}

/// Whether the builder runs: every pipeline made, and either 64-lane waves
/// run whole or `VICTORY_BVH_BUILDER=1` opts in (`allow_halves`).
fn opens(states: &[Option<bool>], halves: bool, allow_halves: bool) -> bool {
    (!halves || allow_halves) && all_made(states)
}

/// Tests that set the builder's image, one at a time; each starts with
/// no decision made.
#[cfg(test)]
pub(crate) fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    *DECIDED.lock().unwrap_or_else(|e| e.into_inner()) = (u64::MAX, true);
    guard
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{S_OK, slot};
    use crate::pipeline::{
        Bytecode, ComputeHead, PIPELINE_IIDS, This, builder_runs, create_compute, held_back,
        pipeline,
    };

    #[test]
    fn the_builder_runs_only_with_every_pipeline_made() {
        assert!(all_made(&[]), "no builder: nothing to hold back");
        assert!(all_made(&[Some(true), Some(true)]));
        assert!(!all_made(&[Some(true), Some(false)]), "one refused");
        assert!(!all_made(&[Some(true), None]), "one still being made");
        assert!(opens(&[Some(true)], false, false));
        assert!(
            !opens(&[Some(true)], true, false),
            "64-lane programs run as halves"
        );
        assert!(
            opens(&[Some(true)], true, true),
            "VICTORY_BVH_BUILDER=1 override"
        );
        assert!(
            !opens(&[Some(false)], true, true),
            "override still requires all made"
        );
        assert!(!opens(&[], true, false));
    }

    #[test]
    fn builder_pipelines_are_known_by_their_origin() {
        let _one = test_lock();
        let here = builder_pipelines_are_known_by_their_origin as *const () as usize;
        let elsewhere = [0x7_0000_0000usize as *const u8];
        *IMAGE.lock().unwrap() = None;
        assert!(!created_by_builder(&elsewhere), "no builder mapped");
        *IMAGE.lock().unwrap() = Some((0x7_0000_0000, 0x7_0001_0000));
        assert!(created_by_builder(&elsewhere));
        assert!(!created_by_builder(&[0x8_0000_0000usize as *const u8]));
        *IMAGE.lock().unwrap() = Some((here - 0x10_0000, here + 0x10_0000));
        assert!(created_by_builder(&[]));
        *IMAGE.lock().unwrap() = None;
    }

    #[test]
    fn builder_dispatches_wait_for_every_builder_pipeline() {
        let _one = test_lock();
        let code = [0xB0u8; 64];
        let create = |code: &[u8]| {
            let desc = ComputeHead {
                root: std::ptr::null(),
                stage: Bytecode {
                    code: code.as_ptr(),
                    size: code.len(),
                },
            };
            let mut p: This = std::ptr::null();
            assert_eq!(
                create_compute(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p),
                S_OK
            );
            p
        };
        note_title_dll("xboxbvhbuilder.DLL", code.as_ptr() as usize, code.len());
        let p = create(&code);
        let other_code = [0xC0u8; 64];
        let other = create(&other_code);
        note_title_dll("XboxBvhBuilder.dll", 0, 0);
        assert!(
            pipeline(p).builder,
            "its bytecode is in the builder's image"
        );
        assert!(!pipeline(other).builder);
        let refused = (0..500).any(|_| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            make::state(&pipeline(p).host).is_some()
        });
        assert!(refused, "the builder's pipeline was started");
        assert_eq!(make::state(&pipeline(p).host), Some(false));
        assert!(!builder_runs());
        assert!(
            held_back(p),
            "one builder pipeline missing: the builder waits"
        );
        assert!(!held_back(other));
        for q in [p, other] {
            slot::<extern "system" fn(This) -> u32>(q, 0x10)(q);
        }
    }
}
