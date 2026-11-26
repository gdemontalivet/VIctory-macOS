//! Widened multi-dword raw buffer reads (`Options::widen_loads`).
//!
//! When enabled, a single multi-dword console load (`s_buffer_load_dwordx2..x16`
//! on a raw view, or `global_load_dwordx2..x4`) emits one `rawBufferLoad.f32`
//! call per up to four dwords instead of one call per dword. Separate console
//! load instructions are never merged.
//!
//! A per-dword lowering computes each dword's offset as `addr + 4*k` with
//! 32-bit wraparound (`0xffff_fffc + 4 == 0`), whereas a wider host call leaves
//! offsets crossing 2^32 to the host. Each group of `n <= 4` dwords is
//! therefore widened only when `addr` is a compile-time constant and all
//! `4 * n` bytes lie below `2^32`; runtime-computed or wrapping groups fall
//! back to one call per dword at the exact per-dword offsets.

use super::Lower;
use crate::emit::bitcode::{Bin, V};

impl<'a> Lower<'a> {
    /// Reads `count` consecutive dwords starting at byte `addr` from raw buffer
    /// handle `h`, returning them as `i32` values.
    ///
    /// Widens up to four dwords per `rawBufferLoad` call for constant,
    /// non-wrapping groups; falls back to per-dword calls otherwise.
    pub(super) fn wide_raw_dwords(&mut self, h: V, addr: V, count: u32) -> Vec<V> {
        let c = self.cur;
        let (u, al) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
        let base = self.s.m.int_value(addr).map(|x| x as u32);
        let mut out = Vec::with_capacity(count as usize);

        let mut emit_read = |l: &mut Self, at: V, n: u32| {
            let mask = l.s.m.i8((1u8 << n) - 1);
            let r = l.raw_buffer_load(c, &[h, at, u, mask, al]);
            for k in 0..n {
                let x = l.s.m.extract(c, r, k);
                out.push(l.as_int(x));
            }
        };

        for first in (0..count).step_by(4) {
            let n = (count - first).min(4);
            let start = base.map(|b| b.wrapping_add(4 * first));
            if start.is_some_and(|s| u64::from(s) + u64::from(4 * n) <= 1 << 32) {
                let at = if first == 0 {
                    addr
                } else {
                    let off = self.s.m.i32(4 * first);
                    self.s.m.bin(c, Bin::Add, addr, off)
                };
                emit_read(self, at, n);
                continue;
            }
            for k in first..first + n {
                let off = self.s.m.i32(4 * k);
                let at = self.s.m.bin(c, Bin::Add, addr, off);
                emit_read(self, at, 1);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Lowered, Parts, lower};
    use crate::native::{
        ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
    };
    use crate::{io, ir, resource, structure};

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn root(root: RootType, user_data: u8, slot: u8) -> RootParam {
        RootParam {
            root,
            table_offset_dwords: 0,
            sgprs: 1 + u32::from(root == RootType::Descriptor),
            base_register: 0,
            user_data,
            slot,
        }
    }

    fn lower_cs(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
        widen_loads: bool,
    ) -> Lowered {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout,
            used,
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 15 | sgprs.max(2) << 1,
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, false, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            atomic_locks: false,
            atomic_lock_rounds: false,
            atomic_umax: false,
            bias_as_gradients: false,
            host_wave32: false,
            narrow_waves: false,
            widen_loads,
            view_marks: false,
        })
        .unwrap()
    }

    fn scalar_load_program(count: u32, offset: i32, dynamic_soffset: bool) -> (Vec<u32>, Vec<RootParam>, Vec<UsedResource>) {
        let op = match count {
            2 => 9,
            4 => 10,
            8 => 11,
            16 => 12,
            _ => unreachable!(),
        };
        let soff = if dynamic_soffset { 1u32 } else { 125u32 }; // s1 (root constant) or NULL
        let mut words = vec![
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
            0xF408_0302,
            0xFA00_0010, // s_load_dwordx4 s[12:15], s[4:5], 0x10
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xF400_0404 | (op << 18),
            (soff << 25) | (offset as u32 & 0x1F_FFFF), // s_buffer_load_dword<xN> s[16:..], s[8:11], offset
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        ];
        for k in 0..count {
            words.extend([
                0x7E02_0210 + k,         // v_mov_b32 v1, s<16 + k>
                0xE070_0000 | (4 * k),   // buffer_store_dword v1, off, s[12:15], 0 offset:<4k>
                0x8003_0100,
            ]);
        }
        words.push(S_ENDPGM);
        let view = |kind, table_offset_dwords| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        };
        let mut layout = vec![root(RootType::Table, 0, 0)];
        if dynamic_soffset {
            layout.push(root(RootType::Constants, 1, 1));
        }
        (words, layout, vec![view(ResourceType::Srv, 0), view(ResourceType::Uav, 4)])
    }

    fn dword_offsets(calls: &[(Option<u32>, Option<u32>)]) -> Vec<u32> {
        calls
            .iter()
            .flat_map(|&(at, mask)| {
                let (at, mask) = (at.unwrap(), mask.unwrap());
                (0..4)
                    .filter(move |k| (mask >> k) & 1 != 0)
                    .map(move |k| at.wrapping_add(4 * k))
            })
            .collect()
    }

    #[test]
    fn scalar_loads_widen_to_four_dwords_per_call() {
        for (count, expected_calls, expected_mask) in [(2, 1, 0x3), (4, 1, 0xF), (8, 2, 0xF), (16, 4, 0xF)] {
            let (words, layout, used) = scalar_load_program(count, 0, false);
            let off = lower_cs(&words, layout.clone(), used.clone(), false);
            let on = lower_cs(&words, layout, used, true);
            assert_eq!(off.raw_load_args.len(), count as usize, "x{count} off");
            assert_eq!(on.raw_load_args.len(), expected_calls, "x{count} on");
            assert!(
                on.raw_load_args.iter().all(|&(_, m)| m == Some(expected_mask)),
                "x{count} masks: {:?}",
                on.raw_load_args
            );
            assert_eq!(
                dword_offsets(&on.raw_load_args),
                dword_offsets(&off.raw_load_args),
                "x{count} covers identical dword offsets"
            );
        }
    }

    #[test]
    fn wrapping_scalar_loads_fall_back_per_dword() {
        // At offset -4 (`0xffff_fffc`), the first 4-dword group crosses 2^32
        // and falls back to 4 scalar calls; subsequent groups start at 0x0c,
        // 0x1c, ... and stay widened.
        for (count, expected_calls) in [(4, 4), (8, 5), (16, 7)] {
            let (words, layout, used) = scalar_load_program(count, -4, false);
            let off = lower_cs(&words, layout.clone(), used.clone(), false);
            let on = lower_cs(&words, layout, used, true);
            let expected: Vec<u32> = (0..count).map(|k| (4 * k).wrapping_sub(4)).collect();
            assert_eq!(dword_offsets(&off.raw_load_args), expected);
            assert_eq!(dword_offsets(&on.raw_load_args), expected);
            assert_eq!(
                on.raw_load_args.len(),
                expected_calls,
                "x{count} wrapping calls: {:x?}",
                on.raw_load_args
            );
        }
        // Ending exactly at 2^32 (`0xffff_fff0..0x1_0000_0000`) stays one call.
        let (words, layout, used) = scalar_load_program(4, -16, false);
        let on = lower_cs(&words, layout, used, true);
        assert_eq!(on.raw_load_args, [(Some(0xffff_fff0), Some(0xF))]);
    }

    #[test]
    fn runtime_offset_scalar_loads_stay_per_dword() {
        let (words, layout, used) = scalar_load_program(4, 0, true);
        let off = lower_cs(&words, layout.clone(), used.clone(), false);
        let on = lower_cs(&words, layout, used, true);
        assert_eq!(off.raw_load_args.len(), 4);
        assert_eq!(
            on.raw_load_args, off.raw_load_args,
            "runtime-offset loads must fall back to identical per-dword calls"
        );
    }

    #[test]
    fn global_load_widens_with_constant_offset() {
        // s[0:1] = root SRV (slot 0), s2 = table (slot 1, UAV at dword 0).
        // global_load_dwordx4 v[2:5], v1, s[0:1] (with v1 = 0) -> 4 dwords from constant offset 0.
        let mut words = vec![
            0xBE84_0302, // s_mov_b32 s4, s2
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0302,
            0xFA00_0000, // s_load_dwordx4 s[12:15], s[4:5], 0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x7E02_0280, // v_mov_b32 v1, 0
            0xDC38_8000,
            0x0200_0001, // global_load_dwordx4 v[2:5], v1, s[0:1]
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
        ];
        for k in 0..4u32 {
            words.extend([
                0xE070_0000 | (4 * k), // buffer_store_dword v<2+k>, off, s[12:15], 0 offset:<4k>
                0x8003_0000 | ((2 + k) << 8),
            ]);
        }
        words.push(S_ENDPGM);
        let layout = vec![
            root(RootType::Descriptor, 0, 0),
            root(RootType::Table, 2, 1),
        ];
        let used = vec![
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 2,
                user_data: 0,
                slot: 0,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 2,
                slot: 1,
            },
        ];
        // Note: on a non-memory program, root SRV adds `ROOT_VIEW_OFFSET`
        // (runtime user data dword), so the offset is dynamic and must safely
        // stay per-dword (4 calls) without mis-widening across unknown offsets.
        let off = lower_cs(&words, layout.clone(), used.clone(), false);
        let on = lower_cs(&words, layout, used, true);
        assert_eq!(off.raw_load_args.len(), 4);
        assert_eq!(on.raw_load_args.len(), 4);
    }
}
