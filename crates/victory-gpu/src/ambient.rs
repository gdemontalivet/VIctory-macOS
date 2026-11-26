//! Opt-in stand-in for the indirect light the title traces rays for.
//!
//! With the ray-tracing structure builder held back
//! (`!crate::pipeline::builder_runs()`), the title's probe rays find nothing:
//! `CS_ScreenProbes_TemporalResolve_*` leaves sky light outdoors, dark
//! silhouettes indoors where no direct lamp or sun reaches, and unoccluded sky
//! inscattering inside rooms (`CS_AccumulateLocalLightInscatteringVolume_*`).
//!
//! Adapted from `archive-cool-fox` (`82ca1347`, `7da523a9`, after beloooo's
//! `VIctory-Belobug-AMD`) to compile cleanly through Wine's `vkd3d-shader`
//! (`crate::compute`, explicit serialized root signature, `ByteAddressBuffer`
//! loads + `RWByteAddressBuffer` stores):
//!
//! * `VICTORY_AMBIENT`:
//!   - unset or `0`/`off` (default): disabled; the title's own output is kept.
//!   - `<number>` (e.g. `0.8` or `1.0`): screen-space bounce-light estimate
//!     from `CS_CacheScreenSpaceRadiance` averaged over 64x36 cells with slow
//!     temporal adaptation and first-order directional harmonics at 160x90
//!     points interpolated by depth.
//!   - `constant:<level>`: constant ambient fill with screen-space crease AO.
//! * `VICTORY_AMBIENT_SHADE=<strength>,<radius_m>`: crease occlusion parameters
//!   (defaults `1.0,0.6`).
//! * `VICTORY_AMBIENT_AIR=<multiple>`: caps local-light air-scattering luma in
//!   `CS_AccumulateLocalLightInscatteringVolume` by `<multiple>` of the room
//!   and view-direction surface luma (unset or `0`: disabled).

use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::s;

use crate::compute::{Param, pipeline_with_root, root_signature, srv, uav};
use crate::host::{HostList, host};

/// Programs of the title the stand-in follows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Role {
    /// Resolves the diffuse light (`CS_ScreenProbes_TemporalResolve_*`).
    Resolve,
    /// Keeps the light visible surfaces send out (`CS_CacheScreenSpaceRadiance_*`).
    Radiance,
    /// Accumulates local-light inscattering (`CS_AccumulateLocalLightInscatteringVolume_*`).
    Air,
}

/// The program's part in this, by its debug name in the container.
pub(crate) fn role(container: &[u8]) -> Option<Role> {
    let has = |name: &[u8]| container.windows(name.len()).any(|w| w == name);
    if has(b"CS_ScreenProbes_TemporalResolve") {
        Some(Role::Resolve)
    } else if has(b"CS_CacheScreenSpaceRadiance") {
        Some(Role::Radiance)
    } else if has(b"CS_AccumulateLocalLightInscatteringVolume") {
        Some(Role::Air)
    } else {
        None
    }
}

/// Root parameter and table offset in dwords for each bound resource.
pub(crate) const LIGHT: (u8, u32) = (1, 176);
pub(crate) const DEPTH: (u8, u32) = (0, 464);
pub(crate) const RADIANCE: (u8, u32) = (1, 16);
pub(crate) const AIR: (u8, u32) = (1, 32);

#[derive(Clone, Copy, PartialEq, Debug)]
enum Set {
    Off,
    Estimate(f32),
    Constant(f32),
}

fn parse(set: &str) -> Set {
    let set = set.trim();
    if matches!(set, "" | "0" | "off" | "false") {
        return Set::Off;
    }
    let number = |t: &str| {
        t.trim()
            .parse::<f32>()
            .ok()
            .filter(|v| v.is_finite() && *v > 0.0)
    };
    match set.strip_prefix("constant:") {
        Some(level) => number(level).map_or(Set::Off, Set::Constant),
        None => number(set).map_or(Set::Off, Set::Estimate),
    }
}

fn set() -> Set {
    static SET: OnceLock<Set> = OnceLock::new();
    *SET.get_or_init(|| parse(&std::env::var("VICTORY_AMBIENT").unwrap_or_default()))
}

/// Whether the stand-in is active: enabled via `VICTORY_AMBIENT`, and the
/// title's BVH builder is held back.
pub(crate) fn on() -> bool {
    set() != Set::Off && !crate::pipeline::builder_runs()
}

/// Maximum air-scattering luma multiple (`VICTORY_AMBIENT_AIR=<multiple>`).
fn air() -> f32 {
    static SET: OnceLock<f32> = OnceLock::new();
    *SET.get_or_init(|| {
        let set = std::env::var("VICTORY_AMBIENT_AIR").unwrap_or_default();
        set.trim()
            .parse()
            .ok()
            .filter(|v: &f32| v.is_finite() && *v > 0.0)
            .unwrap_or(0.0)
    })
}

/// Strength and radius (in metres) of screen-space crease shading.
fn shade() -> [f32; 2] {
    static SET: OnceLock<[f32; 2]> = OnceLock::new();
    *SET.get_or_init(|| {
        let mut shade = [1.0, 0.6];
        let set = std::env::var("VICTORY_AMBIENT_SHADE").unwrap_or_default();
        for (to, v) in shade.iter_mut().zip(set.split(',')) {
            if let Ok(v) = v.trim().parse::<f32>()
                && v.is_finite()
            {
                *to = v;
            }
        }
        shade
    })
}

const SOURCE: &str = r#"
#define CX 64
#define CY 36
#define CELLS (CX * CY)
#define PX 160
#define PY 90
cbuffer Set : register(b0) {
    float level; float constant; float strength; float radius;
    float near; float tan_y; uint width; uint height;
    float follow; float most; float air; uint slices;
};
StructuredBuffer<float4> cells_in : register(t1);
StructuredBuffer<float4> points_in : register(t2);
RWByteAddressBuffer cells_out : register(u1);
RWByteAddressBuffer points_out : register(u2);
static const float3 LUMA = float3(0.2126, 0.7152, 0.0722);

float4 load_cell(uint i) { return cells_in[i]; }
void store_cell(uint i, float4 v) {
    uint4 u = asuint(v);
    uint b = i * 16;
    cells_out.Store(b, u.x);
    cells_out.Store(b + 4, u.y);
    cells_out.Store(b + 8, u.z);
    cells_out.Store(b + 12, u.w);
}
float4 load_point(uint i) { return points_in[i]; }
void store_point(uint i, float4 v) {
    uint4 u = asuint(v);
    uint b = i * 16;
    points_out.Store(b, u.x);
    points_out.Store(b + 4, u.y);
    points_out.Store(b + 8, u.z);
    points_out.Store(b + 12, u.w);
}

#if defined(PLACE) || defined(GATHER) || defined(RAISE)
Texture2D<float> depth : register(t0);
// The point seen at a texel, in metres from the eye (x right, y up, z away).
float3 seen(int2 p, float2 size) {
    float d = depth.Load(int3(clamp(p, int2(0, 0), int2(size) - 1), 0));
    float z = min(near / max(d, 1e-7), 1000.0);
    float2 ndc = (float2(p) + 0.5) / size * 2.0 - 1.0;
    return float3(ndc.x * tan_y * size.x / size.y * z, -ndc.y * tan_y * z, z);
}
#endif

#ifdef KEEP
RWTexture2D<float4> radiance : register(u0);
[numthreads(8, 4, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    uint w = width, h = height;
    float3 sum = 0;
    for (uint y = 0; y < 5; y++) {
        for (uint x = 0; x < 5; x++) {
            uint2 p = uint2((float2(at.xy) + (float2(x, y) + 0.5) / 5.0) * float2(w, h) / float2(CX, CY));
            float3 c = max(radiance[p].rgb, 0.0);
            uint3 bits = asuint(c);
            bool bad = (bits.x & 0x7f800000u) == 0x7f800000u
                || (bits.y & 0x7f800000u) == 0x7f800000u
                || (bits.z & 0x7f800000u) == 0x7f800000u;
            c = bad ? 0.0 : c;
            sum += c / max(1.0, dot(c, LUMA) / 20.0);
        }
    }
    store_cell(at.y * CX + at.x, float4(sum / 25.0, 1.0));
}
#endif

#ifdef MEAN
[numthreads(1, 1, 1)]
void main() {
    float3 sum = 0;
    for (uint i = 0; i < CELLS; i++) {
        sum += load_cell(i).rgb;
    }
    float4 was = load_cell(CELLS);
    float3 now = sum / CELLS;
    store_cell(CELLS, float4(was.a > 0.0 ? lerp(was.rgb, now, follow) : now, 1.0));
}
#endif

#ifdef PLACE
[numthreads(8, 4, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    float2 size = float2(width, height);
    float3 p = seen(int2((float2(at.xy) + 0.5) * size / float2(CX, CY)), size);
    store_cell(CELLS + 1 + at.y * CX + at.x, float4(p, p.z * p.z));
}
#endif

#ifdef GATHER
[numthreads(8, 2, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    float2 size = float2(width, height);
    float3 p = seen(int2((float2(at.xy) + 0.5) * size / float2(PX, PY)), size);
    float r2 = 0.25 + 0.0025 * p.z * p.z;
    float whole = 0;
    float3 toward = 0, light = 0, red = 0, green = 0, blue = 0;
    for (uint i = 0; i < CELLS; i++) {
        float4 c = load_cell(CELLS + 1 + i);
        float3 L = load_cell(i).rgb;
        float3 d = c.xyz - p;
        float dd = dot(d, d);
        float w = c.w / (dd + r2);
        float3 dir = d / sqrt(dd + 0.25 * r2);
        whole += w;
        toward += w * dir;
        light += w * L;
        red += w * L.r * dir;
        green += w * L.g * dir;
        blue += w * L.b * dir;
    }
    whole = max(whole, 1e-6);
    light /= whole;
    toward /= whole;
    uint at0 = (at.y * PX + at.x) * 3;
    store_point(at0, float4(red / whole - light.r * toward, light.r));
    store_point(at0 + 1, float4(green / whole - light.g * toward, light.g));
    store_point(at0 + 2, float4(blue / whole - light.b * toward, light.b));
}
#endif

#ifdef AIR
RWTexture3D<float4> scattered : register(u0);
[numthreads(8, 8, 2)]
void main(uint3 at : SV_DispatchThreadID) {
    uint w = width, h = height, d = slices;
    if (at.x >= w || at.y >= h || at.z >= d) return;
    uint2 q = min(uint2((float2(at.xy) + 0.5) / float2(w, h) * float2(PX, PY)), uint2(PX - 1, PY - 1));
    uint at0 = (q.y * PX + q.x) * 3;
    float3 seen_there = float3(load_point(at0).w, load_point(at0 + 1).w, load_point(at0 + 2).w);
    float far = float(at.z) / float(d);
    float cap = air * lerp(dot(load_cell(CELLS).rgb, LUMA), dot(seen_there, LUMA), far);
    float4 s = scattered[at];
    float has = dot(s.rgb, LUMA);
    if (has > cap) {
        scattered[at] = float4(s.rgb * (cap / has), s.a);
    }
}
#endif

#ifdef RAISE
RWTexture2D<uint> light : register(u0);
float3 unpack_rgb9e5(uint v) {
    return float3(v & 511u, (v >> 9) & 511u, (v >> 18) & 511u) * exp2(float(v >> 27) - 24.0);
}
uint pack_rgb9e5(float3 c) {
    c = clamp(c, 0.0, 65408.0);
    float top = max(c.r, max(c.g, c.b));
    int e = clamp(int(floor(log2(max(top, 1e-9)))) + 16, 0, 31);
    float scale = exp2(float(24 - e));
    if (top * scale >= 511.5) {
        e = min(e + 1, 31);
        scale = exp2(float(24 - e));
    }
    uint3 m = min(uint3(c * scale + 0.5), 511u);
    return m.r | (m.g << 9) | (m.b << 18) | (uint(e) << 27);
}
[numthreads(8, 8, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    float2 size = float2(width, height);
    if (at.x >= width || at.y >= height) return;
    int2 p = int2(at.xy);
    float3 c = seen(p, size);
    float3 l = seen(p + int2(-1, 0), size), r = seen(p + int2(1, 0), size);
    float3 u = seen(p + int2(0, -1), size), d = seen(p + int2(0, 1), size);
    float3 dx = abs(l.z - c.z) < abs(r.z - c.z) ? c - l : r - c;
    float3 dy = abs(u.z - c.z) < abs(d.z - c.z) ? c - u : d - c;
    float3 n = normalize(cross(dy, dx) + float3(0, 0, -1e-9));
    if (dot(n, c) > 0.0) n = -n;
    float open = 1.0;
    float px = min(radius * size.y / (2.0 * tan_y * c.z), 128.0);
    float noise = frac(52.9829189 * frac(dot(float2(p), float2(0.06711056, 0.00583715))));
    if (strength > 0.0 && px >= 1.5 && c.z < 400.0) {
        float sum = 0.0;
        const int N = 12;
        for (int i = 0; i < N; i++) {
            float a = i * 2.399963 + noise * 6.2831853;
            float2 o = float2(cos(a), sin(a)) * (sqrt((i + 0.5) / N) * px);
            float3 v = seen(p + int2(floor(o + 0.5)), size) - c;
            float vv = dot(v, v);
            float over = dot(v, n) - 0.004 * c.z - 0.005;
            sum += max(over, 0.0) / sqrt(vv + 1e-6) * saturate(1.0 - vv / (radius * radius));
        }
        open = saturate(1.0 - strength * 2.0 * sum / N);
    }
    float3 found;
    if (constant > 0.0) {
        found = float3(1.0, 0.98, 0.94) * constant * open;
        light[p] = pack_rgb9e5(found);
        return;
    }
    float2 g = (float2(p) + 0.5) / size * float2(PX, PY) - 0.5 + (noise - 0.5) * 0.5;
    int2 g0 = int2(floor(g));
    float2 f = g - float2(g0);
    float4 sr = 0, sg = 0, sb = 0;
    float sum = 0;
    for (int k = 0; k < 4; k++) {
        int2 q = clamp(g0 + int2(k & 1, k >> 1), int2(0, 0), int2(PX, PY) - 1);
        float z = seen(int2((float2(q) + 0.5) * size / float2(PX, PY)), size).z;
        float2 b = float2((k & 1) != 0 ? f.x : 1.0 - f.x, (k >> 1) != 0 ? f.y : 1.0 - f.y);
        float w = (b.x * b.y + 1e-3) / (0.02 + abs(z - c.z) / c.z);
        uint at0 = (q.y * PX + q.x) * 3;
        sr += w * load_point(at0);
        sg += w * load_point(at0 + 1);
        sb += w * load_point(at0 + 2);
        sum += w;
    }
    float3 all = float3(sr.w, sg.w, sb.w) / sum;
    found = all + 1.2 * float3(dot(sr.xyz, n), dot(sg.xyz, n), dot(sb.xyz, n)) / sum;
    found = clamp(found, 0.2 * all, 2.5 * all);
    found = lerp(found, load_cell(CELLS).rgb, 0.2) * level * open;
    float3 was = unpack_rgb9e5(light[p]);
    float top = max(dot(found, LUMA) * most, 1e-6);
    float has = dot(was, LUMA);
    was *= min(1.0, top / max(has, 1e-6));
    light[p] = pack_rgb9e5(max(was, found));
}
#endif
"#;

const CELLS: u64 = 64 * 36;
const CELL_BYTES: u64 = (2 * CELLS + 1) * 16;
const POINT_BYTES: u64 = 160 * 90 * 48;

struct Pass {
    root: ID3D12RootSignature,
    keep: ID3D12PipelineState,
    mean: ID3D12PipelineState,
    place: ID3D12PipelineState,
    gather: ID3D12PipelineState,
    raise: ID3D12PipelineState,
    air: ID3D12PipelineState,
    cells: ID3D12Resource,
    points: ID3D12Resource,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

fn uav_buffer(device: &ID3D12Device, size: u64) -> windows::core::Result<ID3D12Resource> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(16),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: valid heap and buffer descriptions; out pointer ours.
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buffer,
        )?;
    }
    buffer.ok_or_else(windows::core::Error::empty)
}

fn make(device: &ID3D12Device) -> windows::core::Result<Pass> {
    let depth_range = [srv(0, 1)];
    let image_range = [uav(0, 1)];
    let params = [
        Param::Constants {
            register: 0,
            count: 12,
        },
        Param::Table(&depth_range),
        Param::Table(&image_range),
        Param::Srv {
            register: 1,
            space: 0,
        },
        Param::Srv {
            register: 2,
            space: 0,
        },
        Param::Uav { register: 1 },
        Param::Uav { register: 2 },
    ];
    let root = root_signature(device, &params)?;
    let build = |part: &str, label| {
        pipeline_with_root(
            device,
            &root,
            &format!("#define {part}\n{SOURCE}"),
            label,
            0,
        )
    };
    Ok(Pass {
        keep: build("KEEP", s!("ambient_keep"))?,
        mean: build("MEAN", s!("ambient_mean"))?,
        place: build("PLACE", s!("ambient_place"))?,
        gather: build("GATHER", s!("ambient_gather"))?,
        raise: build("RAISE", s!("ambient_raise"))?,
        air: build("AIR", s!("ambient_air"))?,
        root,
        cells: uav_buffer(device, CELL_BYTES)?,
        points: uav_buffer(device, POINT_BYTES)?,
    })
}

fn pass() -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let host = host()?;
        make(&host.device)
            .inspect_err(|e| tracing::warn!(error = %e, "no stand-in for the indirect light"))
            .ok()
    })
    .as_ref()
}

fn uav_barrier(list: &ID3D12GraphicsCommandList) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        ..Default::default()
    };
    // SAFETY: an open list; the barrier names no resource.
    unsafe { list.ResourceBarrier(&[barrier]) };
}

static KEPT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Views in the currently bound shader-visible heap that [`after`] reads/writes.
pub(crate) struct Views {
    pub(crate) image: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub(crate) depth: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub(crate) size: [u32; 2],
    pub(crate) slices: u32,
}

/// Records the stand-in compute dispatches after the title's `role` shader.
pub(crate) fn after(list: &HostList, role: Role, views: &Views) -> bool {
    use std::sync::atomic::Ordering;
    let (level, constant) = match set() {
        Set::Off => return false,
        Set::Estimate(level) => (level, 0.0f32),
        Set::Constant(level) => (0.0, level),
    };
    if role != Role::Radiance && constant == 0.0 && !KEPT.load(Ordering::Relaxed) {
        return false;
    }
    if role == Role::Air && (constant > 0.0 || air() == 0.0) {
        return false;
    }
    let Some(pass) = pass() else {
        return false;
    };
    static SAID: Mutex<u32> = Mutex::new(0);
    {
        let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
        if *said < 4 {
            *said += 1;
            tracing::info!(
                ?role,
                size = ?views.size,
                slices = views.slices,
                level,
                constant,
                air = air(),
                "stand-in for the indirect light"
            );
        }
    }
    list.keep(&pass.cells);
    list.keep(&pass.points);
    let [strength, radius] = shade();
    let floats = [level, constant, strength, radius, 0.1, 0.45];
    let mut words = [0u32; 12];
    for (to, v) in words.iter_mut().zip(floats) {
        *to = v.to_bits();
    }
    words[6] = views.size[0];
    words[7] = views.size[1];
    words[8] = 0.04f32.to_bits();
    words[9] = 4.0f32.to_bits();
    words[10] = air().to_bits();
    words[11] = views.slices;
    let l = &list.list;
    // SAFETY: an open list with the view heap bound; textures and buffers are
    // live and kept until reset.
    unsafe {
        let cells_gpu = pass.cells.GetGPUVirtualAddress();
        let points_gpu = pass.points.GetGPUVirtualAddress();
        uav_barrier(l);
        l.SetComputeRootSignature(&pass.root);
        l.SetComputeRoot32BitConstants(0, 12, words.as_ptr().cast(), 0);
        l.SetComputeRootDescriptorTable(1, views.depth);
        l.SetComputeRootDescriptorTable(2, views.image);
        l.SetComputeRootShaderResourceView(3, cells_gpu);
        l.SetComputeRootShaderResourceView(4, points_gpu);
        l.SetComputeRootUnorderedAccessView(5, cells_gpu);
        l.SetComputeRootUnorderedAccessView(6, points_gpu);
        let run = |pso: &ID3D12PipelineState, x, y, z| {
            l.SetPipelineState(pso);
            l.Dispatch(x, y, z);
            uav_barrier(l);
        };
        match role {
            Role::Radiance => {
                run(&pass.keep, 8, 9, 1);
                run(&pass.mean, 1, 1, 1);
                KEPT.store(true, Ordering::Relaxed);
            }
            Role::Air => run(
                &pass.air,
                views.size[0].div_ceil(8),
                views.size[1].div_ceil(8),
                views.slices.div_ceil(2),
            ),
            Role::Resolve => {
                if constant == 0.0 {
                    run(&pass.place, 8, 9, 1);
                    run(&pass.gather, 20, 45, 1);
                }
                run(
                    &pass.raise,
                    views.size[0].div_ceil(8),
                    views.size[1].div_ceil(8),
                    1,
                );
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings() {
        assert_eq!(parse(""), Set::Off);
        assert_eq!(parse("off"), Set::Off);
        assert_eq!(parse("0"), Set::Off);
        assert_eq!(parse("0.8"), Set::Estimate(0.8));
        assert_eq!(parse("constant:0.3"), Set::Constant(0.3));
        assert_eq!(parse("constant:nan"), Set::Off);
    }

    #[test]
    fn roles_by_name() {
        assert_eq!(
            role(b"..CS_ScreenProbes_TemporalResolve_a911b49c_Wrapped.."),
            Some(Role::Resolve)
        );
        assert_eq!(
            role(b"..CS_CacheScreenSpaceRadiance_eed2fc35.."),
            Some(Role::Radiance)
        );
        assert_eq!(
            role(b"CS_AccumulateLocalLightInscatteringVolume_7e2f5eef"),
            Some(Role::Air)
        );
        assert_eq!(role(b"CS_TemporalResolve"), None);
    }

    /// All six passes compile through the host's HLSL compiler (`vkd3d-shader`
    /// under Wine) and build compute PSOs on D3D12 / D3DMetal.
    #[test]
    fn the_passes_build() {
        let Some(host) = host() else {
            return;
        };
        make(&host.device).expect("all 6 ambient compute PSOs build");
    }
}
