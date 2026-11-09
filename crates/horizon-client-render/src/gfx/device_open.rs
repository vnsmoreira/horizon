//! Opening the renderer's device: the backends tried in turn, the adapter each offers, the
//! retries for hidden and non-conformant adapters, and DirectX 12's shader compiler.
use super::*;

/// Opens a device on the first of `backend_order`'s backends that gives one.
/// Offscreen rendering needs no surface, so it can use a GPU the window's
/// display cannot present from (a headless compositor).
pub(super) async fn open_device(
    window: &Arc<Window>,
    offscreen: bool,
    api: GraphicsApi,
    layers: u32,
) -> Result<Opened> {
    let wgpu_backend = std::env::var_os("WGPU_BACKEND").is_some();
    let mut first = None;
    for (order, allow_cpu) in device_search(
        api.with_env(),
        std::env::consts::OS,
        // The machine's: the x64 build emulated on a Windows ARM64 PC has its GPU too (#7513).
        horizon_system::machine_arch(),
        wgpu_backend,
    ) {
        match open_device_with(window, offscreen, &order, layers, allow_cpu).await {
            Ok(opened) => {
                if let Some(first) = &first {
                    log::warn!(
                        "Horizon renderer: the chosen graphics API gave no GPU ({first}); {}",
                        if allow_cpu {
                            "rendering on the CPU, which is slow"
                        } else {
                            "using the GPU through another one"
                        }
                    );
                }
                return Ok(opened);
            }
            Err(e) => {
                first.get_or_insert(e);
            }
        }
    }
    Err(first.unwrap_or_else(|| NO_GPU_DRIVER.into()))
}
/// The tries `open_device` makes, in turn: the chosen API's backends (`backend_order`) on a
/// real GPU; when the player chose one API, the other backends `Auto` would try, on a real
/// GPU; and last the chosen backends with a software adapter (WARP, llvmpipe) allowed.
/// Software rendering takes most of a weak CPU even on the title screen: an Intel HD 2500 has
/// DirectX 12 only through WARP and its GPU through OpenGL (#5185), and a Fermi GPU has no
/// Vulkan or DirectX 12 driver but OpenGL (#3978). DirectX 12 never falls back to Vulkan: the
/// launcher chooses it after Vulkan crashed in the driver (#746), which the game can't catch.
pub(super) fn device_search(
    api: GraphicsApi,
    os: &str,
    arch: &str,
    wgpu_backend: bool,
) -> Vec<(Vec<wgpu::Backends>, bool)> {
    let chosen = backend_order(api, os, arch, wgpu_backend);
    let others: Vec<_> = backend_order(GraphicsApi::Auto, os, arch, wgpu_backend)
        .into_iter()
        .filter(|b| !chosen.contains(b))
        .filter(|&b| !(api == GraphicsApi::Dx12 && b == wgpu::Backends::VULKAN))
        .collect();
    let mut tries = vec![(chosen.clone(), false)];
    if !others.is_empty() {
        tries.push((others, false));
    }
    tries.push((chosen, true));
    tries
}
/// `open_device` on `order`'s backends in turn, taking an adapter of the CPU type
/// (software rendering) only when `allow_cpu`.
async fn open_device_with(
    window: &Arc<Window>,
    offscreen: bool,
    order: &[wgpu::Backends],
    layers: u32,
    allow_cpu: bool,
) -> Result<Opened> {
    let mut failures = Vec::new();
    for (attempt, backends) in order.iter().copied().enumerate() {
        let name = backends_name(backends);
        if attempt > 0 {
            log::warn!("Horizon renderer: trying {name} ({})", failures.join("; "));
        }
        match open_with_retry(backends, window, offscreen, layers, allow_cpu).await {
            Ok(opened) => return Ok(opened),
            Err(e) => failures.push(e),
        }
    }
    Err(format!("{NO_GPU_DRIVER} ({})", failures.join("; ")).into())
}
/// Why a backend gave no device: no adapter at all, which a second try with the
/// adapters wgpu hides can still find, or a failure after an adapter.
enum NoDevice {
    NoAdapter(String),
    Failed(String),
}
/// Opens a device on one backend, and on a backend whose adapters wgpu all hid
/// for reporting no Vulkan conformance version tries it again with them shown:
/// Mesa's hasvk drives Ivy Bridge and Haswell, works, and reports 0.0.0.0, so
/// without this the player is told they have no graphics driver while Vulkan
/// runs (a `gl not compatible with provided surface` fallback, or nothing, is
/// all they get). Returns why the backend gave no device.
async fn open_with_retry(
    backends: wgpu::Backends,
    window: &Arc<Window>,
    offscreen: bool,
    layers: u32,
    allow_cpu: bool,
) -> std::result::Result<Opened, String> {
    match open_backend(backends, window, offscreen, false, layers, allow_cpu).await {
        Err(NoDevice::NoAdapter(hidden)) => {
            log::warn!(
                "Horizon renderer: {} has no conformant adapter ({hidden}); asking again for the \
                 ones wgpu hides as non-conformant",
                backends_name(backends)
            );
            match open_backend(backends, window, offscreen, true, layers, allow_cpu).await {
                Ok(opened) => {
                    let info = opened.adapter.get_info();
                    if is_broken_haswell_vulkan(&info) {
                        log::warn!(
                            "Horizon renderer: skipping {} on Vulkan because Mesa hasvk suffers from \
                             vertex corruption on Haswell; falling back to next backend",
                            info.name
                        );
                        return Err(format!(
                            "Mesa hasvk on Haswell ({}) has broken vertex buffers under Vulkan; falling back",
                            info.name
                        ));
                    }
                    log::warn!(
                        "Horizon renderer: using {} although its driver reports no Vulkan \
                         conformance version",
                        info.name
                    );
                    Ok(opened)
                }
                Err(NoDevice::NoAdapter(e)) | Err(NoDevice::Failed(e)) => Err(e),
            }
        }
        Err(NoDevice::Failed(e)) => Err(e),
        Ok(opened) => Ok(opened),
    }
}
/// Intel Haswell (Gen 7.5) Mesa Vulkan driver (`hasvk`) reports 0.0.0.0 and suffers
/// from severe vertex buffer corruption and polygon tearing (#9912), but its OpenGL driver
/// (`crocus`) is fully conformant and stable. Ivy Bridge (`IVB`) keeps Vulkan (#4059).
pub(crate) fn is_broken_haswell_vulkan(info: &wgpu::AdapterInfo) -> bool {
    info.backend == wgpu::Backend::Vulkan
        && info.vendor == 0x8086
        && (matches!(
            info.device,
            0x0400..=0x042f // Haswell Desktop / Mobile
                | 0x0a00..=0x0a2f // Haswell ULT (Ultrabook)
                | 0x0d00..=0x0d2f // Haswell CRW (Crystal Well)
        ) || ["HSW", "Haswell"].iter().any(|tag| info.name.contains(tag)))
}
/// Opens a device on `backends` alone: its surface, adapter and device, or what
/// stopped it. `allow_nonconformant` is `open_with_retry`'s second try; a
/// software adapter is taken only when `allow_cpu`.
async fn open_backend(
    backends: wgpu::Backends,
    window: &Arc<Window>,
    offscreen: bool,
    allow_nonconformant: bool,
    layers: u32,
    allow_cpu: bool,
) -> std::result::Result<Opened, NoDevice> {
    let name = backends_name(backends);
    let open = |dx12_compiler| {
        wgpu::Instance::new(instance_descriptor(
            backends,
            Some(window),
            allow_nonconformant,
            dx12_compiler,
        ))
    };
    let instance = open(wgpu::Dx12Compiler::Fxc);
    let (surface, adapter) = match surface_and_adapter(&instance, window, offscreen).await {
        Ok(found) => found,
        Err(NoDevice::Failed(e)) => return Err(NoDevice::Failed(format!("{name} surface: {e}"))),
        Err(NoDevice::NoAdapter(e)) => return Err(NoDevice::NoAdapter(format!("{name}: {e}"))),
    };
    // NoAdapter, so `open_with_retry` asks again: with the non-conformant
    // adapters shown, wgpu ranks a real GPU (hasvk on Ivy Bridge or Haswell)
    // above a software one. The second try's error is final either way.
    if !allow_cpu && is_software(adapter.get_info().device_type) {
        return Err(NoDevice::NoAdapter(format!(
            "{name}: only a software adapter ({})",
            adapter.get_info().name
        )));
    }
    // An integrated GPU on DirectX 12 compiles with DXC: the same request
    // again, on an instance that loads the DLL beside the game. Anything that
    // goes wrong keeps the FXC instance.
    let forced = dx12_compiler::forced();
    let dxc = if dx12_compiler::wants_dxc(&adapter.get_info(), forced) {
        open_dxc(&adapter.get_info(), window, offscreen, open).await
    } else {
        None
    };
    let dxc_found = dxc.is_some();
    let (instance, surface, adapter) = dxc.unwrap_or((instance, surface, adapter));
    if adapter.get_info().backend == wgpu::Backend::Dx12 {
        crate::crash::set_detail(
            "DirectX 12 shader compiler",
            match (forced, dxc_found) {
                (true, _) => "WGPU_DX12_COMPILER",
                (false, true) => "DXC",
                (false, false) => "FXC",
            },
        );
    }
    // Adapters that could not present to the window: the next one of this backend
    // that can is tried before the next backend (#13493).
    let mut tried = Vec::new();
    let mut adapter = adapter;
    loop {
        let why = match open_adapter(name, window, surface.as_ref(), &adapter, layers).await {
            Ok(adapted) => {
                return Ok(Opened {
                    instance,
                    surface,
                    adapter,
                    device: adapted.device,
                    queue: adapted.queue,
                    device_lost: adapted.device_lost,
                    configured: adapted.configured,
                });
            }
            Err(AdapterFailure::Failed(e)) => return Err(NoDevice::Failed(e)),
            Err(AdapterFailure::CannotPresent(why)) => why,
        };
        let info = adapter.get_info();
        tried.push(info.clone());
        let others = instance.enumerate_adapters(wgpu::Backends::all()).await;
        let choices: Vec<_> = others
            .iter()
            .map(|other| AdapterChoice {
                kind: other.get_info().device_type,
                presents: surface
                    .as_ref()
                    .is_some_and(|s| !s.get_capabilities(other).formats.is_empty()),
                tried: tried.contains(&other.get_info()),
            })
            .collect();
        let Some(next) = pick_adapter(&choices, allow_cpu) else {
            log::warn!(
                "Horizon renderer: skipping {} on {name}: it can't present to the window \
                 ({why}), and no other {name} adapter can",
                info.name
            );
            return Err(NoDevice::Failed(format!(
                "{name}: {} can't present to the window ({why})",
                info.name
            )));
        };
        adapter = others[next].clone();
        log::warn!(
            "Horizon renderer: skipping {} on {name}: it can't present to the window ({why}); \
             trying {}",
            info.name,
            adapter.get_info().name
        );
    }
}
/// A surface for `window` (none `offscreen`) and the adapter `open_backend`
/// asks `instance` for. `Failed` is the surface's error, `NoAdapter` the
/// request's.
async fn surface_and_adapter(
    instance: &wgpu::Instance,
    window: &Arc<Window>,
    offscreen: bool,
) -> std::result::Result<(Option<wgpu::Surface<'static>>, wgpu::Adapter), NoDevice> {
    let surface = if offscreen {
        None
    } else {
        Some(
            instance
                .create_surface(window.clone())
                .map_err(|e| NoDevice::Failed(e.to_string()))?,
        )
    };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: surface.as_ref(),
            force_fallback_adapter: false,
            ..Default::default()
        })
        .await
        .map_err(|e| NoDevice::NoAdapter(e.to_string()))?;
    Ok((surface, adapter))
}
/// The DirectX 12 instance, surface and adapter with DXC for the integrated
/// GPU `fxc` (`gfx/dx12_compiler.rs`), `open` making the instance; `None`, with
/// one line in the log, when the DLL is missing, doesn't load, or the instance
/// offers another GPU: the FXC instance stays.
async fn open_dxc(
    fxc: &wgpu::AdapterInfo,
    window: &Arc<Window>,
    offscreen: bool,
    open: impl FnOnce(wgpu::Dx12Compiler) -> wgpu::Instance,
) -> Option<(
    wgpu::Instance,
    Option<wgpu::Surface<'static>>,
    wgpu::Adapter,
)> {
    let Some(dll) = dx12_compiler::bundled_dll() else {
        log::info!(
            "Horizon renderer: no {} beside the game; DirectX 12 compiles shaders with FXC",
            dx12_compiler::DLL
        );
        return None;
    };
    let instance = open(dx12_compiler::dxc(&dll));
    let why = match surface_and_adapter(&instance, window, offscreen).await {
        Ok((surface, adapter)) if dx12_compiler::same_adapter(fxc, &adapter.get_info()) => {
            log::info!(
                "Horizon renderer: DirectX 12 compiles shaders with DXC ({}) on this integrated GPU",
                dll.display()
            );
            return Some((instance, surface, adapter));
        }
        Ok((_, adapter)) => format!("it offered another GPU, {}", adapter.get_info().name),
        Err(NoDevice::Failed(e) | NoDevice::NoAdapter(e)) => e,
    };
    log::warn!(
        "Horizon renderer: DXC from {} didn't load ({why}); DirectX 12 compiles shaders with FXC",
        dll.display()
    );
    None
}
/// What to tell a player whose system offers no usable graphics adapter.
/// Vulkan is loaded at run time (the game links no graphics library), so a
/// missing driver shows here rather than as a missing-DLL dialog.
#[cfg(target_os = "windows")]
const NO_GPU_DRIVER: &str = "No graphics driver with Vulkan, DirectX 12 or OpenGL found: Horizon needs \
     a GPU driver with one of them. Install the latest driver for your NVIDIA, AMD or Intel GPU and \
     start the game again";
#[cfg(target_os = "macos")]
const NO_GPU_DRIVER: &str = "No Metal graphics device found";
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
const NO_GPU_DRIVER: &str = "No Vulkan or OpenGL graphics driver found: Horizon needs the Vulkan loader \
     (libvulkan.so.1) and a Vulkan driver for your GPU (Mesa's, or NVIDIA's), or OpenGL through EGL";
