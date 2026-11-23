//! CUDA DLL 的窄接口；常规 gpu-mine 编译不要求 CUDA 工具链。
use std::{
    ffi::{c_char, c_void, CStr},
    os::windows::ffi::OsStrExt,
    path::Path,
    ptr,
    time::{Duration, Instant},
};
use crate::{
    core::{self, Evaluation, Thresholds},
    gpu_input::GpuInput,
    mining_gpu::{GpuBatch, GpuRecord},
};

type Open = unsafe extern "C" fn(*const u32, *const u32, u32, u32, u32, *const [f32; 2], u32, *mut *mut c_void) -> i32;
type Run = unsafe extern "C" fn(*mut c_void, u64, u32, u32, u32, u32, u32, u32, u32, *mut Readback) -> i32;
type Close = unsafe extern "C" fn(*mut c_void);
type ErrorText = unsafe extern "C" fn(i32) -> *const c_char;

#[repr(C)]
#[derive(Default)]
struct Readback {
    count: u32,
    prop_pass: u32,
    scored: u32,
    shadows: u32,
    kernel_ms: f32,
    records: *const RawRecord,
}

#[repr(C)]
struct RawRecord {
    index: u32,
    props: [u32; 8],
    ids: [u32; 16],
    freqs: [u32; 16],
    shadow: i32,
    xp: f32,
    xd: f32,
}
const _: () = assert!(std::mem::size_of::<RawRecord>() == 176);

#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryExW(path: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *const c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
}

struct Library(*mut c_void);
impl Library {
    fn load(path: &Path) -> Result<Self, String> {
        let path = path.canonicalize().map_err(|e| format!("CUDA DLL {}: {e}", path.display()))?;
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
        // 绝对路径；依赖只从 DLL 所在目录与 Windows 默认安全目录解析。
        let handle = unsafe { LoadLibraryExW(wide.as_ptr(), ptr::null_mut(), 0x100 | 0x1000) };
        if handle.is_null() {
            return Err(format!("加载 CUDA DLL 失败：{}", std::io::Error::last_os_error()));
        }
        Ok(Self(handle))
    }
    fn symbol(&self, name: &CStr) -> Result<*const c_void, String> {
        let address = unsafe { GetProcAddress(self.0, name.as_ptr()) };
        if address.is_null() {
            return Err(format!("CUDA DLL 缺少符号：{name:?}"));
        }
        Ok(address)
    }
}
impl Drop for Library {
    fn drop(&mut self) {
        unsafe {
            FreeLibrary(self.0);
        }
    }
}

pub struct Miner {
    _library: Library,
    handle: *mut c_void,
    invoke: Run,
    close: Close,
    error_text: ErrorText,
    lanes: u32,
    mode: u32,
}
impl Miner {
    pub fn new(path: &Path, input: &GpuInput, capacity: u32, lanes: u32, mode: u32) -> Result<Self, String> {
        if input.encoding != 0 || input.template_map.len() != input.name_len as usize {
            return Err("XP/XD CUDA 后端只支持 GPU 生成的 Base16384".into());
        }
        let library = Library::load(path)?;
        // 这些签名与配套 cuda-mine.cu 的 x64 导出一一对应。
        let (open, invoke, close, error_text) = unsafe {
            (
                std::mem::transmute::<*const c_void, Open>(library.symbol(c"cuda_mine_open")?),
                std::mem::transmute::<*const c_void, Run>(library.symbol(c"cuda_mine_run")?),
                std::mem::transmute::<*const c_void, Close>(library.symbol(c"cuda_mine_close")?),
                std::mem::transmute::<*const c_void, ErrorText>(library.symbol(c"cuda_mine_error")?),
            )
        };
        let weights = core::model_weights();
        assert_eq!(weights.len(), 1035);
        let mut handle = ptr::null_mut();
        let status = unsafe {
            open(
                input.prefix_state.as_ptr(),
                input.template_map.as_ptr(),
                input.name_len,
                input.prefix_steps,
                input.prefix_j,
                weights.as_ptr(),
                capacity,
                &mut handle,
            )
        };
        check(status, error_text)?;
        Ok(Self {
            _library: library,
            handle,
            invoke,
            close,
            error_text,
            lanes,
            mode,
        })
    }

    pub fn run(&mut self, first: u64, count: u32, t: Thresholds, capture: bool) -> Result<GpuBatch, String> {
        let start = Instant::now();
        let mut raw = Readback::default();
        let status = unsafe {
            (self.invoke)(
                self.handle,
                first,
                count,
                t.prop.min(10000),
                t.xp,
                t.xd,
                u32::from(capture),
                self.lanes,
                self.mode,
                &mut raw,
            )
        };
        check(status, self.error_text)?;
        if raw.count > count
            || raw.prop_pass > count
            || raw.scored > raw.prop_pass
            || raw.shadows > raw.scored
            || raw.records.is_null()
        {
            return Err("CUDA 返回了无效的计数或指针".into());
        }
        // DLL 已同步 stream，记录内存至少存活到下次调用。
        let data = unsafe { std::slice::from_raw_parts(raw.records, raw.count as usize) };
        let mut records = Vec::with_capacity(data.len());
        let mut max_error = [0.0f64; 2];
        for row in data {
            let approximate = [row.xp, row.xd];
            if row.index >= count || row.ids.iter().any(|&id| id >= 40) || !approximate.iter().all(|v| v.is_finite()) {
                return Err(format!("CUDA 返回无效记录：批内下标 {}", row.index));
            }
            // 和 WGSL 的 --verify 相同：所有回传候选用同源 f64 精算再做阈值判定。
            let exact = core::scores(&row.props, &row.ids, &row.freqs, row.shadow);
            for i in 0..2 {
                let error = (exact[i] - f64::from(approximate[i])).abs();
                if error >= 10.0 {
                    return Err(format!("CUDA f32 误差 {error} ≥ 10：批内下标 {}, 模型 {i}", row.index));
                }
                max_error[i] = max_error[i].max(error);
            }
            let value = Evaluation {
                props: row.props,
                skill_ids: row.ids,
                skill_freqs: row.freqs,
                shadow_numerator: row.shadow,
                xp: exact[0],
                xd: exact[1],
            };
            if capture || value.matches(t) {
                records.push(GpuRecord {
                    index: row.index as usize,
                    value,
                    approximate,
                });
            }
        }
        records.sort_unstable_by_key(|r| r.index);
        if records.windows(2).any(|pair| pair[0].index == pair[1].index) {
            return Err("CUDA 返回重复记录".into());
        }
        let kernel_time = Duration::try_from_secs_f64(f64::from(raw.kernel_ms) / 1000.0).map_err(|e| e.to_string())?;
        Ok(GpuBatch {
            records,
            prop_pass: raw.prop_pass,
            scored: raw.scored,
            shadows: raw.shadows,
            kernel_time: Some(kernel_time),
            wall_time: start.elapsed(),
            max_error,
            refined: data.len(),
        })
    }
}
impl Drop for Miner {
    fn drop(&mut self) {
        unsafe {
            (self.close)(self.handle);
        }
    }
}
fn check(status: i32, error_text: ErrorText) -> Result<(), String> {
    if status == 0 {
        return Ok(());
    }
    let p = unsafe { error_text(status) };
    let message = if p.is_null() {
        "未知错误".into()
    } else {
        unsafe { CStr::from_ptr(p) }.to_string_lossy()
    };
    Err(format!("CUDA {status}: {message}"))
}
