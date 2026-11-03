//! Rust 产线外观检测推理器(AOI)
//!
//! 目标:替掉 Python + Anomalib + OpenVINO 三件套,让工控机只装一个二进制。
//! 模型:PatchCore 导出的 ONNX(输入 [1,3,256,256],输出 pred_score/pred_label/anomaly_map/pred_mask)
//!
//! 用法:
//!   aoi --model model.onnx --image a.jpg [--threshold 0.5] [--repeat 10]
//!   aoi --model model.onnx --dir samples/ --threshold 0.5
//!
//! 2026-11-01 实测:Mac mini M4,与 Python OpenVINO 134ms/张 对比。

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;
use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// 图片尺寸 —— 必须与模型导出的 input_size 一致
const SIZE: usize = 256;

/// 预处理说明(2026-11-01 实测):
/// - resize 到 256x256(双线性),送 0-255 原始像素
/// - ImageNet 归一化 (x-mean)/std **已经烘进 ONNX 图**,
///   证据:图里有 /pre_processor/export_transform/Sub 和 /Div 节点,
///   常量分别是 [0.485 0.456 0.406] 和 [0.229 0.224 0.225]
/// - 自己再做一次归一化会导致分数完全对不上(PyTorch 侧实测 30813 vs Rust 侧 330)
const SIZE_INPUT_SCALE: f32 = 1.0;

struct Args {
    model: PathBuf,
    images: Vec<PathBuf>,
    threshold: f32,
    repeat: usize,
    threads: usize,
    ep: String,
    dump_amap: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut model = None;
    let mut images = Vec::new();
    let mut threshold = 0.5f32;
    let mut repeat = 1usize;
    let mut threads = 0usize;
    let mut ep = "cpu".to_string();
    let mut dump_amap: Option<String> = None;

    let argv: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                i += 1;
                model = argv.get(i).map(PathBuf::from);
            }
            "--image" => {
                i += 1;
                if let Some(p) = argv.get(i) {
                    images.push(PathBuf::from(p));
                }
            }
            "--dir" => {
                i += 1;
                if let Some(d) = argv.get(i) {
                    if let Ok(rd) = std::fs::read_dir(d) {
                        let mut v: Vec<PathBuf> = rd
                            .filter_map(|e| e.ok().map(|e| e.path()))
                            .filter(|p| {
                                let s = p.to_string_lossy().to_lowercase();
                                s.ends_with(".jpg") || s.ends_with(".jpeg") || s.ends_with(".png")
                            })
                            .collect();
                        v.sort();
                        images.extend(v);
                    }
                }
            }
            "--threshold" => {
                i += 1;
                threshold = argv.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.5);
            }
            "--repeat" => {
                i += 1;
                repeat = argv.get(i).and_then(|s| s.parse().ok()).unwrap_or(1);
            }
            "--threads" => {
                i += 1;
                threads = argv.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "--ep" => {
                i += 1;
                ep = argv.get(i).cloned().unwrap_or_else(|| "cpu".to_string());
            }
            "--dump-amap" => {
                i += 1;
                dump_amap = argv.get(i).cloned();
            }
            "--help" | "-h" => {
                println!("aoi --model <model.onnx> (--image <jpg> | --dir <dir>) [--threshold 0.5] [--repeat N] [--threads N]");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    Ok(Args {
        model: model.ok_or("缺 --model")?,
        images,
        threshold,
        repeat,
        threads,
        ep,
        dump_amap,
    })
}

/// 解码 JPEG → RGB8 → resize 到 256x256(双线性)→ 归一化 → CHW f32
fn preprocess(path: &Path) -> Result<Vec<f32>, String> {
    // 用 image crate 统一解 JPEG 和 PNG
    // (2026-11-01 实测:MVTecAD 样本是 .png,只支持 JPEG 会全部解码失败)
    let img = image::open(path).map_err(|e| format!("解码失败 {}: {e}", path.display()))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    if w == 0 || h == 0 {
        return Err(format!("图片尺寸为 0: {}", path.display()));
    }
    let src = rgb.as_raw(); // &[u8] RGB8

    // 双线性 resize 到 SIZE x SIZE
    let mut out = vec![0f32; SIZE * SIZE * 3];
    let sx = w as f32 / SIZE as f32;
    let sy = h as f32 / SIZE as f32;
    for y in 0..SIZE {
        let fy = (y as f32 + 0.5) * sy - 0.5;
        let y0 = fy.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(h - 1);
        let wy = fy - y0 as f32;
        for x in 0..SIZE {
            let fx = (x as f32 + 0.5) * sx - 0.5;
            let x0 = fx.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(w - 1);
            let wx = fx - x0 as f32;
            for c in 0..3 {
                let p00 = src[(y0 * w + x0) * 3 + c] as f32;
                let p01 = src[(y0 * w + x1) * 3 + c] as f32;
                let p10 = src[(y1 * w + x0) * 3 + c] as f32;
                let p11 = src[(y1 * w + x1) * 3 + c] as f32;
                let top = p00 * (1.0 - wx) + p01 * wx;
                let bot = p10 * (1.0 - wx) + p11 * wx;
                let v = top * (1.0 - wy) + bot * wy;
                // 关键(2026-11-01 实测):ONNX 图里已经内嵌了 (x - mean) / std
                // (节点名 /pre_processor/export_transform/Sub 和 /Div)
                // 所以这里只送 0-255 的原始像素,归一化交给 ONNX 做 —— 自己做会归一化两遍
                out[c * SIZE * SIZE + y * SIZE + x] = v;
            }
        }
    }
    Ok(out)
}

    // 动态加载 ONNX Runtime 动态库。
// 路径优先级:ORT_DYLIB_PATH 环境变量 → 可执行文件同目录 → 常见安装位置 → ort 默认搜索路径。
// 「可执行文件同目录」这一条是 Windows 工控机部署的关键:
// 把 onnxruntime.dll 和 aoi.exe 放在同一个文件夹,双击就能跑,不需要配任何环境变量。
fn load_env() -> Result<ort::environment::Environment, Box<dyn std::error::Error>> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()));

    let mut candidates: Vec<String> = Vec::new();
    // 1) 环境变量优先
    if let Ok(v) = std::env::var("ORT_DYLIB_PATH") {
        if !v.is_empty() {
            candidates.push(v);
        }
    }
    // 2) 可执行文件同目录(Windows 部署主力路径)
    if let Some(dir) = &exe_dir {
        for name in [
            "onnxruntime.dll",            // Windows
            "libonnxruntime.dylib",       // macOS
            "libonnxruntime.so",          // Linux
            "libonnxruntime.so.1",        // Linux 版本号后缀
        ] {
            candidates.push(dir.join(name).to_string_lossy().to_string());
        }
    }
    // 3) 常见系统安装位置
    candidates.extend([
        "/opt/homebrew/lib/libonnxruntime.dylib".to_string(),
        "/usr/local/lib/libonnxruntime.dylib".to_string(),
        "/opt/homebrew/lib/libonnxruntime.1.dylib".to_string(),
        "/usr/lib/libonnxruntime.so".to_string(),
    ]);
    for c in candidates {
        if Path::new(&c).exists() {
            eprintln!("# onnxruntime: {c}");
            let b = ort::init_from(&c)?;
            return Ok(b.with_name("aoi").build()?);
        }
    }
    // 兜底:让 ort 自己在系统路径里找(Windows 会找当前目录 + PATH)
    eprintln!("# onnxruntime: 用 ort 默认搜索路径");
    Ok(ort::init().with_name("aoi").build()?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(|e| format!("{e}\n用法: aoi --model <model.onnx> --image <jpg>"))?;

    if args.images.is_empty() {
        eprintln!("没有输入图片。用 --image 或 --dir。");
        std::process::exit(2);
    }

    // 加载模型
    let t_load = Instant::now();
    let env = load_env()?;
    let mut builder = Session::builder(&env)?.with_optimization_level(GraphOptimizationLevel::Level3)?;
    if args.threads > 0 {
        builder = builder.with_intra_threads(args.threads)?;
    }
    // execution provider:cpu(默认) | coreml(Apple Silicon GPU/ANE) | xnnpack(纯 CPU 优化)
    match args.ep.as_str() {
        "coreml" => {
            builder = builder.with_execution_providers([ort::ep::CoreML::default().build()])?;
            eprintln!("# EP: CoreML");
        }
        "xnnpack" => {
            builder = builder.with_execution_providers([ort::ep::XNNPACK::default().build()])?;
            eprintln!("# EP: XNNPACK");
        }
        "directml" => {
            // Windows 工控机专用:用核显/独显跑 GPU 加速,不依赖 CUDA
            // 需要把 DirectML.dll 和 onnxruntime.dll 放在同一目录
            // (官方包:Microsoft.ML.OnnxRuntime.DirectML)
            builder = builder.with_execution_providers([ort::ep::DirectML::default().build()])?;
            eprintln!("# EP: DirectML");
        }
        _ => {
            eprintln!("# EP: CPU");
        }
    }
    let mut session = builder.commit_from_file(args.model.as_path())?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

    println!("# Roadway Rust AOI 推理器");
    println!("model: {}", args.model.display());
    println!("images: {}", args.images.len());
    println!("load_ms: {load_ms:.1}");
    println!("threads: {}", if args.threads > 0 { args.threads.to_string() } else { "auto".into() });
    println!();

    let mut all_times: Vec<f64> = Vec::new();
    let mut rows: Vec<(String, f32, bool, f64)> = Vec::new();

    for img_path in &args.images {
        let input = match preprocess(img_path) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("跳过 {e}");
                continue;
            }
        };

        for r in 0..args.repeat {
            let t0 = Instant::now();
            let outputs = session.run(ort::inputs![
                "input" => TensorRef::from_array_view(([1usize, 3usize, SIZE, SIZE], &input[..]))?
            ])?;
            let ms = t0.elapsed().as_secs_f64() * 1000.0;

            if r == 0 {
                let (_s_shape, score_data) = outputs["pred_score"].try_extract_tensor::<f32>()?;
                let score: f32 = score_data[0];
                // anomaly_map 最大/最小值 —— 用来和 PyTorch 端做数学等价验证
                let am_shape = outputs["anomaly_map"].try_extract_tensor::<f32>()?;
                let am_data = am_shape.1;
                let am_max = am_data.iter().cloned().fold(f32::MIN, f32::max);
                let am_min = am_data.iter().cloned().fold(f32::MAX, f32::min);
                // 先取 name / label,dump 块要用
                let name = img_path.file_name().unwrap_or_default().to_string_lossy().to_string();
                let (_l_shape_b, label_data_b) = outputs["pred_label"].try_extract_tensor::<bool>()?;
                let name_disp = name.clone();
                let label_bool: bool = label_data_b[0];
                let label: i64 = if label_bool { 1 } else { 0 };
                // 导出 anomaly_map 原始 f32 数据 + JSON 元信息(给 Python 侧画热图)
                if let Some(dir) = &args.dump_amap {
                    let d = Path::new(dir);
                    std::fs::create_dir_all(d).ok();
                    let stem = img_path.file_stem().unwrap_or_default().to_string_lossy().to_string();
                    // anomaly_map
                    let (am_shape, am_data) = outputs["anomaly_map"].try_extract_tensor::<f32>()?;
                    let mut bin = Vec::with_capacity(am_data.len() * 4);
                    for v in am_data {
                        bin.extend_from_slice(&v.to_le_bytes());
                    }
                    std::fs::write(d.join(format!("{stem}.amap.f32")), &bin).ok();
                    let dims: Vec<i64> = am_shape.iter().copied().collect();
                    let meta = format!(
                        "{{\"name\":\"{}\",\"score\":{},\"label\":{},\"dims\":[{}],\"am_min\":{},\"am_max\":{},\"src\":\"{}\"}}",
                        name_disp, score, label,
                        dims.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","),
                        am_min, am_max,
                        img_path.display().to_string().replace('\\', "/")
                    );
                    std::fs::write(d.join(format!("{stem}.json")), meta).ok();
                }
                // pred_label 在 ONNX 里是 BOOL 类型(不是 i64)—— 2026-11-01 实测踩到
                let is_ng = label == 1 || score > args.threshold;
                rows.push((name, score, is_ng, ms));
                println!(
                    "{:<32} score={score:.4} am_max={am_max:.4} am_min={am_min:.4} label={label} {} infer={ms:.1}ms",
                    name_disp,
                    if is_ng { "NG" } else { "OK" }
                );
            }
            all_times.push(ms);
        }
    }

    if !all_times.is_empty() {
        let n = all_times.len() as f64;
        let sum: f64 = all_times.iter().sum();
        let mean = sum / n;
        let mut sorted = all_times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = sorted[(sorted.len() as f64 * 0.50) as usize % sorted.len()];
        let p95 = sorted[(sorted.len() as f64 * 0.95) as usize % sorted.len()];
        let min = sorted[0];
        let max = sorted[sorted.len() - 1];

        println!();
        println!("# 性能汇总 (n={})", all_times.len());
        println!("mean_ms: {mean:.1}");
        println!("p50_ms: {p50:.1}");
        println!("p95_ms: {p95:.1}");
        println!("min_ms: {min:.1}");
        println!("max_ms: {max:.1}");
        println!("fps: {:.2}", 1000.0 / mean);

        let ng = rows.iter().filter(|r| r.2).count();
        println!();
        println!("# 判定汇总 threshold={}", args.threshold);
        println!("total: {}", rows.len());
        println!("ng: {ng}");
        println!("ok: {}", rows.len() - ng);
    }

    Ok(())
}
