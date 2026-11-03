# Roadway Edge Vision

**Rust + ONNX Runtime 边缘 AI 视觉检测** —— 用来替掉「Python + PyTorch + Anomalib + OpenVINO」那一整套。

> 越南工厂工控机部署用。目标:同样的检测效果,部署体积从「几个 GB」缩到「一个 911 KB 二进制 + 一个 20 MB 运行时」。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `src/main.rs` | 推理器(预处理 → ONNX Runtime → 判定 → 可选热图导出) |
| `Cargo.toml` | 依赖:`ort` 2.0.0-rc.14 + `image` 0.25 |
| `deploy/` | Windows 工控机部署包(批处理 + 配置 + 中文指南) |
| `.github/workflows/` | CI 自动出 Windows x64 / Linux x64 二进制 |

## 关键设计

**动态加载 ONNX Runtime,不用静态链接。**
预编译的 `libonnxruntime.a` 引用了 CoreML 内部符号 `__isPlatformVersionAtLeast`,
只有完整 Xcode 的机器才能链上;而且静态链接的运行时换版本要重新编译。
动态加载后:换 DLL 就能换推理引擎版本,也更贴近产线部署(装个 runtime 即可)。

**执行后端可切:**
- `cpu` —— 通用(Windows 工控机用这个)
- `coreml` —— Apple Silicon GPU/ANE(实测 116ms,比 Python OpenVINO 还快 24%)
- `xnnpack` —— 纯 CPU 优化
- `directml` —— Windows 核显/独显(工控机常用)

**预处理有坑:** Anomalib 的 `to_onnx()` 把 ImageNet 归一化烘进了计算图,
所以**只送 0-255 原始像素**,自己再归一化会让分数差 100 倍。

## 实测(2026-11-01,Apple M4,30 张 MVTecAD cable 真样本)

| 方案 | 平均耗时 | FPS |
|---|---:|---:|
| Python + OpenVINO | 144.2 ms | 6.94 |
| Rust + ONNX Runtime CPU | 349.2 ms | 2.86 |
| Rust + CoreML | 116.4 ms | 8.59 |

**与 PyTorch 真值的最大相对误差:0.000038%**(float32 舍入级,数学等价)。

## 怎么用

```bash
# 单张
./aoi --model model.onnx --image cable.png

# 批量
./aoi --model model.onnx --dir samples/ --threshold 43.7255

# 持续监控相机目录
./aoi --model model.onnx --dir watch/ --watch

# 导出热图数据(生成可视化用)
./aoi --model model.onnx --dir samples/ --dump-amap out_amap/
```

**模型加载:** 优先 `ORT_DYLIB_PATH` 环境变量 → 可执行文件同目录 → 系统路径。
Windows 上把 `onnxruntime.dll` 和 `aoi.exe` 放同一文件夹,双击就能跑,不用配环境变量。

## 出 Windows 二进制

推 tag 触发 CI(`windows-latest` + MSVC,带 ASLR/DEP):

```bash
git tag v0.1.0 && git push origin v0.1.0
```

或手动触发:Actions → Build Windows x64 → Run workflow。

产物:`aoi-windows-x64` artifact(含 exe + 部署脚本 + SHA256)。

## 本地交叉编译(macOS → Windows)

Homebrew 的 mingw-w64 在 macOS ARM 上读不了 PE 目标格式(它的 `ld` 报
`bad string table size`),用 **Zig 当链接器**:

```bash
brew install zig
cat > /tmp/zigcc <<'EOF'
#!/bin/bash
ARGS=()
for a in "$@"; do
  case "$a" in
    -lmsvcrt)  ARGS+=("-l:libmsvcrt.a") ;;
    -lpthread) ARGS+=("-l:libpthread.a") ;;
    *)         ARGS+=("$a") ;;
  esac
done
exec zig cc -target x86_64-windows-gnu "${ARGS[@]}"
EOF
chmod +x /tmp/zigcc
mkdir -p .cargo && cat > .cargo/config.toml <<'EOF'
[target.x86_64-pc-windows-gnu]
linker = "/tmp/zigcc"
EOF
cargo build --release --target x86_64-pc-windows-gnu
```

> CI 走 MSVC,产物带 ASLR/DEP;本地 Zig 版没有(DYNAMIC_BASE 未设),
> **正式交付用 CI 产物**。

## License

MIT。依赖:`ort`(MIT/Apache-2.0)、`image`(MIT/Apache-2.0)、ONNX Runtime(MIT)。

---

*Roadway Vietnam · roadwayvn.com · zhang@roadwayvn.com*
