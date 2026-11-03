# Roadway AOI · Windows 工控机部署包

> Rust 边缘视觉检测 · 线束(cable)外观检测 · 2026-11-01
> 适用于 Windows 10 / 11 / Server 2019+ / 工控机 IPC

---

## 一、这是什么

一个**不需要装 Python 的外观检测程序**。传统做法要在工控机上装 Python + PyTorch +
Anomalib + OpenVINO(几个 GB),这里只需要:

```
AOI-Windows/
├── aoi.exe                    ← 检测程序(约 1 MB)
├── onnxruntime.dll            ← 推理引擎(16 MB)
├── onnxruntime_providers_shared.dll
├── model.onnx                 ← 检测模型(364 MB)
├── config.toml                ← 参数(阈值 / 后端 / 线程)
├── deploy_windows.bat         ← 一键部署向导
├── samples/                   ← 测试样本
└── watch/                     ← 相机图片监控目录
```

**把整个文件夹拷到工控机,双击 `deploy_windows.bat` 就行。**

---

## 二、三步部署

### 第 1 步:拷贝文件

整个 `AOI-Windows` 文件夹拷到工控机(建议 `D:\AOI-Windows`)。

> **注意**:必须**整个文件夹一起拷**。`aoi.exe` 和 `onnxruntime.dll` 必须在同一目录。

### 第 2 步:装 VC++ 运行库(一次性)

双击 `deploy_windows.bat` 会自动检查。如果报缺 `vcruntime140.dll`,去这个地址装:

https://aka.ms/vs/17/release/vc_redist.x64.exe

装完不用重启。

### 第 3 步:双击 `deploy_windows.bat`

脚本会依次:

1. 检查文件完整性
2. 显示系统版本 + CPU 核心数
3. 检查 VC++ 运行库
4. **自检**:加载模型 + 跑 5 张样本,打印每张的分数和耗时
5. 创建开机自启计划任务(`RoadwayAOI`)
6. 打印使用说明

**自检通过就算装好了。**

---

## 三、日常使用

### 批量检测一个文件夹

```bat
aoi.exe --model model.onnx --dir samples --threshold 43.7255
```

输出示例:

```
01_good_000.png              score=31239.99 am_max=30813.97 label=1 NG infer=349.2ms
02_good_001.png              score=29753.67 am_max=29253.68 label=1 NG infer=331.5ms
```

### 监控相机输出目录(产线模式)

```bat
aoi.exe --model model.onnx --dir watch --threshold 43.7255 --watch
```

程序轮询 `watch\` 目录,发现新图片就检测,结果写 `results.csv`。

**相机直接存到 `watch\` 就行,不需要别的采集软件。**

### 单张测试

```bat
aoi.exe --model model.onnx --image D:\test\cable.png
```

### 参数一览

| 参数 | 说明 | 默认 |
|---|---|---|
| `--model <路径>` | 模型文件 | 必填 |
| `--image <路径>` | 单张图片 | — |
| `--dir <路径>` | 批量目录 | — |
| `--threshold <数>` | 判定阈值 | `43.7255` |
| `--ep cpu\|coreml\|xnnpack` | 执行后端 | `cpu` |
| `--threads <数>` | 线程数,0=自动 | `0` |
| `--watch` | 持续监控目录 | 关 |
| `--dump-amap <目录>` | 导出热图数据 | 关 |

---

## 四、⚠️ 必须知道的两件事

### 1. 阈值必须重新标定

**当前默认阈值 `43.7255` 不能直接上产线。**

模型实际输出分数在 **30000 量级**,而阈值是 **43.7** —— 会把所有图都判成 NG,
包括良品。

原因:模型 checkpoint 里的归一化参数(`image_min` / `image_max`)在导出 ONNX 时
没有正确写入后处理节点,导致分数尺度对不上。这是 anomalib 2.6.2 的已知问题
(Issue #3726)。

**解决办法(必须做):**

1. 收集 **100 张以上 NG 样本**(各种缺陷类型都要有)
2. 收集 **100 张以上 OK 样本**
3. 用 `--dump-amap` 导出全部 `am_max` 值
4. 取 OK 和 NG 两组分数分布的中间值作为新阈值
5. 写进 `config.toml`

**没做这一步就上产线,结果是「全部报警」或「全部放行」,两个都是事故。**

### 2. 速度预期

| 场景 | 预期耗时 |
|---|---|
| 首次启动(加载 364MB 模型) | 10-30 秒 |
| 单张推理 | 约 350 ms(Intel CPU) |
| 产线节拍 1500 ms | 冗余 4.3 倍,够用 |

**实测数据来自 Apple M4,Intel 工控机需现场复测。**

Windows 上还有个额外优势:如果工控机有核显,可以装 DirectML 版本
(`Microsoft.ML.OnnxRuntime.DirectML`)用 GPU 加速,需要现场测试收益。

---

## 五、故障排查

| 现象 | 原因 | 解决 |
|---|---|---|
| 提示找不到 `onnxruntime.dll` | DLL 不在 exe 同目录 | 把 DLL 拷到同目录 |
| 提示缺 `vcruntime140.dll` | 没装 VC++ 运行库 | 装 vc_redist.x64.exe |
| 加载模型很慢 / 卡住 | 磁盘慢或模型损坏 | 确认 model.onnx 是 364MB,拷到 SSD |
| 全部判 NG | 阈值没标定 | 见上文第 4 节 |
| 中文显示乱码 | 系统区域设置 | 控制面板 → 区域 → 改成 UTF-8 / 越南语 |

日志:程序所有输出都是 stderr,重定向到文件即可:

```bat
aoi.exe --model model.onnx --dir watch --watch > run.log 2>&1
```

---

## 六、开机自启

`deploy_windows.bat` 用**计划任务**实现,不需要管理员权限,不需要装任何服务软件。

查看:`schtasks /query /tn "RoadwayAOI"`
删除:`schtasks /delete /tn "RoadwayAOI" /f`
手动启动:`schtasks /run /tn "RoadwayAOI"`

---

## 七、想自己重新编译

```bash
# 需要 Rust + mingw-w64
rustup target add x86_64-pc-windows-gnu
cargo build --release --target x86_64-pc-windows-gnu
```

产物在 `target/x86_64-pc-windows-gnu/release/aoi.exe`。

> `ort` crate 用 `load-dynamic` 特性 —— **不要**改成静态链接,
> 官方预编译的 `onnxruntime.lib` 依赖完整 MSVC 工具链,在只有 CommandLineTools
> 的机器上会链接失败。动态加载反而更适合产线部署:换 DLL 就能换推理引擎版本。

---

*Roadway · 越南制造业 AI 落地 · roadwayvn.com*
