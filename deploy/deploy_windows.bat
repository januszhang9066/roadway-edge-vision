@echo off
chcp 65001 > nul
REM ============================================================
REM  Roadway · Rust 边缘视觉检测 —— Windows 工控机部署脚本
REM  适用于:Windows 10/11 / Windows Server 2019+ / 工控机 IPC
REM  日期:2026-11-01
REM ============================================================

setlocal
cd /d "%~dp0"

echo.
echo ==========================================
echo   Roadway AOI · Windows 部署向导
echo ==========================================
echo.

REM --- 检查文件是否齐全 ---
echo [1/6] 检查必需文件...
set MISSING=0

if not exist "aoi.exe" (
  echo   [缺失] aoi.exe
  set MISSING=1
)
if not exist "onnxruntime.dll" (
  echo   [缺失] onnxruntime.dll
  set MISSING=1
)
if not exist "model.onnx" (
  echo   [缺失] model.onnx
  set MISSING=1
)
if not exist "config.toml" (
  echo   [缺失] config.toml
  set MISSING=1
)

if "%MISSING%"=="1" (
  echo.
  echo   缺少必需文件,请把整个文件夹完整拷贝过来。
  echo.
  pause
  exit /b 1
)
echo   [OK] 全部文件齐全
echo.

REM --- 系统信息 ---
echo [2/6] 检测系统环境...
for /f "tokens=*" %%a in ('wmic os get Caption,Version,OSArchitecture /value 2^>nul ^| find "="') do echo   %%a
echo   处理器核心数: %NUMBER_OF_PROCESSORS%
echo.

REM --- Visual C++ 运行库检查 ---
echo [3/6] 检查 VC++ 运行库...
where /q vcruntime140.dll
if errorlevel 1 (
  echo   [警告] 未检测到 vcruntime140.dll
  echo   请安装 VC++ 2015-2022 x64 运行库:
  echo   https://aka.ms/vs/17/release/vc_redist.x64.exe
  echo.
) else (
  echo   [OK] VC++ 运行库已安装
)
echo.

REM --- 自检 ---
echo [4/6] 运行自检(加载模型 + 测推理速度)...
echo   首次运行需要加载 364MB 模型,可能耗时 10-30 秒...
echo.
aoi.exe --model model.onnx --dir samples --threshold 43.7255
if errorlevel 1 (
  echo.
  echo   [失败] 自检未通过。请检查:
  echo     1) onnxruntime.dll 是否和 aoi.exe 在同一文件夹
  echo     2) VC++ 运行库是否已安装
  echo     3) 模型文件 model.onnx 是否完整(应为 364MB)
  echo.
  pause
  exit /b 1
)
echo.
echo   [OK] 自检通过
echo.

REM --- 注册为开机自启服务 ---
echo [5/6] 安装开机自启...
echo   方式:计划任务(不需要管理员权限,不需要额外软件)
echo.
schtasks /create /tn "RoadwayAOI" /tr "\"%~dp0aoi.exe\" --model \"%~dp0model.onnx\" --dir \"%~dp0watch\" --threshold 43.7255 --watch" /sc onstart /rl highest /f > nul 2>&1
if errorlevel 1 (
  echo   [提示] 计划任务创建失败(可能需要管理员权限)
  echo   可手动运行 aoi.exe --model model.onnx --dir watch --watch
) else (
  echo   [OK] 计划任务 RoadwayAOI 已创建,开机自动启动
)
echo.

REM --- 完成 ---
echo [6/6] 部署完成
echo.
echo ==========================================
echo   使用方式
echo ==========================================
echo.
echo   1) 单张测试:
echo      aoi.exe --model model.onnx --image D:\test\cable.png
echo.
echo   2) 批量测试:
echo      aoi.exe --model model.onnx --dir samples --threshold 43.7255
echo.
echo   3) 实时监控(接相机):
echo      aoi.exe --model model.onnx --dir watch --threshold 43.7255 --watch
echo.
echo   4) 改阈值:编辑 config.toml 里的 threshold
echo.
echo   常用参数:
echo      --model <路径>     模型文件
echo      --image <路径>     单张图片
echo      --dir <路径>       批量目录
echo      --threshold <数>   判定阈值(默认 43.7255)
echo      --ep cpu|coreml    执行后端(Windows 用 cpu)
echo      --threads <数>     线程数
echo      --watch            持续监控目录新增图片
echo      --dump-amap <目录> 导出热图数据(用于生成效果图)
echo.
pause
