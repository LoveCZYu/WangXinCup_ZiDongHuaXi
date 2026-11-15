# T000906 v1 程序包

本包对应已冻结的 `b02-point-v1-combined-e525-m050`：8000 步权重、起始阈值 0.525、维持阈值 0.05、固定组合关联策略。包内保留算法源文件的原始字节与 SHA-256，不含任何训练、开发、封存或正式验证输入。`PACKAGE.json` 记录本包文件校验值，参赛端每次启动均检查。

## 构建

在本目录（Dockerfile 所在目录）运行：

```bash
docker build -t t000906/b02:v1 .
```

构建需要联网安装依赖；运行不需要联网。基础镜像固定为 Python 3.12 slim-bookworm 的内容摘要，使用 PyTorch 2.8.0 + CUDA 12.8 运行库、NumPy 1.26.4、SciPy 1.15.3。镜像按官方样例安装 Wine/Xvfb，以执行原版 `serve.exe`。本版本固定 **CPU 推理、4 个线程**，无需 `--gpus`，不会初始化 CUDA 设备。此选择针对 9153 参数的小模型，未宣称符合所有“基础模型架构”审核口径。

Debian 软件包默认从清华公开镜像下载，保留 Debian 签名验证。海外环境可加 `--build-arg DEBIAN_MIRROR=https://deb.debian.org` 使用 Debian 官方源。不要关闭签名或 TLS 校验。

`environment.lock.txt` 锁定容器验收时实际安装的全部 Python 包；构建最后会按这份清单安装并执行 `pip check`。正式结果应由同一冻结镜像生成，保留镜像 ID、ZIP 校验值及运行日志。每次运行报告中的 `package_manifest_sha256` 可与 ZIP 中 `PACKAGE.json` 校验值核对。

## 运行

准备只含待评测 `.dat` 的输入文件夹，以及另一个空输出文件夹。Linux/macOS 示例：

```bash
docker run --rm --network none --cpus 4 --memory 6g \
  --mount type=bind,source=/absolute/input,target=/input,readonly \
  --mount type=bind,source=/absolute/empty-output,target=/output \
  t000906/b02:v1 --task all --test --out-dir /output
```

Windows PowerShell 示例（修改实际输入和输出目录）：

```powershell
$b02Input = (Resolve-Path 'C:\path\input').Path
$b02Output = (Resolve-Path 'C:\path\empty-output').Path
docker --context desktop-linux run --rm --network none --cpus 4 --memory 6g `
  --mount "type=bind,source=$b02Input,target=/input,readonly" `
  --mount "type=bind,source=$b02Output,target=/output" `
  t000906/b02:v1 --task all --test --out-dir /output
```

`--test` 与官方模板一致，仅为兼容参数，会处理全部输入，并不缩短运行。每次使用空输出目录。输入目录不得与输出目录相同；将输入挂载为只读。可以通过 `--input-dir /input`、`--out-dir /output`、`--port 9902` 显式指定官方服务参数。不要通过 TCP 连接检查端口健康状态，原始服务只接受一次参赛端连接。

## 输出和验收

只有官方 `serve.exe` 读取 DAT，参赛端通过回环 TCP 逐圈收取点迹，并只使用本圈和历史圈。输出保留原始点的时间、点号、圈号与坐标；杂点不输出，目标使用正整数 BatchID；每包重新初始化历史和航迹状态。`src/client.py`、`serve.py`、`exe_runtime.py`、`windows_job.py` 与官方样例逐字节一致。

`src/launch_server.py` 仅对 Linux 启动做兼容适配：通过系统 `script` 命令为 Wine 子进程提供伪终端，并使用镜像中有效的 `C.UTF-8` locale。原版 EXE 在直接重定向输出时会报 `init_sys_streams / WinError 6`，未生成的中文 locale 又会造成日志乱码；该层解决标准句柄和编码问题，不改变协议、点迹、航迹结果或官方核验。构建时会用真实 EXE 的 `--help` 验证非交互启动与中文日志。

成功时退出码为 0。输出目录包含官方航迹 `.dat`、`*_rt.dat`、`run_summary.json`、`logs/`，另附 `logs/client-summary.json` 记录模型与依赖版本、逐圈处理耗时、进程峰值常驻内存。应检查官方和客户端 summary 均为 success，全部数据包与圈数齐全、无超时。官方 30 秒单圈期限由原始程序检查，参赛客户端也保留 29 秒保护。

同一镜像重复运行，应比较航迹 DAT 的 SHA-256。`*_rt.dat` 和日志记录实际时间，重复运行无需相同，不能人工修改。正式验证结果包须完整保留官方输出，具体压缩结构按官网要求操作。

```bash
docker run --rm --network none --entrypoint python t000906/b02:v1 \
  -m unittest discover -s tests -v
```

包内测试验证文件完整性、篡改拒绝、合成数据逐包状态隔离和官方编码。实际官方样例、离线双跑、内存和失败退出验证在工作区交付验收报告中记录；本 README 的流程说明本身不代表已经完成正式验证集验收。正式验证集和官网接收仍需实际执行，不能用本地估算分冒充官方成绩。
