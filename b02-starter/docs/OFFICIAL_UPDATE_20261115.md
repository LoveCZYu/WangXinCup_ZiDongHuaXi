# 11.15 官方收发包适配（程序包 v1.1）

> 11.16 补充：五份官方 DAT 已由主办方确认为 A 榜输入。下面“只用 train_005”指此前环境联调，不是打榜范围；正式 A 榜须完整运行五包，见 [A 榜提交说明](A_BOARD_SUBMISSION.md)。

本次输入是用户提供的《代码样例及数据收发程序（11.15更新） (1).zip》。与旧官方 ZIP 按相对路径比较，不执行包内 IDE 文件或其他无关内容。更新只涉及数据收发环境和交付，模型仍为冻结 v1。

## 已核对的变化

| 项目 | 旧包 | 11.15 包 | 处理 |
| --- | --- | --- | --- |
| ZIP 根目录 | `participant_demo_v2/participant_demo/` | `participant_demo/` | 准备入口自动识别两种目录；若同时存在两套完整程序则拒绝歧义 |
| 官方 Dockerfile | Wine / Wine64 | 新增 i386 架构与 `wine32:i386` | 镜像同时保留 Wine32、Wine64 |
| 官方 README | 含“当前示例不需要显卡”一句 | 删除该句 | 不据此推断协议或模型硬性要求变化 |
| TCP 协议、六个官方源码文件、官方测试 | 原版 | 字节相同 | 无需改动点迹字段、点序号或 ID 协议 |
| `serve.exe` | 原版 | 字节相同，PE 类型 AMD64 | 二进制原样复制，使用匹配的 64 位加载器 |
| `train_005.dat` | 原版 | 字节相同 | 继续仅用此场景进行容器联调，其余场景不参与本轮评估 |
| 官方许可 | Apache-2.0 | 字节相同 | 程序包附 `OFFICIAL_LICENSE` 原文 |

新增 `.idea` 文件不是程序依赖，准备工具不提取。仅提取明确列出的程序、说明和官方 DAT 样例；不从 ZIP 路径任意写出文件。

## Wine32 与当前容器的兼容

Debian 的 `wine` 包装脚本在 Wine32 存在时会优先启动 32 位加载器。本机 Docker 环境中，这条路径导致 `wineboot --init` 报 `socket : Function not implemented`。同一依赖镜像中的对照结果：默认加载器失败，显式 Wine64 启动相同官方 EXE 成功，说明问题也存在于普通容器，并非只发生在构建沙箱。

`runtime/wine64-loader.sh` 安装到 `/usr/local/bin/wine` 和 `wine-stable`，选择 `/usr/lib/wine/wine64`，与官方 AMD64 EXE 匹配；Wine32 仍保持安装。没有关闭 seccomp、没有增加特权。沿用原有 PTY 与 `C.UTF-8` 适配，以保留正常标准句柄与中文日志。官方 EXE、四个交互源码均未修改。

本机 Wine 初始化日志仍可能记录 32 位辅助进程的非致命兼容提示；是否成功由初始化退出码、真实 EXE、全部官方逐圈记录和结果文件共同验证。不能把本机现象泛化为所有环境均不能运行 Wine32。

## 队友使用

从 GitHub 的 `b02-association-calibration` 分支更新代码，使用新版官方 ZIP：

```text
python -m b02 prepare-official "新版官方代码样例及数据收发程序.zip"
python tools/build_submission.py --out runs/submission-build
```

已有 `.official` 时，准备工具会拒绝覆盖；可改用 `--demo-dir .official-20261115`，构建时相应加 `--official .official-20261115`。默认产物名为 `T000906_v1.1_code.zip`，镜像运行命令见 [模板说明](../submission-template/README.md)。程序包版本 v1.1 不代表模型重新训练。

准备后的官方目录含五份样例。运行联调时显式指定只含 `train_005.dat` 的输入目录；不要为验证环境而默认跑所有场景。正式验证集应另放独立目录，使用同一冻结镜像产生正式结果。

更新后的完整容器验收结论见 [交付验收报告](SUBMISSION_ACCEPTANCE.md)，来源与校验值见 [官方包比对记录](official-release-20261115.json)。
