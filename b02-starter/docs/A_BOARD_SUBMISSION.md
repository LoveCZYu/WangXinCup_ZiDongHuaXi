# A 榜输入与结果 ZIP（11.16 确认）

2026-11-16，用户转述主办方确认：**A 榜输入就是官方收发程序附带的五包，结果 ZIP 只需上传全部航迹 DAT。** 不需要另找一份名为“验证集”的下载包。此前将这五包只当环境样例、认为尚缺 A 榜输入的判断现已纠正。

当前 v1.1 的五包双跑与输出校验已完成，见 [交付记录](A_BOARD_RUN.md)。代理未替用户上传；用户现已提交结果并反馈官网总分 75.24，详细记录见交付报告。

## 对应文件

输入取自 11.15 更新的官方 ZIP 的 `participant_demo/data/inputs/`：`train_001.dat` 至 `train_005.dat`。官方来源及每份输入的 SHA-256 见 [A 榜数据角色登记](../splits/a_board.json)。输入文件只交给官方收发服务；模型仍仅通过 TCP 接收本圈点迹。

提交页面第 2 行“可运行代码包”使用 `T000906_v1.1_code.zip`。第 3 行“结果文件 ZIP”使用同一冻结代码生成的 `T000906_v1.1_result.zip`，ZIP 根目录应只有：

```text
train_001.dat
train_002.dat
train_003.dat
train_004.dat
train_005.dat
```

上传 ZIP 不含 `_rt.dat`、日志、JSON、CSV、查看器、代码或输入文件，也不额外套文件夹。五个文件均为官方收发程序保存的航迹输出，不能拿同名输入 DAT 代替。完整输出目录（包括计时与日志）仍在本地保留，供复核追溯，不人工修改航迹二进制。

## 复现与打包

用既定 v1.1 镜像分别运行到两个空输出目录；每次输入目录都包含上述五包。Docker 命令见 [程序包说明](../submission-template/README.md)。当前代码、权重、阈值及关联规则保持冻结，第二次运行用于逐字节复现，不用 A 榜数据再调参。

从 `b02-starter` 目录执行（首次交付使用的原始输出仍保存在工作区 `analysis/a-leaderboard-v1.1-20261116/`）：

```text
python tools/build_results.py --code-zip ../T000906_v1.1_code.zip --input-dir .official-20261115/data/inputs --run ../analysis/a-leaderboard-v1.1-20261116/run-1 --repeat ../analysis/a-leaderboard-v1.1-20261116/run-2 --out ../T000906_v1.1_result.zip --report runs/a-board-v1.1/report.json
```

工具会核对五份输入的版本、官方与客户端完整成功状态、运行 manifest 与代码 ZIP 的对应关系，以及两次轨迹字节是否相同，再只打包这五个航迹文件。已有 ZIP 或报告不会覆盖。重复运行时换输出文件名或目录。

程序包与结果包要成对留存，包括 SHA-256、镜像 ID、运行日志和后续官网成绩。生成结果不等于已上传；当前工具不访问官网，也不会消耗提交次数；用户已完成一次实际提交并反馈 75.24，总次数以官网为准。原 79.456 是本地开发集双种子估算，不是 A 榜成绩。

## 内部封存范围调整

官方公开五包与历史训练池及封存划分有重合。`train_003` 原在内部 30 场景封存清单中，按此次官方口径参与冻结 A 榜推理后，不再标为完全未处理的独立测试场景。本轮不读取该场景的 CSV 真值、不计算其本地指标、不训练或调参。

`splits/v1.json` 保留历史分组，`--split holdout` 仍表示原 30 场景，以便复现历史记录。当前独立保留的 29 场景登记在 `splits/a_board.json` 的 `remaining_independent_holdout`，未来获准评估时使用 `--split holdout_unseen`。本轮不运行它们。公开榜分数与独立泛化评估应分开解释。
