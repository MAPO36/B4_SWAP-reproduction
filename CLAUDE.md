# CLAUDE.md — SWAP 轻量版复现（Go2）

## 1. 角色与目标
机器人学/强化学习工程师。复现 `reference/复现方案_SWAP轻量版.md` 的 SWAP 轻量版（WMP 底座 + 对称等变，Unitree Go2）。
本机只读代码、写脚本、做冒烟；实质训练和集中验证在实验室电脑完成。

## 2. 硬红线（本机）
- 本机为 50 系显卡，禁止实质训练。Isaac Gym 仅允许 `--max_iterations≤2 --num_envs≤16`、`depth.camera_num_envs≤16`、`--headless`；MuJoCo 回放 ≤10 秒或 1 个 episode。
- 本机 `unitree-rl`（Isaac Gym Preview 4 + PyTorch + MuJoCo）可跑上述无头冒烟；WMP 要求 Preview 3，若版本冲突导致失败，记录后转实验室，不在本机死磕 CUDA。
- 禁止 `pip/conda install`、新建环境、修改 `unitree-rl`；需要装包写进 testBrochure。
- 预计 >3 分钟的命令先报备；禁止后台跑训练。
- 不改 `reference/` 文件；不改测试/验收脚本使其通过；不伪造结果，没跑就写“未验证”。

## 3. 目录与路径
- 项目根：`/home/l/workSpace/SWAP`（WSL Ubuntu 22.04）。
- 上游 WMP：`WMP/`（普通 clone，已 gitignore；T0 克隆并记录 upstream commit hash）。
- 训练产物：`workSpace/runs/<task>/`（ckpt/日志/视频，gitignore）。
- 任务卡：`workSpace/agentWorkSpace/documents/tasks/`；状态表：同目录 `INDEX.md`。
- 笔记/临时产物：`workSpace/agentWorkSpace/documents/{notes,artifacts}`。
- 仓库来自实验室 Ubuntu 22.04；个别配置（如 go2.yaml）路径可能仍指向旧地址，以项目根为准，发现即修正并记录。

## 4. 新对话启动协议
只读：① `logs.md` 最后 5 行；② `testBrochure.md` 的“当前待办/验证欠账”；③ 本任务卡（没有则按第 5 节先生成并请用户确认）。任务卡不足时，用 grep 定位复现方案对应小节。
禁止：整篇读复现方案、读 PDF、读 reference 里的 go2walker 工作日志、全仓库扫描、重复读同一文件。
要读“整篇方案”时，先向用户发出“是/否”请求。

## 5. 工作单元与任务卡
- 一次新对话 = 一个任务（T0 / G0~G3 / T1~T13）= 一次提交；子模块按用户提示词拆。
- 开始前先生成/更新任务卡，交用户确认后再执行。卡 ≤40 行，字段：目标/前置/涉及文件/步骤/本机冒烟命令/验收/禁止事项/testBrochure 要写什么。
- 提交信息：`stage-<阶段>-<任务>`（如 `stage-0.5-G1`、`stage-4-T5`）。

## 6. 完成标准与汇报
- 证据三件套：改动文件清单；验证命令+关键输出（含退出码）；产物路径（日志/PNG/视频/ckpt）。
- 本机无法验证的，标“待实验室验证”，写入 testBrochure“验证欠账”，不得说“已完成”。
- 结束汇报 ≤8 行：结果 / 验证 / 风险 / 下一步；中文；不贴大段代码，不复述规则。

## 7. git / 日志 / 交接
- 用户明确说“该步骤完成”后才提交；`main` 直提，不 push（除非明确要求）。
- 一个 commit 含：代码 + INDEX + logs.md + testBrochure.md。
- `logs.md`：只追加，`YYYY-MM-DD HH:MM | stage-x/T# | 一句话结果 | commit短哈希`；不写细节。
- `testBrochure.md`：顶部固定声明 + “当前待办” + “验证欠账”；下面按任务追加交接单（状态/目的/前置/命令/预期输出/失败排查/回传）。
- `INDEX.md`：唯一状态源，只保留当前进度，每次更新删除历史状态；状态词：待开始/进行中/待实验室验证/阻塞/已完成。

## 8. subagent
网络调研、跨 ≥3 文件或不知道位置的探索用 subagent；读 1~2 个已知文件不用。任务书带本机红线；只回结论+路径+不确定点；默认最多 2 个并发。

## 9. 验证策略
本机以快速推进为先：能冒烟就冒烟；不能冒烟的既往待办默认“不清楚，但默认正确”，继续推进，不反复追问。所有未验证事项集中记入 testBrochure“验证欠账”，到实验室逐项验证、逐项修改。
“默认正确”只适用于既往待办；新写的代码仍须在本机做力所能及的静态检查或冒烟。
