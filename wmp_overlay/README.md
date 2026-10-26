# wmp_overlay

`WMP/` 是上游 clone，且在 `.gitignore` 里。**我们对 WMP 的所有改动都以本目录为准**，这样备份、回滚、换机器都不会丢。

## 文件

- `WMP_COMMIT`：上游固定 commit `c232c115`。
- `apply.sh`：还原 WMP（clone/checkout + 覆盖 `files/` + 复制 Go2 资产）。
- `files/`：镜像 WMP 根目录的相对路径；新增或整文件替换的文件放这里。
- `SETUP.md`：4090 环境安装、验收、故障排查。
- `WMP_code_map.md`：WMP 代码地图（285 维观测、RSSM/AMP/深度预测器、Go2 改动点）。

## 改 WMP 的标准流程

1. 在 `WMP/` 里改文件、跑通。
2. 把改过的文件复制到 `wmp_overlay/files/<相同相对路径>`。
3. 运行 `bash wmp_overlay/apply.sh` 验证 overlay 能还原。
4. 只提交 `wmp_overlay/`，不要提交 `WMP/`。

## 注意

- 不要往 `files/` 放 ckpt、视频、大资产；Go2 资产由 `apply.sh` 从 `unitree_rl_gym/resources/robots/go2/` 复制。
- 直接改上游后不同步回 overlay，下一次 `apply.sh` 或换人后改动会丢。
- 如果实验室不能访问 GitHub：先从 U 盘快照把 `WMP/` 整目录复制到 `$SWAP/WMP`（可以没有 `.git`），再运行 `apply.sh`；脚本会把它当作 vendored copy，跳过 checkout。
- `apply.sh --ref <tag>` 会先按 tag 还原 `wmp_overlay/`，再应用；这是回滚动作，执行前先问使用者。
