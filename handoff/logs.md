# 交接日志（只追加，不写细节）

2026-10-26 22:00 | workflow-v2 | MAPO | 工作流重构：AGENTS.md + handoff/ + wmp_overlay/；T0 待实验室验证 | next: 00-env | commit: workflow-v2 | backup: 待做
2026-10-28 16:31 | stage-0 | 陈旺 | T0 三条验收通过（Preview 4 + torch 2.3.1 路线，显存峰值 8946 MiB） | next: 由使用者决定 | commit: a4f099c | backup: 待做
2026-10-28 17:41 | stage-0.5 | 陈旺 | Go2 迁移 G0-G3 完成；300 iter 短训练 reward 0.72→1.99、无 nan；play 可加载；遗留站立不稳/WASD 未实现 | next: teleop + 长训练 | commit: 待提交 | backup: 待做
2026-11-05 20:51 | stage-0.5 | 陈旺 | 长训练 899/2000 手动停止；reward 0.02→11.16 无 nan；play iter800 total_reward 4.07（400=2.91, 600=1.98）；最佳 ckpt 已存 best_ckpt/ | next: 续训+三地形评估+视频 | commit: 待提交 | backup: 待做
2026-11-08 13:09 | T1 | 陈旺 | 基线 512env×10k iter 完成（22h18m，reward 15.93，无 nan）；slope 93.8% / stair 98.4% / gap 18.2%；材料见 Materials/SE | next: T2/T3 | commit: 待提交 | backup: 待做
2026-11-12 17:55 | T3 | 王 | 镜像评测 A 组(WMP 原地形)掉点≈0——WMP 地形左右对称、任务卡退出门槛不成立；B 组单侧横坡(OOD)无方向性掉点(7/9 mirror 更好，唯一明显 ramp_gap 5° +14pp)；T1 复现 3 次全部 ±5pp 内；修掉 trimesh 绕序 bug（假掉点 64.5pp→修复后 0pp） | next: T4/T5（对称变换+SE-MLP，先 CPU 单测）；或单侧倾斜地形重训 R0 后再严格对标 Fig.3 | commit: c833743 | backup: SWAP-backup-20261112-1809
