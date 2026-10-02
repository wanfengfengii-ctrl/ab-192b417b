# Peptide De Novo Sequencing Service

从含**漏峰（missing peaks）**和**杂峰（spurious peaks）**的串联质谱整数信号中恢复短肽
序列。不采用"逐峰贪心匹配最近残基再拼接"的做法（那样会拼出总质量不守恒的序列），
而是：

1. **质量守恒枚举** —— 仅枚举残基质量之和**恰好等于前体质量**、长度落在区间内的序列
   （按残基标签字典序生成）；
2. **互补裂解离子** —— 对长度 L 的候选，每个内部裂解 k 产生一对互补离子
   `prefix_k` 与 `precursor − prefix_k`（整数理论质量）；
3. **峰至多使用一次** —— 观测峰 ↔ 理论离子构成二分图，每个峰最多分配给一个离子；
4. **两级目标精确优化** —— 先**最大化同时获得前缀与后缀支持的裂解数 D**，
   再在所有 D 最优的方案中**最小化全部已分配峰的绝对误差之和 E**；
5. **预算约束** —— 无任一离子支持的裂解数 ≤ `max_missing_cleavages`，
   未分配峰数 ≤ `max_spurious_peaks`；
6. **歧义见证** —— 当多条不同残基序列在 (D, E) 两项目标下并列最优时，
   返回字典序最前的两份见证；不存在合格解释或输入自相矛盾时，返回**可定位的失败原因**
   （`location` 指向具体请求字段）。

求解器是**精确**的（分支定界 + Dijkstra 势函数最小费用流），并用一个独立的暴力参考
实现做随机交叉验证（`tests/test_solver.py`，120+400 组随机谱）。

## 目录

```
app/solver.py    质量守恒枚举 + 分支定界 + 最小费用流核心
app/schemas.py   请求/响应校验（含可定位的跨字段矛盾检查）
app/main.py      FastAPI: POST /api/spectra/sequence, GET /health
scripts/verify.py 一次性校验服务（健康等待 + 测试 + 构建 + 三类冒烟）
tests/           单元、随机暴力对照、HTTP 端到端测试
Dockerfile, docker-compose.yml
```

## 运行

宿主机端口可配置（默认 8000）：

```bash
docker compose up -d --build api
# 或自定义端口
HOST_PORT=9000 docker compose up -d --build api

curl http://localhost:8000/health
```

## 一次性校验服务 verify

`verify` 服务在 `api` 通过健康检查后才启动，依次执行：构建/导入检查、`pytest`
代码测试、以及成功 / 歧义 / 无解三类 HTTP 冒烟，**以退出码报告结果**（0 全部通过）：

```bash
docker compose build api verify
# --exit-code-from 让命令以 verify 服务的退出码为准（0 = 全部通过）
docker compose up --abort-on-container-exit --exit-code-from verify verify
docker compose down
```

## 请求

`POST /api/spectra/sequence`

| 字段 | 约束 |
| --- | --- |
| `peaks` | 8–28 个峰；`id` 唯一，`mass` 为正整数 |
| `precursor_mass` | 正整数前体质量 |
| `residues` | 4–8 种残基；标签唯一、**质量两两互异**的正整数 |
| `length_range` | `min`/`max`，2–12，min ≤ max |
| `tolerance` | 统一对称绝对容差（非负整数） |
| `max_missing_cleavages` | 无离子支持裂解的上限 |
| `max_spurious_peaks` | 未分配杂峰的上限 |

### 成功响应要点

- `canonical_sequence`：规范（字典序最先）最优序列；
- `optimal_objective`：`{double_supported_cleavages, total_abs_error}`；
- `witnesses`：1–2 份见证，每份含
  - `sequence`、`cleavages[]`（`k` 与互补的 `prefix_mass`/`suffix_mass`）；
  - `peak_assignments[]`（峰、裂解、前缀/后缀、理论质量、`error = 观测−理论`、
    `abs_error`）；
  - `unassigned_peaks`、`missing_cleavages`、`spurious_peaks`；
- `ambiguous`：是否存在并列最优的不同序列。

### 失败响应

- `400 invalid_request`：字段校验/跨字段矛盾，`errors[].location` 可定位；
- `422 no_solution`：输入本身合法但不存在合格解释，`error.location` 可定位；
- `422 error`：搜索空间/节点预算超限（建议缩小容差或长度/残基范围）。

## 本地开发

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
pytest -q
BASE_URL=http://127.0.0.1:8000 python scripts/verify.py
```
