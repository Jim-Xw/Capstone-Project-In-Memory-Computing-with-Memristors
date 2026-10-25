# memristor_measure — Agent 说明

给后续 agent 用的操作手册。改脚本、跑测量、读 CSV 之前先读完「接线 / 判据 / 禁令」三节。

工作目录：本文件夹或仓库根目录。

```text
python <脚本>.py
python memristor_measure/<脚本>.py
```

需要 PyVISA + Keithley 2450 或 2461（SCPI）。`--address auto` 自动选一台；多台必须显式传 VISA 地址。2461 前面板须为 SCPI，不要 TSP。

---

## 接线（硬规则）

- 器件接 **前面板 FORCE HI / FORCE LO**，仪器选 **FRONT**。
- **不要**用 SENSE HI/LO 代替 FORCE。SENSE 是四线电压端，当电流回路会读成开路或乱跳。
- FORCE HI = 顶电极，FORCE LO = 底电极。默认 **正压 SET、负压 RESET**。
- 输出开启时禁止换线。`init_R.py` 每颗测完会关输出，再换下一颗。
- 软件保护不能代替正确接线和人工值守。

---

## 判据（读数据时必须用）

1. **阻态只用 0.1 V 读电阻比。** 扫描/写脉冲中途的 `R = V/I` 是该偏压下的表观电阻。高阻氧化物非线性，0.6 V 的 120 kΩ 不等于已经 SET。
2. **SET / RESET 是否发生，看写完后 0.1 V 电阻是否锁住。**  
   SET：明显下降并保持（这批器件大约 100–500 kΩ → 10–30 kΩ）。  
   RESET：明显上升并保持（曾到约 0.8–1.3 MΩ）。
3. **图和 CSV 的电压以 `voltage_actual_V` / `pulse_voltage_actual_V` 为准。** 设定 ±1.8 V 时，低阻器件上往往只有 ±0.6～1.0 V（`V ≈ Icc × R`）。
4. **compliance / 回读电压塌缩 ≠ RESET 成功。** 电流顶满 Icc、电压掉下去，通常是细丝已通、限流在工作。
5. **V≈0 的电阻不可信。** 电流 nA 级时 `R` 会乱跳到几 kΩ 或几十 kΩ。用 |V|≈0.1 V 的点。
6. **脚本里的「RESET 1.5–3×」诊断常是不同偏压下的表观 R，不是阻态。** 不要据此建议加大 RESET 电流或更负电压。
7. **I–V 开口要分清：** 去程高阻、SET 后回程被夹在 +Icc，看起来像三角形开口，负半轴若仍是低阻则没有 RESET 窗口。
8. **超流停止：** 多数测量脚本在 `|I| > 1.02 × Icc` 或脉冲未完成且非 compliance 时停并回零。compliance 本身通常继续，电阻记 `nan`。

---

## 安全与默认值

| 项 | 实际行为 |
|---|---|
| `|V|` | 默认上限 2 V。超过须确认；`forming.py` 还要 `--allow-overvoltage`。 |
| Icc | `keithley_smu.MAX_SAFE_CURRENT_A` 是默认推荐（当前约 70 µA），**不是硬上限**。`validate_safety_limits` 不再因 Icc>50 µA 拒绝。仍应尽量用 50 µA。 |
| software 脉冲 | 主机定时，**最小 20 ms**。2450 只能用这个。 |
| hardware 脉冲 | 仅 2461 原生单脉冲，可短至约 150 µs。2450 会报错。 |
| 退出 | 任何正常/异常路径应 `safe_shutdown`：电压 0、输出 OFF。 |

低阻时能加到器件上的电压：

```text
V_actual_max ≈ Icc × R_read
```

例：R=12 kΩ、Icc=50 µA → 最多约 0.60 V。此时设 −1.0 V RESET 加不满。不要靠把 Icc 加到几百 µA 去「凑电压」，细丝会被焊死。

---

## 这批器件（实验室上下文）

堆叠：Ni / AlOx / TiOx /（Ti）一类，正压 SET。

| 量 | 一次实测（2026-09-17，新器件） | 说明 |
|---|---|---|
| SET | 约 **+1.01 V**，Icc 50 µA | ±1.0 V 扫描未锁定；±1.2 V 去程 +1.008 V 电流跳升后限流 |
| RESET | 约 **−1.0 V**，20 ms | −0.8 V 加不满或略降 R；−0.9 V 弱升；−1.0 V 后可到约 0.8–1.3 MΩ（可有延迟） |
| 读电压 | **+0.1 V** | 非破坏读 |
| 安全窗口 | 0.1 V 电阻大约 **24 kΩ～1 MΩ** | `pulse_set_reset.py` 默认：`R_min ≈ \|V\|/Icc × 1.2` |

上一只器件 ±1.8 V 长扫 + 高 Icc 负向脉冲后锁在约 2.3 kΩ，50–80 µA 下 RESET 电压加不上，按报废处理。

**不要**用加大 `V_MAX`、加点数、拉长 compliance 来「把回滞开口做大」。限流时间越长细丝越粗，开口越小。

---

## 脚本怎么选

| 目的 | 用 | 不要用 |
|---|---|---|
| 新器件先看初始 R | `init_R.py` | I–V / 脉冲 |
| 固定 0.1 V 看接触是否稳 | `Fixed_V_Read.py` | 大电压扫描 |
| 找 SET 电压、看回滞 | `IV_Sweep.py`，从 ±0.6 V 往上加一圈 | 直接 ±1.8 V、多圈 |
| 只读当前阻态 | `pulse_resistance_modulation.py --n-pot 0 --n-dep 0` | 开扫描 |
| 单次摸 RESET/SET | 同上，`--n-dep 1` 或 `--n-pot 1` | 一次 20 个高压脉冲 |
| 闭环维持阻态窗口 | `pulse_set_reset.py` | 无保护的长串 SET |
| 模拟 LTP/LTD 脉冲串 | `modulation.py` | 未确认 V_SET/V_RESET 之前 |
| 写后保持 | `resistance_switch_retention.py` | 器件已卡死 LRS |
| 未成形样品渐进成形 | `forming.py` | 已经能 SET 的器件 |
| 导出仪器 buffer | `buffer_read.py` | 当测量脚本 |
| 离线重画 I–V | `bulkplot.py` / `replot_trial4_icc_hold.py` | 连仪器 |

推荐顺序：`init_R` → 小电压 `IV_Sweep`（±0.6 → ±0.8 → ±1.0，每档一圈）→ 出现 SET 后 **停扫** → 0.1 V 只读 → 单脉冲摸 RESET → 确认窗口后才用 `pulse_set_reset`。

---

## keithley_smu.py

**不直接运行。** 所有测量脚本的 VISA、限流、回读、compliance、关断、软件/硬件脉冲都走这里。

常用符号：`connect_smu`、`configure_voltage_source`、`read_vi`、`apply_voltage_pulse`、`safe_shutdown`、`validate_safety_limits`。

改限流/脉冲/回读逻辑只改这个文件，并跑仓库根目录：

```text
python -m unittest test_keithley_smu test_pulse_set_reset
```

CSV 里 `source_status`：136 常见为正常，168 常见为 compliance（以 `compliance` 列为准）。

---

## init_R.py

**目的：** 成形前批量测初始电阻，区分正常 / 短路（<1 kΩ）/ 开路（>100 MΩ）。

**方法：** 0.1 V、Icc 50 µA、每颗约 10 次平均。每颗：设读电压 → 开输出 → 采样 → 回零关闭。换下一颗后按 Enter。

```text
python init_R.py
python init_R.py --n_devices 20 --output initial.csv
```

输出：终端摘要；`--output` 时写 CSV。

注意：`safe_shutdown` 后必须重新施加 `V_READ` 再开输出，否则会在 0 V 下读。

---

## Fixed_V_Read.py

**目的：** 固定电压连续读 I、R，检查接触、保持、漂移。

```text
python Fixed_V_Read.py --voltage 0.1 --icc 50e-6 --duration 5
```

先 `--pre-time` 用 `--baseline-voltage`（默认 0.1 V）采基线，再切到目标电压。`--duration 0` 表示直到 Ctrl+C。

输出目录：`Fixed_V_Read/`。

固定电阻或滑线变阻器校准：R 应稳；大振荡优先查 SENSE/FORCE 接错或接触，不要先改算法。

---

## IV_Sweep.py

**目的：** 双极性 I–V。路径：0 → +Vmax → −Vmin → 0。默认每 1/4 段 200 点，整圈 800 点。

**当前文件默认：** ±1.2 V、1 圈、NPLC=0.5、SET/RESET Icc 各 50 µA、正压 SET。

```text
python IV_Sweep.py
python IV_Sweep.py --v-max 0.8 --v-min -0.8 --cycles 1 --points 200 --icc-set 50e-6 --icc-reset 50e-6
```

`|V|>2` 时会多一行警告再 Enter/q。compliance **继续扫**，该点电阻 `nan`。`|I|>1.02×Icc` 才中止。

输出：`IV_sweep/V{min}_{max}_cyc…/`  
`iv_sweep_*.csv`、`iv_cycleN_*.csv`、对应 PNG。图横轴是回读电压。

找 V_SET：正向上升支路电流突然跳升或第一次 `compliance=1` 的 `voltage_set_V` / `voltage_actual_V`。再对照回程 0.1 V 电阻是否掉到低阻。

新器件不要一上来 ±1.8 V、两圈、200 点——限流段会把细丝打粗，随后 RESET 加不满。

---

## forming.py

**目的：** 未通器件渐进成形：电压台阶上升，电流到约 `0.8×Icc` 视为成形，停并读 0.1 V LRS。

默认 V_MAX=2 V、步进 0.05 V、驻留 0.5 s。`ICC_LADDER` 当前只有 50 µA。超过 2 V 必须 `--allow-overvoltage` 并再次确认。

```text
python forming.py --v-max 2 --dev DEV-01
```

输出：`Forming/`。已能量子化开关的器件用 `IV_Sweep` / 脉冲，不要再成形。

---

## pulse_resistance_modulation.py

**目的：** 逐个写脉冲，每次写后 0.1 V 多次平均读 R。无闭环保护。适合只读、单脉冲摸阈值。

```text
# 只读基线
python pulse_resistance_modulation.py --n-pot 0 --n-dep 0 --n-read-avg 20

# RESET-only 1 个
python pulse_resistance_modulation.py --n-pot 0 --n-dep 1 --v-dep -1.0 --t-pulse 0.02 --icc 50e-6 --n-read-avg 20

# SET-only 1 个
python pulse_resistance_modulation.py --n-pot 1 --n-dep 0 --v-pot 1.0 --t-pulse 0.02 --icc 50e-6
```

要点参数：`--v-pot` / `--v-dep` / `--n-pot` / `--n-dep` / `--t-pulse` / `--icc` / `--n-read-avg` / `--pulse-mode`。

看 `pulse_voltage_actual_V` 和 `compliance`。RESET 时回读远小于设定，说明 Icc×R 不够，不是「器件完全不能 RESET」。

输出：`Pulse_Modulation/`。compliance 默认仍读后电阻并继续（与早期「一限流就停」不同）；超流仍停。

---

## pulse_set_reset.py

**目的：** 闭环 +1 V SET / −1 V RESET，把 0.1 V 电阻留在能加满写电压的窗口，防止 LRS 过低后无法 RESET。

写前：`I_pred = |V|/R_read`。将顶满 Icc 时：SET → 改打 RESET 或停（`--on-limit reverse|stop`）；RESET 加不满 → **只停**，不再 SET。

```text
python pulse_set_reset.py --mode cycle --n-set 4 --n-reset 4 --icc 50e-6 --n-read-avg 20
python pulse_set_reset.py --mode set --n-set 6
python pulse_set_reset.py --mode reset --n-reset 6
python pulse_set_reset.py --mode hold --n-hold 8
```

`--mode`：`cycle` SET 再 RESET；`set` / `reset` 单向；`hold` 把 R 推回 `[R_min, R_max]`。

默认 `R_min ≈ 1 V / 50 µA × 1.2 ≈ 24 kΩ`，`R_max = 1 MΩ`。不要为了 SET 把 Icc 加到 95 µA（R_min 会掉到约 13 kΩ）。

输出：`Pulse_Set_Reset/`。CSV 额外列：`planned`、`action`、`guard_reason`、`i_pred_A`、`r_floor_Ohm`。

保护函数单测：`python -m unittest test_pulse_set_reset`。

---

## modulation.py

**目的：** LTP/LTD 脉冲串（默认 +0.5 / −0.5 V，各 5 个，1 圈）。多数参数在文件顶部，CLI 主要是 `--dev`、`--address`、`--pulse-mode`。

输出：`Pulse_LTP_LTD/`。未测到真实 V_SET/V_RESET 前，±0.5 V 往往只是高阻非线性，不要写成已调制。

可选 `ENABLE_INIT_RESET`：LTP 前先打一串较强负向脉冲。默认关。

---

## resistance_switch_retention.py

**目的：** 每圈一次 SET、一次 RESET，写后按间隔重复 0.1 V 读，看保持。

默认写电压 ±0.6 V、保持 5 s。这批器件 SET 约 1 V，默认 ±0.6 V 可能不够，需把 `--v-set` / `--v-reset` 改到已验证值。

```text
python resistance_switch_retention.py --v-set 1.0 --v-reset -1.0 --icc 50e-6 --retention-time 5
```

输出：`Switch_Retention/`。

---

## buffer_read.py

**目的：** 读仪器已有 buffer（`defbuffer1` / `defbuffer2`），不主动加压。导出前确认输出已关。

```text
python buffer_read.py --buffer defbuffer1 --output buffer_read/my_data
```

---

## bulkplot.py

**目的：** 离线。读取某目录下全部 `iv_cycle*.csv`，画 |I|–V（默认可对数）和指定读电压的开关比。不连仪器。

```text
python bulkplot.py --folder ./IV_sweep/<run_dir> --out iv_summary.png
```

---

## replot_trial4_icc_hold.py

**目的：** 离线重画 Trial 4 指定目录的 I–V，限流区电阻按 Icc 修正。不改原始 CSV，覆盖同名 PNG。路径写死在脚本里，不是通用绘图工具。

---

## 改代码时

- 测量/限流/脉冲：改 `keithley_smu.py`，各脚本只做流程。
- 不要恢复「Icc>50 µA 直接 ValueError」除非用户明确要求硬帽。
- 不要为「开口更大」自动加大 V_MAX、点数或 Icc。
- Windows 终端用 ASCII 打日志（`uA`/`Ohm`），避免 `µ`/`Ω` 在 GBK 下 `UnicodeEncodeError`。
- 零偏/小电流电阻、跨电压比 R、compliance 当 RESET：分析里不要当开关证据。
- 未要求时不要 commit、不要改计划文件。

---

## 输出目录一览

| 目录 | 来源 |
|---|---|
| `IV_sweep/` | `IV_Sweep.py` |
| `Forming/` | `forming.py` |
| `Fixed_V_Read/` | `Fixed_V_Read.py` |
| `Pulse_Modulation/` | `pulse_resistance_modulation.py` |
| `Pulse_Set_Reset/` | `pulse_set_reset.py` |
| `Pulse_LTP_LTD/` | `modulation.py` |
| `Switch_Retention/` | `resistance_switch_retention.py` |
| `buffer_read/` | `buffer_read.py` |

CSV 一般含：设定电压、回读电压、电流、电阻、Icc、`compliance`、源状态、停止原因。限流点电阻多为 `nan`。
