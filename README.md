# Memristor Measurement and Simulation

This repository contains Python scripts for memristor device measurement,
data plotting, and offline convolution simulation.

## Project Structure

```text
.
├── memristor_measure/       # Keithley 2450/2461 measurement and plotting scripts
├── memristor_simulation/    # Offline memristor convolution simulation
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Main Scripts

- `memristor_measure/forming.py`: progressive forming workflow for memristor devices.
- `memristor_measure/IV_Sweep.py`: I-V sweep measurement with configurable compliance current.
- `memristor_measure/Fixed_V_Read.py`: fixed-voltage current monitoring.
- `memristor_measure/modulation.py`: pulse-based LTP/LTD modulation experiment.
- `memristor_measure/resistance_switch_retention.py`: SET/RESET resistance switching and retention test.
- `memristor_measure/pulse_resistance_modulation.py`: pulse-by-pulse resistance modulation test.
- `memristor_measure/pulse_set_reset.py`: closed-loop +1 V SET / -1 V RESET with Icc voltage-window guard.
- `memristor_measure/bulkplot.py`: bulk plotting utility for I-V CSV results.
- `memristor_simulation/iv_convolution_sim.py`: offline convolution simulation using measured I-V data.

## Environment

Recommended Python version:

```bash
python --version
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Measurement scripts support Keithley 2450 and 2461 SMUs in SCPI mode. They
require a working PyVISA backend.

### Windows VISA setup

1. Install NI-VISA or the Keithley I/O Layer so Windows can use the USBTMC
   interface. The vendor VISA backend is preferred over `pyvisa-py` for USB.
2. On a Keithley 2461, select the **SCPI** command set from the front-panel
   settings. TSP mode is not supported by these scripts.
3. Connect the SMU and verify that PyVISA can see it:

```powershell
python -c "import pyvisa; rm=pyvisa.ResourceManager(); print(rm.list_resources())"
```

If this prints `()`, confirm that the 2461 is powered on, connected through its
rear-panel USB device port (not the front USB host port), and visible in Windows
Device Manager. Reinstall the Keithley I/O Layer or NI-VISA driver if Windows
does not identify it as a USBTMC instrument.

A 2461 USB resource normally resembles:

```text
USB0::0x05E6::0x2461::<serial-number>::INSTR
```

The default `--address auto` probes visible VISA resources and selects a single
Keithley 2450 or 2461. If more than one supported SMU is connected, pass the
exact address explicitly:

```powershell
python memristor_measure/init_R.py --address "USB0::0x05E6::0x2461::<serial-number>::INSTR"
```

## Usage

Run measurement scripts from their folder or from the repository root:

```bash
python memristor_measure/IV_Sweep.py
python memristor_measure/forming.py --v-max 2
python memristor_measure/Fixed_V_Read.py --voltage 0.1 --icc 50e-6
python memristor_measure/modulation.py --pulse-mode software
python memristor_measure/resistance_switch_retention.py --pulse-mode hardware --dev DEV-1-1
python memristor_measure/pulse_resistance_modulation.py --pulse-mode hardware --dev DEV-1-1
python memristor_measure/pulse_set_reset.py --mode cycle --n-set 4 --n-reset 4 --icc 50e-6
```

All commands above use automatic instrument discovery. Every measurement
script also accepts `--address <VISA-resource>`.

## Keithley 2461 notes

- The source-voltage current limit uses the 2461 SCPI command
  `:SOUR:VOLT:ILIM`.
- `modulation.py`, `pulse_resistance_modulation.py`,
  `pulse_set_reset.py`, and
  `resistance_switch_retention.py` accept `--pulse-mode software|hardware`.
  Software mode uses host timing and requires pulse widths of at least 20 ms.
  Hardware mode uses the 2461 native single-pulse trigger sequence and is
  rejected on a 2450.
- Source readback is enabled and CSV output records programmed voltage, source
  readback voltage, measured current, current limit, compliance, source status,
  and the automatic stop reason. Resistance is recorded as `nan` while the
  source is in compliance.
- The project enforces a hard current limit of 50 uA. Ordinary measurements are
  limited to +/-2 V. Forming above 2 V requires both an explicit command-line
  override and a second interactive confirmation.
- Before the first device measurement, verify polarity, voltage range, current
  limit, wiring, and whether the front or rear terminals are selected. Start
  with conservative limits. The measurement scripts can enable the SMU output.
- Automatic discovery and `*IDN?` checks do not enable the output.

### 安全调制流程

器件必须接在所选面板的 `FORCE HI` 与 `FORCE LO` 之间。推荐先使用
`0.1 V` 重复读取确认读操作不会改变阻态，再从 `+/-0.5 V`、20 ms 的单脉冲
开始逐步寻找 SET/RESET 窗口。每个写脉冲后程序都会回零、等待弛豫并进行多次
小信号读取。发生 compliance、读数错误或脉冲未完成时，程序停止后续写入并执行
回零、关闭输出。

Run the offline simulation:

```bash
python memristor_simulation/iv_convolution_sim.py
```

List simulation device data:

```bash
python memristor_simulation/iv_convolution_sim.py --list-devices
```

## Data

Generated measurement and simulation outputs are ignored by git by default to
keep the repository lightweight. If specific CSV, PNG, or PDF result files need
to be published, update `.gitignore` and add only the selected files.

## TODO

- Add project background and research objective.
- Add hardware setup details.
- Add device stack and fabrication notes.
- Add example figures or selected result files.
- Add citation or license information.
