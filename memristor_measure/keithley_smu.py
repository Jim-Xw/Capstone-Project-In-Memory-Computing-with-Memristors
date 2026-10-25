"""Shared communication, safety-limit and pulse-control utilities for the Keithley 2450/2461.

This module is not run directly. All measurement scripts use it for instrument
identification, source-voltage readback, compliance detection and safe shutdown.
50 uA is only the recommended default current and is no longer a hard limit.
The 2461 can use native hardware single pulses; the 2450 supports only the
software-timed pulses implemented in this project.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable, Literal


AUTO_ADDRESS = "auto"
SUPPORTED_MODELS = ("2450", "2461")
MAX_SAFE_CURRENT_A = 200e-6  # Backward-compatible default, not a hard cap.
DEFAULT_MAX_VOLTAGE_V = 2.0
DEFAULT_COMPLIANCE_RATIO = 0.985
SOFTWARE_MIN_PULSE_S = 0.02
HARDWARE_MIN_PULSE_S = 150e-6


class InstrumentConnectionError(RuntimeError):
    """Raised when a supported Keithley SMU cannot be selected or opened."""


def _resource_manager():
    try:
        import pyvisa
    except ImportError as exc:
        raise InstrumentConnectionError(
            "Missing dependency. Run: python -m pip install -r requirements.txt"
        ) from exc
    return pyvisa.ResourceManager()


def model_from_idn(idn: str) -> str | None:
    """Return the supported model number found in an IEEE-488.2 ID string."""
    normalized = idn.upper()
    if "KEITHLEY" not in normalized:
        return None
    for model in SUPPORTED_MODELS:
        if model in normalized:
            return model
    return None


def _configure_resource(inst, timeout_ms: int) -> None:
    inst.write_termination = "\n"
    inst.read_termination = "\n"
    inst.timeout = timeout_ms


def _identify_resource(rm, address: str, timeout_ms: int) -> str | None:
    inst = None
    try:
        inst = rm.open_resource(address)
        _configure_resource(inst, timeout_ms)
        return inst.query("*IDN?").strip()
    except Exception:
        return None
    finally:
        if inst is not None:
            try:
                inst.close()
            except Exception:
                pass


def discover_supported_smus(rm=None, timeout_ms: int = 3000) -> list[tuple[str, str]]:
    """Return ``(VISA address, *IDN?)`` pairs for reachable 2450/2461 SMUs."""
    rm = rm or _resource_manager()
    matches: list[tuple[str, str]] = []
    for address in rm.list_resources():
        idn = _identify_resource(rm, address, timeout_ms)
        if idn and model_from_idn(idn):
            matches.append((address, idn))
    return matches


def _format_resources(resources: Iterable[str]) -> str:
    values = list(resources)
    return ", ".join(values) if values else "(none)"


def connect_smu(address: str = AUTO_ADDRESS, timeout_ms: int = 15000, rm=None):
    """Open and validate a Keithley 2450/2461 without enabling its output."""
    rm = rm or _resource_manager()
    selected = address

    if not address or address.lower() == AUTO_ADDRESS:
        matches = discover_supported_smus(rm)
        if not matches:
            resources = _format_resources(rm.list_resources())
            raise InstrumentConnectionError(
                "No reachable Keithley 2450/2461 was found. "
                f"VISA resources: {resources}"
            )
        if len(matches) > 1:
            choices = _format_resources(item[0] for item in matches)
            raise InstrumentConnectionError(
                "Multiple supported Keithley SMUs were found. "
                f"Choose one with --address. Matches: {choices}"
            )
        selected = matches[0][0]

    try:
        inst = rm.open_resource(selected)
        _configure_resource(inst, timeout_ms)
        try:
            inst.clear()
            time.sleep(0.1)
        except Exception:
            pass
        inst.write("*CLS")
        idn = inst.query("*IDN?").strip()
    except Exception as exc:
        try:
            inst.close()
        except (Exception, UnboundLocalError):
            pass
        resources = _format_resources(rm.list_resources())
        raise InstrumentConnectionError(
            f"Cannot connect to {selected}: {exc}. VISA resources: {resources}"
        ) from exc

    model = model_from_idn(idn)
    if model is None:
        try:
            inst.close()
        except Exception:
            pass
        raise InstrumentConnectionError(
            f"Resource {selected} is not a supported Keithley 2450/2461: {idn}"
        )

    print(f"[Connect] {selected}")
    print(f"[Connect] OK -> {idn}\n")
    try:
        inst._memristor_model = model
        inst._memristor_idn = idn
    except Exception:
        pass
    return inst


def parse_current_response(raw: str) -> float:
    """Parse current from a 2450/2461 READ? response or a single value."""
    fields = [field.strip() for field in raw.strip().split(",")]
    if not fields or not fields[0]:
        raise ValueError("Instrument returned an empty reading")
    value = fields[1] if len(fields) >= 2 else fields[0]
    return float(value)


def read_current(inst) -> float:
    return parse_current_response(inst.query(":READ?"))


def query_error(inst) -> str:
    return inst.query(":SYST:ERR?").strip()


def error_is_clear(response: str) -> bool:
    """Accept common Keithley zero-error response formats."""
    try:
        code = int(response.split(",", 1)[0].strip())
    except (TypeError, ValueError):
        return False
    return code == 0


@dataclass(frozen=True)
class ViReading:
    """Result of one source-voltage readback and current measurement."""

    voltage_set_v: float
    voltage_actual_v: float
    current_a: float
    icc_a: float
    compliance: bool
    relative_time_s: float | None = None
    source_status: int | None = None

    @property
    def resistance_ohm(self) -> float:
        if self.compliance or not math.isfinite(self.current_a) or abs(self.current_a) <= 1e-14:
            return float("nan")
        voltage = self.voltage_actual_v
        if not math.isfinite(voltage):
            return float("nan")
        return abs(voltage / self.current_a)


@dataclass(frozen=True)
class PulseResult:
    mode: str
    voltage_set_v: float
    voltage_actual_v: float
    current_a: float
    current_limit_a: float
    compliance: bool
    completed: bool
    relative_time_s: float | None = None
    source_status: int | None = None
    stopped_reason: str = ""


class ResistanceReadError(RuntimeError):
    """A rejected read burst, retaining samples for diagnosis, never control."""

    def __init__(self, message: str, *, readings: tuple[ViReading, ...],
                 issues: tuple[str, ...], sample_index: int):
        super().__init__(message)
        self.readings = readings
        self.issues = issues
        self.sample_index = sample_index


def instrument_model(inst) -> str:
    model = getattr(inst, "_memristor_model", None)
    if model in SUPPORTED_MODELS:
        return model
    idn = inst.query("*IDN?").strip()
    model = model_from_idn(idn)
    if model is None:
        raise InstrumentConnectionError(f"Unsupported instrument: {idn}")
    return model


def validate_safety_limits(
    voltage_v: float,
    current_limit_a: float,
    *,
    max_voltage_v: float = DEFAULT_MAX_VOLTAGE_V,
    allow_overvoltage: bool = False,
) -> None:
    if not math.isfinite(current_limit_a) or current_limit_a <= 0:
        raise ValueError("Current limit must be positive and finite")
    if not math.isfinite(voltage_v):
        raise ValueError("Voltage must be finite")
    if abs(voltage_v) > max_voltage_v and not allow_overvoltage:
        raise ValueError(
            f"|Voltage|={abs(voltage_v):.6g} V exceeds safety limit "
            f"{max_voltage_v:.6g} V"
        )


# --- Series resistor in the HI line ---------------------------------------
# Measured quantities include the resistor; these give the device alone.

def add_series_argument(parser) -> None:
    parser.add_argument(
        "--r-series", type=float, default=0.0,
        help="Resistor in series with the device (HI line), Ohm; "
             "reads report the device alone and output names get an RS tag",
    )


def validate_series(r_series: float) -> None:
    if not math.isfinite(r_series) or r_series < 0:
        raise ValueError("--r-series must be a finite value >= 0 Ohm")


def series_tag(r_series: float) -> str:
    """Name tag for outputs, e.g. 'RS10k'; empty when no resistor is used."""
    return f"RS{r_series / 1e3:g}k" if r_series else ""


def device_voltage(voltage_v: float, current_a: float, r_series: float) -> float:
    """Voltage left across the device: applied minus the drop on the resistor."""
    return voltage_v - current_a * r_series


def device_resistance(voltage_v: float, current_a: float, r_series: float) -> float:
    """|V/I| of the whole path minus the series resistor (nan if I is ~0)."""
    if not math.isfinite(current_a) or abs(current_a) <= 1e-14:
        return float("nan")
    return abs(voltage_v / current_a) - r_series


def set_current_limit(inst, requested_a: float) -> float:
    validate_safety_limits(0.0, requested_a)
    errors: list[str] = []
    for set_cmd, query_cmd in (
        (":SOUR:VOLT:ILIM", ":SOUR:VOLT:ILIM?"),
        (":SOUR:VOLT:ILIM:LEV", ":SOUR:VOLT:ILIM:LEV?"),
    ):
        inst.write(f"{set_cmd} {requested_a:.9e}")
        err = query_error(inst)
        if not error_is_clear(err):
            errors.append(f"{set_cmd}: {err}")
            continue
        actual = float(inst.query(query_cmd).strip())
        if not math.isfinite(actual) or actual <= 0 or actual > requested_a:
            safe_shutdown(inst)
            raise RuntimeError(
                f"Unsafe current limit readback: requested {requested_a:.9g} A, "
                f"instrument returned {actual:.9g} A"
            )
        return actual
    raise RuntimeError("Instrument rejected current limit: " + " | ".join(errors))


def configure_voltage_source(inst, current_limit_a: float, nplc: float) -> float:
    """Configure a safe voltage source; keep the output off and enable source readback."""
    validate_safety_limits(0.0, current_limit_a)
    if nplc <= 0:
        raise ValueError("NPLC must be > 0")
    for cmd in (
        "*RST",
        "*CLS",
        ":SOUR:FUNC VOLT",
        ":SOUR:VOLT:RANG:AUTO 1",
        ":SOUR:VOLT:READ:BACK ON",
        ":SOUR:VOLT:LEV 0",
        ':SENS:FUNC "CURR"',
        ":SENS:CURR:RANG:AUTO 1",
        f":SENS:CURR:NPLC {nplc:.6g}",
        ":SENS:AZER:ONCE",
        ":OUTP OFF",
    ):
        inst.write(cmd)
    actual = set_current_limit(inst, current_limit_a)
    err = query_error(inst)
    if not error_is_clear(err):
        raise RuntimeError(f"Instrument initialization failed: {err}")
    return actual


def _query_trip(inst) -> bool:
    try:
        return bool(int(float(inst.query(":SOUR:VOLT:ILIM:TRIP?").strip())))
    except Exception:
        return False


def is_compliance(current_a: float, icc_a: float, trip: bool = False) -> bool:
    if trip:
        return True
    return (
        math.isfinite(current_a)
        and icc_a > 0
        and abs(current_a) >= abs(icc_a) * DEFAULT_COMPLIANCE_RATIO
    )


def parse_vi_response(
    raw: str,
    *,
    voltage_set_v: float,
    icc_a: float,
    trip: bool = False,
) -> ViReading:
    """Parse a 2461 ASCII reply in ``SOUR,READ,REL,SOURSTAT`` order."""
    fields = [field.strip() for field in raw.strip().split(",")]
    if len(fields) < 2:
        raise ValueError(f"Expected at least SOUR,READ; got: {raw!r}")
    actual_v = float(fields[0])
    current_a = float(fields[1])
    relative = float(fields[2]) if len(fields) >= 3 and fields[2] else None
    status = int(float(fields[3])) if len(fields) >= 4 and fields[3] else None
    return ViReading(
        voltage_set_v=voltage_set_v,
        voltage_actual_v=actual_v,
        current_a=current_a,
        icc_a=icc_a,
        compliance=is_compliance(current_a, icc_a, trip),
        relative_time_s=relative,
        source_status=status,
    )


def read_vi(inst, voltage_set_v: float, icc_a: float) -> ViReading:
    raw = inst.query(':READ? "defbuffer1", SOUR, READ, REL, SOURSTAT')
    return parse_vi_response(
        raw,
        voltage_set_v=voltage_set_v,
        icc_a=icc_a,
        trip=_query_trip(inst),
    )


def safe_shutdown(inst) -> None:
    """Best-effort abort of the trigger model, return to zero and output off."""
    for cmd in (":ABOR", ":SOUR:VOLT:LEV 0", ":OUTP OFF"):
        try:
            inst.write(cmd)
        except Exception:
            pass


def read_resistance_samples(
    inst, *, current_limit_a: float, count: int = 10, settle_s: float = 0.02,
    voltage_v: float = 0.1, voltage_tolerance: float = 0.1,
) -> tuple[ViReading, ...]:
    """Strict read burst for resistance verification and write authorization."""
    return _read_low_bias_samples(
        inst, current_limit_a=current_limit_a, count=count, settle_s=settle_s,
        voltage_v=voltage_v, voltage_tolerance=voltage_tolerance,
        require_positive_current=True,
    )


def read_diagnostic_current_samples(
    inst, *, current_limit_a: float, count: int = 10, settle_s: float = 0.2,
    voltage_v: float = 0.1,
) -> tuple[ViReading, ...]:
    """Signed low-bias currents for diagnostics only; never authorize a write.

    Accept zero/negative current, but retain compliance, current magnitude,
    finite-reading and voltage checks. No resistance is inferred at zero bias.
    """
    return _read_low_bias_samples(
        inst, current_limit_a=current_limit_a, count=count, settle_s=settle_s,
        voltage_v=voltage_v, voltage_tolerance=0.1, require_positive_current=False,
    )


def _read_low_bias_samples(
    inst, *, current_limit_a: float, count: int, settle_s: float,
    voltage_v: float, voltage_tolerance: float, require_positive_current: bool,
) -> tuple[ViReading, ...]:
    """Read a complete, validated low-bias sample burst, then switch output off.

    Used by the trend controller. A failed or clamped sample invalidates the
    burst; callers must not continue programming from a partial/stale read.
    ResistanceReadError retains prior samples and the rejected sample, if one
    was received, for diagnostics only.
    Resistance is computed from measured voltage, not the requested voltage.
    """
    validate_safety_limits(voltage_v, current_limit_a)
    if voltage_v < 0 or voltage_v > 0.1 or (require_positive_current and voltage_v == 0):
        raise ValueError("Read voltage must be in [0, 0.1] V; resistance verification requires > 0")
    if not isinstance(count, int) or count <= 0:
        raise ValueError("Read count must be a positive integer")
    if not math.isfinite(settle_s) or settle_s < 0:
        raise ValueError("Read settle must be finite and nonnegative")
    if not math.isfinite(voltage_tolerance) or not 0 < voltage_tolerance < 1:
        raise ValueError("Read voltage tolerance must be in (0, 1)")
    allowed_error = voltage_tolerance * (voltage_v if voltage_v > 0 else 0.1)
    readings = []
    try:
        inst.write(f":SOUR:VOLT:LEV {voltage_v:.9g}")
        inst.write(":OUTP ON")
        time.sleep(settle_s)
        for sample_index in range(1, count + 1):
            try:
                reading = read_vi(inst, voltage_v, current_limit_a)
            except Exception as exc:
                raise ResistanceReadError(
                    f"Resistance read sample {sample_index}/{count} failed: {exc}",
                    readings=tuple(readings), issues=("read_failed",), sample_index=sample_index,
                ) from exc
            readings.append(reading)
            issues = []
            if reading.compliance:
                issues.append("compliance")
            if not math.isfinite(reading.current_a):
                issues.append("nonfinite_current")
            elif abs(reading.current_a) >= current_limit_a:
                issues.append("current_at_or_above_limit")
            elif require_positive_current and reading.current_a <= 1e-14:
                issues.append("nonpositive_or_unresolved_current")
            if not math.isfinite(reading.voltage_actual_v):
                issues.append("nonfinite_voltage")
            elif abs(reading.voltage_actual_v - voltage_v) > allowed_error:
                issues.append("read_voltage_out_of_tolerance")
            if issues:
                raise ResistanceReadError(
                    f"Invalid resistance read sample {sample_index}/{count}: {', '.join(issues)}; "
                    f"Vset={voltage_v:.9g} V, Vactual={reading.voltage_actual_v:.9g} V, "
                    f"I={reading.current_a:.9g} A, Icc={current_limit_a:.9g} A, "
                    f"compliance={reading.compliance}, source_status={reading.source_status}; "
                    f"allowed V={voltage_v - allowed_error:.9g}.."
                    f"{voltage_v + allowed_error:.9g} V",
                    readings=tuple(readings), issues=tuple(issues), sample_index=sample_index,
                )
    finally:
        safe_shutdown(inst)
    return tuple(readings)


def apply_software_pulse(
    inst,
    voltage_v: float,
    width_s: float,
    relax_s: float,
    current_limit_a: float,
) -> PulseResult:
    validate_safety_limits(voltage_v, current_limit_a)
    if width_s < SOFTWARE_MIN_PULSE_S:
        raise ValueError(
            f"Software pulse width must be >= {SOFTWARE_MIN_PULSE_S:.3f} s"
        )
    set_current_limit(inst, current_limit_a)
    started = time.perf_counter()
    try:
        inst.write(f":SOUR:VOLT:LEV {voltage_v:.9g}")
        inst.write(":OUTP ON")
        reading = read_vi(inst, voltage_v, current_limit_a)
        remaining = width_s - (time.perf_counter() - started)
        # Do not finish the requested hold after detecting compliance or an
        # invalid/at-limit reading. This acts only after the blocking read.
        invalid = (not math.isfinite(reading.current_a)
                   or not math.isfinite(reading.voltage_actual_v))
        at_limit = abs(reading.current_a) >= current_limit_a
        if remaining > 0 and not (reading.compliance or invalid or at_limit):
            time.sleep(remaining)
    finally:
        safe_shutdown(inst)
    if relax_s > 0:
        time.sleep(relax_s)
    reason = ("invalid_pulse_read" if invalid else "current_limit_reached" if at_limit
              else "compliance" if reading.compliance else "")
    return PulseResult(
        mode="software",
        voltage_set_v=voltage_v,
        voltage_actual_v=reading.voltage_actual_v,
        current_a=reading.current_a,
        current_limit_a=current_limit_a,
        compliance=reading.compliance,
        completed=not bool(reason),
        relative_time_s=reading.relative_time_s,
        source_status=reading.source_status,
        stopped_reason=reason,
    )


def apply_hardware_pulse_2461(
    inst,
    voltage_v: float,
    width_s: float,
    relax_s: float,
    current_limit_a: float,
) -> PulseResult:
    validate_safety_limits(voltage_v, current_limit_a)
    if instrument_model(inst) != "2461":
        raise ValueError("Hardware pulse mode requires Keithley 2461")
    if width_s < HARDWARE_MIN_PULSE_S:
        raise ValueError(
            f"2461 hardware pulse width must be >= {HARDWARE_MIN_PULSE_S:.6f} s"
        )
    set_current_limit(inst, current_limit_a)
    original_timeout = getattr(inst, "timeout", 15000)
    completed = False
    raw = ""
    try:
        inst.write(":ABOR")
        inst.write(':TRAC:CLE "defbuffer1"')
        inst.write(
            ":SOUR:PULS:TR:VOLT "
            f"0,{voltage_v:.9g},{width_s:.9g},1,ON,\"defbuffer1\","
            f"0,{max(relax_s, 0.0):.9g},{current_limit_a:.9e},"
            f"{current_limit_a:.9e},ON"
        )
        inst.timeout = max(original_timeout, int((width_s + relax_s + 2.0) * 1000))
        inst.write(":INIT")
        inst.query("*OPC?")
        count = int(float(inst.query(':TRAC:ACT? "defbuffer1"').strip()))
        if count > 0:
            raw = inst.query(
                f':TRAC:DATA? 1,{count},"defbuffer1",SOUR,READ,REL,SOURSTAT'
            ).strip()
            completed = True
    finally:
        inst.timeout = original_timeout
        safe_shutdown(inst)
    if not raw:
        return PulseResult(
            mode="hardware",
            voltage_set_v=voltage_v,
            voltage_actual_v=float("nan"),
            current_a=float("nan"),
            current_limit_a=current_limit_a,
            compliance=_query_trip(inst),
            completed=False,
            stopped_reason="no_pulse_reading",
        )
    fields = raw.split(",")
    raw_last = ",".join(fields[-4:])
    reading = parse_vi_response(
        raw_last,
        voltage_set_v=voltage_v,
        icc_a=current_limit_a,
        trip=_query_trip(inst),
    )
    reason = "compliance" if reading.compliance else ""
    return PulseResult(
        mode="hardware",
        voltage_set_v=voltage_v,
        voltage_actual_v=reading.voltage_actual_v,
        current_a=reading.current_a,
        current_limit_a=current_limit_a,
        compliance=reading.compliance,
        completed=completed and not reading.compliance,
        relative_time_s=reading.relative_time_s,
        source_status=reading.source_status,
        stopped_reason=reason,
    )


def apply_voltage_pulse(
    inst,
    *,
    mode: Literal["software", "hardware"],
    voltage_v: float,
    width_s: float,
    relax_s: float,
    current_limit_a: float,
) -> PulseResult:
    if mode == "software":
        return apply_software_pulse(
            inst, voltage_v, width_s, relax_s, current_limit_a
        )
    if mode == "hardware":
        return apply_hardware_pulse_2461(
            inst, voltage_v, width_s, relax_s, current_limit_a
        )
    raise ValueError(f"Unknown pulse mode: {mode}")
