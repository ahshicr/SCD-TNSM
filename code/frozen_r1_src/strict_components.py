from __future__ import annotations

"""Reference state machines used to specify and test the compiled simulator."""

from dataclasses import dataclass, field
import math
from typing import Iterable

import numpy as np

from src.scheduler_core import hop_jitter_mean


@dataclass(frozen=True)
class CacheDecision:
    cold: bool
    feasible: bool
    evicted_services: tuple[int, ...] = ()


@dataclass
class MemoryBoundLRUCache:
    """A TTL cache whose capacity is resident memory, not an entry count.

    The timestamp is the completion of the last invocation.  A queued request
    tests the cache at its actual or predicted execution start.  No reservation
    is made at release or dispatch time.
    """

    capacity_mb: float
    ttl_s: float
    entries: dict[int, tuple[float, float]] = field(default_factory=dict)

    def _live_entries(self, start_s: float) -> dict[int, tuple[float, float]]:
        return {
            service: value
            for service, value in self.entries.items()
            if start_s - value[0] <= self.ttl_s
        }

    def preview(self, service: int, footprint_mb: float, start_s: float) -> CacheDecision:
        if footprint_mb <= 0:
            raise ValueError("container footprint must be positive")
        if footprint_mb > self.capacity_mb:
            return CacheDecision(cold=True, feasible=False)
        live = self._live_entries(start_s)
        if service in live:
            return CacheDecision(cold=False, feasible=True)
        occupied = sum(value[1] for value in live.values())
        evicted: list[int] = []
        for victim, (_, memory) in sorted(live.items(), key=lambda item: item[1][0]):
            if occupied + footprint_mb <= self.capacity_mb:
                break
            occupied -= memory
            evicted.append(victim)
        return CacheDecision(cold=True, feasible=True, evicted_services=tuple(evicted))

    def access(
        self,
        service: int,
        footprint_mb: float,
        start_s: float,
        completion_s: float,
    ) -> CacheDecision:
        if completion_s < start_s:
            raise ValueError("completion precedes execution start")
        decision = self.preview(service, footprint_mb, start_s)
        if not decision.feasible:
            return decision
        self.entries = self._live_entries(start_s)
        for victim in decision.evicted_services:
            self.entries.pop(victim, None)
        self.entries[service] = (completion_s, footprint_mb)
        return decision

    @property
    def occupied_mb(self) -> float:
        return sum(value[1] for value in self.entries.values())


@dataclass(frozen=True)
class RoutePreview:
    completion_s: float
    queue_s: float
    jitter_s: float
    link_ids: tuple[int, ...]
    link_available_after_s: tuple[float, ...]


@dataclass
class FluidPacketTrainNetwork:
    """FCFS store-and-forward network matching the manuscript recursion."""

    rate_bps: np.ndarray
    fixed_s: np.ndarray
    propagation_s: np.ndarray
    background_fraction: np.ndarray
    residual_exponent: float = 1.0
    minimum_capacity_fraction: float = 0.05
    mtu_bytes: int = 1500
    available_s: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        lengths = {
            len(self.rate_bps),
            len(self.fixed_s),
            len(self.propagation_s),
            len(self.background_fraction),
        }
        if len(lengths) != 1:
            raise ValueError("network parameter arrays must have equal length")
        if np.any(self.rate_bps <= 0):
            raise ValueError("link rates must be positive")
        if np.any((self.background_fraction < 0) | (self.background_fraction >= 1)):
            raise ValueError("background fractions must lie in [0, 1)")
        if self.residual_exponent <= 0:
            raise ValueError("residual-capacity exponent must be positive")
        if not 0 < self.minimum_capacity_fraction <= 1:
            raise ValueError("minimum capacity fraction must lie in (0, 1]")
        self.available_s = np.zeros(len(self.rate_bps), dtype=np.float64)

    def preview(
        self,
        route: Iterable[int],
        release_s: float,
        payload_bytes: float,
        z_by_hop: np.ndarray | None = None,
    ) -> RoutePreview:
        if payload_bytes <= 0:
            raise ValueError("payload must be positive")
        link_ids = tuple(int(link) for link in route)
        if not link_ids:
            raise ValueError("route must contain at least one link")
        if z_by_hop is not None and len(z_by_hop) < len(link_ids):
            raise ValueError("one network variate per hop is required")
        hop_arrival = float(release_s)
        last_bit_completion = float(release_s)
        total_queue = 0.0
        total_jitter = 0.0
        after: list[float] = []
        for hop, link in enumerate(link_ids):
            if link < 0 or link >= len(self.rate_bps):
                raise IndexError(f"invalid link id: {link}")
            capacity_fraction = max(
                (1.0 - self.background_fraction[link]) ** self.residual_exponent,
                self.minimum_capacity_fraction,
            )
            effective_rate = self.rate_bps[link] * capacity_fraction
            serialization = payload_bytes * 8.0 / effective_rate
            service = serialization + self.fixed_s[link]
            start = max(hop_arrival, float(self.available_s[link]))
            queue = start - hop_arrival
            link_finish = start + service
            jitter_mean = hop_jitter_mean(
                serialization, float(self.background_fraction[link])
            )
            if z_by_hop is None:
                jitter = jitter_mean
            else:
                probability = 0.5 * (
                    1.0 + math.erf(float(z_by_hop[hop]) / math.sqrt(2.0))
                )
                probability = min(max(probability, 1.0e-12), 1.0 - 1.0e-12)
                jitter = -jitter_mean * math.log1p(-probability)
            total_queue += queue
            total_jitter += jitter
            # Residence jitter is part of this FCFS service reservation.  This
            # prevents a later train from overtaking an earlier train merely
            # because its sampled residence time is shorter.
            link_busy_finish = link_finish + jitter
            hop_arrival = link_busy_finish + self.propagation_s[link]
            last_bit_completion = link_busy_finish + self.propagation_s[link]
            after.append(link_busy_finish)
        return RoutePreview(
            completion_s=last_bit_completion,
            queue_s=total_queue,
            jitter_s=total_jitter,
            link_ids=link_ids,
            link_available_after_s=tuple(after),
        )

    def commit(self, preview: RoutePreview) -> None:
        for link, available in zip(
            preview.link_ids, preview.link_available_after_s, strict=True
        ):
            if available + 1.0e-12 < self.available_s[link]:
                raise ValueError("network commit would move link state backward")
            self.available_s[link] = available


def thermal_advance(
    temperature_c: float,
    duration_s: float,
    supplied_power_w: float,
    idle_power_w: float,
    idle_temperature_c: float,
    gain_c_per_w: float,
    tau_s: float,
) -> float:
    if duration_s < 0 or tau_s <= 0:
        raise ValueError("thermal duration must be nonnegative and tau positive")
    steady = idle_temperature_c + gain_c_per_w * (supplied_power_w - idle_power_w)
    return steady + (temperature_c - steady) * math.exp(-duration_s / tau_s)


@dataclass
class ThermalState:
    idle_power_w: float
    active_power_w: float
    idle_temperature_c: float
    gain_c_per_w: float
    tau_s: float
    temperature_c: float | None = None
    state_time_s: float = 0.0

    def __post_init__(self) -> None:
        if self.temperature_c is None:
            self.temperature_c = self.idle_temperature_c

    def preview(self, start_s: float, active_duration_s: float) -> tuple[float, float]:
        if start_s < self.state_time_s:
            raise ValueError("thermal reservations must be time ordered")
        at_start = thermal_advance(
            float(self.temperature_c),
            start_s - self.state_time_s,
            self.idle_power_w,
            self.idle_power_w,
            self.idle_temperature_c,
            self.gain_c_per_w,
            self.tau_s,
        )
        at_finish = thermal_advance(
            at_start,
            active_duration_s,
            self.active_power_w,
            self.idle_power_w,
            self.idle_temperature_c,
            self.gain_c_per_w,
            self.tau_s,
        )
        return at_start, at_finish

    def commit(self, start_s: float, active_duration_s: float) -> tuple[float, float]:
        at_start, at_finish = self.preview(start_s, active_duration_s)
        self.temperature_c = at_finish
        self.state_time_s = start_s + active_duration_s
        return at_start, at_finish


@dataclass
class TokenBucket:
    rate_per_s: float
    capacity: float
    tokens: float | None = None
    state_time_s: float = 0.0

    def __post_init__(self) -> None:
        if self.rate_per_s < 0 or self.capacity < 0:
            raise ValueError("token bucket parameters must be nonnegative")
        if self.tokens is None:
            self.tokens = self.capacity

    def advance(self, time_s: float) -> None:
        if time_s < self.state_time_s:
            raise ValueError("token bucket time cannot move backward")
        self.tokens = min(
            self.capacity,
            float(self.tokens) + (time_s - self.state_time_s) * self.rate_per_s,
        )
        self.state_time_s = time_s

    def consume(self, amount: float, time_s: float) -> bool:
        if amount < 0:
            raise ValueError("token amount must be nonnegative")
        self.advance(time_s)
        if float(self.tokens) + 1.0e-12 < amount:
            return False
        self.tokens = float(self.tokens) - amount
        return True

    def refund(self, amount: float, time_s: float) -> None:
        if amount < 0:
            raise ValueError("refund must be nonnegative")
        self.advance(time_s)
        self.tokens = min(self.capacity, float(self.tokens) + amount)
