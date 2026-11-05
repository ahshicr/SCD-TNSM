"""Causal live-board adapter to the same SCD/DPP score and selection primitives.

Live energy charges are incremental supplied-energy estimates above an explicitly
separate idle baseline. Hardware health and bounded in-flight work constrain all
methods; they are never disabled by an algorithmic thermal ablation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
from src.scheduler_core import (operating_penalty, candidate_score, select_candidate,
                                refill_energy_credit, energy_eligible, path_timing)


METHODS = {'round_robin': 0, 'compute_only': 1, 'energy_greedy': 2,
           'link_greedy': 3, 'deadline_first': 4, 'dpp_matched': 5,
           'scd_no_debt': 6, 'scd_no_link': 7, 'scd_no_forecast': 8,
           'scd_static_weights': 9, 'scd': 10, 'scd_no_thermal': 11,
           'scd_no_guard': 12, 'scd_no_fairness': 13,
           'scd_independent_network': 14, 'scd_no_debt_feedback': 15,
           'scd_elastic_energy': 16}


@dataclass
class LiveNode:
    port: int
    service_by_batch: dict
    idle_w: float
    active_increment_w: float
    link_mbps: float
    transport_fixed_s: float
    transport_seconds_per_mb: float
    power_provenance: str
    temperature_c: float
    healthy: bool = True
    predicted_compute_free: float = 0.
    predicted_port_free: float = 0.
    pending: list = field(default_factory=list)
    service_by_workload: dict = field(default_factory=dict)


class PhysicalPolicy:
    def __init__(self, nodes, method='scd', budget_factor=.5, tenants=4,
                 v_joules=1., energy_margin=1.1, guard=True,
                 constraint_mode='legacy'):
        self.nodes = nodes
        self.method = METHODS[method]
        self.v = v_joules
        self.tenants = tenants
        self.margin = energy_margin
        if constraint_mode not in ('legacy', 'matched'):
            raise ValueError('constraint_mode must be legacy or matched')
        self.constraint_mode = constraint_mode
        self.guard = guard and (constraint_mode == 'matched' or self.method >= 5) and self.method != 12
        # R2 holds request screening constant even in the energy-guard ablation.
        self.request_guard = constraint_mode == 'matched' or self.guard
        self.isolate = self.method != 13
        self.budget = np.array([60.*node.active_increment_w*budget_factor for node in nodes])
        if len(nodes) == 0 or np.any(self.budget <= 0) or v_joules <= 0 or energy_margin < 1:
            raise ValueError('Invalid live model or incremental-energy budget')
        self.debt = np.zeros(len(nodes))
        self.tenant_debt = np.zeros((len(nodes), tenants))
        self.credit = self.budget*2
        self.tenant_credit = np.repeat((self.budget*2/tenants)[:, None], tenants, axis=1)
        self.committed = np.zeros(len(nodes))
        self.screened = np.zeros(tenants)
        self.ingress_free = 0.
        self.last_time = 0.
        self.window = 0
        self.round_robin = 0

    def advance(self, now):
        if now < self.last_time:
            raise ValueError('Causal policy clock moved backwards')
        elapsed = now-self.last_time
        for i in range(len(self.nodes)):
            self.credit[i] = refill_energy_credit(self.credit[i], self.budget[i], elapsed, 2.)
            self.debt[i] = max(0., self.debt[i]-self.budget[i]*elapsed/60.)
            for k in range(self.tenants):
                self.tenant_credit[i, k] = refill_energy_credit(
                    self.tenant_credit[i, k], self.budget[i]/self.tenants, elapsed, 2.)
                self.tenant_debt[i, k] = max(0., self.tenant_debt[i, k]-self.budget[i]*elapsed/(60.*self.tenants))
        if int(now//60) != self.window:
            self.committed[:] = 0.
            self.screened[:] = 0.
            self.window = int(now//60)
        self.last_time = now

    def decide(self, job, now):
        self.advance(now)
        tenant, batch = job['tenant'], job['batch']
        if not 0 <= tenant < self.tenants or batch not in (1, 4, 16, 64):
            raise ValueError('Unsupported tenant or calibrated batch')
        if self.request_guard:
            available = 4096.-self.screened[tenant] if self.isolate else 4096.*self.tenants-self.screened.sum()
            if available < batch:
                return dict(selected=None, reason='request_ceiling')
            self.screened[tenant] += batch
        count = len(self.nodes)
        scores = np.full(count, math.inf)
        feasible = np.zeros(count, dtype=np.bool_)
        candidates = []
        service_id = job.get('service_id', 'mlp')
        services = [n.service_by_workload.get(service_id, n.service_by_batch)[str(batch)] for n in self.nodes]
        energies = np.array([self.margin*n.active_increment_w*s for n,s in zip(self.nodes,services)])
        minimum = energies.min()
        fastest = min(services)
        for i, node in enumerate(self.nodes):
            service = services[i]
            serial_ingress = job['payload_bytes']*8/1e9
            serial_port = job['payload_bytes']*8/(node.link_mbps*1e6)
            ingress, _, departure, queue_s = path_timing(now,
                now if self.method == 14 else self.ingress_free, 0.,
                now if self.method == 14 else node.predicted_port_free,
                serial_ingress, 0., serial_port, 0., 0., 0.)
            observed_path = node.transport_fixed_s+node.transport_seconds_per_mb*job['payload_bytes']/1e6
            # Residual calibration includes SSH/controller service, not another
            # fictional switch hop. Shared serialization/FCFS still appears once.
            residual = max(0., observed_path-serial_ingress-serial_port)
            network = departure-now+residual
            finish = max(now+network, node.predicted_compute_free)+service
            latency = finish-job['release_s']
            debt_ratio = self.debt[i]/self.budget[i]
            tenant_ratio = self.tenant_debt[i, tenant]/(self.budget[i]/self.tenants)
            observed_temp = node.temperature_c
            valid_temp = observed_temp is not None and math.isfinite(observed_temp)
            score_temp = observed_temp if valid_temp else 65.
            h = operating_penalty(latency, job['deadline_s'], energies[i], minimum,
                max(0., node.predicted_compute_free-now), network, service, queue_s,
                job.get('forecast', 0.), fastest/service*math.sqrt(node.link_mbps/1000.),
                self.committed[i]/self.budget[i], (self.committed[i]+energies[i])/self.budget[i],
                debt_ratio, score_temp, 65., 10 if self.method in (14, 16) else self.method)
            if self.method == 1:
                score = latency
            elif self.method == 2:
                score = energies[i]/minimum+2.*max(0., min(latency/job['deadline_s'], 8.)-1.)
            elif self.method == 3:
                score = network+.20*queue_s
            elif self.method == 4:
                normalized = min(latency/job['deadline_s'], 8.)
                score = normalized+4.*max(0., normalized-1.)
            else:
                score = candidate_score(h, energies[i], debt_ratio, tenant_ratio, self.v,
                                        10 if self.method in (14, 16) else self.method)
            health = node.healthy and valid_temp and observed_temp < 65.
            eligible = health and len(node.pending) < 4
            if self.guard:
                eligible = eligible and energy_eligible(energies[i], self.credit[i],
                                                         self.tenant_credit[i, tenant],
                                                         self.isolate and self.method != 16)
            feasible[i] = eligible and latency <= job['deadline_s']
            if eligible:
                scores[i] = score
            candidates.append(dict(port=node.port, eligible=bool(eligible), feasible=bool(feasible[i]),
                operating_penalty=float(h), score=float(score), predicted_energy_j=float(energies[i]),
                predicted_latency_s=latency, predicted_network_s=network, predicted_service_s=service,
                predicted_finish_s=finish, predicted_network_finish_s=now+network,
                predicted_ingress_free_s=ingress, predicted_port_free_s=departure,
                device_debt_j=float(self.debt[i]), tenant_debt_j=float(self.tenant_debt[i, tenant]),
                device_credit_j=float(self.credit[i]), tenant_credit_j=float(self.tenant_credit[i, tenant]),
                observed_temperature_c=observed_temp if valid_temp else None, inflight=len(node.pending)))
        if not np.isfinite(scores).any():
            return dict(selected=None, reason='no_eligible_candidate', candidates=candidates)
        if self.method == 0:
            selected = next((self.round_robin+j)%count for j in range(count)
                            if np.isfinite(scores[(self.round_robin+j)%count]))
            self.round_robin = (selected+1)%count
        else:
            selected = int(select_candidate(scores, feasible, self.method >= 6))
        return dict(selected=selected, reason='dispatch', candidates=candidates)

    def commit(self, job, decision):
        i = decision['selected']
        if i is None:
            raise ValueError('Cannot commit rejected work')
        node, state = self.nodes[i], decision['candidates'][i]
        tenant, charge = job['tenant'], state['predicted_energy_j']
        self.debt[i] += charge
        self.tenant_debt[i, tenant] += charge
        self.committed[i] += charge
        if self.guard:
            self.credit[i] -= charge
            self.tenant_credit[i, tenant] -= charge
        self.ingress_free = state['predicted_ingress_free_s']
        node.predicted_port_free = state['predicted_port_free_s']
        node.predicted_compute_free = state['predicted_finish_s']
        node.pending.append(dict(id=job['id'], service_s=state['predicted_service_s'],
                                 network_finish_s=state['predicted_network_finish_s']))

    def completed(self, index, result, observed_time):
        # Caller delivers only received results. No future execution timestamp
        # is read to update any candidate's current state.
        self.advance(observed_time)
        node = self.nodes[index]
        if not node.pending or node.pending[0]['id'] != result['id']:
            raise ValueError('Physical completion does not match ordered reservations')
        node.pending.pop(0)
        telemetry = result.get('after', result.get('before', {}))
        node.temperature_c = telemetry.get('cpu_c')
        if result.get('status') != 'ok' or node.temperature_c is None or node.temperature_c >= 65.:
            node.healthy = False
        cursor = observed_time
        for pending in node.pending:
            cursor = max(cursor, pending['network_finish_s'])+pending['service_s']
        node.predicted_compute_free = cursor
        if not node.pending:
            node.predicted_port_free = observed_time
        if not any(n.pending for n in self.nodes):
            self.ingress_free = observed_time
