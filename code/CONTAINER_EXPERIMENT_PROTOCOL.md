# Container startup experiment — fixed before collecting outcomes

The complete 280-trial scheduler campaign, its analysis and its byte-verified
archive finished before any container timing. Preliminary or incomplete
container acquisitions remain separate and cannot supply final statistics. The
reported four-platform matrix starts from block zero and includes only boards
that pass the predeclared read--write, numerical-parity, temperature and cleanup
checks. The algorithms, digest, paired-block count, acquisition count, seeds,
safety latch and analysis rules below are fixed for the complete matrix.

## Scope and design

- Four separately identified generations: Pi 5 port 5, Pi 4 port 9, Pi 3 B+
  port 7, Pi 2 port 8. This is a measured platform subset, not a claim of runtime
  support or container performance for Nano/RDK or all 16 boards.
- One board at a time, fixed order 5/9/7/8. No concurrent policy campaign or
  controller simulation. Within a board use 20 matched blocks with one fresh
  cached-image container and one host-process activation in randomized order.
- Each activation loads the identical trained 66–256–128–64–1 GELU MLP and
  performs a first batch-1 inference, then 20 resident batch-1 inferences.
  Inputs rotate deterministically through the same 128 validation rows.
  Every prediction must pass the existing relative tolerance 1e-4.
- Ten seconds idle between activations. Record temperature at start and after
  each inference. A missing identified CPU sensor or >=65 C stops that board;
  do not relabel a guarded incomplete block as a completed repetition.
- Twenty fresh cached-image containers per board measure creation plus Python
  and model activation. Within-container calls separately measure warm service.
  Model JSON transmission is separately timed; all boundaries are reported.
- Host and container Python build/version strings are retained. Their paired
  difference compares the complete deployment stacks, not pure kernel isolation
  overhead. The pinned initial image tag is the official multi-architecture
  `docker.io/library/python:3.11.16-slim-bookworm`; after the first successful
  acquisition every pull and activation uses the inspected immutable digest.
- Five image-uncached pulls per board are a separate acquisition experiment,
  not included silently in the cached-container interval. The exact image
  reference, resolved immutable digest, compressed/downloaded bytes where
  exposed, network failures, elapsed times and cache-inspection outputs are
  retained. No OS-page-cache flush, reboot or global image pruning is allowed.
- Registry/CDN caches and the outside Internet are not experimentally reset.
  "Image-uncached" means absent from this study's private daemon store, not
  globally uncached content. Failed downloads remain explicit observations.
- Statistics use 20 block-level means/differences per board, not 400 correlated
  resident requests as independent startup trials. Report all five acquisition
  observations; these are small-sample network-conditioned intervals, not a
  universal cold-start latency distribution.
- Run the 20 activation blocks in the first successfully acquired private store.
  Then finish the remaining predeclared five acquisition attempts in fresh
  stores. This availability rule does not select a fast acquisition. Failed
  attempts are retained and are never replaced with extra favorable trials.

## Isolated runtime and safety

Use the existing noninteractive sudo authorization on the selected Pi boards;
stop if it is unavailable. Do not change Nano's Docker permissions or use its
existing daemon. Download the Docker 29.8.0 static distribution from the official
HTTPS host only after the policy campaign. Record the download SHA256 and exact
architecture (`aarch64` or `armhf`). The URL's existence does not prove that the
binary works on a given installed kernel; preflight must establish that.
The preflight also requires two GiB of free local storage, cgroups, identified
CPU sensors and no existing Docker daemon/socket. Where AppArmor is enabled,
the default enforcing Docker profile must already be present; do not disable
or replace a host security profile to make this experiment run.

All executable files, explicit daemon configuration, data/exec roots, PID file,
Unix socket and client configuration belong to one freshly created study-specific
temporary directory. Refuse a symlink, ambiguous path or existing default daemon.
Start no system service and change no `/etc`, `/usr/bin`, login groups, IP
forwarding, switch configuration, firewall or host security policy.

The private daemon has bridge `none`, no TCP listener, disabled iptables/ip6tables
and IP-forward/masquerade management, its own socket and roots, classic `vfs`
storage and the containerd image-store feature disabled. `vfs` avoids pretending
that different kernels offer the same overlay support; its explicit storage
cost must be reported and must not be generalized to production overlayfs.
Inference containers use network `none`, read-only rootfs, dropped capabilities,
no-new-privileges and default seccomp. Only the worker/model source and read-only
sensor filesystem are mounted; no Docker socket is mounted into a container.
Container processes run as the original non-root SSH UID/GID, with a 512-MiB
memory ceiling and at most 64 processes. Temperature is checked during pulls
and archive extraction as well as before and after every inference.

Before cleanup, stop only containers created by this study on the explicit
private socket. Terminate only the validated child daemon PID. Confirm no child
process or mount remains under the verified temporary directory. Preserve raw
logs, model/source hashes, runtime/image identities and failure records locally
before removing generated runtime state. Never use default-socket pruning or
broad filesystem deletion. Unexpected existing services or permission needs
are a stop condition, not permission to change the machine.
The driver deliberately retains the named stopped runtime directories until
their copied evidence and exact safe cleanup targets can be reviewed. It does
not recursively delete any directory. Retention is not an unreported active
service: daemon, container and mount shutdown are checked for each store.

## Primary implementation sources

- https://docs.docker.com/engine/install/binaries/
- https://docs.docker.com/reference/cli/dockerd/
- https://docs.docker.com/engine/storage/drivers/select-storage-driver/
- https://download.docker.com/linux/static/stable/aarch64/
- https://download.docker.com/linux/static/stable/armhf/

These describe installation/runtime behavior, not experimental results. The
complete container evidence and its corresponding measured-startup simulation
must be produced before closing Reviewer 1's cold-start comment.
