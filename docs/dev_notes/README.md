# Dev notes (audit trail, not the shipped spec)

These documents record the reasoning behind the D-metric design, not what the
pipeline currently publishes. Kept for historical/audit purposes; the release
docs are one level up (`docs/dataset_formats.md`, `docs/evaluation_protocol.md`,
`docs/metrics_reference.md`).

- `d_metrics_audit.md` — v1 audit of the D-suite against the v1 report. Superseded
  by `d_metrics_proposal_audit.md` for the D-suite.
- `d_metrics_proposal_audit.md` — v2 audit that produced the construct-valid
  successor design in `d_metrics_methodology.md`. References a scratch working
  file (`.rr_audit_tmp/proposal.md`) that is not part of this repo.
- `d_metrics_methodology.md` — the audited successor methodology. Describes a
  proposed/gated design, not the numbers the live engine (`evaluation/d_metrics.py`)
  publishes today — see its "What this is NOT" note.
- `yolopx_adapter.md` — empty stub, kept only so the historical filename isn't lost.
