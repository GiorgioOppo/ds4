# DwarfStar Shared

Cross-feature application support lives here. Code in `Shared/` must be usable
by more than one feature and must not depend on a feature-specific view or
controller.

- `Support/` contains process and engine-log utilities.

Prefer `DS4Core` or `DS4Engine` for reusable domain logic. This directory is
only for app-layer adapters and helpers.

