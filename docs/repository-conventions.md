# Repository conventions

Every solution record names its status, model revision, runtime, input contract, output schema, device, and known limits. Use these status values consistently:

- `verified`: the stated acceptance checks passed on the named device and runtime.
- `loaded-only`: the artifact loaded and ran, but product quality is not established.
- `experimental`: the design is still changing.
- `failed`: the tested approach did not meet its stated requirement.
- `not-verified`: the required device, fixture, or measurement is missing.

Keep preview throughput, model inference rate, and rendered result age as separate measurements. A stale or malformed result must be rejected before it reaches the UI.
