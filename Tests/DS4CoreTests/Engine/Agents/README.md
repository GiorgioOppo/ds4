**English** | [Italiano](README.it.md)

# Agent tests

This folder verifies the predefined profiles of `DS4Engine/Agents` without
starting inference or real tools.

## Coverage

- shared operational contract: the user's language, untrusted tool data,
  multiple rounds and limited side effects;
- least privilege of the Reviewer, which must expose read-only tools only;
- absence of the mutating `git` tool from roles that do not need it.
- explicit delegation scope of the Orchestrator, always a subset of the
  grantable tools and without `git`, cancellation or nested orchestration.

When adding an agent or changing the shared contract, update these
tests together with `Sources/DS4Engine/Agents/README.md`.
