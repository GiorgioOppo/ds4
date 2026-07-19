**English** | [Italiano](README.it.md)

# Builtins/Agents

Exposes the agent catalog and delegation of isolated work.

## Tools

- `agents_list`: available roles and their associated tools.
- `subagent_search`: delegated orientation/search.
- `subagent_run`: complete request with target, role, and limits.

## Flow and dependencies

The specs read [`AgentRegistry`](../../../Agents/README.md); the actual
sub-agent execution is handled by
[`Inference/Subagents`](../../../Inference/Subagents/README.md), so the tool
does not own the decoder directly.
The role and the `tools` list are model inputs and can only narrow the parent
profile's `delegatedToolNames` scope, never widen it.

## Extension

Do not make the orchestration tools themselves grantable to a sub-agent.
Limit depth, rounds, tokens, and granted tools to prevent recursion or
uncontrolled amplification.
