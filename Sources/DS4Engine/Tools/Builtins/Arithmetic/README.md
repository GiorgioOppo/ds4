# Builtins/Arithmetic

Local, deterministic tools with no external side effects.

## Tools

- `Add.swift`, `Subtract.swift`, `Multiply.swift`: binary operations shared
  through the `binaryTool` helper.
- `Calculator.swift`: expressions with operators, parentheses, constants, and
  functions.
- `Clock.swift`: ISO-8601 date and time (`now`).

## Dependencies and flow

The closures receive JSON, use the helpers from
[`Tools/Core`](../../Core/README.md), and return `ToolOutput`. They do not
access the project, the network, or the decoder.

## Extension

Enforce numeric domains and clear messages for invalid input. Functions added
to the parser must have deterministic arity and tests for precedence,
parentheses, NaN/infinity, and division by zero.
