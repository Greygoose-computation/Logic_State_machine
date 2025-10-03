# Automata Supervisor Synthesis

This project implements basic automata operations and a supervisor synthesis algorithm for Discrete Event Systems, using Python.

## Functions

### `reach(events, trans, start_states, forbidden)`
- **Purpose**: Computes all *reachable states* in an automaton from a given set of start states.
- **How it works**:
  - Start from the initial (or given) states.
  - Follow all transitions step by step.
  - Exclude any forbidden states.
- **Usage**: Helps identify the states actually reachable in the system.

---

### `synch(automaton1, automaton2)`
- **Purpose**: Computes the *synchronous product* of two automata.
- **How it works**:
  - States are combined as pairs (`qA.qB`).
  - Shared events synchronize (both automata must take them together).
  - Private events of one automaton interleave freely with the other.
- **Usage**: Used to combine plant models and specifications into one system.

---

### `supervisor(P, Sp, sigma_u)`
- **Purpose**: Synthesizes a *nonblocking* and *controllable* supervisor for the plant `P` under specification `Sp`.
- **How it works**:
  1. Build product `P || Sp`.
  2. Mark unsafe states (forbidden or uncontrollable violations).
  3. Compute reachable states.
  4. Iteratively prune:
     - **Nonblocking**: keep only states that can reach a marked state.
     - **Controllability**: remove states where uncontrollable events escape the safe set.
  5. Return restricted automaton `S`.
- **Usage**: Guarantees the system behaves according to the specification and respects uncontrollable events.

---

## Example: Stick-Picking Game

We model a simple game:

- **Rules**:  
  - 5 sticks on the ground.  
  - Players A and B take turns removing 1 or 2 sticks.  
  - The player forced to take the **last stick loses**.  
  - A always starts.

- **Plant `P`**:  
  - States represent `(sticks_remaining, turn)`.  
  - Events: `a1, a2` (A takes 1 or 2), `b1, b2` (B takes 1 or 2).  
  - Transitions reduce the stick count and swap turns.

- **Specification `Sp`**:  
  - Mark states where **A wins** (e.g., `(1, B)` — B is forced to take the last stick).  

- **Uncontrollable events**:  
  - B’s moves (`b1, b2`) are uncontrollable.  
  - A’s moves are controllable.  

- **Supervisor `S = supervisor(P, Sp, sigma_u)`**:  
  - Disables losing choices for A.  
  - Ensures A follows a winning strategy (e.g., always start with `a1`).  

---

## Running the Example

```python
P, Sp, sigma_u = build_stick_picking_game()
S = supervisor(P, Sp, sigma_u)
plot_automaton(S, "StickPickingSupervisor")
