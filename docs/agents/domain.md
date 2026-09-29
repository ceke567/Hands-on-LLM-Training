# Domain Docs

The repository uses a **single-context** layout for model training. This file tells engineering skills where to find domain terms and architecture decisions.

## Before exploring

- Read root `CONTEXT.md` when it exists. Use its definitions for terms such as token, step, batch, run, checkpoint, and experiment.
- Read relevant decisions in `docs/adr/` before changing a design they cover.
- If either location does not exist, continue the task. Create a glossary or ADR only when a term or decision actually needs to be settled.

## Layout

- `CONTEXT.md` at the repository root: shared vocabulary and domain boundaries.
- `docs/adr/`: numbered architecture decision records affecting this training project.

Use the terms defined in `CONTEXT.md` in issue titles, plans, code, tests, and reports. If a proposed change conflicts with an ADR, name the ADR and explain the conflict before changing direction.
