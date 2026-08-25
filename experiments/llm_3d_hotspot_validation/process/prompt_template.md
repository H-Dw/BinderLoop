# Blind prediction-agent contract

You are one fresh `gpt-5.6-sol` agent running at `xhigh` reasoning. Analyze only
the assigned anonymous target-only mmCIF and identity-free feature JSON. You may
write and execute local Python in the assigned scratch directory. Do not access,
list, or search any parent/workspace/repository path, any other run, or any
unassigned file. Do not browse the web, use network tools, install packages,
reverse-identify the structure, or search target literature.

Select a compact, solvent-accessible protein-binding patch using three-dimensional
geometry and physicochemical evidence. Return exactly three ordered primary local
residue tokens and three further ordered, unique alternates. Do not assume or
infer the size of the hidden label set.

Create both `output/prediction.json` and `output/process.md`. The prediction must
be valid JSON matching `prediction_schema.json`, and every residue token must
occur in the assigned `features.json`. State whether identity recognition
occurred without naming a suspected identity. Set all compliance booleans
truthfully and list every file read, command run, and file created.

Anonymous local chains are named `T1`, `T2`, `T3`, ... . Use residue tokens in
the exact form `T<positive-int>:<positive-int>` (for example, `T1:7`).

`process.md` must be non-empty and briefly summarize the local structural basis
for the ranking, files read and written, commands run, any supplemental code or
logs generated in `scratch/`, and a compliance self-check. Use opaque local
residue tokens only: do not name, propose, or discuss any guessed target identity,
protein, family, partner, organism, disease, function, or provenance.
