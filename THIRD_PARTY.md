# Third-party sources and design provenance

This repository contains a new implementation. It does not vendor source code from the projects
below. Their public interfaces and experiment organization informed the design:

- FLIP (`J-SNACKKB/FLIP`, AFL-3.0): GB1 schema and benchmark provenance. Raw GB1 data is
  CC BY 4.0. Downloaded data keeps the upstream license and citation metadata.
- ALDE (`jsunn-y/ALDE`, MIT): discrete batch optimization concepts (Greedy/UCB/TS), uncertainty
  recording, and fixed-budget campaign organization.
- Kermut (`petergroth/kermut`, MIT): inspiration for the optional sequence/structure GP adapter.
- BioDesignBench (`RomeroLab/BioDesignBench`, Apache-2.0): typed agent output, intervention tests,
  and auditable tool-call traces.
- protein-design-mcp (`jasonkim8652/protein-design-mcp`, Apache-2.0): small typed tool boundaries
  and separation of optional structure services.
- Virtual Lab (`zou-group/virtual-lab`, MIT): analyst/hypothesis/critic role separation.

EVOLVEpro is intentionally not included in this version. In particular, this repository does not
implement its lightweight PLM + random-forest architecture.
