# Public API

The names below are the complete supported export surface in `frend.__all__` for
0.1.0. Signatures and data-class fields remain available through Python's normal
`help()` and inspection tools.

- `BehaviorLoadError` — reports every refusal from one transactional behavior-schema validation.
- `BudgetExhausted` — reports that a declared tiergraph work budget was exhausted.
- `ChoiceGraph` — joins a choice lattice to the spoken alternatives for each edge.
- `ChoiceLattice` — carries every distinct scored reading edge without selecting a route.
- `DEFAULT_MAX_INPUT_CHARS` — the default document input limit, 4,194,304 code points.
- `DEFAULT_MAX_UNIT_CHARS` — the default graph-building unit limit, 8,192 code points.
- `DEFAULT_SYMBOL_RUN_THRESHOLD` — the default adjacent-symbol run threshold, 3.
- `DEFAULT_TIERGRAPH_WORK_BUDGET` — the default deterministic tiergraph work budget.
- `InputValidationError` — reports input that is not bounded plain Unicode text.
- `InputFold` — the type of a named length-preserving input fold.
- `LatticeNode` — identifies one source-text code-point boundary in a lattice.
- `MAX_NON_TEXT_SHARE` — the maximum accepted share of controls and unassigned code points, 0.01.
- `NormalizedText` — returns normalized text with source alignments and fold metadata.
- `NormalizedUnit` — maps one output span to its source span, reader, and provenance.
- `NormalizationBatch` — returns newly settled stream text, alignments, and source watermark.
- `NormalizationStream` — incrementally normalizes chunks after sentence decisions settle.
- `PriorSummary` — records per-reading priors and their supported count for a path.
- `ReadingEdge` — represents one candidate reading or passthrough lattice transition.
- `ReadingLattice` — stores the resolved lattice and its ranked public path projection.
- `ReadingPath` — represents one complete ranked route through reading and passthrough edges.
- `ReadingRank` — holds decomposed geometry and semantic rank components for an edge.
- `Resolution` — returns the best detections, ranked covers, margins, and ambiguity state.
- `ResolvedBehavior` — holds immutable composed behavior arguments and provenance.
- `SemanticRank` — holds exact measured, generated, or unsupported semantic ranking data.
- `SentenceBase` — the type of an optional named sentence-ranking model.
- `SpokenAlternative` — holds one spoken form with provenance, weight, group, and role.
- `VerbalizedLattice` — holds verbalized projected paths and their best path.
- `VerbalizedPath` — holds one ranked route and its factored verbalized units.
- `VerbalizedUnit` — holds one edge's spoken alternatives and reading-prior trace.
- `__version__` — gives the installed frend release version.
- `compose_choices` — attaches spoken forms to every edge in a choice lattice.
- `apply_input_fold` — applies a declared length-preserving fold to input text.
- `freeze_after_setup` — moves currently live objects into Python's permanent GC generation.
- `normalize` — converts a plain-text document to its first-choice spoken form or alignment.
- `prewarm` — loads and compiles the production path once for each requested locale.
- `resolve` — resolves detections into ranked, non-overlapping covers.
- `resolve_behavior` — loads and composes behavior presets or JSON files in caller order.
- `resolve_lattice` — recognizes or accepts detections and returns a distilled reading lattice.
- `resolve_choices` — carries all distinct readings as scored edges without choosing a route.
- `route_geometry` — computes the cover score of a selected choice-lattice route.
- `register_curated_alternative` — registers a caller-owned additive spoken alternative.
- `verbalize_edge` — validates source text and verbalizes one reading edge.
- `verbalize_lattice` — verbalizes every projected path without multiplying unit alternatives.
- `validate_input` — validates and returns bounded plain Unicode text.
- `validate_unit_length` — enforces document and per-unit graph-work length bounds.
