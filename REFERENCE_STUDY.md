# Reference Study: IEEE VGC AI Competition

Working notes on the competition we are targeting. Citations only — no inferred conclusions. Researched 2026-05-11.

## Competition identity

- **Name**: VGC AI Competition (a.k.a. "Pokemon VGC AI Competition 2.0").
- **Organizing body**: IEEE Conference on Games (CoG), under IEEE Computational Intelligence Society.
- **Organizer**: Simão Reis (Vortex-CoLab + LIACC, Univ. Porto). Co-organizers: A. Lucas Martins, Rita Novais, Fernando Alves.
- **Current edition**: **4th**, at **CoG 2026**, Madrid, **Sept 1-4, 2026**.
- **Prior editions**: 1st (2023), 2nd (CoG 2024 Milan), 3rd (CoG 2025 Lisbon, INESC-ID, Feb 1-June 30 submission).
- **Status as of 2026-05-19**: 4th-edition wiki page still unpublished, but Reis posted a Discord announcement confirming 3-track structure + paradigm shifts and shipping framework v2.1.0 (Feb 2026) / v2.1.1 (Apr 2026). Submission timeline still TBD.
- **Contact**:
  - Discord: <https://discord.gg/GwKHqXpdjf>
  - Email: simao.reis@vortex-colab.com
  - CoG 2026: <https://cog2026.org/competitions>
  - 2025 site (reference): <https://cog2025.inesc-id.pt/vgc-ai-competition/>

## Framework

- **Repo**: <https://gitlab.com/DracoStriker/pokemon-vgc-engine> (MIT, 460+ commits, active).
- **Package**: `vgc2` (also called "VGC AI Framework 2", version 2.1.1 on master).
- **Pinned commit (this project)**: `b0b77f9ba0b6b1ae297255fd867a6a866e74bb66` — this **is** v2.1.1 (Apr 2026 release: "Fixed initial reserve on BattlingTeam"), the 4th-edition framework Reis announced for the 2026 competition.
- **Companion baselines**: <https://gitlab.com/DracoStriker/vgc-agents>.
- **Python**: 3.10.12 declared, works on 3.12 in practice.
- **Deps**: `gymnasium~=1.0`, `numpy~=2.2`, `setuptools~=75.8`. **No PyTorch required by the framework.**
- **Not Showdown-compatible.** Standalone simulator. Fictional roster, parametric moves. No real species, abilities, items, dynamax, or tera. Supports both singles and doubles via the `n_active` parameter — **Battle Track default is `n_active=2` (doubles)** per `organization/run_battle_track.py`. (Earlier reading of this study said "singles only"; that was incorrect.)

### Key abstractions (`vgc2`)

- `Competitor` (in `vgc2.competition`) — Battle/Championship submission. Override `battlepolicy`, `selectionpolicy`, `teambuildpolicy`, `name`.
- `DesignCompetitor` (in `vgc2.competition`) — Balance Track submission. Override `metabalancepolicy`, `rulebalancepolicy`, `name`.
- `BattlePolicy`, `SelectionPolicy`, `TeamBuildPolicy` (in `vgc2.agent.*`) — strategy components.
- `MetaBalancePolicy`, `RuleBalancePolicy` (in `vgc2.agent.*`) — Balance Track ABCs (new in v2.1.x).
- New `BattlePolicy` / `SelectionPolicy` lifecycle hooks (v2.1.x): `set_params(params)` (rule perturbations for general game-playing), `set_meta(meta)` (meta priors for Championship), `on_new_battle()` (per-battle reset).
- Random defaults: `RandomBattlePolicy`, `RandomSelectionPolicy`, `RandomTeamBuildPolicy`.
- `CompetitorManager` wraps a Competitor for evaluation.
- `BattleEcosystem`, `ChampionshipEcosystem`, `MetaDesign`, `RuleDesign` — simulation drivers.
- `PkmRoster`, `MetaData` — define the universe per epoch.
- `RemoteCompetitorManager` — serve a Competitor as a network process (see `template/main.py`).
- Track entry points in `organization/`: `run_battle_track.py`, `run_championship_track.py`, `run_meta_balance_track.py`, `run_rules_balance_track.py` (framework retains separate meta + rules entry points even though they're merged at the submission level).

## Tracks (2026 — confirmed via Reis Discord announcement, 2026-05-19)

| Track                     | What you submit                                                         | Key API                                                       | Eval                                                                              |
| ------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Battle**                | `BattlePolicy` + `SelectionPolicy` (general game-playing)               | `self.params` reads dynamically generated rules per tournament| Adversarial play under unseen mechanics; overfitting to fixed rules is penalised  |
| **Championship**          | `TeamBuildPolicy` + battle + selection                                  | `BattlePolicy.set_meta(meta)` consumes team-build meta priors | Round-robin across championship epochs; meta-game reasoning is the new emphasis   |
| **Balance (Meta + Rule)** | `DesignCompetitor` exposing `MetaBalancePolicy` + `RuleBalancePolicy`   | Designer agents access the battle agents used for evaluation  | Behaviour-based: meta-balance targets usage diversity; rule-balance shapes moves  |

**What changed from 2025:** the 2025 edition had two public tracks (Battle, Championship). For 2026 Reis added the Balance Track (merging the framework's separate Meta-Balance and Rules-Balance evaluators into one submission), reframed Battle around general game-playing (dynamic rules), and gave Championship explicit hidden-information / meta-history reasoning.

## Submission & evaluation

- Submission was a Google Form in 2025 (procedure for 2026 TBD).
- Python implementation of `Competitor`, no Docker requirement in 2025.
- No published memory/time limits; aim for sub-second per turn to stay safe.
- Stochastic battles (damage rolls, generation) — high variance is intentional.

## Past results

Public records are thin. No official leaderboard on the wiki. Ask Reis directly on Discord for full rankings.

- **2024, 3rd place**: AurelianTactics. Approach: **tabular first-visit Monte Carlo**, ~30M trials, 11-dim collapsed observation space, action sampling with `p=0.05` / min 100 visits per (state, action). **Explicitly stated deep RL failed due to game randomness.** ([writeup](https://medium.com/@aureliantactics/vgc-ai-competition-2024-edition-3rd-place-submission-5420d2f6aafe))
- 2024 notable names referenced in search snippets but unverified: "EnhancedBot", "Punisher".
- 2023, 1st place: "Dominik Baziuk" (one search snippet, unverified).

## Foundational reading

- Reis, Reis, Lau, *VGC AI Competition - A New Model of Meta-Game Balance AI Competition*, CoG 2021 — [IEEE 9618985](https://ieeexplore.ieee.org/document/9618985).
- Reis et al., *An Adversarial Approach for Automated Pokemon Team Building*, IEEE ToG 2023 — [IEEE 10115492](https://ieeexplore.ieee.org/document/10115492).
- Reis et al., *A New Rules Balance Track for the Pokemon VGC AI Competition 2.0*, CoG 2025 — [IEEE 11114412](https://ieeexplore.ieee.org/document/11114412). Also: [Vortex-CoLab summary](https://www.vortex-colab.com/publications/a-new-rules-balance-track-for-the-pokemon-vgc-ai-competition-2-0/).

## Honest assessment (don't sugar-coat)

- **Small, niche academic competition.** No public leaderboard, results not posted on wiki, only one published place-holder writeup. Entries per track in 2024 were almost certainly under 10.
- **Prize**: historically $1,000 IEEE CIS Education prize.
- **One primary maintainer** (Reis).
- **Realistic effort to win**: 4-8 weeks focused work on (a) MCTS/expectimax + heuristic eval for Battle and Championship, (b) GA or LP for team building, (c) optional Rules Balance entry. One competent engineer, no GPU needed. Top-3 is credible; top-1 hinges on polish and avoiding submission-format mistakes.

## Implications for code

- `pyproject.toml` pins `vgc2` at the commit above (= v2.1.1, the 4th-edition framework). Re-pin only when Reis publishes a v2.2.x successor or names a new SHA.
- All Pokemon-Showdown / poke-env code was removed in the pivot commit; see `docs/archive/study_pokeenv.md` for the historical record of that direction.
- No GPU dependency. PyTorch and similar are not required deps; add only if a specific learned component proves necessary later.

## Measured baselines (2026-05-11, doubles, team_size=4, max_pkm_moves=4)

| Matchup            | n  | Win rate (a) | Per-battle wall time |
| ------------------ | -- | ------------ | -------------------- |
| Greedy vs Random   | 20 | 90.0%        | 0.04 s               |
| Tree   vs Random   | 3  | 100.0%       | 170.9 s              |

Key takeaway: `TreeSearchBattlePolicy(max_depth=1)` is unusable for live submission in doubles — ~11 s per turn. `GreedyBattlePolicy` is the submission-viable floor. Our default `VgcAiBattlePolicy` is therefore aliased to Greedy; replacing it requires both (a) beating Greedy in head-to-head duels and (b) fitting a sub-second-per-turn budget.
