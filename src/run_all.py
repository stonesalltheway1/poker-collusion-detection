"""End-to-end reproduction of the SHIPPED submissions (prize-eligibility requirement).

Reproduces, from data/raw only:
    submissions/sub025_fusebags_exp042ramp.csv (public LB 0.91942)  risk exp048_fuse_bags    + evidence exp042_ramp
      ^ SELECTED FINAL 1
    submissions/sub028_famspecw15_exp042ramp.csv (public LB 0.91935) risk exp068_famspecmix_w15 + evidence exp042_ramp
      ^ SELECTED FINAL 2
    submissions/sub027_fusebags_exp036flat.csv (public LB 0.91918)  risk exp048_fuse_bags    + evidence exp036 (flat)
    submissions/sub024_fuse_exp042ramp.csv     (public LB 0.91954)  risk exp044_fuse_lgb_xgb + evidence exp042_ramp

This file REPLACES src/run_all.py, which reproduced an early-September configuration
(exp027_bag risk + exp008 evidence, LB ~0.911) and is missing 11 of the stages the shipped
models depend on.  Every stage below was reverse-engineered from the scripts themselves, from
experiments/LEDGER.md, from scripts/exp034_samefold.sh, and from the metadata written next to
the artefacts (oof/*_views.json, data/derived/evidence_cache/listing_rule_exp036.json).
The four fusion recipes were verified numerically: re-running scripts/fuse.py with the member
lists below reproduces the shipped oof/<exp>_eval_scores.parquet rank vector to corr = 1.00000000.

--------------------------------------------------------------------------------------------
PRE-CONDITIONS FOR A FRESH CLONE  (see "REPRO BLOCKERS" at the bottom -- must be fixed in git)
--------------------------------------------------------------------------------------------
  1. data/raw/*.parquet from the Kaggle competition page.
  2. data/folds_tables_5.csv            -- regenerated automatically and bit-identical
                                           (common.get_table_folds, numpy default_rng(42)); verified.
  3. data/derived/spies_exp005.parquet (128 rows) and spies2_exp022.parquet (26 rows).
     No script in the repo writes either; every pair model reads them (ADD_SPIES_POS=1).
     Both were force-added to git in commit 5e6e69e, so a fresh clone now has them.
  4. research/forensics/oc_family_feats.json -- committed, good (oc_hunt.famflags needs it).
  5. Packages: polars, pandas, numpy, lightgbm, xgboost, catboost(unused here), numba, duckdb,
     scikit-learn, pyarrow.  A CUDA GPU is required by the XGBoost stages as written
     (device="cuda"); set XGB_DEVICE=cpu below to run without one -- this changes the numbers
     slightly and must be disclosed if used.

Wall clock on the campaign machine (i5-10400F / 12 threads / 32 GB / GTX 1660S): ~10-14 h.

Usage:  python run_all.py [stage ...] [--force] [--list] [--dry-run]
        (no stage argument = every stage, in order; each is skipped when its output exists)
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DER = BASE / "data" / "derived"
OOF = BASE / "oof"
PY = sys.executable

# XGBoost device.  The shipped models were trained with device="cuda"; exp_xgb_pairmodel.py reads
# this env var only if you apply the one-line patch noted in REPRO BLOCKER #5.
XGB_DEVICE = os.environ.get("XGB_DEVICE", "cuda")

# ---------------------------------------------------------------------------- shared env blocks
E_ENGINE_V1 = dict(PH_DIR="ph", NUMBA_NUM_THREADS="6", POLARS_MAX_THREADS="4")
E_ENGINE = dict(PH_DIR="ph_v2", NUMBA_NUM_THREADS="6", POLARS_MAX_THREADS="4")

# the pair-model feature switches, identical for every shipped pair model (LGBM and XGB):
#   CFG=C                 feature configuration C
#   PF_TAG=v2             engine pair features from ph_v2   -> pfv2_<window>.parquet
#   PFH_TAG=v4sf          hand-detector pair features, round-4 self-trained + SAME-FOLD eval scoring
#   MIX_TAG=v4sf          mixture-LLR pair features from the same detector
#   USE_OC=1              other-coordination pair scores
#   ADD_SPIES_POS=1 W_SPY=0.5   frozen likely-hidden-positive pairs promoted to weak positives
E_PAIR = dict(PH_DIR="ph_v2", CFG="C", PF_TAG="v2", PFH_TAG="v4sf", MIX_TAG="v4sf",
              USE_OC="1", USE_HANDDET="1", USE_MIX="1",
              ADD_SPIES_POS="1", W_SPY="0.5", POLARS_MAX_THREADS="4")

E_HD = dict(NTHR="5", POLARS_MAX_THREADS="4")
SELF = dict(SELFTRAIN="1", W_EXTRA="8")


def _handids():
    """Write the two hand-id lists src/hand_replay.py consumes (nothing else in the repo creates them)."""
    import polars as pl
    h = pl.read_parquet(DER / "hands.parquet", columns=["hand_idx", "phase"])
    h.filter(pl.col("phase") == 0).select("hand_idx").write_parquet(DER / "_dev_hand_ids.parquet")
    h.filter(pl.col("phase") == 1).select("hand_idx").write_parquet(DER / "_eval_hand_ids.parquet")
    print("wrote _dev_hand_ids.parquet / _eval_hand_ids.parquet", flush=True)


def _mk_v4sf():
    """hs_v4sf = hs_v4's dev phase verbatim + an eval phase re-scored SAME-FOLD (exp034).

    Dev hand scores are single-model OOF; scoring eval with the mean of the 5 fold models thins the
    score tail (eval/dev rate above the negative q999: .78-.88), and the pair model's count/threshold
    features were learned on the sharper dev scores.  SAMEFOLD_EVAL=1 scores each eval table with the
    fold model that held that table out -> ratios .92-1.07.  (scripts/exp034_samefold.sh)
    """
    (DER / "hs_v4sf").mkdir(parents=True, exist_ok=True)
    if not (DER / "hs_v4sf" / "phase0").exists():
        shutil.copytree(DER / "hs_v4" / "phase0", DER / "hs_v4sf" / "phase0")
        print("copied hs_v4/phase0 -> hs_v4sf/phase0", flush=True)


# --------------------------------------------------------------------------------------- stages
# (name, command, extra env, output that marks the stage done)
STAGES = [
    # ---------------------------------------------------------------- 1. raw -> int-coded cache
    ("cache", [PY, "src/build_cache.py"], {}, DER / "actions.parquet"),
    ("strength", [PY, "src/build_seat_strength.py"], {}, DER / "preflop_equity_169.parquet"),
    ("handids", _handids, {}, DER / "_eval_hand_ids.parquet"),

    # ---------------------------------------------------------------- 2. omniscient equity replay
    # ~750 s (dev, 1.2M hands / 11.1M actions) and ~500 s (eval, 800k hands).
    # THE EVAL LEG IS ABSENT FROM THE CURRENT src/run_all.py -- engine.py phase 1 asserts on it.
    ("equity_dev", [PY, "src/hand_replay.py", "dev", "data/derived/_dev_hand_ids.parquet"], {},
     DER / "action_equity_dev.parquet"),
    ("equity_eval", [PY, "src/hand_replay.py", "eval", "data/derived/_eval_hand_ids.parquet"], {},
     DER / "action_equity_eval.parquet"),

    # ---------------------------------------------------------------- 3. label-free normal-play policy
    ("policy", [PY, "src/policy.py", "all"], {}, DER / "action_policy.parquet"),

    # ---------------------------------------------------------------- 4. pair-hand engine, BOTH variants
    # ph    = engine WITHOUT the policy block; its pair features pf_<window>.parquet are what
    #         scripts/oc_hunt.py:famflags() and src/exp004_baseline.py read (both hardcode pf_, not pfv2_).
    # ph_v2 = engine WITH the policy block; pfv2_<window>.parquet feeds every shipped pair model.
    # NEITHER "ph" NOR pf_* IS BUILT BY THE CURRENT src/run_all.py -> its ochunt stage cannot run.
    ("engine_v1", [PY, "src/engine.py", "0", "1"], E_ENGINE_V1, DER / "ph" / "phase1"),
    ("pairfeat_v1", [PY, "src/pairfeat.py"], E_ENGINE_V1 | {"PF_TAG": ""}, DER / "pf_eval.parquet"),
    ("engine", [PY, "src/engine.py", "0", "1"], E_ENGINE, DER / "ph_v2" / "phase1"),
    ("pairfeat", [PY, "src/pairfeat.py"], E_ENGINE | {"PF_TAG": "v2"}, DER / "pfv2_eval.parquet"),

    # ---------------------------------------------------------------- 5. generic pair/player stats
    # scripts/audit_structure.py `build` writes pair_generic_{dev,dev_w2000,eval}.parquet and
    # player_stats_*.parquet.  src/exp004_baseline.py reads pair_generic_* and nothing else creates
    # them, so this stage is a hard dependency of the tie-break score below.  It is absent from the
    # current src/run_all.py entirely.
    ("audit_build", [PY, "scripts/audit_structure.py", "build"], {"POLARS_MAX_THREADS": "4"},
     DER / "pair_generic_eval.parquet"),

    # ---------------------------------------------------------------- 6. exp004 baseline
    # Only needed for oof/exp004_model_eval.parquet: src/assemble.py uses its s_z column as the
    # legitimate secondary score that breaks risk ties (the host would otherwise break them by
    # pair_id, an ID artefact).  Without this file assemble.py raises.
    ("baseline", [PY, "src/exp004_baseline.py", "model"], {}, OOF / "exp004_model_eval.parquet"),

    # ---------------------------------------------------------------- 7. other-coordination pair scores
    # events -> per-table event arrays;  stats -> pairstats_w{0,1,2,3}.npz (+ permutation nulls);
    # export -> oc_pair_scores_<window>.parquet for ALL FOUR windows.
    # The current src/run_all.py runs only `export`, with no env:
    #   * `events` and `stats` never run  -> load_ps() fails on a missing pairstats_w0.npz;
    #   * OC_EXPORT_WINDOWS defaults to "0,2", so dev_w2000a / dev_w2000b are never written, and
    #     exp005_pairmodel.py TRAIN=all3 needs all four;
    #   * with OC_EXPORT_WINDOWS unset, export falls through to a hand-candidate step that reads
    #     oof/exp005_eval_scores.parquet -- a circular dependency on a pair model that does not
    #     exist yet on a fresh clone.  Setting OC_EXPORT_WINDOWS short-circuits that block.
    ("oc_events", [PY, "scripts/oc_hunt.py", "events"], {"POLARS_MAX_THREADS": "4"},
     DER / "oc" / "events_ev.npy"),
    ("oc_stats", [PY, "scripts/oc_hunt.py", "stats"], {"OC_WINDOWS": "0,1,2,3", "POLARS_MAX_THREADS": "4"},
     DER / "oc" / "pairstats_w3.npz"),
    ("oc_export", [PY, "scripts/oc_hunt.py", "export"], {"OC_EXPORT_WINDOWS": "0,1,2,3"},
     DER / "oc_pair_scores_dev_w2000b.parquet"),

    # ---------------------------------------------------------------- 8. hand detector, 4 self-training rounds
    # Round 0 = listed evidence hands vs confirmed-clean-pair hands.
    # Rounds 1-3 add, as weak positives (weight 8), the UNLISTED hands of positive pairs whose
    # previous-round family score exceeds 0.5 (there are ~2x more planted hands than the host lists).
    # Self-training converged at round 3 (exp029/exp030); round 4 is the shipped one, with the
    # same-fold eval rescoring on top.  The current src/run_all.py stops at hs_v2.
    #
    # ORDERING IS LOAD-BEARING: handdet.py writes its fold models and thresholds.json to the SINGLE
    # shared directory data/derived/handdet_models/, so each round overwrites the last.  Running
    # these stages out of order, or re-running an early one later, silently mixes a round's hand
    # scores with another round's thresholds.  Do not use --force on one round alone.
    ("handdet_r0", [PY, "src/handdet.py", "all"], E_HD | {"HS_IN": "hs", "HS_OUT": "hs"}, DER / "pfh_eval.parquet"),
    ("handdet_r1", [PY, "src/handdet.py", "all"],
     E_HD | SELF | {"HS_IN": "hs", "HS_OUT": "hs_v2", "PFH_TAG": "v2"}, DER / "pfhv2_eval.parquet"),
    ("handdet_r2", [PY, "src/handdet.py", "all"],
     E_HD | SELF | {"HS_IN": "hs_v2", "HS_OUT": "hs_v3", "PFH_TAG": "v3"}, DER / "pfhv3_eval.parquet"),
    ("handdet_r3", [PY, "src/handdet.py", "all"],
     E_HD | SELF | {"HS_IN": "hs_v3", "HS_OUT": "hs_v4", "PFH_TAG": "v4"}, DER / "pfhv4_eval.parquet"),

    # same-fold eval rescoring of the round-4 detector -> hs_v4sf, then its pair aggregation
    ("handdet_sf_prep", _mk_v4sf, {}, DER / "hs_v4sf" / "phase0"),
    ("handdet_sf_score", [PY, "src/handdet.py", "score"],
     {"SAMEFOLD_EVAL": "1", "PHASES": "1", "HS_IN": "hs_v4", "HS_OUT": "hs_v4sf",
      "NTHR": "6", "POLARS_MAX_THREADS": "4"}, DER / "hs_v4sf" / "phase1"),
    ("handdet_sf_agg", [PY, "src/handdet.py", "agg"],
     {"HS_OUT": "hs_v4sf", "PFH_TAG": "v4sf", "NTHR": "4", "POLARS_MAX_THREADS": "5"},
     DER / "pfhv4sf_eval.parquet"),

    # mixture-LLR pair features over the same detector scores (the current run_all builds the v2 tag,
    # which no shipped model uses).
    ("mixllr", [PY, "scripts/mixllr.py", "hs_v4sf", "v4sf"], {"POLARS_MAX_THREADS": "5"},
     DER / "pfmv4sf_eval.parquet"),
    # hs_v2 mixture features: not consumed by the shipped models, kept because evidence_v4 reads hs_v2
    # scores directly and some CV tables in the ledger reference pfmv2_*.  Cheap; drop if you like.
    ("mixllr_v2", [PY, "scripts/mixllr.py", "hs_v2", "v2"], {"POLARS_MAX_THREADS": "5"},
     DER / "pfmv2_eval.parquet"),

    # ---------------------------------------------------------------- 9. evidence chain
    # evidence.py      -> exp007 / exp008 rankers + evidence_cache/cv_results.json
    #                     (cv_results.json["features_007"] and ["best007_single"] are read by BOTH
    #                      evidence_v4.py and scripts/listing_rule.py -- the cv stage is mandatory)
    # evidence_v2.py   -> evidence_cache/cv_results_v2.json["features_base"] (read by evidence_v4)
    # evidence_v4.py   -> exp027ev, the learned in-pair ranker the listing rule is blended with.
    #                     run_eval needs evidence_cache/v4_store.npz, written by run_cv -> run cv first.
    # listing_rule.py  -> the shipped evidence file.
    ("evidence", [PY, "src/evidence.py", "cache", "cv", "eval"], {"EVID_PH": "ph_v2"},
     DER / "evidence_scores_eval_exp008.parquet"),
    ("evidence_v2", [PY, "src/evidence_v2.py", "cv"], {"EVID_PH": "ph_v2"},
     DER / "evidence_cache" / "cv_results_v2.json"),
    # EVID_S1MODE=outer is REQUIRED to reproduce the SHIPPED exp027ev partner: the submitted
    # evidence_scores_{dev,eval}_exp027ev.parquet were built with run_eval(s1_mode="outer")
    # (stage-2 trained AND served on the cached 80% outer-OOF stage-1).  run_eval's own default
    # is "inner" (the honest-protocol estimator), which produces a DIFFERENT model worth -0.0039
    # dev MAP in the blend.  Proof of the provenance: scripts/exp027ev_provenance.py rebuilds the
    # shipped dev file bit-for-bit (max |diff| 0.000e+00 over all 45,129 rows x 3 families).
    ("evidence_v4", [PY, "src/evidence_v4.py", "stage1", "cv", "eval"],
     {"EVID_PH": "ph_v2", "EVID_S1MODE": "outer"},
     DER / "evidence_scores_eval_exp027ev.parquet"),

    # exp042_ramp = the listing-rule DP blended with exp027ev, with a SMOOTH family-strength ramp:
    #   w_dp = 0.7 * clip((strength - 0.45) / 0.20, 0, 1),  strength = max_f top5-mean partner s_f.
    # Rationale (exp037/exp042): on a pair whose family is not one of the three disclosed ones the
    # family-specific DP is peaked and confidently wrong (LOFO .258 vs .419 for the plain ranker), so
    # the DP is faded out there; the hard 0.60 cliff of exp037 cost -.00065 on the LB, the ramp gained
    # +.00025.  LR_GATE_RAMP is an UNCOMMITTED addition to scripts/listing_rule.py -- see BLOCKER #2.
    ("listing_rule_cv", [PY, "scripts/listing_rule.py", "cv"],
     {"LR_TAG": "exp042_ramp", "LR_GATE_RAMP": "0.45,0.65", "LR_PARTNER": "exp027ev"},
     DER / "evidence_scores_dev_exp042_ramp_exp027ev.parquet"),
    ("listing_rule", [PY, "scripts/listing_rule.py", "eval"],
     {"LR_TAG": "exp042_ramp", "LR_GATE_RAMP": "0.45,0.65", "LR_PARTNERS": "exp027ev"},
     DER / "evidence_scores_eval_exp042_ramp_exp027ev.parquet"),

    # ---------------------------------------------------------------- 10. pair risk models (6)
    # 3 x LightGBM GOSS (src/exp005_pairmodel.py, 600 rounds) and 3 x GPU XGBoost hist
    # (src/exp_xgb_pairmodel.py, 500 rounds), seeds 42 / 101 / 202, all on TRAIN=all3 with W_U=0.1.
    # The two uncommitted files src/exp_xgb_pairmodel.py and src/exp_cb_pairmodel.py MUST be
    # committed -- the champion cannot be rebuilt without the first (BLOCKER #3).
    ("lgb_s42", [PY, "src/exp005_pairmodel.py", "exp034_v4sf", "0.1", "600", "all3"], E_PAIR,
     OOF / "exp034_v4sf_eval_scores.parquet"),
    ("lgb_s101", [PY, "src/exp005_pairmodel.py", "exp035_sf101", "0.1", "600", "all3"],
     E_PAIR | {"SEED": "101"}, OOF / "exp035_sf101_eval_scores.parquet"),
    ("lgb_s202", [PY, "src/exp005_pairmodel.py", "exp035_sf202", "0.1", "600", "all3"],
     E_PAIR | {"SEED": "202"}, OOF / "exp035_sf202_eval_scores.parquet"),
    ("xgb_s42", [PY, "src/exp_xgb_pairmodel.py", "exp043_xgb", "0.1", "500", "all3"],
     E_PAIR | {"XGB_DEVICE": XGB_DEVICE}, OOF / "exp043_xgb_eval_scores.parquet"),
    ("xgb_s101", [PY, "src/exp_xgb_pairmodel.py", "exp043_xgb101", "0.1", "500", "all3"],
     E_PAIR | {"SEED": "101", "XGB_DEVICE": XGB_DEVICE}, OOF / "exp043_xgb101_eval_scores.parquet"),
    ("xgb_s202", [PY, "src/exp_xgb_pairmodel.py", "exp043_xgb202", "0.1", "500", "all3"],
     E_PAIR | {"SEED": "202", "XGB_DEVICE": XGB_DEVICE}, OOF / "exp043_xgb202_eval_scores.parquet"),

    # ---------------------------------------------------------------- 11. rank-average fusions
    # Verified against the shipped artefacts: each reproduces the shipped eval rank vector exactly
    # (Pearson corr of the rank vectors = 1.00000000).
    # NOTE exp048 is a NESTED fusion -- mean(rank(LGBM bag), rank(XGB bag)) -- i.e. the two model
    # families get 50/50, not each of the 6 models 1/6.  The ledger's "full 6-model bag" wording is
    # loose; a flat 6-way mean is a DIFFERENT file (rank corr .99999963 with the shipped one).
    ("fuse_lgb_bag", [PY, "scripts/fuse.py", "exp035_bagsf", "exp034_v4sf", "exp035_sf101", "exp035_sf202"], {},
     OOF / "exp035_bagsf_eval_scores.parquet"),
    ("fuse_xgb_bag", [PY, "scripts/fuse.py", "exp047_xgbag", "exp043_xgb", "exp043_xgb101", "exp043_xgb202"], {},
     OOF / "exp047_xgbag_eval_scores.parquet"),
    ("fuse_sub024", [PY, "scripts/fuse.py", "exp044_fuse_lgb_xgb", "exp035_bagsf", "exp043_xgb"], {},
     OOF / "exp044_fuse_lgb_xgb_eval_scores.parquet"),
    ("fuse_sub025", [PY, "scripts/fuse.py", "exp048_fuse_bags", "exp035_bagsf", "exp047_xgbag"], {},
     OOF / "exp048_fuse_bags_eval_scores.parquet"),

    # ---------------------------------------------------------------- 12. assemble + validate
    ("sub024", [PY, "src/assemble.py", "sub024_fuse_exp042ramp", "exp044_fuse_lgb_xgb",
                "scores:data/derived/evidence_scores_eval_exp042_ramp_exp027ev.parquet"], {},
     BASE / "submissions" / "sub024_fuse_exp042ramp.csv"),
    ("sub025", [PY, "src/assemble.py", "sub025_fusebags_exp042ramp", "exp048_fuse_bags",
                "scores:data/derived/evidence_scores_eval_exp042_ramp_exp027ev.parquet"], {},
     BASE / "submissions" / "sub025_fusebags_exp042ramp.csv"),

    # sub027 = the same risk model with the UNGATED (flat w = 0.7) listing-rule evidence.  It is the
    # second selected final, kept deliberately decorrelated from sub025 on the evidence component.
    # Its evidence file comes from the same listing_rule.py run with the ramp switched off.
    ("listing_rule_flat", [PY, "scripts/listing_rule.py", "eval"],
     {"LR_TAG": "exp036", "LR_PARTNERS": "exp027ev"},
     DER / "evidence_scores_eval_exp036_exp027ev.parquet"),
    ("sub027", [PY, "src/assemble.py", "sub027_fusebags_exp036flat", "exp048_fuse_bags",
                "scores:data/derived/evidence_scores_eval_exp036_exp027ev.parquet"], {},
     BASE / "submissions" / "sub027_fusebags_exp036flat.csv"),

    # ---------------------------------------------------------------- 12b. the second selected final
    # sub028 = the same evidence and behaviour as sub025, with the pair risk rank-blended 0.85/0.15 with the
    # sum of three FAMILY-SPECIALIST models.  It is ticked as final slot 2 because it is a genuinely different
    # risk ranking (Spearman 0.9955 vs sub025) at an essentially zero measured cost (public LB -0.00007), which
    # makes it the best hedge available -- NOT because it is expected to score higher.  Its dev-side gain
    # (view C' +0.0028 at P 1.000) did NOT transfer: see experiments/LEDGER.md, row exp068_famspecmix_w15.
    # Each specialist copies exp048_fuse_bags' behaviour posteriors verbatim (DONOR), so the behaviour column
    # of the assembled submission is bit-identical to sub025's.
    ("famspec_dt", [PY, "src/exp053_famspec.py", "exp061_dtonly", "0.1", "600", "all3"],
     E_PAIR | {"FAM_TARGET": "directed_transfer", "OTHER_MODE": "drop", "SPY_MODE": "drop",
               "DONOR": "exp048_fuse_bags"}, OOF / "exp061_dtonly_eval_scores.parquet"),
    ("famspec_sp", [PY, "src/exp053_famspec.py", "exp061_sponly", "0.1", "600", "all3"],
     E_PAIR | {"FAM_TARGET": "soft_play", "OTHER_MODE": "drop", "SPY_MODE": "drop",
               "DONOR": "exp048_fuse_bags"}, OOF / "exp061_sponly_eval_scores.parquet"),
    ("famspec_ci", [PY, "src/exp053_famspec.py", "exp061_cionly", "0.1", "600", "all3"],
     E_PAIR | {"FAM_TARGET": "coordinated_isolation", "OTHER_MODE": "drop", "SPY_MODE": "drop",
               "DONOR": "exp048_fuse_bags"}, OOF / "exp061_cionly_eval_scores.parquet"),
    ("famspec_mix", [PY, "scripts/mixmax.py", "rmix", "exp068_famspecmix_w15", "exp048_fuse_bags",
                     "exp061_dtonly", "exp061_sponly", "exp061_cionly", "0.15", "--eval"], {},
     OOF / "exp068_famspecmix_w15_eval_scores.parquet"),
    ("sub028", [PY, "src/assemble.py", "sub028_famspecw15_exp042ramp", "exp068_famspecmix_w15",
                "scores:data/derived/evidence_scores_eval_exp042_ramp_exp027ev.parquet"], {},
     BASE / "submissions" / "sub028_famspecw15_exp042ramp.csv"),

    # ---------------------------------------------------------------- 13. prize deliverable
    ("case_reviews", [PY, "scripts/case_review2.py", "submissions/sub025_fusebags_exp042ramp.csv", "5",
                      "research/CASE_REVIEWS.md"], {}, BASE / "research" / "CASE_REVIEWS.md"),
    ("figures", [PY, "scripts/make_figures.py"], {}, BASE / "research" / "figures" / "fig1_phase_locality.png"),
]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    dry = "--dry-run" in sys.argv
    if "--list" in sys.argv:
        for name, cmd, env, out in STAGES:
            mark = "have" if out.exists() else "MISSING"
            print(f"{mark:8s} {name:18s} -> {out.relative_to(BASE)}")
        return
    for name, cmd, env, out in STAGES:
        if args and name not in args:
            continue
        if out.exists() and not force:
            print(f"[skip] {name} (exists: {out.name})", flush=True)
            continue
        shown = "<python callable>" if callable(cmd) else " ".join(cmd)
        print(f"[run ] {name}: {shown}", flush=True)
        if dry:
            continue
        if callable(cmd):
            cmd()
            continue
        r = subprocess.run(cmd, cwd=BASE, env={**os.environ, **env, "PYTHONIOENCODING": "utf-8"})
        if r.returncode:
            sys.exit(f"stage {name} failed ({r.returncode})")
    print("done")


if __name__ == "__main__":
    main()


# ============================================================================================
# REPRO BLOCKERS -- things that must change in the repository, not in this file
# ============================================================================================
# 1/2/3. FIXED by commit 5e6e69e ("Track the champion pipeline") during this review:
#    data/folds_tables_5.csv, data/derived/spies_exp005.parquet and spies2_exp022.parquet are now
#    force-added to git despite the `data/` ignore rule; src/exp_xgb_pairmodel.py and
#    src/exp_cb_pairmodel.py are tracked; scripts/listing_rule.py's LR_GATE_RAMP branch is
#    committed.  Residual: .gitignore still says `data/` with no negation, so the three artefacts
#    survive only because they are already in the index -- a `git rm --cached` or a fresh
#    `git add .` elsewhere will not re-add them.  Adding explicit `!` exceptions is safer.
#    Residual: spies2_exp022.parquet's selection rule is documented NOWHERE in the repo; the
#    writeup should state what the 26 pairs are (and spies_exp005's rule, "unlabelled dev pairs
#    with exp005 OOF > 0.3", which IS in the ledger).
#
# 4. data/derived/_dev_hand_ids.parquet is referenced by the old run_all.py and written by nothing.
#    The `handids` stage above replaces it.
#
# 5. src/exp_xgb_pairmodel.py hardcodes {"device": "cuda"}.  On a CPU-only reviewer machine the
#    stage dies.  One-line fix:
#        "device": os.environ.get("XGB_DEVICE", "cuda"),
#    Without it, the "public reproducible code" claim is conditional on owning an NVIDIA GPU.
#
# 6. src/handdet.py writes every self-training round's fold models and thresholds.json into the one
#    directory data/derived/handdet_models/.  Nothing records which round is in there.  Tagging it
#    (MOD = C.DER / os.environ.get("HD_MODELS", "handdet_models")) would make rounds independently
#    re-runnable; as it stands, only a clean in-order run is reproducible.
#
# 7. Known nondeterminism: handdet.py's `agg` double-centring sums are nondeterministic at ~1e-8,
#    which moves view C' by about +-0.0001 between re-aggregations of identical hand scores
#    (LEDGER exp033 REPRO NOTE).  A rebuilt submission will differ from the shipped CSV in the low
#    bits of risk_score and, for a handful of near-tied pairs, in row order.  Say so in the writeup.
#
# 8. oof/ and submissions/*.csv are git-ignored.  That is fine for reproduction, but it means the
#    exact shipped CSVs are not archived anywhere in git; keep a copy of
#    sub024_fuse_exp042ramp.csv and sub025_fusebags_exp042ramp.csv outside the repo, or add
#    a `!submissions/sub024_*.csv` exception, before the deadline.
