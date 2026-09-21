"""Evidence-hand forensics for behavior family 'coordinated_isolation' (CI).  Notes: research/forensics/coordinated_isolation.md

Re-runnable, stage-based (run from the comp root; outputs -> research/forensics/ci_*.parquet):
  extract                  CI positive pairs + 400 confirmed-negative controls: dev shared pair-hands + hand subsets
  print N M [fam] [seed]   print N CI evidence hands and M non-evidence shared hands (roles A/B/o1..o4)
  printq "<polars expr>" N print hands from ci_feat_sub matching an expression (e.g. grp / feature filters)
  printids out.txt h1 h2.. print specific hand_idx (archetypes -> research/forensics/ci_archetypes.txt)
  baseline                 population preflop P(aggr/fold/call) per 169 class x #prior aggr [x position x #prior callers]
  features                 numba pair-hand feature engine on the subset -> ci_feat_sub.parquet (+group means)
  sigtable                 coverage / rates of candidate signatures (EV_CI, same-pair non-evidence, NEG, EV_DT, EV_SP)
  aggrtable                every preflop aggressive action (all hands) with position-aware population p_aggr
  playerrates              trash-raise rate per member with vs without partner seated (labelled pairs)
  evsim                    evidence MAP@5 simulation on dev CI pairs for ranking recipes (GroupKFold by pair)
  pairextract / pairlevel  all labelled pairs + 3000 unknown dev pairs: pair-level signature rates + AUCs
  lift PHASE / pairlift    directed co-player lift of trash raises (all players, per phase) + pair AUC/AP
  confound [reps]          shared-hand-count confound via subsampling (AUC/AP at n=20/38/60/100)
Memory-light: polars scans with is_in filters, duckdb threads=3 memory_limit=4GB, POLARS_MAX_THREADS=3.
Roles: labels always have player_1 < player_2 (string canonical) -> A = player_1, B = player_2 carries no meaning.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "3")
import numpy as np
import polars as pl

BASE = Path(__file__).resolve().parent.parent
DER = BASE / "data" / "derived"
OUT = BASE / "research" / "forensics"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE / "src"))
FAM = "coordinated_isolation"
RANKS, SUITS = "23456789TJQKA", "cdhs"
ANAME = {0: "fold", 1: "check", 2: "call", 3: "bet", 4: "raise", 5: "all_in"}
SNAME = {0: "PRE", 1: "FLOP", 2: "TURN", 3: "RIVER"}
SEED = 42
N_NEG_CTRL = 400


def scan(n):
    return pl.scan_parquet(DER / f"{n}.parquet")


def cs(c):
    return RANKS[c // 4] + SUITS[c % 4]


def P(name):
    return OUT / f"ci_{name}.parquet"


# ============================================================================ extract
def stage_extract():
    labels = pl.read_parquet(DER / "labels.parquet")
    ev = pl.read_parquet(DER / "evidence.parquet")
    pos = labels.filter((pl.col("label") == 1) & (pl.col("behavior_family") == FAM))
    neg = labels.filter(pl.col("label") == 0).sample(n=N_NEG_CTRL, seed=SEED)
    sel = pl.concat([pos, neg]).select("pair_id", pl.col("p1").alias("A"), pl.col("p2").alias("B"), "label",
                                       "behavior_family")
    players = set(sel["A"].to_list()) | set(sel["B"].to_list())
    s = (scan("seats").select("hand_idx", "player_idx")
         .filter(pl.col("player_idx").is_in(list(players)))
         .join(scan("hands").filter(pl.col("phase") == 0).select("hand_idx"), on="hand_idx").collect())
    ph = (sel.lazy().select("pair_id", "A", "B", "label").join(s.rename({"player_idx": "A"}).lazy(), on="A")
          .join(s.rename({"player_idx": "B"}).lazy(), on=["B", "hand_idx"]).collect())
    ph = ph.join(ev.select("pair_id", "hand_idx", "evidence_rank"), on=["pair_id", "hand_idx"], how="left")
    print("pair-hands", ph.shape, "evidence matched", ph["evidence_rank"].is_not_null().sum())
    ph.write_parquet(P("pair_hands"))
    sel.write_parquet(P("pairs"))
    # other-family evidence hands with their pair roles
    oth = ev.join(labels.select("pair_id", pl.col("p1").alias("A"), pl.col("p2").alias("B")), on="pair_id")
    oth.write_parquet(P("evidence_all"))
    hidx = sorted(set(ph["hand_idx"].to_list()) | set(ev["hand_idx"].to_list()))
    print("distinct hands", len(hidx))
    for name in ("hands", "seats", "actions"):
        df = scan(name).filter(pl.col("hand_idx").is_in(hidx)).collect()
        df.write_parquet(P(f"sub_{name}"))
        print(name, df.shape)


# ============================================================================ printing
class Sub:
    def __init__(self):
        self.hands = pl.read_parquet(P("sub_hands"))
        self.seats = pl.read_parquet(P("sub_seats"))
        self.actions = pl.read_parquet(P("sub_actions")).sort("hand_idx", "action_no")
        self.h = {r["hand_idx"]: r for r in self.hands.iter_rows(named=True)}
        self.s = {k[0]: v for k, v in self.seats.partition_by("hand_idx", as_dict=True).items()}
        self.a = {k[0]: v for k, v in self.actions.partition_by("hand_idx", as_dict=True).items()}


def hand_text(sub, hidx, A, B, extra=""):
    from phevaluator import evaluate_cards
    h = sub.h[hidx]
    seats = sub.s[hidx].sort("seat_no")
    acts = sub.a.get(hidx)
    board = [cs(c) for c in (h["board"] or [])]
    role = {}
    oth_i = 0
    for r in seats.iter_rows(named=True):
        if r["player_idx"] == A:
            role[r["player_idx"]] = "A"
        elif r["player_idx"] == B:
            role[r["player_idx"]] = "B"
        else:
            oth_i += 1
            role[r["player_idx"]] = f"o{oth_i}"
    bb = h["bb"]
    lines = [f"--- hand_idx {hidx} seq {h['hand_seq']} bb {bb} button {h['button_seat']} pot {h['final_pot']} "
             f"({h['final_pot'] / bb:.1f}bb) dealt {h['players_dealt']} sd {h['players_at_showdown']} "
             f"board {' '.join(board)} {extra}"]
    for r in seats.iter_rows(named=True):
        hc = [cs(r["c1"]), cs(r["c2"])]
        rk = ""
        if len(board) >= 3:
            rk = f"rank@end={evaluate_cards(*(hc + board))}"
        lines.append(f"   seat{r['seat_no']} {role[r['player_idx']]:>3} {hc[0]}{hc[1]} stack {r['starting_stack'] / bb:6.1f}bb "
                     f"contrib {r['total_contribution'] / bb:6.1f} net {r['net_chips'] / bb:+7.1f} "
                     f"fold={int(r['folded'])} sd={int(r['went_to_showdown'])} won={r['won_share']:.2f} {rk}")
    if acts is not None:
        for r in acts.iter_rows(named=True):
            lines.append(f"     {SNAME[r['street']]:5} {role.get(r['player_idx'], '?'):>3} {ANAME[r['action']]:6} "
                         f"amt {r['amount'] / bb:6.1f} to {r['amount_to'] / bb:6.1f} pot {r['pot_before'] / bb:6.1f} "
                         f"tocall {r['to_call'] / bb:5.1f} stk {r['stack_before'] / bb:6.1f} act {r['players_active']}")
    return "\n".join(lines)


def stage_print(n="25", m="15", fam=FAM, seed=str(SEED)):
    n, m = int(n), int(m)
    sub = Sub()
    ph = pl.read_parquet(P("pair_hands"))
    pairs = pl.read_parquet(P("pairs"))
    evp = ph.filter(pl.col("evidence_rank").is_not_null() & (pl.col("label") == 1))
    rng = np.random.default_rng(int(seed))
    pids = evp["pair_id"].unique().sort().to_list()
    rng.shuffle(pids)
    shown = 0
    for pid in (pids if n > 0 else []):
        rows = evp.filter(pl.col("pair_id") == pid).sort("evidence_rank")
        for r in rows.iter_rows(named=True):
            print(hand_text(sub, r["hand_idx"], r["A"], r["B"], f"EVIDENCE pair {pid} rank {r['evidence_rank']}"))
            shown += 1
        if shown >= n:
            break
    if m:
        ne = ph.filter(pl.col("evidence_rank").is_null() & (pl.col("label") == 1)).sample(n=m, seed=int(seed))
        for r in ne.iter_rows(named=True):
            print(hand_text(sub, r["hand_idx"], r["A"], r["B"], f"NONEVIDENCE pair {r['pair_id']}"))


def stage_printq(expr, n="12", seed=str(SEED)):
    """Print hands from ci_feat_sub matching a polars expression string, e.g.
    "(pl.col('grp')=='EV_CI') & (pl.min_horizontal('pA_weak','pB_weak')>0.05)"."""
    sub = Sub()
    F = pl.read_parquet(P("feat_sub"))
    x = F.filter(eval(expr))
    print("matching", x.height)
    x = x.sample(n=min(int(n), x.height), seed=int(seed))
    cols = ["pA_weak", "pB_weak", "pfoldA_facing_B", "pfoldB_facing_A", "surpA", "surpB", "reraiseA_over_B",
            "reraiseB_over_A", "sandwich", "hsA_fold_facing_B", "hsB_fold_facing_A", "airbetA", "airbetB"]
    for r in x.iter_rows(named=True):
        info = " ".join(f"{c}={r[c]:.3g}" for c in cols)
        print(hand_text(sub, r["hand_idx"], r["A"], r["B"], f"{r['grp']} pair {r['pair_id']} rank {r['evidence_rank']}\n   {info}"))


def stage_aggrtable(min_n="150"):
    """All preflop aggressive actions (all players, all hands) with position-aware population P(aggr)
    -> research/forensics/ci_pre_aggr_all.parquet (hand_idx, player_idx, phase, hand_seq, p_aggr, nr)."""
    import duckdb
    con = duckdb.connect()
    con.sql("SET threads=3; SET memory_limit='4GB'; SET preserve_insertion_order=false")
    a_p, s_p, h_p = [(DER / f"{n}.parquet").as_posix() for n in ("actions", "seats", "hands")]
    b1 = (OUT / "ci_baseline_pre.parquet").as_posix()
    b2 = (OUT / "ci_baseline_pre_pos.parquet").as_posix()
    con.sql(f"""COPY (
      WITH a AS (
        SELECT hand_idx, action_no, player_idx, action,
               (action IN (3,4) OR (action=5 AND amount>to_call))::INT AS aggr,
               (action=2 OR (action=5 AND amount<=to_call))::INT AS isc
        FROM '{a_p}' WHERE street=0),
      b AS (
        SELECT *, coalesce(sum(aggr) OVER w, 0) AS nr_before, coalesce(sum(isc) OVER w, 0) AS nc_before
        FROM a WINDOW w AS (PARTITION BY hand_idx ORDER BY action_no ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)),
      c AS (
        SELECT b.hand_idx, b.player_idx, b.action_no, {CLASS_SQL}::SMALLINT AS class_id, least(nr_before, 3)::TINYINT AS nr,
               ((s.seat_no - h.button_seat + 6) % 6)::TINYINT AS pos, least(nc_before, 2)::TINYINT AS nc,
               h.phase, h.hand_seq
        FROM b JOIN '{s_p}' s USING (hand_idx, player_idx) JOIN '{h_p}' h USING (hand_idx) WHERE b.aggr = 1)
      SELECT c.hand_idx, c.player_idx, c.action_no, c.phase, c.hand_seq, c.nr, c.pos, c.class_id,
             (CASE WHEN p2.n >= {int(min_n)} THEN p2.p_aggr ELSE p1.p_aggr END)::FLOAT AS p_aggr
      FROM c LEFT JOIN '{b2}' p2 USING (class_id, nr, pos, nc) LEFT JOIN '{b1}' p1 USING (class_id, nr)
    ) TO '{(OUT / "ci_pre_aggr_all.parquet").as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
    t = pl.read_parquet(OUT / "ci_pre_aggr_all.parquet")
    print(t.shape, t.select((pl.col("p_aggr") < 0.02).mean().alias("lt02"), (pl.col("p_aggr") < 0.1).mean().alias("lt10")))


def stage_playerrates():
    """Trash-raise rate (preflop aggr with population p<thr) per hand dealt, with vs without partner seated."""
    import duckdb
    labels = pl.read_parquet(DER / "labels.parquet")
    con = duckdb.connect()
    con.sql("SET threads=3; SET memory_limit='4GB'; SET preserve_insertion_order=false")
    pairs = labels.select("pair_id", pl.col("p1").alias("A"), pl.col("p2").alias("B"), "label", "behavior_family")
    con.register("pairs", pairs.to_arrow())
    s_p, h_p = (DER / "seats.parquet").as_posix(), (DER / "hands.parquet").as_posix()
    ag = (OUT / "ci_pre_aggr_all.parquet").as_posix()
    res = con.sql(f"""
      WITH mem AS (SELECT pair_id, label, behavior_family, A AS me, B AS partner FROM pairs
                   UNION ALL SELECT pair_id, label, behavior_family, B AS me, A AS partner FROM pairs),
      sh AS (SELECT s.hand_idx, s.player_idx FROM '{s_p}' s JOIN '{h_p}' h USING (hand_idx) WHERE h.phase = 0),
      mh AS (SELECT m.pair_id, m.label, m.behavior_family, m.me, sh.hand_idx,
                    (p.player_idx IS NOT NULL) AS with_partner
             FROM mem m JOIN sh ON sh.player_idx = m.me
             LEFT JOIN sh p ON p.hand_idx = sh.hand_idx AND p.player_idx = m.partner),
      tr AS (SELECT hand_idx, player_idx, min(p_aggr) AS pmin FROM '{ag}' WHERE phase = 0 GROUP BY ALL)
      SELECT mh.pair_id, mh.label, mh.behavior_family, mh.me, mh.with_partner, count(*)::BIGINT AS n,
             sum((tr.pmin < 0.02)::INT)::BIGINT AS n_t02, sum((tr.pmin < 0.1)::INT)::BIGINT AS n_t10, sum((tr.pmin IS NOT NULL)::INT)::BIGINT AS n_aggr
      FROM mh LEFT JOIN tr ON tr.hand_idx = mh.hand_idx AND tr.player_idx = mh.me
      GROUP BY ALL""").pl()
    res.write_parquet(P("player_rates"))
    agg = (res.group_by("behavior_family", "with_partner")
           .agg(pl.col("n").sum(), (pl.col("n_t02").sum() / pl.col("n").sum()).alias("t02_rate"),
                (pl.col("n_t10").sum() / pl.col("n").sum()).alias("t10_rate"),
                (pl.col("n_aggr").sum() / pl.col("n").sum()).alias("aggr_rate"))
           .sort("behavior_family", "with_partner"))
    print(agg)
    # per member: rate without partner distribution
    w = res.pivot(on="with_partner", index=["pair_id", "label", "behavior_family", "me"], values=["n", "n_t02"])
    print(w.head())


def ap_at5(ranked_is_rel, n_rel):
    hits, s = 0, 0.0
    for r, v in enumerate(ranked_is_rel[:5], start=1):
        if v:
            hits += 1
            s += hits / r
    return s / min(n_rel, 5) if n_rel else 0.0


def add_chrono(F, cand_expr=None, phase_len=3000):
    """Per-pair chronological context features (computable in eval as well, within the eval phase)."""
    if cand_expr is None:
        cand_expr = pl.min_horizontal("pA_weak", "pB_weak") < 0.02
    F = F.sort("pair_id", "hand_seq").with_columns(cand_expr.cast(pl.Int32).alias("cand"))
    F = F.with_columns((pl.col("hand_seq").diff().over("pair_id").fill_null(10 ** 6) > 10).cast(pl.Int32)
                       .cum_sum().over("pair_id").alias("session"))
    F = F.with_columns(
        pl.col("cand").sum().over("pair_id", "session").alias("sess_n_cand"),
        pl.len().over("pair_id", "session").alias("sess_len"),
        (pl.col("session").rank("dense").over("pair_id")).alias("sess_rank"),
        ((pl.col("cand").sum().over("pair_id", "session") > 0).cast(pl.Int32)).alias("sess_has_cand"),
    )
    F = F.with_columns(
        # number of earlier sessions (in phase) that contained >=1 candidate hand
        (pl.col("sess_has_cand") * (pl.col("hand_seq") == pl.col("hand_seq").min().over("pair_id", "session")).cast(pl.Int32))
        .cum_sum().over("pair_id").alias("_sess_cand_cum"),
    ).with_columns((pl.col("_sess_cand_cum") - pl.col("sess_has_cand")).alias("prior_cand_sessions")).drop("_sess_cand_cum")
    scen = ((pl.col("fpa_ord") == 0) & (pl.col("fpa_p") < 0.3)).cast(pl.Int32)
    scen2 = ((pl.col("fpa_ord") <= 2) & (pl.col("pair_open") > 0) & (pl.col("fpa_p") < 0.3)).cast(pl.Int32)
    F = F.with_columns(scen.alias("scen"), scen2.alias("scen2")).with_columns(
        (pl.col("scen").cum_sum().over("pair_id") - pl.col("scen")).alias("prior_scen"),
        (pl.col("scen2").cum_sum().over("pair_id") - pl.col("scen2")).alias("prior_scen2"))
    return F.with_columns(
        pl.int_range(pl.len()).over("pair_id").alias("shared_idx"),
        (pl.col("cand").cum_sum().over("pair_id") - pl.col("cand")).alias("prior_cand"),
        (pl.col("hand_seq") / phase_len).alias("seq_frac"),
        pl.len().over("pair_id").alias("n_shared"),
        pl.col("cand").sum().over("pair_id").alias("n_cand"),
    )


HAND_X = ["pw_log", "n_pair_trash", "first_pair_vol_raise", "pair_open", "fap", "rr", "sandwich", "tf_pre",
          "tf_both_active", "third_fold_facing_pair", "third_aggr", "third_vol", "players_vol", "pot_log",
          "showdown", "pair_win_nosd", "pf_min", "aggA_post", "aggB_post", "pair_hu_post_checks", "isoA", "isoB",
          "fpa_ord", "fpa_nr", "fpa_p", "fpa_partner_acted_before"]


def hand_matrix(F):
    return F.with_columns(
        (-pl.min_horizontal("pA_weak", "pB_weak").clip(1e-4, 1).log()).alias("pw_log"),
        ((pl.col("foldA_after_aggB") + pl.col("foldB_after_aggA")) > 0).cast(pl.Int32).alias("fap"),
        ((pl.col("reraiseA_over_B") + pl.col("reraiseB_over_A")) > 0).cast(pl.Int32).alias("rr"),
        pl.col("pot_bb").log1p().alias("pot_log"),
        pl.min_horizontal("pfoldA_facing_B", "pfoldB_facing_A").alias("pf_min"),
    )


def stage_evsim():
    """Simulate evidence MAP@5 on dev CI positive pairs for several ranking recipes (GroupKFold by pair)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    import lightgbm as lgb
    F = pl.read_parquet(P("feat_sub")).filter((pl.col("src") == "ci_or_neg") & (pl.col("label") == 1))
    F = add_chrono(hand_matrix(F)).with_columns(pl.col("evidence_rank").is_not_null().cast(pl.Int32).alias("ev"))
    y = F["ev"].to_numpy()
    groups = F["pair_id"].to_numpy()
    rng = np.random.default_rng(SEED)
    scores = {}
    scores["random"] = rng.random(F.height)
    scores["earliest"] = -F["hand_seq"].to_numpy().astype(float)
    pwl = F["pw_log"].to_numpy()
    scores["pw_only(tie=earliest)"] = pwl + 1e-6 * -F["hand_seq"].to_numpy()
    scores["cand_then_earliest"] = (pwl > -np.log(0.02)).astype(float) * 10 - F["hand_seq"].to_numpy() / 3000
    hand_rule = (np.minimum(pwl, 6) + 2 * F["first_pair_vol_raise"].to_numpy() + 1.0 * F["fap"].to_numpy()
                 + 0.5 * F["tf_pre"].to_numpy())
    scores["rule_hand"] = hand_rule
    scores["rule_hand+chrono"] = hand_rule - 0.35 * np.minimum(F["prior_cand"].to_numpy(), 12)
    # simple transparent rule: planted gate (trash raise) + pressure-success + chronology
    cand = (pwl > -np.log(0.02)).astype(float)
    tfp = F["tf_pre"].to_numpy()
    fap = F["fap"].to_numpy()
    pc = np.minimum(F["prior_cand"].to_numpy(), 15)
    best = None
    for w_tf in [0.0, 0.5, 1.0]:
        for w_fap in [0.0, 1.0, 2.0]:
            for w_pc in [0.0, 0.15, 0.3, 0.6]:
                sc = 10 * cand + np.minimum(pwl, 6) * 0.3 + w_tf * tfp + w_fap * fap - w_pc * pc - 1e-4 * F["hand_seq"].to_numpy()
                scores[f"grid_tf{w_tf}_fap{w_fap}_pc{w_pc}"] = sc
    Xh = F.select(HAND_X).to_numpy().astype(float)
    Xc = np.column_stack([Xh, np.minimum(F["prior_cand"].to_numpy(), 16), F["seq_frac"].to_numpy(),
                          np.minimum(F["shared_idx"].to_numpy(), 200)])
    Xpc = np.column_stack([Xh, np.minimum(F["prior_cand"].to_numpy(), 16)])
    Xsq = np.column_stack([Xh, F["seq_frac"].to_numpy()])
    Xsc = np.column_stack([Xpc, np.minimum(F["prior_scen"].to_numpy(), 12), np.minimum(F["prior_scen2"].to_numpy(), 12)])
    scores["rule_first5_UTG_p<.3"] = (F["scen"].to_numpy() * (10 - np.minimum(F["prior_scen"].to_numpy(), 9))
                                      + 0.01 * np.minimum(pwl, 6) - 1e-5 * F["hand_seq"].to_numpy())
    Xss = np.column_stack([Xpc, np.minimum(F["sess_n_cand"].to_numpy(), 10), np.minimum(F["prior_cand_sessions"].to_numpy(), 10),
                           np.minimum(F["sess_len"].to_numpy(), 60)])
    for name, X in [("logit_hand", Xh), ("logit_hand+chrono", Xc), ("lgb_hand", Xh), ("lgb_hand+chrono", Xc),
                    ("logit_hand+prior_cand", Xpc), ("lgb_hand+prior_cand", Xpc), ("lgb_hand+seq_frac", Xsq), ("lgb_hand+prior_cand+session", Xss), ("lgb_hand+prior_cand+prior_scen", Xsc), ("logit_hand+prior_cand+prior_scen", Xsc),
                    ("logit_hand+prior_cand+session", Xss)]:
        oof = np.zeros(F.height)
        for tr, va in GroupKFold(5).split(X, y, groups):
            if name.startswith("logit"):
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
                m = LogisticRegression(max_iter=5000, C=1.0).fit((X[tr] - mu) / sd, y[tr])
                oof[va] = m.decision_function((X[va] - mu) / sd)
            else:
                m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=3)
                m.fit(X[tr], y[tr])
                oof[va] = m.predict_proba(X[va])[:, 1]
        scores[name] = oof
        if name == "logit_hand+chrono":
            mu, sd = Xc.mean(0), Xc.std(0) + 1e-9
            m = LogisticRegression(max_iter=5000).fit((Xc - mu) / sd, y)
            names = HAND_X + ["prior_cand", "seq_frac", "shared_idx"]
            print("logit_hand+chrono standardized coefs:")
            for nm, w in sorted(zip(names, m.coef_[0]), key=lambda t: -abs(t[1])):
                print(f"   {nm:24s} {w:+.3f}")
    from sklearn.metrics import roc_auc_score
    res = []
    pid = F["pair_id"].to_numpy()
    idx_by_pair = {}
    for i, p in enumerate(pid):
        idx_by_pair.setdefault(p, []).append(i)
    for name, sc in scores.items():
        aps = []
        for p, ids in idx_by_pair.items():
            ids = np.array(ids)
            order = ids[np.argsort(-sc[ids], kind="mergesort")]
            aps.append(ap_at5(y[order].tolist(), int(y[ids].sum())))
        res.append((name, float(np.mean(aps)), roc_auc_score(y, sc)))
        print(f"{name:28s} MAP@5 {np.mean(aps):.4f}   hand AUC(within CI pos pairs) {roc_auc_score(y, sc):.4f}")
    pl.DataFrame(res, schema=["recipe", "map5", "auc"], orient="row").write_parquet(P("evsim"))


N_UNK = 3000


def stage_pairextract():
    """All labelled pairs + 3000 unknown dev pairs (eval-like exclusion) -> dev pair-hands + hand subsets (tag 'pl')."""
    import duckdb
    labels = pl.read_parquet(DER / "labels.parquet")
    con = duckdb.connect()
    con.sql("SET threads=3; SET memory_limit='4GB'; SET preserve_insertion_order=false")
    s_p, h_p = (DER / "seats.parquet").as_posix(), (DER / "hands.parquet").as_posix()
    dev_pairs = con.sql(f"""
        WITH s AS (SELECT s.hand_idx, s.player_idx FROM '{s_p}' s JOIN '{h_p}' h USING (hand_idx) WHERE h.phase = 0)
        SELECT a.player_idx AS lo, b.player_idx AS hi, count(*)::INT AS dev_shared
        FROM s a JOIN s b ON a.hand_idx = b.hand_idx AND a.player_idx < b.player_idx GROUP BY ALL""").pl()
    lab = labels.with_columns(pl.min_horizontal("p1", "p2").alias("lo"), pl.max_horizontal("p1", "p2").alias("hi"))
    pos_players = set(labels.filter(pl.col("label") == 1).select(pl.concat_list("p1", "p2").explode()).to_series().to_list())
    unk = (dev_pairs.join(lab.select("lo", "hi", "pair_id"), on=["lo", "hi"], how="left")
           .filter(pl.col("pair_id").is_null() & (pl.col("dev_shared") >= 38)
                   & ~pl.col("lo").is_in(list(pos_players)) & ~pl.col("hi").is_in(list(pos_players)))
           .sample(n=N_UNK, seed=SEED)
           .select((pl.lit("U") + pl.col("lo").cast(pl.Utf8) + "_" + pl.col("hi").cast(pl.Utf8)).alias("pair_id"),
                   pl.col("lo").alias("A"), pl.col("hi").alias("B"), pl.lit(-1).cast(pl.Int64).alias("label"),
                   pl.lit("unknown").alias("behavior_family")))
    sel = pl.concat([labels.select("pair_id", pl.col("p1").alias("A"), pl.col("p2").alias("B"), "label", "behavior_family"), unk])
    sel = sel.join(dev_pairs, left_on=[pl.min_horizontal("A", "B"), pl.max_horizontal("A", "B")], right_on=["lo", "hi"],
                   how="left").select("pair_id", "A", "B", "label", "behavior_family", "dev_shared")
    sel.write_parquet(P("pl_pairs"))
    print(sel.group_by("behavior_family").agg(pl.len(), pl.col("dev_shared").mean().round(1).alias("mean_sh"), pl.col("dev_shared").median().alias("med_sh")))
    players = list(set(sel["A"].to_list()) | set(sel["B"].to_list()))
    s = (scan("seats").select("hand_idx", "player_idx").filter(pl.col("player_idx").is_in(players))
         .join(scan("hands").filter(pl.col("phase") == 0).select("hand_idx"), on="hand_idx").collect())
    ph = (sel.lazy().select("pair_id", "A", "B", "label", "behavior_family").join(s.rename({"player_idx": "A"}).lazy(), on="A")
          .join(s.rename({"player_idx": "B"}).lazy(), on=["B", "hand_idx"]).collect())
    ev = pl.read_parquet(DER / "evidence.parquet")
    ph = ph.join(ev.select("pair_id", "hand_idx", "evidence_rank"), on=["pair_id", "hand_idx"], how="left")
    ph.write_parquet(P("pl_pair_hands"))
    hidx = ph["hand_idx"].unique()
    print("pair-hands", ph.shape, "distinct hands", hidx.len(), "evidence matched", ph["evidence_rank"].is_not_null().sum())
    for name in ("hands", "seats", "actions"):
        df = scan(name).filter(pl.col("hand_idx").is_in(hidx.implode())).collect()
        df.write_parquet(P(f"pl_{name}"))
        print(name, df.shape)
        del df


def stage_pairlevel():
    import time
    from sklearn.metrics import roc_auc_score
    t = time.time()
    store = load_store("pl")
    ph = pl.read_parquet(P("pl_pair_hands"))
    F = compute_features(store, ph)
    del store
    F = hand_matrix(F)
    F.write_parquet(P("pl_feat"))
    print("pair-hand features", F.shape, f"{time.time() - t:.0f}s")
    pw = pl.min_horizontal("pA_weak", "pB_weak")
    agg = F.group_by("pair_id", "label", "behavior_family").agg(
        pl.len().alias("n_shared"),
        (pw < 0.02).sum().alias("n_C02"),
        (pw < 0.05).sum().alias("n_C05"),
        ((pw < 0.02) & ((pl.col("fap") + pl.col("rr")) > 0)).sum().alias("n_Cfap"),
        pl.col("first_pair_vol_raise").sum().alias("n_fpvr"),
        pl.col("pair_open").sum().alias("n_open"),
        pl.col("rr").sum().alias("n_rr"),
        pl.col("sandwich").sum().alias("n_sandwich"),
        (pl.col("volA") + pl.col("volB")).sum().alias("n_vol"),
        (pl.col("aggA_pre") + pl.col("aggB_pre")).sum().alias("n_aggr_pre"),
        pl.col("pw_log").sum().alias("sum_pwlog"),
        pl.col("pw_log").top_k(5).mean().alias("top5_pwlog"),
        signature_exprs()["S14_UTG_open_p<.3"].sum().alias("n_S14"),
        signature_exprs()["S16_S15&fpa_p<.3"].sum().alias("n_S16"),
    )
    # member-level trash-raise rate without partner (player baseline, all dev hands)
    pr = pl.read_parquet(P("player_rates")) if P("player_rates").exists() else None
    agg = agg.with_columns(
        (pl.col("n_C02") / pl.col("n_shared")).alias("rate_C02"),
        (pl.col("n_C05") / pl.col("n_shared")).alias("rate_C05"),
        (pl.col("n_Cfap") / pl.col("n_shared")).alias("rate_Cfap"),
        (pl.col("n_fpvr") / pl.col("n_shared")).alias("rate_fpvr"),
        (pl.col("n_open") / pl.col("n_shared")).alias("rate_open"),
        (pl.col("n_rr") / pl.col("n_shared")).alias("rate_rr"),
        (pl.col("n_sandwich") / pl.col("n_shared")).alias("rate_sandwich"),
        (pl.col("n_aggr_pre") / pl.col("n_shared")).alias("rate_aggr_pre"),
        (pl.col("sum_pwlog") / pl.col("n_shared")).alias("mean_pwlog"),
        (pl.col("n_S14") / pl.col("n_shared")).alias("rate_S14"),
        (pl.col("n_S16") / pl.col("n_shared")).alias("rate_S16"),
        # binomial surprise of n_C02 vs population rate 0.004 per shared hand (confound-aware count statistic)
    )
    from scipy.stats import binom
    p0 = 0.004
    agg = agg.with_columns(pl.Series("C02_logsf", -binom.logsf(agg["n_C02"].to_numpy().astype(np.int64) - 1, agg["n_shared"].to_numpy().astype(np.int64), p0)))
    agg.write_parquet(P("pl_pair_agg"))
    stats = ["n_shared", "n_C02", "rate_C02", "C02_logsf", "rate_C05", "rate_Cfap", "rate_fpvr", "rate_open", "rate_rr",
             "rate_sandwich", "rate_aggr_pre", "mean_pwlog", "top5_pwlog", "rate_S14", "rate_S16"]
    print(agg.group_by("behavior_family").agg([pl.len()] + [pl.col(c).mean().round(4) for c in stats]).sort("behavior_family")
          .transpose(include_header=True).to_pandas().to_string())
    rows = []
    for fam in ["coordinated_isolation", "directed_transfer", "soft_play"]:
        for negname, negexpr in [("conf_neg", pl.col("label") == 0), ("unknown", pl.col("label") == -1)]:
            x = agg.filter((pl.col("behavior_family") == fam) | negexpr)
            y = (x["behavior_family"] == fam).to_numpy().astype(int)
            for c in stats:
                a = roc_auc_score(y, x[c].to_numpy())
                rows.append((fam, negname, c, a))
    R = pl.DataFrame(rows, schema=["family", "vs", "stat", "auc"], orient="row")
    R.write_parquet(P("pl_auc"))
    print(R.pivot(on=["family", "vs"], index="stat", values="auc").to_pandas().round(3).to_string())


def stage_lift(phase="0", thr="0.02"):
    """Directed co-player lift of trash raises for ALL players in a phase:
    X trash-raise rate when Y seated vs when Y not seated. Output ci_lift_phase{phase}.parquet (X, Y, n_with, t_with,
    n_X, t_X). Pair features derived in pairlift (min/max over directions, argmax-partner flags)."""
    import duckdb
    con = duckdb.connect()
    con.sql("SET threads=3; SET memory_limit='4GB'; SET preserve_insertion_order=false")
    s_p, h_p = (DER / "seats.parquet").as_posix(), (DER / "hands.parquet").as_posix()
    ag = (OUT / "ci_pre_aggr_all.parquet").as_posix()
    out = (OUT / f"ci_lift_phase{phase}.parquet").as_posix()
    con.sql(f"""COPY (
      WITH s AS (SELECT s.hand_idx, s.player_idx FROM '{s_p}' s JOIN '{h_p}' h USING (hand_idx) WHERE h.phase = {int(phase)}),
      tr AS (SELECT DISTINCT hand_idx, player_idx FROM '{ag}' WHERE phase = {int(phase)} AND p_aggr < {float(thr)}),
      st AS (SELECT s.hand_idx, s.player_idx, (tr.player_idx IS NOT NULL)::INT AS t FROM s LEFT JOIN tr USING (hand_idx, player_idx)),
      tot AS (SELECT player_idx, count(*)::INT AS n_X, sum(t)::INT AS t_X FROM st GROUP BY 1),
      d AS (SELECT a.player_idx AS X, b.player_idx AS Y, count(*)::INT AS n_with, sum(a.t)::INT AS t_with
            FROM st a JOIN s b ON a.hand_idx = b.hand_idx AND a.player_idx <> b.player_idx GROUP BY 1, 2)
      SELECT d.*, tot.n_X, tot.t_X FROM d JOIN tot ON tot.player_idx = d.X
    ) TO '{out}' (FORMAT parquet, COMPRESSION zstd)""")
    L = pl.read_parquet(out)
    print(L.shape, L.select(pl.col("n_with").mean(), pl.col("t_X").mean()))


def lift_pair_features(L, pairs):
    """pairs: df with pair_id, A, B. Returns pair-level lift features (symmetric)."""
    a = 0.5  # smoothing pseudo-counts
    L = L.with_columns(
        ((pl.col("t_with") + a) / (pl.col("n_with") + a / 0.003)).alias("r_with"),
        ((pl.col("t_X") - pl.col("t_with") + a) / (pl.col("n_X") - pl.col("n_with") + a / 0.003)).alias("r_without"),
    ).with_columns((pl.col("r_with") / pl.col("r_without")).log().alias("loglift"))
    L = L.with_columns(
        (pl.col("loglift").rank("ordinal", descending=True).over("X")).alias("lift_rank_X"),
        (pl.col("t_with") - pl.col("n_with") * (pl.col("t_X") - pl.col("t_with")) /
         (pl.col("n_X") - pl.col("n_with")).clip(1, None)).alias("excess_t"),
    )
    ab = pairs.join(L.select(pl.col("X").alias("A"), pl.col("Y").alias("B"), pl.col("loglift").alias("ll_AB"),
                             pl.col("lift_rank_X").alias("rk_AB"), pl.col("excess_t").alias("ex_AB"),
                             pl.col("t_with").alias("t_AB"), pl.col("n_with").alias("n_dev_shared")), on=["A", "B"], how="left")
    ab = ab.join(L.select(pl.col("X").alias("B"), pl.col("Y").alias("A"), pl.col("loglift").alias("ll_BA"),
                          pl.col("lift_rank_X").alias("rk_BA"), pl.col("excess_t").alias("ex_BA"),
                          pl.col("t_with").alias("t_BA")), on=["A", "B"], how="left")
    return ab.with_columns(
        pl.min_horizontal("ll_AB", "ll_BA").alias("loglift_min"),
        pl.max_horizontal("ll_AB", "ll_BA").alias("loglift_max"),
        (pl.col("ll_AB") + pl.col("ll_BA")).alias("loglift_sum"),
        pl.max_horizontal("rk_AB", "rk_BA").alias("lift_rank_worst"),
        ((pl.col("rk_AB") == 1) & (pl.col("rk_BA") == 1)).alias("mutual_top_lift"),
        (pl.col("ex_AB") + pl.col("ex_BA")).alias("excess_trash"),
        pl.min_horizontal("ex_AB", "ex_BA").alias("excess_trash_min"),
    )


def stage_pairlift():
    from sklearn.metrics import roc_auc_score
    L = pl.read_parquet(OUT / "ci_lift_phase0.parquet")
    pairs = pl.read_parquet(P("pl_pairs")).select("pair_id", "A", "B", "label", "behavior_family")
    F = lift_pair_features(L, pairs)
    agg = pl.read_parquet(P("pl_pair_agg")).select("pair_id", "n_shared", "rate_C02", "C02_logsf", "rate_rr", "rate_sandwich", "rate_Cfap")
    F = F.join(agg, on="pair_id", how="left")
    F.write_parquet(P("pl_pair_lift"))
    stats = ["loglift_min", "loglift_max", "loglift_sum", "excess_trash", "excess_trash_min", "lift_rank_worst",
             "mutual_top_lift", "rate_C02", "C02_logsf"]
    print(F.group_by("behavior_family").agg([pl.len()] + [pl.col(c).cast(pl.Float64).mean().round(3) for c in stats]).sort("behavior_family"))
    rows = []
    for fam in ["coordinated_isolation", "directed_transfer"]:
        for negname, negexpr in [("conf_neg", pl.col("label") == 0), ("unknown", pl.col("label") == -1)]:
            x = F.filter((pl.col("behavior_family") == fam) | negexpr)
            y = (x["behavior_family"] == fam).to_numpy().astype(int)
            for c in stats:
                v = x[c].cast(pl.Float64).to_numpy()
                if c == "lift_rank_worst":
                    v = -v
                rows.append((fam, negname, c, roc_auc_score(y, v)))
            # AP at eval-like prevalence: positives vs unknown (unknown treated as negative)
    R = pl.DataFrame(rows, schema=["family", "vs", "stat", "auc"], orient="row")
    print(R.pivot(on=["family", "vs"], index="stat", values="auc").to_pandas().round(4).to_string())
    # hard negatives: confirmed negatives containing a positive player
    labels = pl.read_parquet(DER / "labels.parquet")
    posp = set(labels.filter(pl.col("label") == 1).select(pl.concat_list("p1", "p2").explode()).to_series().to_list())
    x = F.filter(pl.col("label") == 0).with_columns((pl.col("A").is_in(list(posp)) | pl.col("B").is_in(list(posp))).alias("has_pos_player"))
    print(x.group_by("has_pos_player").agg(pl.len(), pl.col("rate_C02").mean().round(4).alias("rateC_mean"),
                                          pl.col("rate_C02").max().round(4).alias("rateC_max"),
                                          pl.col("excess_trash").max().round(3).alias("excess_max"),
                                          pl.col("loglift_min").mean().round(3).alias("llmin_mean"),
                                          pl.col("loglift_min").max().round(3).alias("llmin_max"),
                                          pl.col("mutual_top_lift").mean().round(3).alias("mutual_top")))
    from sklearn.metrics import average_precision_score
    for c in ["rate_C02", "C02_logsf", "excess_trash", "loglift_sum"]:
        for fam in ["coordinated_isolation"]:
            xx = F.filter((pl.col("behavior_family") == fam) | (pl.col("label") <= 0))
            yy = (xx["behavior_family"] == fam).to_numpy().astype(int)
            print(f"AP {fam} vs (conf_neg+unknown, prevalence {yy.mean():.3f}) {c:14s} {average_precision_score(yy, xx[c].to_numpy()):.4f}")
    ci = F.filter(pl.col("behavior_family") == "coordinated_isolation")
    u = F.filter(pl.col("label") <= 0)
    print("CI excess_trash min", ci["excess_trash"].min(), "| negatives+unknown above that:", (u["excess_trash"] >= ci["excess_trash"].min()).sum())
    print(u.sort("excess_trash", descending=True).head(8).select("pair_id", "behavior_family", "n_shared", "rate_C02", "excess_trash", "loglift_min", "mutual_top_lift"))
    ci = F.filter(pl.col("behavior_family") == "coordinated_isolation")
    print("CI loglift_min min/median:", ci["loglift_min"].min(), ci["loglift_min"].median(), " mutual_top_lift frac", ci["mutual_top_lift"].mean())
    print("neg loglift_min max:", F.filter(pl.col("label") == 0)["loglift_min"].max(), " unknown with loglift_min > CI min:",
          (F.filter(pl.col("label") == -1)["loglift_min"] > ci["loglift_min"].min()).sum())


def signature_exprs():
    pw = pl.min_horizontal("pA_weak", "pB_weak")
    fap = (pl.col("foldA_after_aggB") + pl.col("foldB_after_aggA")) > 0
    rr = (pl.col("reraiseA_over_B") + pl.col("reraiseB_over_A")) > 0
    fpvr = pl.col("first_pair_vol_raise") > 0
    return {
        "S1_trash_raise_p02": pw < 0.02,
        "S2_weak_raise_p10": pw < 0.10,
        "S3_first_pair_vol_is_raise": fpvr,
        "S4_partner_fold_after_partner_aggr": fap,
        "S5_partner_reraise": rr,
        "S6_sandwich": pl.col("sandwich") > 0,
        "S7_trash&(fold|reraise)": (pw < 0.02) & (fap | rr),
        "S8_firstvolraise&partnerfold": fpvr & fap,
        "S9_3+_third_folds_pre": pl.col("tf_pre") >= 3,
        "S10_trash&3+_third_folds": (pw < 0.02) & (pl.col("tf_pre") >= 3),
        "S11_weak10|reraise": (pw < 0.10) | rr,
        "S12_firstvolraise&(fold|reraise)&(weak30|reraise)": fpvr & (fap | rr) & ((pw < 0.30) | rr),
        "S13_UTG_first_pair_aggr": pl.col("fpa_ord") == 0,
        "S14_UTG_open_p<.3": (pl.col("fpa_ord") == 0) & (pl.col("fpa_p") < 0.3),
        "S15_early(ord<=2)_pair_aggr_partner_behind&(fold|reraise)": (pl.col("fpa_ord") <= 2) & (pl.col("fpa_partner_behind") > 0) & (fap | rr),
        "S16_S15&fpa_p<.3": (pl.col("fpa_ord") <= 2) & (pl.col("fpa_partner_behind") > 0) & (fap | rr) & (pl.col("fpa_p") < 0.3),
    }


def stage_sigtable():
    F = pl.read_parquet(P("feat_sub"))
    labels = pl.read_parquet(DER / "labels.parquet")
    rows = []
    groups = {"EV_CI": pl.col("grp") == "EV_CI", "CI_nonev_shared": pl.col("grp") == "CI_nonev",
              "NEG_shared": pl.col("grp") == "NEG_shared", "EV_DT": pl.col("grp") == "EV_directed_transfer",
              "EV_SP": pl.col("grp") == "EV_soft_play"}
    for name, e in signature_exprs().items():
        r = {"signature": name}
        for g, ge in groups.items():
            r[g] = F.filter(ge).select(e.cast(pl.Float64).mean()).item()
        rows.append(r)
    T = pl.DataFrame(rows)
    T.write_parquet(P("sigtable"))
    print(T.to_pandas().round(4).to_string())
    # early vs late dev rates of S1 within CI pos pairs (non-evidence) and neg
    x = F.filter(pl.col("src") == "ci_or_neg").with_columns((pl.col("hand_seq") < 1500).alias("early"))
    print(x.group_by("grp", "early").agg(pl.len(), signature_exprs()["S1_trash_raise_p02"].mean().alias("S1_rate"),
                                          signature_exprs()["S7_trash&(fold|reraise)"].mean().alias("S7_rate")).sort("grp", "early"))


def stage_confound(reps="20"):
    """Shared-hand-count confound: subsample each pair to n shared hands and measure AUC/AP of S1/S7 rates."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    F = pl.read_parquet(P("pl_feat"))
    sig = signature_exprs()
    F = F.select("pair_id", "label", "behavior_family", sig["S1_trash_raise_p02"].alias("s1"),
                 sig["S7_trash&(fold|reraise)"].alias("s7"))
    grp = {k[0]: (v["s1"].to_numpy(), v["s7"].to_numpy()) for k, v in F.partition_by("pair_id", as_dict=True).items()}
    meta = F.group_by("pair_id").agg(pl.col("behavior_family").first(), pl.col("label").first(), pl.len().alias("n"))
    rng = np.random.default_rng(SEED)
    A = pl.read_parquet(P("pl_pair_agg")).select("pair_id", "n_shared", "rate_C02", "behavior_family")
    # full-data AUC within n_shared terciles
    x = A.filter(pl.col("behavior_family").is_in(["coordinated_isolation", "none", "unknown"]))
    q1, q2 = x["n_shared"].quantile(1 / 3), x["n_shared"].quantile(2 / 3)
    for lo, hi in [(0, q1), (q1, q2), (q2, 10 ** 6)]:
        xx = x.filter((pl.col("n_shared") >= lo) & (pl.col("n_shared") < hi))
        yy = (xx["behavior_family"] == "coordinated_isolation").to_numpy()
        print(f"n_shared in [{lo:.0f},{hi:.0f}): n={xx.height} CI={yy.sum()} AUC(rate_C02)={roc_auc_score(yy, xx['rate_C02'].to_numpy()):.4f}")
    for n in [20, 38, 60, 100]:
        m = meta.filter((pl.col("n") >= n) & pl.col("behavior_family").is_in(["coordinated_isolation", "none", "unknown"]))
        y = (m["behavior_family"] == "coordinated_isolation").to_numpy().astype(int)
        aucs, aps, aps7 = [], [], []
        for _ in range(int(reps)):
            r1 = np.empty(m.height)
            r7 = np.empty(m.height)
            for i, pid in enumerate(m["pair_id"].to_list()):
                s1, s7 = grp[pid]
                idx = rng.choice(len(s1), n, replace=False)
                r1[i] = s1[idx].mean() + 1e-6 * rng.random()
                r7[i] = s7[idx].mean() + 1e-6 * rng.random()
            aucs.append(roc_auc_score(y, r1))
            aps.append(average_precision_score(y, r1))
            aps7.append(average_precision_score(y, r7))
        print(f"subsample n={n:3d}: pairs={m.height} CI={y.sum()} prevalence={y.mean():.3f}  AUC(S1 rate)={np.mean(aucs):.4f}  "
              f"AP(S1 rate)={np.mean(aps):.4f}  AP(S7 rate)={np.mean(aps7):.4f}")


def stage_printids(outname="ci_archetypes.txt", *hids):
    sub = Sub()
    ph = pl.read_parquet(P("pair_hands"))
    lines = []
    for h in hids:
        r = ph.filter(pl.col("hand_idx") == int(h)).row(0, named=True)
        lines.append(hand_text(sub, int(h), r["A"], r["B"], f"pair {r['pair_id']} evidence_rank {r['evidence_rank']}"))
    txt = "\n".join(lines)
    (OUT / outname).write_text(txt, encoding="utf-8")
    print(txt)


def stage_printother(n="10", fam="soft_play"):
    sub = Sub()
    ev = pl.read_parquet(P("evidence_all")).filter(pl.col("behavior_family") == fam)
    for r in ev.sample(n=int(n), seed=SEED).iter_rows(named=True):
        print(hand_text(sub, r["hand_idx"], r["A"], r["B"], f"EVIDENCE[{fam}] pair {r['pair_id']} rank {r['evidence_rank']}"))


# ============================================================================ population baseline (preflop)
CLASS_SQL = """(CASE WHEN c1//4 = c2//4 THEN (c1::INT//4)*13 + c1//4
      WHEN c1%4 = c2%4 THEN greatest(c1::INT//4, c2::INT//4)*13 + least(c1//4, c2//4)
      ELSE least(c1::INT//4, c2::INT//4)*13 + greatest(c1//4, c2//4) END)"""


def stage_baseline():
    """Population preflop action rates per 169 class x #aggressive actions before (0,1,2,3+).
    Uses gameplay only (all hands, no labels)."""
    import duckdb
    con = duckdb.connect()
    con.sql("SET threads=3; SET memory_limit='4GB'; SET preserve_insertion_order=false")
    a_p, s_p = (DER / "actions.parquet").as_posix(), (DER / "seats.parquet").as_posix()
    df = con.sql(f"""
      WITH a AS (
        SELECT hand_idx, action_no, player_idx, action,
               (action IN (3,4) OR (action=5 AND amount>to_call))::INT AS aggr
        FROM '{a_p}' WHERE street=0),
      b AS (
        SELECT *, coalesce(sum(aggr) OVER (PARTITION BY hand_idx ORDER BY action_no
                   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS nr_before FROM a)
      SELECT {CLASS_SQL}::SMALLINT AS class_id, least(nr_before, 3)::TINYINT AS nr,
             count(*)::INT AS n, avg(aggr) AS p_aggr, avg((action=0)::INT) AS p_fold,
             avg((action=2 OR (action=5 AND aggr=0))::INT) AS p_call
      FROM b JOIN '{s_p}' s USING (hand_idx, player_idx)
      GROUP BY ALL ORDER BY 1, 2""").pl()
    df.write_parquet(OUT / "ci_baseline_pre.parquet")
    h_p = (DER / "hands.parquet").as_posix()
    df2 = con.sql(f"""
      WITH a AS (
        SELECT hand_idx, action_no, player_idx, action,
               (action IN (3,4) OR (action=5 AND amount>to_call))::INT AS aggr,
               (action=2 OR (action=5 AND amount<=to_call))::INT AS isc
        FROM '{a_p}' WHERE street=0),
      b AS (
        SELECT *, coalesce(sum(aggr) OVER w, 0) AS nr_before, coalesce(sum(isc) OVER w, 0) AS nc_before
        FROM a WINDOW w AS (PARTITION BY hand_idx ORDER BY action_no ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING))
      SELECT {CLASS_SQL}::SMALLINT AS class_id, least(nr_before, 3)::TINYINT AS nr,
             ((s.seat_no - h.button_seat + 6) % 6)::TINYINT AS pos, least(nc_before, 2)::TINYINT AS nc,
             count(*)::INT AS n, avg(aggr) AS p_aggr, avg((action=0)::INT) AS p_fold, avg(isc) AS p_call
      FROM b JOIN '{s_p}' s USING (hand_idx, player_idx) JOIN '{h_p}' h USING (hand_idx)
      GROUP BY ALL ORDER BY 1, 2, 3, 4""").pl()
    df2.write_parquet(OUT / "ci_baseline_pre_pos.parquet")
    print("positional baseline cells", df2.height, "cells n>=150:", df2.filter(pl.col("n") >= 150).height)
    from poker_equity import class_name
    piv = df.filter(pl.col("nr") == 0).with_columns(
        pl.col("class_id").map_elements(class_name, return_dtype=pl.Utf8).alias("cls")).sort("p_aggr")
    print(piv.head(10))
    print(piv.tail(10))
    print(df.group_by("nr").agg(pl.col("n").sum(), (pl.col("p_aggr") * pl.col("n")).sum() / pl.col("n").sum()))


# ============================================================================ pair-hand feature engine (numba)
FEATS = [
    "aggA_pre", "aggB_pre", "aggA_post", "aggB_post",            # aggressive action counts
    "pA_weak", "pB_weak",                                        # min baseline P(aggr) over own preflop aggr (1 if none)
    "foldA_after_aggB", "foldB_after_aggA",                      # loose: fold later in hand than partner's first aggr
    "foldA_facing_B", "foldB_facing_A",                          # strict: fold while partner is last aggressor on street
    "pfoldA_facing_B", "pfoldB_facing_A",                        # baseline P(fold) of that folded hand (low = strong fold)
    "reraiseA_over_B", "reraiseB_over_A",                        # aggr while partner is last aggressor on street
    "third_fold_facing_pair", "third_aggr", "third_vol",         # third-party responses
    "volA", "volB",                                              # voluntary preflop entry (call/raise)
    "winA", "winB", "showdown", "pot_bb", "netA_bb", "netB_bb",
    "hu_pair_post_bets", "hu_pair_post_checks", "hu_pair_post_actions",  # postflop, only A and B active
    "airbetA", "airbetB",                                        # postflop bet/raise with HS vs random < 0.5
    "hsA_min_aggr_post", "hsB_min_aggr_post",
    "hsA_fold_facing_B", "hsB_fold_facing_A",                    # HS vs random when folding to partner postflop (max)
    "first_aggr_role",                                           # 0 none, 1 A, 2 B, 3 other (preflop first raiser)
    "third_fold_after_pair_pre",                                 # third-party folds preflop after a pair aggr
    "surpA", "surpB",                                            # sum -log(P_aggr) over preflop aggr
    "n_act_pre", "players_vol",
    "isoA", "isoB",                                              # X aggr pre with >=1 third player voluntarily in before it
    "sandwich",                                                  # X aggr, third player acts (call/raise), Y aggr over it
    "pair_aggr_then_partner_fold_then_third_fold",
    "streetA_fold", "streetB_fold",
    "tf_both_active",                                            # third-party folds facing pair aggr while A and B both live
    "tf_pre",                                                    # third-party preflop folds facing pair aggr
    "pair_open",                                                 # first voluntary preflop action of the hand = pair raise
    "first_pair_vol_raise",                                      # first voluntary pair action preflop is a raise
    "pair_win_nosd", "n_pair_trash",                             # pair member wins w/o showdown; # pair preflop aggr with p<.02
    "pair_hu_post_actions", "pair_hu_post_bets", "pair_hu_post_checks",  # postflop, exactly A and B live
    "fpa_ord",                                                   # preflop acting order of first pair aggr (0 UTG..5 BB; 9 none)
    "fpa_nr", "fpa_p",                                           # #prior aggr and population p_aggr at that action
    "fpa_partner_acted_before",                                  # partner already acted preflop before that aggr
    "fpa_partner_behind",                                        # partner yet to act (sits behind) at that aggr
]
NF = len(FEATS)
FI = {f: i for i, f in enumerate(FEATS)}
globals().update({f"F_{f}": i for i, f in enumerate(FEATS)})


def _build_numba():
    from numba import njit
    from poker_equity import eval_cards, hand_strength_vs_random, class_id



    @njit(cache=False)
    def hs_rand(c1, c2, board, nb):
        dead = np.full(1, -1, dtype=np.int64)
        return hand_strength_vs_random(c1, c2, board, nb, dead, 0)

    @njit(cache=False)
    def run(ph_hrow, ph_A, ph_B, a_start, a_end, s_start, s_end,
            a_street, a_player, a_action, a_amount, a_to_call, a_pactive,
            s_player, s_c1, s_c2, s_net, s_won, s_folded, s_sd, s_seat,
            h_bb, h_pot, h_board, h_nboard, h_button, p_aggr_tab, p_fold_tab):
        n = ph_hrow.shape[0]
        out = np.zeros((n, NF), dtype=np.float64)
        board = np.zeros(5, dtype=np.int64)
        for i in range(n):
            hr = ph_hrow[i]
            A = ph_A[i]
            B = ph_B[i]
            bb = h_bb[hr]
            nbd = h_nboard[hr]
            for j in range(5):
                board[j] = h_board[hr, j]
            # seats
            cA1 = -1; cA2 = -1; cB1 = -1; cB2 = -1
            clsA = -1; clsB = -1
            nseats = s_end[hr] - s_start[hr]
            seat_pl = np.empty(nseats, dtype=np.int64)
            seat_cls = np.empty(nseats, dtype=np.int64)
            seat_pos = np.empty(nseats, dtype=np.int64)
            for k in range(s_start[hr], s_end[hr]):
                kk = k - s_start[hr]
                seat_pl[kk] = s_player[k]
                seat_pos[kk] = (s_seat[k] - h_button[hr] + 6) % 6
                seat_cls[kk] = class_id(s_c1[k], s_c2[k])
                if s_player[k] == A:
                    cA1 = s_c1[k]; cA2 = s_c2[k]; clsA = seat_cls[kk]
                    out[i, F_netA_bb] = s_net[k] / bb
                    if s_won[k] > 0: out[i, F_winA] = 1
                    if s_sd[k]: out[i, F_showdown] = 1
                if s_player[k] == B:
                    cB1 = s_c1[k]; cB2 = s_c2[k]; clsB = seat_cls[kk]
                    out[i, F_netB_bb] = s_net[k] / bb
                    if s_won[k] > 0: out[i, F_winB] = 1
            out[i, F_pot_bb] = h_pot[hr] / bb
            out[i, F_pA_weak] = 1.0
            out[i, F_pB_weak] = 1.0
            out[i, F_hsA_min_aggr_post] = 1.0
            out[i, F_hsB_min_aggr_post] = 1.0
            out[i, F_pfoldA_facing_B] = 1.0
            out[i, F_pfoldB_facing_A] = 1.0
            # action walk
            cur_street = -1
            last_aggr = -1          # player idx of last aggressor on street
            nr = 0                  # aggr count on street
            first_aggr_A = 10**9    # action position of first aggr
            first_aggr_B = 10**9
            first_pre_aggr = -2
            third_vol_before = 0    # third players voluntarily in preflop before current point
            third_vol_pl = np.full(nseats, 0, dtype=np.int64)
            last_pair_aggr_role = 0  # 1 A, 2 B (preflop, latest)
            third_since_pair_aggr = 0
            partner_fold_seen = 0
            foldedA = 0
            foldedB = 0
            actedA = 0
            actedB = 0
            fpa_done = 0
            out[i, F_fpa_ord] = 9
            out[i, F_fpa_nr] = -1
            out[i, F_fpa_p] = 1.0
            any_vol = 0
            pair_vol_seen = 0
            pos = 0
            ncall = 0
            nvol = 0
            vol_seen = np.zeros(nseats, dtype=np.int64)
            for k in range(a_start[hr], a_end[hr]):
                st = a_street[k]
                if st != cur_street:
                    cur_street = st
                    last_aggr = -1
                    nr = 0
                    third_since_pair_aggr = 0
                p = a_player[k]
                act = a_action[k]
                aggr = (act == 3) or (act == 4) or (act == 5 and a_amount[k] > a_to_call[k])
                isA = p == A
                isB = p == B
                if st == 0:
                    out[i, F_n_act_pre] += 1
                    if act == 2 or aggr or act == 5:
                        for q in range(nseats):
                            if seat_pl[q] == p and vol_seen[q] == 0:
                                vol_seen[q] = 1
                                nvol += 1
                # hand strength for postflop pair actions
                hs = -1.0
                if st > 0 and (isA or isB) and (aggr or act == 0):
                    nb = 3 if st == 1 else (4 if st == 2 else 5)
                    if nb > nbd:
                        nb = nbd
                    if nb >= 3:
                        if isA:
                            hs = hs_rand(cA1, cA2, board, nb)
                        else:
                            hs = hs_rand(cB1, cB2, board, nb)
                if st == 0 and (act == 2 or aggr or act == 5):
                    if any_vol == 0 and (isA or isB) and aggr:
                        out[i, F_pair_open] = 1
                    any_vol = 1
                    if (isA or isB) and pair_vol_seen == 0:
                        pair_vol_seen = 1
                        if aggr:
                            out[i, F_first_pair_vol_raise] = 1
                if st > 0 and a_pactive[k] == 2 and (isA or isB) and foldedA == 0 and foldedB == 0:
                    out[i, F_pair_hu_post_actions] += 1
                    if aggr:
                        out[i, F_pair_hu_post_bets] += 1
                    elif act == 1:
                        out[i, F_pair_hu_post_checks] += 1
                if isA or isB:
                    me = 0 if isA else 1
                    cls = clsA if isA else clsB
                    partner = B if isA else A
                    ppos = 0
                    for q in range(nseats):
                        if seat_pl[q] == p:
                            ppos = seat_pos[q]
                    if aggr:
                        if st == 0:
                            out[i, F_aggA_pre + me] += 1
                            pa = p_aggr_tab[cls, min(nr, 3), ppos, min(ncall, 2)]
                            if pa < out[i, F_pA_weak + me]:
                                out[i, F_pA_weak + me] = pa
                            out[i, F_surpA + me] += -np.log(max(pa, 1e-4))
                            if fpa_done == 0:
                                fpa_done = 1
                                out[i, F_fpa_ord] = (ppos - 3 + 6) % 6
                                out[i, F_fpa_nr] = nr
                                out[i, F_fpa_p] = pa
                                pacted = actedB if isA else actedA
                                out[i, F_fpa_partner_acted_before] = pacted
                                out[i, F_fpa_partner_behind] = 1 - pacted
                            if pa < 0.02:
                                out[i, F_n_pair_trash] += 1
                            if third_vol_before > 0:
                                out[i, F_isoA + me] = 1
                            # sandwich: partner aggr earlier this street, a third player acted voluntarily since
                            if last_pair_aggr_role == (2 - me) and third_since_pair_aggr > 0:
                                out[i, F_sandwich] = 1
                            last_pair_aggr_role = me + 1
                            third_since_pair_aggr = 0
                            if first_pre_aggr == -2:
                                first_pre_aggr = me + 1
                        else:
                            out[i, F_aggA_post + me] += 1
                            if hs >= 0:
                                if hs < 0.5:
                                    out[i, F_airbetA + me] += 1
                                if hs < out[i, F_hsA_min_aggr_post + me]:
                                    out[i, F_hsA_min_aggr_post + me] = hs
                        if last_aggr == partner:
                            out[i, F_reraiseA_over_B + me] += 1
                        if isA and first_aggr_A > pos:
                            first_aggr_A = pos
                        if isB and first_aggr_B > pos:
                            first_aggr_B = pos
                    elif act == 0:
                        if isA and first_aggr_B < pos:
                            out[i, F_foldA_after_aggB] = 1
                            partner_fold_seen = 1
                        if isB and first_aggr_A < pos:
                            out[i, F_foldB_after_aggA] = 1
                            partner_fold_seen = 1
                        out[i, F_streetA_fold + me] = st + 1
                        if isA:
                            foldedA = 1
                        else:
                            foldedB = 1
                        if last_aggr == partner:
                            out[i, F_foldA_facing_B + me] = 1
                            if st == 0:
                                pf = p_fold_tab[cls, min(nr, 3), ppos, min(ncall, 2)]
                                if pf < out[i, F_pfoldA_facing_B + me]:
                                    out[i, F_pfoldA_facing_B + me] = pf
                            elif hs >= 0:
                                if hs > out[i, F_hsA_fold_facing_B + me]:
                                    out[i, F_hsA_fold_facing_B + me] = hs
                    if st == 0 and (act == 2 or aggr or act == 5):
                        if isA: out[i, F_volA] = 1
                        else: out[i, F_volB] = 1
                else:
                    if aggr:
                        out[i, F_third_aggr] += 1
                        if st == 0 and first_pre_aggr == -2:
                            first_pre_aggr = 3
                    if st == 0 and (act == 2 or aggr or act == 5):
                        third_vol_before += 1
                        if last_pair_aggr_role > 0:
                            third_since_pair_aggr += 1
                    if act == 0 and (last_aggr == A or last_aggr == B):
                        out[i, F_third_fold_facing_pair] += 1
                        if st == 0:
                            out[i, F_tf_pre] += 1
                        if foldedA == 0 and foldedB == 0:
                            out[i, F_tf_both_active] += 1
                        if partner_fold_seen:
                            out[i, F_pair_aggr_then_partner_fold_then_third_fold] = 1
                    if act == 0 and st == 0 and last_pair_aggr_role > 0:
                        out[i, F_third_fold_after_pair_pre] += 1
                # heads-up pair postflop: only A and B remain active
                if st > 0 and a_pactive[k] == 2 and (isA or isB):
                    # the two active players include p; check partner still active via folded status later
                    out[i, F_hu_pair_post_actions] += 1
                    if aggr:
                        out[i, F_hu_pair_post_bets] += 1
                    elif act == 1:
                        out[i, F_hu_pair_post_checks] += 1
                if st == 0 and isA:
                    actedA = 1
                if st == 0 and isB:
                    actedB = 1
                if aggr:
                    last_aggr = p
                    nr += 1
                elif st == 0 and (act == 2 or act == 5):
                    ncall += 1
                pos += 1
            out[i, F_first_aggr_role] = 0 if first_pre_aggr == -2 else first_pre_aggr
            out[i, F_third_vol] = third_vol_before
            out[i, F_players_vol] = nvol
            if (out[i, F_winA] > 0 or out[i, F_winB] > 0) and out[i, F_showdown] == 0:
                out[i, F_pair_win_nosd] = 1
        return out

    return run


def load_store(tag="sub"):
    """Load hands/seats/actions (subset parquet files) into numpy arrays with per-hand offsets."""
    if tag == "sub":
        hands = pl.read_parquet(P("sub_hands"))
        seats = pl.read_parquet(P("sub_seats"))
        acts = pl.read_parquet(P("sub_actions"))
    else:
        hands = pl.read_parquet(P(f"{tag}_hands"))
        seats = pl.read_parquet(P(f"{tag}_seats"))
        acts = pl.read_parquet(P(f"{tag}_actions"))
    hands = hands.sort("hand_idx").with_row_index("hrow")
    hrow = dict(zip(hands["hand_idx"].to_list(), hands["hrow"].to_list()))
    acts = acts.sort("hand_idx", "action_no")
    seats = seats.sort("hand_idx", "seat_no")
    nh = hands.height

    def offsets(df):
        hi = df["hand_idx"].to_numpy()
        rows = hands.select("hand_idx").to_series().to_numpy()
        start = np.searchsorted(hi, rows, side="left")
        end = np.searchsorted(hi, rows, side="right")
        return start.astype(np.int64), end.astype(np.int64)

    a_s, a_e = offsets(acts)
    s_s, s_e = offsets(seats)
    board = np.full((nh, 5), -1, dtype=np.int64)
    nboard = np.zeros(nh, dtype=np.int64)
    for r, b in enumerate(hands["board"].to_list()):
        if b:
            board[r, :len(b)] = b
            nboard[r] = len(b)
    st = dict(hands=hands, hrow=hrow, a_s=a_s, a_e=a_e, s_s=s_s, s_e=s_e, board=board, nboard=nboard,
              a_street=acts["street"].to_numpy().astype(np.int64), a_player=acts["player_idx"].to_numpy().astype(np.int64),
              a_action=acts["action"].to_numpy().astype(np.int64), a_amount=acts["amount"].to_numpy().astype(np.int64),
              a_to_call=acts["to_call"].to_numpy().astype(np.int64), a_pactive=acts["players_active"].to_numpy().astype(np.int64),
              s_player=seats["player_idx"].to_numpy().astype(np.int64), s_c1=seats["c1"].to_numpy().astype(np.int64),
              s_c2=seats["c2"].to_numpy().astype(np.int64), s_net=seats["net_chips"].to_numpy().astype(np.int64),
              s_won=seats["won_share"].to_numpy().astype(np.float64), s_folded=seats["folded"].to_numpy(),
              s_sd=seats["went_to_showdown"].to_numpy(), s_seat=seats["seat_no"].to_numpy().astype(np.int64),
              h_button=hands["button_seat"].to_numpy().astype(np.int64),
              h_bb=hands["bb"].to_numpy().astype(np.float64), h_pot=hands["final_pot"].to_numpy().astype(np.float64),
              h_seq=hands["hand_seq"].to_numpy())
    return st


def baseline_tables(min_n=150):
    """4-D population tables [class, nr_before(0..3), pos(0 BTN,1 SB,2 BB,3 UTG,4 HJ,5 CO), ncall_before(0..2)]
    with back-off to [class, nr] when a cell has < min_n decisions."""
    b = pl.read_parquet(OUT / "ci_baseline_pre.parquet")
    b2 = pl.read_parquet(OUT / "ci_baseline_pre_pos.parquet")
    pa = np.full((169, 4, 6, 3), 0.5)
    pf = np.full((169, 4, 6, 3), 0.5)
    for r in b.iter_rows(named=True):
        pa[r["class_id"], r["nr"], :, :] = r["p_aggr"]
        pf[r["class_id"], r["nr"], :, :] = r["p_fold"]
    for r in b2.filter(pl.col("n") >= min_n).iter_rows(named=True):
        pa[r["class_id"], r["nr"], r["pos"], r["nc"]] = r["p_aggr"]
        pf[r["class_id"], r["nr"], r["pos"], r["nc"]] = r["p_fold"]
    return pa, pf


def compute_features(store, ph):
    """ph: polars df with hand_idx, A, B (+ any other columns kept)."""
    run = _build_numba()
    pa, pf = baseline_tables()
    hr = np.array([store["hrow"][h] for h in ph["hand_idx"].to_list()], dtype=np.int64)
    X = run(hr, ph["A"].to_numpy().astype(np.int64), ph["B"].to_numpy().astype(np.int64),
            store["a_s"], store["a_e"], store["s_s"], store["s_e"],
            store["a_street"], store["a_player"], store["a_action"], store["a_amount"], store["a_to_call"],
            store["a_pactive"], store["s_player"], store["s_c1"], store["s_c2"], store["s_net"], store["s_won"],
            store["s_folded"], store["s_sd"], store["s_seat"], store["h_bb"], store["h_pot"], store["board"],
            store["nboard"], store["h_button"], pa, pf)
    return ph.with_columns([pl.Series(f, X[:, j]) for j, f in enumerate(FEATS)]).with_columns(
        pl.Series("hand_seq", store["h_seq"][hr]))


def stage_features():
    store = load_store("sub")
    ph = pl.read_parquet(P("pair_hands"))
    ev = pl.read_parquet(P("evidence_all"))
    oth = ev.filter(pl.col("behavior_family") != FAM).select(
        "pair_id", "A", "B", pl.lit(1).cast(pl.Int64).alias("label"), "hand_idx", "evidence_rank", "behavior_family")
    ph = ph.with_columns(pl.lit(None).cast(pl.Utf8).alias("behavior_family"))
    ph = ph.with_columns(pl.when(pl.col("label") == 1).then(pl.lit(FAM)).otherwise(pl.lit("none")).alias("behavior_family"))
    allph = pl.concat([ph.with_columns(pl.lit("ci_or_neg").alias("src")), oth.with_columns(pl.lit("oth_ev").alias("src"))],
                      how="diagonal_relaxed")
    import time
    t = time.time()
    F = compute_features(store, allph.select("pair_id", "A", "B", "hand_idx", "label", "evidence_rank", "behavior_family", "src"))
    print("features", F.shape, f"{time.time() - t:.1f}s")
    grp = (pl.when(pl.col("src") == "oth_ev").then(pl.lit("EV_") + pl.col("behavior_family"))
           .when((pl.col("label") == 1) & pl.col("evidence_rank").is_not_null()).then(pl.lit("EV_CI"))
           .when(pl.col("label") == 1).then(pl.lit("CI_nonev")).otherwise(pl.lit("NEG_shared")))
    F = F.with_columns(grp.alias("grp"))
    F.write_parquet(P("feat_sub"))
    print(F.group_by("grp").agg([pl.len()] + [pl.col(f).mean().round(3) for f in FEATS]).sort("grp")
          .transpose(include_header=True).to_pandas().to_string())


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "extract"
    globals()[f"stage_{st}"](*sys.argv[2:])
