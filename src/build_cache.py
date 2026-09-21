"""Build the integer-coded data cache in data/derived/ (run once, ~minutes).

String IDs are mapped to dense ints purely for memory/speed. Hand ints follow
(table_id, started_at) chronological order -- gameplay time, not file order.
Nothing here derives features from ID formats or row ordering (rules forbid that).

Outputs (all parquet, zstd):
  players.parquet  player_idx, player_id, metadata...
  tables.parquet   table_idx, table_id
  hands.parquet    hand_idx, hand_id, table_idx, ts (epoch s), phase (0=dev,1=eval),
                   hand_seq (per-table chronological index), button_seat, sb, bb,
                   board_cards, board (list<int8> 0..51), final_pot, players_dealt, players_at_showdown
  seats.parquet    hand_idx, player_idx, seat_no, starting_stack, c1, c2 (int8 0..51),
                   total_contribution, net_chips, folded, went_to_showdown, won_share
  actions.parquet  hand_idx, action_no, street (0..3), player_idx, action (0..5),
                   amount, amount_to, pot_before, stack_before, to_call, players_active
  labels.parquet / evidence.parquet / eval_pairs.parquet  with idx columns added
Card code = rank_idx*4 + suit_idx, rank '23456789TJQKA', suit 'cdhs'.
Action codes: fold0 check1 call2 bet3 raise4 all_in5. Street: preflop0 flop1 turn2 river3.
"""
from pathlib import Path

import duckdb

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "derived"
OUT.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
con.sql("SET threads=10; SET memory_limit='20GB'; SET preserve_insertion_order=false")


def q(sql):
    con.sql(sql)


def save(name, sql):
    con.sql(f"COPY ({sql}) TO '{(OUT / name).as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
    n = con.sql(f"select count(*) from '{(OUT / name).as_posix()}'").fetchone()[0]
    print(f"{name}: {n:,} rows")


r = lambda t: (RAW / t).as_posix()

CARD = """(CASE substr({c},1,1) WHEN '2' THEN 0 WHEN '3' THEN 1 WHEN '4' THEN 2 WHEN '5' THEN 3
 WHEN '6' THEN 4 WHEN '7' THEN 5 WHEN '8' THEN 6 WHEN '9' THEN 7 WHEN 'T' THEN 8 WHEN 'J' THEN 9
 WHEN 'Q' THEN 10 WHEN 'K' THEN 11 WHEN 'A' THEN 12 END)*4 +
 (CASE substr({c},2,1) WHEN 'c' THEN 0 WHEN 'd' THEN 1 WHEN 'h' THEN 2 WHEN 's' THEN 3 END)"""

q(f"CREATE TEMP TABLE pmap AS SELECT player_id, (row_number() OVER (ORDER BY player_id) - 1)::INT AS player_idx FROM '{r('players.parquet')}'")
q(f"CREATE TEMP TABLE tmap AS SELECT table_id, (row_number() OVER (ORDER BY table_id) - 1)::SMALLINT AS table_idx FROM (SELECT DISTINCT table_id FROM '{r('hands.parquet')}')")
q(f"""CREATE TEMP TABLE hmap AS
SELECT h.hand_id,
  (row_number() OVER (ORDER BY t.table_idx, h.started_at, h.hand_id) - 1)::INT AS hand_idx,
  t.table_idx,
  (row_number() OVER (PARTITION BY t.table_idx ORDER BY h.started_at, h.hand_id) - 1)::INT AS hand_seq,
  epoch(h.started_at) AS ts,
  (h.phase = 'evaluation')::TINYINT AS phase,
  h.button_seat::TINYINT AS button_seat, h.small_blind::INT AS sb, h.big_blind::INT AS bb,
  h.board_cards, h.final_pot::INT AS final_pot, h.players_dealt::TINYINT AS players_dealt,
  h.players_at_showdown::TINYINT AS players_at_showdown
FROM '{r('hands.parquet')}' h JOIN tmap t USING (table_id)""")

save("players.parquet", f"SELECT m.player_idx, p.* FROM '{r('players.parquet')}' p JOIN pmap m USING (player_id) ORDER BY player_idx")
save("tables.parquet", "SELECT table_idx, table_id FROM tmap ORDER BY table_idx")
save("hands.parquet", f"""SELECT hand_idx, hand_id, table_idx, hand_seq, ts, phase, button_seat, sb, bb, board_cards,
  list_transform(CASE WHEN board_cards = '' OR board_cards IS NULL THEN [] ELSE string_split(board_cards, ' ') END,
                 c -> ({CARD.format(c='c')})::TINYINT) AS board,
  final_pot, players_dealt, players_at_showdown
FROM hmap ORDER BY hand_idx""")
save("seats.parquet", f"""SELECT h.hand_idx, m.player_idx, s.seat_no::TINYINT AS seat_no, s.starting_stack::INT AS starting_stack,
  ({CARD.format(c='s.hole_card_1')})::TINYINT AS c1, ({CARD.format(c='s.hole_card_2')})::TINYINT AS c2,
  s.total_contribution::INT AS total_contribution, s.net_chips::INT AS net_chips,
  s.folded, s.went_to_showdown, s.won_share::FLOAT AS won_share
FROM '{r('seats.parquet')}' s JOIN hmap h USING (hand_id) JOIN pmap m USING (player_id)
ORDER BY h.hand_idx, s.seat_no""")
save("actions.parquet", f"""SELECT h.hand_idx, a.action_no::SMALLINT AS action_no,
  (CASE a.street WHEN 'preflop' THEN 0 WHEN 'flop' THEN 1 WHEN 'turn' THEN 2 WHEN 'river' THEN 3 END)::TINYINT AS street,
  m.player_idx,
  (CASE a.action WHEN 'fold' THEN 0 WHEN 'check' THEN 1 WHEN 'call' THEN 2 WHEN 'bet' THEN 3 WHEN 'raise' THEN 4 WHEN 'all_in' THEN 5 END)::TINYINT AS action,
  a.amount::INT AS amount, a.amount_to::INT AS amount_to, a.pot_before::INT AS pot_before,
  a.stack_before::INT AS stack_before, a.to_call::INT AS to_call, a.players_active::TINYINT AS players_active
FROM '{r('actions.parquet')}' a JOIN hmap h USING (hand_id) JOIN pmap m USING (player_id)
ORDER BY h.hand_idx, a.action_no""")
save("labels.parquet", f"""SELECT l.*, a.player_idx AS p1, b.player_idx AS p2
FROM '{r('development_labels.csv')}' l JOIN pmap a ON a.player_id = l.player_1 JOIN pmap b ON b.player_id = l.player_2""")
save("evidence.parquet", f"""SELECT e.*, h.hand_idx FROM '{r('development_evidence.csv')}' e JOIN hmap h USING (hand_id)""")
save("eval_pairs.parquet", f"""SELECT e.*, a.player_idx AS p1, b.player_idx AS p2
FROM '{r('evaluation_pairs.csv')}' e JOIN pmap a ON a.player_id = e.player_1 JOIN pmap b ON b.player_id = e.player_2""")
print("done")
