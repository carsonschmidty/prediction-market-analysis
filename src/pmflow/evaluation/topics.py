"""Topic / conflict clusters for markets (evaluation role).

Purpose: a coarser clustering than event_id for bootstrap CIs and for measuring how much of the test split
sits in topics that also appear in train. Rules use ONLY ex-ante text (event title + question) and the
universe; they were written from event titles before any test metric was computed and are not tuned.

Two levels:
* ``topic``    fine cluster (first matching rule wins), e.g. iran_gulf, yemen_redsea, israel_levant,
               russia_ukraine, fed, us_yields, ...
* ``theatre``  coarse cluster: middle_east = iran_gulf + yemen_redsea + israel_levant (one correlated 2026
               conflict), us_rates = fed + us_yields, otherwise = topic.
* ``template`` event-title template with dates, numbers and month names stripped: recurring series such as
               "Israel military action against Lebanon on...?" get one template across many event_ids.
"""
from __future__ import annotations

import re

import pandas as pd

MONTHS = ("january|february|march|april|may|june|july|august|september|october|november|december|"
          "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")

# (topic, regex) in priority order; matched against lower-cased "event_title || ' ' || question"
INSCOPE_RULES = [
    ("yemen_redsea", r"houthi|yemen|bab el-mandeb|bab-el-mandeb|red sea|aden|sanaa|marib|taiz"),
    ("iran_gulf", r"iran|hormuz|khamenei|tehran|persian gulf|gulf state|arab countr|oman|qatar|kharg|irgc|saudi|riyadh|"
                  r"uae|emirat|kuwait|bahrain|sharjah|centcom|al udeid|ukmto|tanker|opec|isfahan|fordow|natanz|"
                  r"abraham accords|mecca"),
    ("israel_levant", r"israel|lebanon|beirut|gaza|hamas|hezbollah|west bank|netanyahu|idf|syria|damascus|"
                      r"iraq|baghdad|jordan|egypt"),
    ("russia_ukraine", r"russia|ukrain|kyiv|kiev|putin|zelensk|donbas|donetsk|crimea|kharkiv|zaporizh|"
                       r"pokrovsk|kostyantyniv|dobropil|kursk|moscow|belarus|lukashenko"),
    ("venezuela_latam", r"venezuela|maduro|machado|cuba|panama|greenland|colombia|mexic|brazil|argentin|"
                        r"haiti|nicaragua|peru|chile|ecuador|bolivia|cartel|sinaloa"),
    ("china_asia", r"china|chinese|xi jinping|\bxi\b|taiwan|korea|kim jong|japan|takaichi|india|pakistan|"
                   r"kashmir|philippin|vietnam|thailand|cambodia|myanmar|afghan|taliban|bangladesh"),
    ("nato_europe", r"nato|europe|\beu\b|france|macron|germany|merz|uk |britain|starmer|poland|romania|"
                    r"hungary|orban|italy|spain|serbia|kosovo|georgia|armenia|azerbaijan|turkey|erdo|"
                    r"denmark|sweden|finland|norway|baltic|moldova"),
    ("africa", r"sudan|congo|ethiopia|somalia|nigeria|niger|mali|libya|south africa|kenya|sahel|rwanda"),
    ("trump_us_exec", r"trump|white house|cabinet|executive order|pardon|tariff|press secretary|hegseth|"
                      r"rubio|vance|national guard|insurrection|doge|musk|secretary|fbi|cia|pentagon|"
                      r"department of|attorney general|biden|harris"),
]
MACRO_RULES = [
    ("us_yields", r"treasury|yield"),
    ("fed", r"\bfed\b|fomc|federal reserve|powell|rate hike|rate cut|dissent"),
    ("ecb", r"\becb\b|european central"),
    ("boe", r"bank of england|\bboe\b"),
    ("boj", r"bank of japan|\bboj\b"),
    ("boc", r"bank of canada"),
    ("rba_rbnz", r"reserve bank of australia|\brba\b|reserve bank of new zealand|rbnz"),
    ("em_cb", r"brazil|mexico|russia|china|pboc|india|turkey|colombia|chile|south africa|korea|indonesia|"
              r"poland|hungary|czech|peru|philippin|thailand"),
    ("other_cb", r"swiss|snb|norges|riksbank|sweden|norway|bank of"),
]
THEATRE = {"iran_gulf": "middle_east", "yemen_redsea": "middle_east", "israel_levant": "middle_east",
           "fed": "us_rates", "us_yields": "us_rates"}


def _match(text: str, rules) -> str:
    for name, rx in rules:
        if re.search(rx, text):
            return name
    return "other"


def template(title: str) -> str:
    t = (title or "").lower()
    t = re.sub(rf"\b({MONTHS})\b\.?", " ", t)
    t = re.sub(r"\d+(st|nd|rd|th)?", " ", t)
    t = re.sub(r"[^a-z ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def assign(markets: pd.DataFrame) -> pd.DataFrame:
    """markets: market_id, universe, event_title, question. Returns market_id, topic, theatre, template."""
    out = []
    for r in markets.itertuples(index=False):
        text = f"{r.event_title or ''} {r.question or ''}".lower()
        topic = _match(text, MACRO_RULES if r.universe == "macro" else INSCOPE_RULES)
        if r.universe == "macro":
            topic = "macro_" + topic
        th = THEATRE.get(topic.replace("macro_", ""), topic) if r.universe != "macro" else (
            "macro_us_rates" if topic in ("macro_fed", "macro_us_yields") else topic)
        out.append((r.market_id, topic, th, template(r.event_title or r.question)))
    return pd.DataFrame(out, columns=["market_id", "topic", "theatre", "template"])


def load(conn=None) -> pd.DataFrame:
    from .harness import connect
    with (conn or connect()) as c:
        cur = c.execute("SELECT market_id, universe, event_title, question FROM eval_markets")
        m = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    return assign(m)
