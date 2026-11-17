"""Transparent rule-based market taxonomy.

Each market gets:
  category       geopolitical | macro_central_bank | elections_domestic_politics | sports | crypto |
                 entertainment | other
  geo_subtype    military | leadership_change | sanctions | treaty_ceasefire | emergency_policy | other
                 (only for category='geopolitical')
  is_scheduled_election  scheduled vote / election / primary / referendum outcome (resolved by public
                 scheduled data) -> NOT core in-scope, tagged separately so it can be included later
  in_scope       category='geopolitical' AND NOT is_scheduled_election
  macro_comparison  central-bank rate/policy decision markets (secondary comparison set). Scheduled
                 statistical releases (CPI, jobs, GDP prints) are category macro but not in this set.
  taxonomy_rule  the rule that fired, for auditing.

Rules are evaluated in order; the first match wins. Inputs: Polymarket tag labels (strongest signal),
the legacy Gamma `category` field (pre-2023 markets often have no tags), and keyword regexes over the
question + event title (descriptions are boilerplate-heavy and are not used).

Usage:
  python -m pmflow.data.taxonomy --apply          # (re)classify every row in pm_markets
  python -m pmflow.data.taxonomy --sample 150     # write a hand-check sample to artifacts/data/
"""
from __future__ import annotations

import argparse
import io
import re

VERSION = "v1"


def _rx(words: list[str]) -> re.Pattern:
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(words) + r")(?![a-z0-9])", re.I)


def _tags(words: list[str]) -> set[str]:
    return {w.lower() for w in words}


SPORTS_TAGS = _tags([
    "Sports", "Soccer", "Basketball", "NBA", "NFL", "NFL (All)", "CFB (All)", "Tennis", "Table Tennis", "Esports",
    "Golf", "Cricket", "Hockey", "NHL", "MLB", "baseball", "UFC", "Formula 1", "f1", "Boxing", "MMA",
    "NCAA", "NCAA Basketball", "football", "EPL", "Premier League", "La Liga", "Bundesliga", "Serie A",
    "Ligue 1", "Champions League", "UCL", "UEL", "MLS", "Chess", "Pickleball", "counter strike 2", "Dota 2",
    "league of legends", "lol", "Valorant", "Olympics", "WNBA", "Fantasy Football", "PGA", "LPGA", "Racing",
    "NASCAR", "Cycling", "Rugby", "Darts", "Snooker", "Games", "World Cup", "FIFA World Cup", "Super Bowl",
    "march madness", "Grand Prix", "Setka", "Wimbledon", "US Open", "props", "Horse Racing", "Volleyball",
])
CRYPTO_TAGS = _tags([
    "Crypto", "Crypto Prices", "Bitcoin", "Ethereum", "Solana", "XRP", "Ripple", "BNB", "Dogecoin", "Memecoins",
    "Up or Down", "hype", "Zcash", "Airdrops", "NFTs", "Stablecoins", "DeFi", "Pump.fun", "Hyperliquid",
])
ENT_TAGS = _tags([
    "Culture", "Pop Culture", "Awards", "Movies", "Music", "Tweet Markets", "Mentions", "Politics Mentions",
    "Celebrities", "TV", "Oscars", "Grammys", "Box Office", "YouTube", "podcast", "Rogan", "joe rogan",
    "Elon Tweets", "Twitter", "Video Games", "Gaming", "Reality TV", "Eurovision", "Spotify", "Taylor Swift",
    "Netflix", "Emmys", "Golden Globes", "Billboard", "Streamers", "MrBeast", "Kanye",
])
MACRO_TAGS = _tags([
    "Fed", "Fed Rates", "FOMC", "Jerome Powell", "Economy", "Inflation", "CPI", "US CPI", "CPI Release",
    "Macro Indicators", "Global Rates", "Interest Rates", "ECB", "Bank of England", "BoJ", "Bank of Japan",
    "Jobs Report", "Unemployment", "GDP", "Recession", "Treasuries", "Central Banks", "Rate Cuts",
])
ELECTION_TAGS = _tags([
    "Elections", "US Election", "Midterms", "Global Elections", "World Elections", "House Elections",
    "Senate Elections", "Governor Elections", "Primaries", "Nov 4 Elections", "Main Election", "election",
    "Election Night", "UK By-Elections", "German Elections", "Israel Election", "UK Elections", "Midterm MOV",
    "Margin of Victory", "Tossups", "Ballot Measures", "Turnout", "Control & Seats", "Down Ballot",
    "deprec USA Election", "International Election Props", "Polls", "Primary", "Mayoral Election",
    "Canadian Election", "Presidential Election", "Electoral College", "Swing States",
])
GEO_TAGS = _tags([
    "Geopolitics", "World", "Ukraine", "Russia", "Israel", "Iran", "Middle East", "Gaza", "China", "Taiwan",
    "NATO", "Military Actions", "Military", "Military Strikes", "Strait of Hormuz", "Hormuz", "U.S. x Iran",
    "Iran Ceasefire", "Iran Regime", "Israel x Iran", "Houthis", "Yemen", "Venezuela", "Peace Deal",
    "Foreign Policy", "Ukraine Map", "North Korea", "Syria", "Lebanon", "Hezbollah", "Hamas", "Sanctions",
    "Tariffs", "Trade War", "Ceasefire", "putin", "zelensky", "zelenskyy", "Kyiv", "Moscow", "Bab el-Mandeb",
    "somalia", "Pakistan", "India-Pakistan", "Cuba", "Greenland", "Panama", "Mexico", "Coup", "War",
])
WEATHER_TAGS = _tags(["Weather", "Daily Temperature", "Highest temperature", "Lowest temperature", "Hurricanes",
                      "Climate", "Earthquakes"])
FIN_TAGS = _tags(["Finance", "Equities", "Stocks", "Stock Prices", "Finance Updown", "Pyth Finance", "Hit Price",
                  "Commodities", "Gold", "Silver", "Oil", "forex", "FX", "Foreign Exchange", "Earnings",
                  "IPOs", "IPO", "Indicies", "Tech", "AI", "Big Tech", "Business", "Equity Daily Pyth", "DXY"])

LEGACY_CAT = {  # pre-2023 Gamma 'category' field
    "sports": "sports", "nba playoffs": "sports", "olympics": "sports", "chess": "sports", "nfts": "crypto",
    "crypto": "crypto", "pop-culture": "entertainment", "pop culture": "entertainment",
    "art": "entertainment", "coronavirus": "other", "science": "other", "tech": "other", "business": "other",
    "us-current-affairs": None, "global politics": None, "politics": None, "world affairs": None,
}

# ---- keyword rules (question + event title)
RX_MENTION = _rx([r"say", r"says", r"said", r"mention\w*", r"tweet\w*", r"post\w* on (?:x|truth social)",
                  r"truth social posts?", r"# of tweets"])
RX_SPORTS = _rx([r"vs\.?", r"fc", r"nba", r"nfl", r"mlb", r"nhl", r"ufc", r"premier league", r"champions league",
                 r"super bowl", r"world series", r"stanley cup", r"grand slam", r"playoffs?", r"touchdowns?",
                 r"world cup", r"olympic\w*", r"medals?", r"esports", r"match(?:es)?", r"tournament", r"goals? scored",
                 r"o/u", r"spread", r"mvp", r"heisman", r"f1", r"formula 1"])
RX_CRYPTO = _rx([r"bitcoin", r"btc", r"ethereum", r"eth", r"solana", r"sol", r"xrp", r"dogecoin", r"doge",
                 r"crypto\w*", r"memecoin", r"token", r"airdrop", r"fdv", r"market cap", r"coinbase", r"binance",
                 r"stablecoin", r"nft\w*", r"up or down"])
RX_ENT = _rx([r"oscars?", r"grammys?", r"emmys?", r"box office", r"album", r"movie", r"film", r"song", r"billboard",
              r"spotify", r"netflix", r"youtube", r"tiktok", r"celebrity", r"taylor swift", r"kanye", r"drake",
              r"mrbeast", r"rotten tomatoes", r"eurovision", r"bachelor\w*", r"survivor", r"gta ?vi?"])
RX_MACRO_DECISION = _rx([
    r"fed", r"fomc", r"federal reserve", r"interest rates?", r"rate (?:cut|hike|decision)s?", r"bps",
    r"basis points?", r"ecb", r"bank of england", r"boe", r"bank of japan", r"boj", r"rba", r"bank of canada",
    r"pboc", r"snb", r"rbi", r"central bank", r"powell", r"lagarde", r"quantitative (?:easing|tightening)",
    r"emergency (?:rate )?cut", r"fed chair"])
RX_MACRO_DATA = _rx([r"cpi", r"inflation", r"gdp", r"unemployment", r"jobs report", r"nonfarm", r"payrolls?",
                     r"recession", r"pce", r"jobless claims", r"treasury yields?", r"10-year", r"ism", r"retail sales"])
RX_ELECTION = _rx([
    r"elections?", r"elected", r"primary", r"primaries", r"caucus\w*", r"referendum", r"ballot", r"electoral",
    r"popular vote", r"win (?:the )?(?:\w+ ){0,4}(?:seat|district|race|governorship|mayoralty|nomination)",
    r"nominee", r"nomination", r"runoff", r"vote share", r"turnout", r"poll\w*", r"margin of victory",
    r"swing state", r"midterms?", r"senate race", r"house race", r"governor race", r"(?:win|wins) (?:the )?presidency",
    r"next (?:us )?president", r"presidential (?:race|winner)", r"by-election", r"snap election", r"seats?"])
RX_DOMESTIC = _rx([
    r"approval rating", r"bill", r"act", r"senate (?:vote|pass\w*|confirm\w*)", r"house (?:vote|pass\w*)",
    r"confirm\w*", r"supreme court", r"scotus", r"impeach\w*", r"indict\w*", r"convicted", r"sentenced",
    r"government shutdown", r"shutdown", r"filibuster", r"speaker", r"cabinet", r"governor", r"mayor",
    r"democrat\w*", r"republican\w*", r"gop", r"dnc", r"rnc", r"congress\w*", r"trump", r"biden", r"harris",
    r"desantis", r"newsom", r"musk", r"doge", r"veto", r"lawsuit", r"court"])

GEO_MILITARY = _rx([
    r"invade\w*", r"invasion", r"military", r"troops", r"airstrikes?", r"air strikes?", r"strikes?", r"struck",
    r"missiles?", r"attacks?", r"attacked", r"war", r"warfare", r"bomb\w*", r"drones?", r"nuclear (?:test|weapon|strike|bomb)",
    r"nuke", r"deploy\w*", r"blockade", r"occupy", r"occupation", r"capture[sd]?", r"offensive", r"ground operation",
    r"military operation", r"special operation", r"shoot down", r"shot down", r"naval", r"navy", r"warship",
    r"hostilities", r"incursion", r"martial", r"annex\w*", r"seize[sd]?", r"enter (?:\w+ ){0,2}(?:city|territory)",
    r"clash\w*", r"conflict", r"assassinat\w*", r"killed", r"kill", r"target\w*", r"military action",
    r"war powers", r"(?:missile|nuclear|weapons?|icbm) tests?", r"tests? (?:a |an )?(?:missile|icbm|nuke|nuclear)",
    r"launch\w* (?:\w+ ){0,3}(?:strikes?|attacks?|missiles?|offensive|invasion|operation)", r"shelling", r"casualt\w*", r"retake[sn]?", r"liberat\w*"])
GEO_LEADER = _rx([
    r"resign\w*", r"out as", r"ousted", r"oust", r"removed", r"step(?:s)? down", r"coup", r"regime (?:change|fall\w*|collapse)",
    r"leave office", r"left office", r"no longer", r"successor", r"replaced", r"next (?:prime minister|pm|leader|supreme leader|president of)",
    r"supreme leader", r"fired", r"exile", r"flee\w*", r"detained", r"arrested", r"overthrow\w*", r"toppled",
    r"dies?", r"death", r"vote of no confidence", r"no-confidence", r"leadership", r"out by", r"out before",
    r"out in \d{4}", r"leader out", r"next (?:united states |us |u\.s\. )?(?:attorney general|secretary|ambassador|chair\w*|director|head|chancellor|premier)",
    r"(?:no longer|remain\w*) (?:in|under) (?:\w+ ){0,2}control", r"in power"])
GEO_SANCTIONS = _rx([r"sanction\w*", r"tariffs?", r"embargo\w*", r"export (?:control|ban)s?", r"trade war",
                     r"trade deal", r"ban(?:s|ned)? (?:\w+ ){0,3}(?:imports?|exports?)", r"price cap", r"swift",
                     r"freeze (?:\w+ )?assets", r"blacklist\w*", r"entity list"])
GEO_TREATY = _rx([r"cease-?fire", r"truce", r"peace (?:deal|talks|agreement|treaty|plan|summit)", r"treaty", r"accords?",
                  r"armistice", r"hostages?", r"prisoner (?:swap|exchange)", r"normali[sz]\w*", r"recogni[sz]e\w*",
                  r"diplomatic relations", r"agreement", r"deal with", r"negotiat\w*", r"summit", r"meet(?:s|ing)? with",
                  r"join nato", r"nato", r"un resolution", r"security council", r"visit\w*", r"call with",
                  r"talks", r"diplomatic", r"article 4", r"article 5"])
GEO_EMERGENCY = _rx([r"martial law", r"state of emergency", r"national emergency", r"emergency (?:powers|declaration|order)",
                     r"executive order", r"insurrection act", r"lockdown", r"curfew", r"pardon\w*", r"invoke\w*",
                     r"declare\w*", r"nationali[sz]\w*", r"default on", r"capital controls", r"travel ban",
                     r"close (?:the )?(?:border|strait|embassy)", r"expel\w*", r"evacuat\w*", r"withdraw\w*",
                     r"announce\w*", r"designate\w*", r"terrorist organization"])
GEO_ACTORS = _rx([
    r"ukraine", r"ukrainian", r"russia\w*", r"putin", r"zelensk\w*", r"kyiv", r"moscow", r"crimea", r"donbas\w*",
    r"israel\w*", r"gaza", r"hamas", r"hezbollah", r"netanyahu", r"idf", r"iran\w*", r"khamenei", r"tehran", r"houthis?",
    r"yemen", r"syria\w*", r"assad", r"lebanon", r"china", r"chinese", r"xi jinping", r"xi", r"taiwan", r"north korea",
    r"kim jong[ -]un", r"pyongyang", r"venezuela\w*", r"maduro", r"nato", r"united nations", r"un", r"pentagon",
    r"cuba", r"greenland", r"panama canal", r"pakistan", r"india", r"afghanistan", r"taliban", r"iraq", r"saudi\w*",
    r"qatar", r"egypt", r"turkey", r"erdogan", r"sudan", r"somalia", r"ethiopia", r"armenia", r"azerbaijan",
    r"serbia", r"kosovo", r"belarus", r"lukashenko", r"poland", r"baltic", r"strait of hormuz", r"hormuz",
    r"red sea", r"south china sea", r"philippines", r"japan", r"south korea", r"mexico", r"cartels?",
    r"colombia", r"brazil", r"argentina", r"milei", r"france", r"macron", r"germany", r"uk", r"britain",
    r"starmer", r"canada", r"trudeau", r"european union", r"eu", r"modi", r"bangladesh", r"myanmar",
    r"haiti", r"nigeria", r"niger", r"mali", r"libya", r"palestin\w*", r"west bank", r"opec", r"bolivia",
    r"peru", r"chile", r"ecuador", r"nicaragua", r"georgia's", r"moldova", r"romania", r"hungary", r"orban",
    r"south africa", r"congo", r"rwanda", r"thailand", r"cambodia", r"vietnam", r"indonesia", r"nepal",
    r"sri lanka", r"kenya", r"zimbabwe", r"canadian", r"german", r"french", r"british", r"japanese", r"korean",
    r"mexican", r"brazilian", r"polish", r"dutch", r"netherlands", r"spain", r"spanish", r"italy", r"italian",
    r"australia\w*", r"portug\w*", r"austria\w*", r"czech", r"greece", r"greek", r"irish", r"ireland",
    r"scotland", r"scottish", r"sweden", r"swedish", r"norway", r"denmark", r"finland", r"foreign", r"embassy", r"ambassador", r"prime minister", r"pm",
    r"head of state", r"regime", r"supreme leader", r"king", r"pope", r"vatican"])
# markets that carry geo tags but resolve on prices, rankings or prizes -> not geopolitical events
PRICE_TAGS = _tags(["Hit Price", "Finance Updown", "Pyth Finance", "Stock Prices", "Equities", "Daily-Close",
                    "Equity Daily Pyth", "AI Rankings", "Commodities"])
RX_NON_EVENT = _rx([r"nobel", r"(?:best|top|third-best|second-best) (?:\w+ ){0,3}(?:ai|model|company)",
                    r"(?:wti|brent|crude oil|gold|silver) (?:\(\w+\) )?(?:hit|above|below|close)",
                    r"hit \((?:high|low)\)", r"time person of the year", r"word of the year"])
RX_US_LEADER = _rx([r"resign\w*", r"out as", r"step(?:s)? down", r"fired", r"fires", r"leave\w*", r"next to leave",
                    r"removed", r"ousted", r"replaced", r"departs?", r"exit\w*",
                    r"next (?:united states |us |u\.s\. )?(?:attorney general|secretary|ambassador|chair\w*|director|head)",
                    r"(?:announce|nominate)\w* (?:\w+ ){1,4}as (?:the )?next"])
RX_ELECTION_STRICT = _rx([r"elections?", r"elected", r"primary", r"primaries", r"caucus\w*", r"referendum", r"ballot",
                          r"runoff", r"vote share", r"seats?", r"nominee", r"nomination", r"polls?", r"popular vote",
                          r"electoral", r"most votes", r"win (?:the )?(?:\w+ ){0,4}(?:seat|district|race|state)"])
RX_NOVELTY = _rx([r"miss universe", r"miss world", r"eurovision", r"shake hands", r"handshake", r"state dinner",
                  r"wear\w*", r"outfit", r"hold hands"])
RX_US_EXEC = _rx([r"trump", r"white house", r"president", r"administration", r"secretary of state", r"rubio",
                  r"hegseth", r"vance", r"biden"])


def classify(question: str | None, event_title: str | None = None, tags: list[str] | None = None,
             gamma_category: str | None = None) -> dict:
    q = " ".join(x for x in [question or "", event_title or ""] if x)
    tl = {t.lower() for t in (tags or []) if t}
    res = {"category": "other", "geo_subtype": None, "is_scheduled_election": False, "in_scope": False,
           "macro_comparison": False, "taxonomy_rule": "default_other", "taxonomy_version": VERSION}

    def done(cat: str, rule: str, **kw) -> dict:
        res.update(category=cat, taxonomy_rule=rule, **kw)
        if cat == "geopolitical" and not res["is_scheduled_election"]:
            res["in_scope"] = True
        return res

    legacy = LEGACY_CAT.get((gamma_category or "").strip().lower(), "unknown")
    # 1. non-political verticals, by tag first (strong), then legacy category
    if tl & SPORTS_TAGS:
        return done("sports", "tag:sports")
    if tl & CRYPTO_TAGS and not (tl & GEO_TAGS):
        return done("crypto", "tag:crypto")
    if tl & WEATHER_TAGS:
        return done("other", "tag:weather")
    if RX_MENTION.search(q) or tl & {"mentions", "politics mentions", "tweet markets"}:
        return done("entertainment", "kw:speech_mention_market")
    if tl & ENT_TAGS and not (tl & GEO_TAGS):
        return done("entertainment", "tag:entertainment")
    if legacy not in ("unknown", None) and not tl:
        return done(legacy, f"legacy_category:{gamma_category}")

    if tl & PRICE_TAGS or RX_NON_EVENT.search(q):
        return done("other", "tag_or_kw:price_ranking_prize")

    # 2. central bank / macro
    if RX_MACRO_DECISION.search(q) and not GEO_MILITARY.search(q):
        return done("macro_central_bank", "kw:central_bank_decision", macro_comparison=True)
    if tl & MACRO_TAGS or RX_MACRO_DATA.search(q):
        if RX_MACRO_DECISION.search(q) or tl & _tags(["Fed", "Fed Rates", "FOMC", "Jerome Powell", "Global Rates"]):
            return done("macro_central_bank", "tag:central_bank", macro_comparison=True)
        return done("macro_central_bank", "tag_or_kw:macro_data_release")

    # 3. scheduled elections (tagged separately; not core in-scope)
    election = bool(tl & ELECTION_TAGS) or bool(RX_ELECTION.search(q))
    geo_tagged = bool(tl & GEO_TAGS)
    geo_actor = bool(GEO_ACTORS.search(q))

    # 4. geopolitical / government decision sub-types
    sub = None
    for name, rx in (("military", GEO_MILITARY), ("treaty_ceasefire", GEO_TREATY), ("sanctions", GEO_SANCTIONS),
                     ("leadership_change", GEO_LEADER), ("emergency_policy", GEO_EMERGENCY)):
        if rx.search(q):
            sub = name
            break
    if RX_NOVELTY.search(q):
        return done("entertainment", "kw:novelty")
    if election and sub == "leadership_change" and not RX_ELECTION_STRICT.search(q):
        # "next PM / next leader out / successor" markets resolve on unscheduled leadership change
        election = False
    if election:
        res["is_scheduled_election"] = True
        if geo_tagged or geo_actor:
            # foreign / national elections: geopolitical category but excluded from core scope
            return done("geopolitical", "election:geo", geo_subtype="leadership_change")
        return done("elections_domestic_politics", "election:domestic")
    if sub and (geo_tagged or geo_actor):
        return done("geopolitical", f"kw:{sub}+{'tag' if geo_tagged else 'actor'}", geo_subtype=sub)
    if sub == "leadership_change" and RX_US_EXEC.search(q) and not RX_US_LEADER.search(q):
        sub = None   # e.g. arrests/indictments of US politicians are domestic legal events, not gov decisions
    if not sub and RX_US_EXEC.search(q) and RX_US_LEADER.search(q):
        sub = "leadership_change"
    if sub in ("sanctions", "emergency_policy", "military", "leadership_change") and RX_US_EXEC.search(q):
        # US executive actions (tariffs, executive orders, pardons, strikes ordered by the President)
        return done("geopolitical", f"kw:{sub}+us_exec", geo_subtype=sub)
    if geo_tagged:
        return done("geopolitical", "tag:geo_no_subtype", geo_subtype="other")
    if tl & FIN_TAGS:
        return done("other", "tag:finance_business")
    if tl & ELECTION_TAGS or "politics" in tl or RX_DOMESTIC.search(q) or legacy is None:
        return done("elections_domestic_politics", "tag_or_kw:domestic_politics")
    if RX_SPORTS.search(q):
        return done("sports", "kw:sports")
    if RX_CRYPTO.search(q):
        return done("crypto", "kw:crypto")
    if RX_ENT.search(q):
        return done("entertainment", "kw:entertainment")
    return res


# ---------------------------------------------------------------- DB application

def apply_all(batch: int = 50000) -> None:
    from .common import connect
    with connect() as conn:
        conn.execute("CREATE TEMP TABLE _tax (market_id bigint PRIMARY KEY, category text, geo_subtype text, "
                     "is_scheduled_election boolean, in_scope boolean, macro_comparison boolean, "
                     "taxonomy_rule text, taxonomy_version text)")
        last = -1
        total = 0
        while True:
            rows = conn.execute(
                "SELECT market_id, question, event_title, tag_labels, gamma_category FROM pm_markets "
                "WHERE market_id > %s ORDER BY market_id LIMIT %s", (last, batch)).fetchall()
            if not rows:
                break
            buf = io.StringIO()
            for mid, q, et, tags, gc in rows:
                r = classify(q, et, tags, gc)
                vals = [mid, r["category"], r["geo_subtype"], r["is_scheduled_election"], r["in_scope"],
                        r["macro_comparison"], r["taxonomy_rule"], r["taxonomy_version"]]
                buf.write("\t".join("\\N" if v is None else ("t" if v is True else "f" if v is False else str(v))
                                    for v in vals) + "\n")
            with conn.cursor() as cur:
                with cur.copy("COPY _tax FROM STDIN") as cp:
                    cp.write(buf.getvalue())
            last = rows[-1][0]
            total += len(rows)
        conn.execute("""UPDATE pm_markets m SET category=t.category, geo_subtype=t.geo_subtype,
                        is_scheduled_election=t.is_scheduled_election, in_scope=t.in_scope,
                        macro_comparison=t.macro_comparison, taxonomy_rule=t.taxonomy_rule,
                        taxonomy_version=t.taxonomy_version
                        FROM _tax t WHERE t.market_id=m.market_id
                          AND (m.taxonomy_rule IS DISTINCT FROM t.taxonomy_rule OR m.in_scope IS DISTINCT FROM t.in_scope
                               OR m.category IS DISTINCT FROM t.category OR m.geo_subtype IS DISTINCT FROM t.geo_subtype
                               OR m.taxonomy_version IS DISTINCT FROM t.taxonomy_version)""")
        print(f"classified {total} markets")


def write_sample(n: int, seed: float = 0.42) -> None:
    import pandas as pd
    from .common import ARTIFACTS, connect
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.execute("SELECT setseed(%s)", (seed,))
        q = """SELECT market_id, question, event_title, array_to_string(tag_labels,'|') tags, category, geo_subtype,
                      is_scheduled_election, in_scope, taxonomy_rule
               FROM pm_markets WHERE closed AND in_scope = %s ORDER BY random() LIMIT %s"""
        a = pd.DataFrame(conn.execute(q, (True, n)).fetchall(), columns=[d for d in
                         ["market_id", "question", "event_title", "tags", "category", "geo_subtype",
                          "is_scheduled_election", "in_scope", "taxonomy_rule"]])
        b = pd.DataFrame(conn.execute(q, (False, n)).fetchall(), columns=a.columns)
    df = pd.concat([a, b])
    df["hand_label_in_scope"] = ""
    df["hand_note"] = ""
    df.to_csv(ARTIFACTS / "taxonomy_handcheck_sample.csv", index=False)
    print(f"wrote {len(df)} rows to {ARTIFACTS / 'taxonomy_handcheck_sample.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()
    if a.apply:
        apply_all()
    if a.sample:
        write_sample(a.sample)
