import pandas as pd

from pmflow.evaluation import topics
from pmflow.evaluation.final_score import holm


def test_template_strips_dates_and_numbers():
    assert topics.template("Israel military action against Lebanon on November 12?") == \
        topics.template("Israel military action against Lebanon on July 3?")
    assert topics.template("How many UKMTO notices October 19-25?") == "how many ukmto notices"


def test_assign_topics_and_theatres():
    m = pd.DataFrame(dict(market_id=[1, 2, 3, 4], universe=["in_scope", "in_scope", "macro", "in_scope"],
                          event_title=["Houthis successfully target shipping on...?", "Will Russia capture Lyman by...?",
                                       "Fed Decision in October?", "DeepSeek V4 released by...?"],
                          question=["", "", "Fed holds?", ""]))
    t = topics.assign(m).set_index("market_id")
    assert t.loc[1, "topic"] == "yemen_redsea" and t.loc[1, "theatre"] == "middle_east"
    assert t.loc[2, "topic"] == "russia_ukraine"
    assert t.loc[3, "topic"] == "macro_fed" and t.loc[3, "theatre"] == "macro_us_rates"
    assert t.loc[4, "topic"] == "other"


def test_holm_monotone_and_capped():
    h = holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert abs(h["a"] - 0.03) < 1e-12 and abs(h["b"] - 0.08) < 1e-12 and h["c"] == 0.5
    assert holm({"a": 0.6, "b": 0.7})["a"] == 1.0
