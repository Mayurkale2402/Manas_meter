"""Input rules for Manas Meter, kept separate from FastAPI so they can be tested on their own."""

TOP_COUNTRIES = ['Other', 'India', 'USA', 'Canada', 'Australia', 'UK', 'Germany', 'Mexico', 'Turkey', 'France']

_ALIASES = {
    'USA': {'us', 'usa', 'u s', 'u s a', 'united states', 'united states of america', 'america'},
    'UK': {'uk', 'u k', 'united kingdom', 'great britain', 'britain', 'england', 'scotland', 'wales'},
    'India': {'bharat'},
    'Germany': {'deutschland'},
    'Turkey': {'turkiye', 'türkiye'},
    'Mexico': {'méxico'},
}
_LOOKUP = {c.casefold(): c for c in TOP_COUNTRIES}
for _canon, _names in _ALIASES.items():
    for _n in _names:
        _LOOKUP[_n] = _canon


def normalize_country(raw: str) -> str:
    """Map what the user typed ('india ', 'United States') to a group the model knows, else 'Other'."""
    key = ' '.join(raw.replace('.', ' ').casefold().split())
    return _LOOKUP.get(key, 'Other')


# Ranges the model was trained on (from the dataset). Outside them, a random forest just
# reuses its nearest known value, so we tell the user instead of pretending it is precise.
TRAINED_RANGES = {
    'age': ('Age', 18, 24),
    'avg_daily_usage_hours': ('Daily screen time', 1, 8.8),
    'daily_unlocks': ('Phone unlocks per day', 62, 273),
    'study_hours': ('Study time', 0.3, 8.3),
    'physical_activity_hours': ('Exercise time', 0, 4.1),
    'sleep_hours_per_night': ('Sleep', 3.6, 9.9),
}


def range_notes(values: dict) -> list:
    notes = []
    for key, (label, lo, hi) in TRAINED_RANGES.items():
        v = values[key]
        if v < lo or v > hi:
            nearest = lo if v < lo else hi
            notes.append(
                f"{label} ({v:g}) is outside the {lo:g}–{hi:g} range this model learned from, "
                f"so it is treated like {nearest:g}."
            )
    return notes


def day_error(sleep: float, study: float, activity: float):
    """Sleep, study and exercise cannot overlap, so together they must fit in a day."""
    total = sleep + study + activity
    if total > 24:
        return (f"Sleep, study and exercise add up to {total:g} hours, which is more than a day. "
                "Please check those three answers.")
    return None
