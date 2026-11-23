"""
Fetch raw COVID-19 case data (JHU CSSE), country/continent reference data,
and World Bank socioeconomic indicators, then build the processed datasets
that the Streamlit dashboard (app.py) reads from data/processed/.

Run once locally to refresh data:

    python build_data.py
"""
import io
from functools import reduce

import numpy as np
import pandas as pd
import requests

JHU_BASE = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series"
)
ISO_URL = (
    "https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-"
    "Regional-Codes/master/all/all.csv"
)
WB_URL = "https://api.worldbank.org/v2/country/all/indicator/{indicator}"

OUT = "data/processed"

BOATS = ["Diamond Princess", "MS Zaandam"]
NON_COUNTRIES = BOATS + ["Summer Olympics 2020", "Winter Olympics 2022"]

# JHU uses simplified/common country names. Align them to the RENAME dict below.
RENAME = {
    "Taiwan*": "Taiwan",
    "Korea, South": "Korea",
    "North Macedonia": "Macedonia",
    "Cabo Verde": "Cape Verde",
    "Congo (Brazzaville)": "Congo",
    "Congo (Kinshasa)": "Congo",
}

# ISO-3166 official names -> JHU country names, so the two datasets join cleanly.
ISO_TO_JHU = {
    "Bolivia, Plurinational State of": "Bolivia",
    "Brunei Darussalam": "Brunei",
    "Myanmar": "Burma",
    "Congo": "Congo (Brazzaville)",
    "Congo, Democratic Republic of the": "Congo (Kinshasa)",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "Iran, Islamic Republic of": "Iran",
    "Korea, Democratic People's Republic of": "Korea, North",
    "Korea, Republic of": "Korea, South",
    "Lao People's Democratic Republic": "Laos",
    "Micronesia, Federated States of": "Micronesia",
    "Moldova, Republic of": "Moldova",
    "Netherlands, Kingdom of the": "Netherlands",
    "Russian Federation": "Russia",
    "Syrian Arab Republic": "Syria",
    "Taiwan, Province of China": "Taiwan*",
    "Tanzania, United Republic of": "Tanzania",
    "Türkiye": "Turkey",
    "United States of America": "US",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "Venezuela, Bolivarian Republic of": "Venezuela",
    "Viet Nam": "Vietnam",
    "Palestine, State of": "West Bank and Gaza",
}

WB_INDICATORS = {
    "SP.DYN.LE00.IN": "Life expectancy",
    "NY.GDP.PCAP.PP.CD": "GDP per capita",
    "SP.RUR.TOTL.ZS": "Rural population %",
    "SP.POP.TOTL": "Population",
    "SH.XPD.CHEX.GD.ZS": "GDP Healthcare %",
}


def fetch_csv(url, **kwargs):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), **kwargs)


def load_jhu(kind):
    return fetch_csv(f"{JHU_BASE}/time_series_covid19_{kind}_global.csv")


def process_cases(df):
    df = df.drop(columns=["Lat", "Long", "Province/State"])
    df = df.rename(columns={"Country/Region": "Country"})
    df["Country"] = df["Country"].replace(RENAME)
    df = df.groupby("Country", as_index=False).sum()
    df = df[~df["Country"].isin(NON_COUNTRIES)]
    df = df.set_index("Country").T
    df.index.name = "Date"
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    cols = [c for c in df.columns if c != "Date"]
    df[cols] = df[cols].astype(int)
    df = df[["Date"] + sorted(cols)]
    return df


def make_cases(conf_raw, recov_raw, dead_raw):
    conf = process_cases(conf_raw)
    recov = process_cases(recov_raw)
    dead = process_cases(dead_raw)

    common = sorted(set(conf.columns) & set(recov.columns) & set(dead.columns) - {"Date"})
    conf = conf[["Date"] + common]
    recov = recov[["Date"] + common]
    dead = dead[["Date"] + common]

    active = conf.drop(columns=["Date"]) - recov.drop(columns=["Date"]) - dead.drop(columns=["Date"])
    active["Date"] = conf["Date"]
    active = active[["Date"] + common]

    return conf, recov, dead, active


def make_coordinates(conf_raw):
    df = conf_raw.rename(columns={"Country/Region": "Country"})
    df = df.loc[~df["Country"].isin(NON_COUNTRIES)]
    df["Country"] = df["Country"].replace(RENAME)
    df = df[["Country", "Lat", "Long"]].groupby("Country", as_index=False).mean()
    return df.sort_values("Country").reset_index(drop=True)


def make_continents(iso):
    df = iso[["name", "alpha-3", "region"]].copy()
    df["name"] = df["name"].replace(ISO_TO_JHU)
    df["name"] = df["name"].replace(RENAME)
    df = df.rename(columns={"name": "Country", "alpha-3": "Country Code", "region": "Continent"})
    # Taiwan has no UN region assigned upstream.
    df.loc[df["Country"] == "Taiwan", "Continent"] = "Asia"
    # Kosovo is not part of ISO-3166 but reports COVID data separately.
    df = pd.concat(
        [df, pd.DataFrame([{"Country": "Kosovo", "Country Code": "XKX", "Continent": "Europe"}])],
        ignore_index=True,
    )
    df = df.dropna(subset=["Continent"])
    df = df.drop_duplicates(subset=["Country"])
    return df


def get_country_ts(country, dataframes, columns):
    cols = ["Date"] + columns
    parts = [df.loc[:, ["Date", country]] for df in dataframes]
    merged = reduce(lambda x, y: pd.merge(x, y, on="Date", how="outer"), parts)
    merged.columns = cols
    merged = merged.bfill()
    return merged


def make_mortality(conf, dead):
    countries = sorted(set(conf.columns) - {"Date"})
    parts = []
    for c in countries:
        df = get_country_ts(c, [conf, dead], ["Confirmed", "Deaths"])
        df = df[df["Confirmed"] > 0].copy()
        df["Mortality"] = np.round(df["Deaths"] / df["Confirmed"] * 100, 2)
        df = df[["Date", "Mortality"]].rename(columns={"Mortality": c})
        parts.append(df)
    return reduce(lambda x, y: pd.merge(x, y, on="Date"), parts)


def make_cases_since_t0(conf, n_cases_start=100):
    countries = sorted(set(conf.columns) - {"Date"})
    parts = []
    for c in countries:
        t0 = conf[[c]]
        t0 = t0.loc[t0[c] >= n_cases_start].reset_index(drop=True)
        parts.append(t0)
    merged = reduce(
        lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how="outer"),
        parts,
    )
    merged.columns = countries
    return merged.head(n_cases_start)


def make_cases_daily_change(conf):
    diff = conf.drop(columns=["Date"]).diff().fillna(0)
    diff["Date"] = conf["Date"]
    return diff[["Date"] + [c for c in diff.columns if c != "Date"]]


def make_country_stats(conf, recov, dead, active, mortality):
    stats = []
    for df, name in zip([conf, recov, dead, active, mortality], ["Confirmed", "Recovered", "Dead", "Active", "Mortality"]):
        tmp = df.tail(1).drop(columns=["Date"]).T.reset_index()
        tmp.columns = ["Country", name]
        stats.append(tmp)
    return reduce(lambda x, y: pd.merge(x, y, on="Country", how="outer"), stats)


def fetch_world_bank_indicator(indicator):
    r = requests.get(
        WB_URL.format(indicator=indicator),
        params={"format": "json", "per_page": 20000, "mrnev": 1},
        timeout=60,
    )
    r.raise_for_status()
    _, rows = r.json()
    return {row["countryiso3code"]: row["value"] for row in rows if row["value"] is not None}


def make_world_bank(continents, country_stats):
    wb = continents[["Country", "Country Code", "Continent"]].copy()
    for indicator, name in WB_INDICATORS.items():
        values = fetch_world_bank_indicator(indicator)
        wb[name] = wb["Country Code"].map(values)

    wb = pd.merge(wb, country_stats, on="Country", how="inner")
    wb = wb[wb["Confirmed"] > 5000]
    wb = wb[wb["Country"] != "Yemen"]

    for col in ["Life expectancy", "GDP Healthcare %", "GDP per capita"]:
        wb[col] = wb[col].fillna(wb[col].median())

    wb["Cases per mln"] = wb["Confirmed"] / wb["Population"] * 1_000_000
    wb["Dead per mln"] = wb["Dead"] / wb["Population"] * 1_000_000
    wb = wb.rename(columns={"Mortality": "Mortality %"})

    wb = wb.drop(columns=["Confirmed", "Active", "Dead", "Recovered", "Country Code"])
    return wb[sorted(wb.columns)]


def main():
    import os

    os.makedirs(OUT, exist_ok=True)

    print("Downloading JHU CSSE case data...")
    conf_raw, recov_raw, dead_raw = load_jhu("confirmed"), load_jhu("recovered"), load_jhu("deaths")

    print("Processing case data...")
    conf, recov, dead, active = make_cases(conf_raw, recov_raw, dead_raw)
    conf.to_csv(f"{OUT}/confirmed_cases.csv", index=False)
    recov.to_csv(f"{OUT}/recovered_cases.csv", index=False)
    dead.to_csv(f"{OUT}/dead_cases.csv", index=False)
    active.to_csv(f"{OUT}/active_cases.csv", index=False)

    print("Computing coordinates...")
    make_coordinates(conf_raw).to_csv(f"{OUT}/coordinates.csv", index=False)

    print("Downloading country/continent reference data...")
    iso = fetch_csv(ISO_URL)
    continents = make_continents(iso)
    continents.to_csv(f"{OUT}/continents.csv", index=False)

    print("Computing derived case datasets...")
    make_cases_since_t0(conf).to_csv(f"{OUT}/confirmed_cases_since_t0.csv", index=False)
    make_cases_daily_change(conf).to_csv(f"{OUT}/confirmed_cases_daily_change.csv", index=False)
    mortality = make_mortality(conf, dead)
    mortality.to_csv(f"{OUT}/mortality_rate.csv", index=False)

    country_stats = make_country_stats(conf, recov, dead, active, mortality)
    country_stats.to_csv(f"{OUT}/country_stats.csv", index=False)

    print("Downloading World Bank indicators...")
    world_bank = make_world_bank(continents, country_stats)
    world_bank.to_csv(f"{OUT}/world_bank.csv", index=False)

    print("Done. Processed CSVs written to", OUT)


if __name__ == "__main__":
    main()
