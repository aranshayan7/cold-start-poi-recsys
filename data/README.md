# Data

## Format

All code reads four files from the directory given by `--data`.

| File | Columns / structure |
|---|---|
| `checkins.parquet` | `user_idx` (int), `venue_idx` (int), `utc_time` (string, Foursquare format `%a %b %d %H:%M:%S %z %Y`) |
| `pois.parquet` | `venue_idx`, `latitude`, `longitude`, `lat_norm`, `lon_norm` (z-scored coordinates), `category_idx` (int) |
| `splits.pkl` | dict with lists `train_users`, `val_users`, `test_users` (user-disjoint) |
| `social_adj.pkl` | dict `user_idx -> list of friend user_idx` |

User and POI indices must be dense integers starting at zero. A user takes part
in the experiments only if the check-in history contains at least
`k_max + q` (default 30) distinct POIs.

## Bundled sample

`data/sample/` contains 2,100 users (1,500 train, 300 validation, 300 test), the
first 120 check-ins of each user and the friendships among the selected users.
It is a random subset of the Japan data, re-indexed so that it works on its own.
Use it to run the pipeline end to end and for the unit tests.

Regenerate it from the full processed data with:

```bash
python scripts/make_sample.py --src /path/to/processed_japan --dst data/sample
```

## Full data

The experiments use the public Foursquare check-in dataset of Yang et al. (WWW 2019,
"Revisiting User Mobility and Social Relationships in LBSNs: A Hypergraph Embedding
Approach"), available from the authors' dataset page. Download these three files into
`data/raw/`:

| File | Content |
|---|---|
| `dataset_WWW_Checkins_anonymized.txt` | 22.8 M check-ins: user, venue, UTC time, timezone offset |
| `raw_POIs.txt` | all venues: venue, latitude, longitude, category, country code |
| `dataset_WWW_friendship_new.txt` | friendships (second snapshot) |

Build the processed files (Japan by default):

```bash
python scripts/prepare_data.py --raw data/raw --out data/processed
python scripts/prepare_data.py --raw data/raw --out data/processed_asia --countries JP,KR,SG,TW,HK,CN
```

Defaults: venues need at least 5 check-ins, users between 20 and 1,000 check-ins,
the 50 most frequent categories are kept (the rest become `Other`), coordinates are
z-scored, and users are split 60/20/20 stratified by activity tercile with seed 42.
Rows with a corrupted timestamp (a few exist in the raw file) are dropped.

### Agreement with the files used in the thesis

The preparation was checked against the thesis files (`scripts/compare_processed.py`)
on the Japan subset:

| Component | Result |
|---|---|
| Users | identical set (8,628), identical indices |
| POIs | 76,160 shared; 24 only in the new build and 14 only in the thesis files, all venues with exactly 4 to 5 check-ins at the filtering boundary |
| Check-ins | 1,714,241 versus 1,714,169 (+72, 0.004%) |
| Categories, coordinates | identical for all shared POIs |
| Friendships | 59,743 versus 59,744 edges |
| Train/val/test | 5,176/1,725/1,727 versus 5,175/1,726/1,727, but the membership overlaps only partly (Jaccard 0.87, 0.77, 0.72) |

The split differs because the stratification uses per-user check-in counts, which
differ slightly for a few hundred users, and the random shuffle then assigns users
differently. Results on a freshly prepared split are therefore statistically
equivalent to, but not numerically identical with, `reference_results/`. To compare
against the reference numbers exactly, use the thesis processed files.

Please cite the original dataset when using the data.
