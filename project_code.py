# NAME: Matthew Jodoin-Siegel
"""
NBA shot make probability model.

Predicts the probability that each shot in testing.csv.gz is made, using
player-tracking features from training.csv.gz, and writes the predictions
to the make_prob column of submission.csv.

Pipeline
    1. Load and explore the data
    2. Clean the data and engineer features
    3. Constant baseline (leave-one-season-out validation)
    4. Logistic regression baseline
    5. LightGBM model with early stopping
    6. Shooter / defender / team IDs as "make rate above expected"
    7. Calibration plot and SHAP feature importance (saved to figures/)
    8. Retrain on all data and write submission.csv

Run from the repo root:  python project_code.py
"""

from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")  # draw straight to image files, no window
import matplotlib.pyplot as plt

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRAIN_PATH = "training.csv.gz"
TEST_PATH = "testing.csv.gz"
SUBMISSION_PATH = "submission.csv"
FIGURE_DIR = Path("figures")

# Court geometry (feet). The rim center sits at x = -41.75, y = 0 and the
# baseline is at x = -47. The corner three line runs straight for 14 ft out
# from the baseline, so corner threes are taken at x <= -33.
RIM_X = -41.75
CORNER_THREE_MAX_X = -47 + 14

# Contester distances are missing when there are fewer contesters than the
# column number (e.g. distcont2 on a one-contester shot). We fill them with a
# distance far beyond a real contest so "no contester" reads as "very open".
# 99.9% of recorded contests are within 13 ft; only 4 of ~450k exceed 30 ft.
NO_CONTESTER_DIST = 30.0


# ---------------------------------------------------------------------------
# Step 1: Load and explore
# ---------------------------------------------------------------------------

def load_data(path: str) -> pl.DataFrame:
    """Read a gzipped shot CSV.

    Two read options matter here:
      - null_values="NA": missing values are written as the text "NA". Without
        this, polars reads distcont1-4 as text columns instead of numbers.
      - infer_schema_length=None: scan every row before choosing column types.
        distcont3/4 are almost always empty, so the default 100-row scan
        guesses the wrong type.
    """
    return pl.read_csv(path, null_values="NA", infer_schema_length=None)


def make_rate_by(df: pl.DataFrame, col: str | pl.Expr) -> pl.DataFrame:
    """Shot count and make rate for each value of a column."""
    return (
        df.group_by(col)
        .agg(shots=pl.len(), make_rate=pl.col("outcome").mean().round(3))
        .sort(df.select(col).columns[0])
    )


def explore(train: pl.DataFrame, test: pl.DataFrame) -> None:
    """Print the summaries that shaped the cleaning and validation choices."""
    print(f"Train: {train.height:,} shots | Test: {test.height:,} shots")
    print(f"Overall make rate: {train['outcome'].mean():.3f}\n")

    # Missing values: only contester columns should have nulls.
    nulls = {c: n for c, n in zip(train.columns, train.null_count().row(0)) if n}
    print("Columns with missing values (train):", nulls, "\n")

    # Seasons: the test season does not appear in training, which drives the
    # validation design (hold out a whole season, not random rows).
    print("Train seasons:", sorted(train["season_id"].unique()))
    print("Test seasons: ", sorted(test["season_id"].unique()), "\n")

    unseen = test.filter(~pl.col("shooter_id").is_in(train["shooter_id"].implode()))
    print(f"Test shooters not seen in training: {unseen['shooter_id'].n_unique()} "
          f"of {test['shooter_id'].n_unique()}\n")

    print(make_rate_by(train, "shottype"))       # heaves are rare and ~9% makes
    print(make_rate_by(train, "gamestate"))
    print(make_rate_by(train, "num_contesters"))
    print(make_rate_by(train, (pl.col("distance") // 3 * 3).alias("distance_3ft_bin")))

    # Shot clock: no nulls, but spikes at exactly 0 and exactly 24 behave
    # differently from their neighbors (see clean_features for how we handle them).
    print(make_rate_by(train, pl.col("shotclock").round(0).alias("shotclock_sec")))
    for value in (0.0, 24.0):
        at_value = train.filter(pl.col("shotclock") == value)
        print(f"shotclock == {value}: {at_value.height:,} shots")
        print(make_rate_by(at_value, "gamestate"))


# ---------------------------------------------------------------------------
# Step 2: Clean and engineer features
# ---------------------------------------------------------------------------

def clean_features(df: pl.DataFrame) -> pl.DataFrame:
    """Turn raw shot rows into model-ready features.

    Works on train and test the same way. It never looks at `outcome`, so it
    can't leak the answer into the features. Categorical columns (shottype,
    gamestate) stay as text here; one_hot() handles them for models that
    need numeric inputs.
    """
    # closestdefapproach is text like "{20.04,18.82,16.56,13.86}": the closest
    # defender's distance 1.0, 0.75, 0.5 and 0.25 seconds before release.
    approach = (
        pl.col("closestdefapproach")
        .str.strip_chars("{}")
        .str.split(",")
        .list.eval(pl.element().cast(pl.Float64))
    )

    return df.with_columns(
        approach.list.get(0).alias("def_dist_1s"),
        approach.list.get(1).alias("def_dist_075s"),
        approach.list.get(2).alias("def_dist_05s"),
        approach.list.get(3).alias("def_dist_025s"),

        # Contester distances: fill "no such contester" with a far-away value.
        # distcont3/4 are dropped below (present on <1% of shots);
        # num_contesters still carries that information.
        pl.col("distcont1").fill_null(NO_CONTESTER_DIST),
        pl.col("distcont2").fill_null(NO_CONTESTER_DIST),

        # Shot angle from the rim in degrees: 0 = along the baseline (corner),
        # 90 = straight on. Left vs right side is folded together with abs().
        pl.arctan2(pl.col("locationy").abs(), pl.col("locationx") - RIM_X)
        .degrees()
        .alias("shot_angle"),

        # Corner three: the shortest three (22 ft), historically efficient.
        (pl.col("three") & (pl.col("locationx") <= CORNER_THREE_MAX_X))
        .cast(pl.Int8)
        .alias("corner_three"),

        # Shot clock has no missing values, but two exact values look like
        # special cases rather than real time remaining:
        #   == 24: mostly tip-ins and putbacks right after a rebound reset
        #   == 0 : mostly end-of-quarter heaves, where the shot clock is off
        # Flags let the model treat these separately from the smooth trend.
        (pl.col("shotclock") == 24).cast(pl.Int8).alias("shotclock_at_24"),
        (pl.col("shotclock") == 0).cast(pl.Int8).alias("shotclock_at_0"),

        # Booleans to 0/1 so every model reads them the same way.
        pl.col("three").cast(pl.Int8),
        pl.col("contested").cast(pl.Int8),
    ).drop(
        "closestdefapproach",            # replaced by the four def_dist columns
        "distcont3", "distcont4",        # too rare to learn from
        "contester1_id", "contester2_id", "contester3_id", "contester4_id",
        "month",                         # make rate is flat across months
    )


# One-hot categories for linear models. The most common category in each
# group is left out as the reference level (see one_hot below).
SHOTTYPE_REFERENCE = "jumper"
SHOTTYPES = ["dunk", "floater", "heave", "layup", "post", "tip"]
GAMESTATE_REFERENCE = "HALFCOURT"
GAMESTATES = ["TRANSITION", "DIRECTSECONDCHANCE"]


def one_hot(df: pl.DataFrame) -> pl.DataFrame:
    """Expand shottype and gamestate into 0/1 columns, minus a reference level.

    Every shot belongs to exactly one category, so a full set of dummies always
    sums to 1. That duplicates the model's intercept and leaves the
    coefficients with no unique answer. Dropping one category (the reference)
    fixes this, and each remaining coefficient then reads "compared with a
    halfcourt jumper".

    Categories are listed explicitly, not discovered from the data, so the fit
    and validation sets always get identical columns.
    """
    return df.with_columns(
        *[(pl.col("shottype") == c).cast(pl.Int8).alias(f"shottype_{c}") for c in SHOTTYPES],
        *[(pl.col("gamestate") == c).cast(pl.Int8).alias(f"gamestate_{c}") for c in GAMESTATES],
    )


# Columns that are identifiers or the target, not features.
ID_COLS = ["shot_id", "season_id", "off_team_id", "def_team_id",
           "shooter_id", "closestdef_id"]
TARGET = "outcome"


# ---------------------------------------------------------------------------
# Validation: leave-one-season-out
# ---------------------------------------------------------------------------
# The test set is a season the training data never sees. To mimic that, each
# model is fit on one training season and scored on the other, then the
# seasons swap. The average of the two scores is our estimate of test error.

# A "model" here is any function that takes a fit set and a validation set
# and returns one make probability per validation shot. Every model in this
# file (constant, logistic, LightGBM) follows this shape, so they all go
# through the same validation loop.
FitPredict = Callable[[pl.DataFrame, pl.DataFrame], np.ndarray]


def season_folds(train: pl.DataFrame):
    """Yield (held_out_season, fit_set, validation_set) for each season."""
    for season in sorted(train["season_id"].unique()):
        is_held_out = pl.col("season_id") == season
        yield season, train.filter(~is_held_out), train.filter(is_held_out)


def evaluate(name: str, fit_predict: FitPredict, train: pl.DataFrame) -> dict:
    """Score a model with leave-one-season-out log-loss.

    Log-loss averages -[y*log(p) + (1-y)*log(1-p)] over shots. It punishes
    confident wrong answers heavily, so it rewards honest probabilities, not
    just picking the right side of 50%. Lower is better.

    Also keeps each shot's prediction from the fold where its season was held
    out ("oof" = out-of-fold), used later for the calibration plot.
    """
    scores = {}
    oof = np.empty(train.height)
    for season, fit_set, val_set in season_folds(train):
        # The model never sees validation outcomes: we drop them before the
        # call and only use them afterward to score the predictions.
        probs = fit_predict(fit_set, val_set.drop(TARGET))
        scores[season] = log_loss(val_set[TARGET].to_numpy(), probs)
        oof[(train["season_id"] == season).to_numpy()] = probs

    result = {"model": name, **scores, "mean": float(np.mean(list(scores.values()))), "oof": oof}
    fold_text = "  ".join(f"held out {s}: {v:.4f}" for s, v in scores.items())
    print(f"{name:<22} {fold_text}  |  mean: {result['mean']:.4f}")
    return result


# ---------------------------------------------------------------------------
# Step 3: Constant baseline
# ---------------------------------------------------------------------------

def constant_baseline(fit_set: pl.DataFrame, val_set: pl.DataFrame) -> np.ndarray:
    """Predict the fit set's overall make rate for every validation shot.

    This uses no features at all. It is the same as a logistic regression
    with only an intercept, so it marks the score any real model must beat.
    """
    make_rate = fit_set[TARGET].mean()
    return np.full(val_set.height, make_rate)


# ---------------------------------------------------------------------------
# Step 4: Logistic regression baseline
# ---------------------------------------------------------------------------
# Logistic regression models the log-odds of a make as a weighted sum of
# features: log(p / (1 - p)) = intercept + w1*x1 + w2*x2 + ...
# It can only learn straight-line effects on the log-odds scale, which keeps
# it easy to interpret but limits accuracy (e.g. make rate vs distance is not
# a straight line). LightGBM will relax that later.

# Continuous features get standardized: (x - mean) / std, using the fit set's
# mean and std. Scaling doesn't change what the model can learn. It puts each
# coefficient in "per standard deviation" units, so they can be compared, and
# it makes the default L2 penalty treat every feature evenly.
LOGIT_CONTINUOUS = [
    "distance",
    "closestdefdist",
    "def_dist_1s", "def_dist_075s", "def_dist_05s", "def_dist_025s",
    "num_contesters",
    "dribblesbefore",
    "shotclock",
    "shooterspeed",
]

# Contester distances only mean something when that contester exists. For
# the linear model they're set to 0 when there's no such contester (instead of
# the 30 ft fill from clean_features), so the term weight * distance adds
# nothing on those shots. The presence of a contest is carried separately by
# `contested` and `num_contesters`, so a 0 here never reads as "0 ft away":
# every row with a 0 also has contested = 0.
#
# These columns are scaled by std only (not centered). Subtracting the mean
# would turn the 0s into a negative number and break that "adds nothing" rule.
LOGIT_CONTEST_DIST = ["distcont1", "distcont2"]

# 0/1 features pass through unscaled, so their coefficients read as
# "this vs not this".
LOGIT_BINARY = (
    ["three", "contested", "shotclock_at_24", "shotclock_at_0"]
    + [f"shottype_{c}" for c in SHOTTYPES]
    + [f"gamestate_{c}" for c in GAMESTATES]
)

LOGIT_FEATURES = LOGIT_CONTINUOUS + LOGIT_CONTEST_DIST + LOGIT_BINARY


def logistic_matrix(df: pl.DataFrame):
    """Select the logistic regression features as a pandas table.

    sklearn works with pandas, and keeping column names lets us label the
    coefficients afterward. Contester distances are zeroed when that contester
    doesn't exist (see LOGIT_CONTEST_DIST).
    """
    return (
        one_hot(df)
        .with_columns(
            pl.when(pl.col("num_contesters") >= 1).then("distcont1").otherwise(0.0).alias("distcont1"),
            pl.when(pl.col("num_contesters") >= 2).then("distcont2").otherwise(0.0).alias("distcont2"),
        )
        .select(LOGIT_FEATURES)
        .to_pandas()
    )


def build_logistic() -> Pipeline:
    """Scaler + logistic regression, packaged so they always run together.

    A Pipeline makes .fit() learn the scaling *and* the coefficients from the
    same data, and .predict_proba() apply the stored scaling to new data. That
    guarantees the validation set is scaled with the fit set's mean and std.
    """
    scaler = ColumnTransformer(
        [
            ("scale", StandardScaler(), LOGIT_CONTINUOUS),
            ("scale_no_center", StandardScaler(with_mean=False), LOGIT_CONTEST_DIST),
        ],
        remainder="passthrough",          # binary columns pass through as-is
        verbose_feature_names_out=False,  # keep plain column names
    )
    # C is the inverse strength of the L2 penalty. With ~200k shots per fit,
    # the default (1.0) barely shrinks anything; max_iter gives the solver
    # room to converge.
    model = LogisticRegression(C=1.0, max_iter=1000)
    return Pipeline([("scaler", scaler), ("model", model)])


def logistic_fit_predict(fit_set: pl.DataFrame, val_set: pl.DataFrame) -> np.ndarray:
    """Fit logistic regression on fit_set and return make probabilities for val_set."""
    pipeline = build_logistic()
    pipeline.fit(logistic_matrix(fit_set), fit_set[TARGET].to_numpy())
    # predict_proba returns [P(miss), P(make)] per row; keep P(make).
    return pipeline.predict_proba(logistic_matrix(val_set))[:, 1]


def logistic_coefficients(train: pl.DataFrame) -> pl.DataFrame:
    """Fit on all training shots and return a readable coefficient table.

    - coefficient: change in log-odds of a make. For continuous features, per
      one standard deviation; for 0/1 features, "this vs not this" (or vs the
      jumper / halfcourt reference).
    - odds_ratio: exp(coefficient). 1.20 = 20% higher odds of a make,
      0.80 = 20% lower odds.

    Caveat: correlated features (e.g. the four def_dist columns and
    closestdefdist) split credit between themselves, so read their individual
    signs with care. The model's predictions aren't affected by this.
    """
    pipeline = build_logistic()
    pipeline.fit(logistic_matrix(train), train[TARGET].to_numpy())

    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["scaler"].get_feature_names_out()
    print(f"Intercept: {model.intercept_[0]:.3f} (log-odds of a make when every "
          f"continuous feature is at its average and every 0/1 feature is 0)")

    return (
        pl.DataFrame({"feature": names, "coefficient": model.coef_[0]})
        .with_columns(pl.col("coefficient").exp().alias("odds_ratio"))
        .sort(pl.col("coefficient").abs(), descending=True)
        .with_columns(pl.col("coefficient", "odds_ratio").round(3))
    )


# ---------------------------------------------------------------------------
# Step 5: LightGBM (gradient boosted trees)
# ---------------------------------------------------------------------------
# Gradient boosting builds many small decision trees in sequence. Each new
# tree is fit to the errors the trees before it still make, and its output is
# shrunk by the learning rate before being added. Trees split features into
# ranges ("distance < 3.2 ft"), so curved effects like make rate vs distance,
# and interactions like "close AND contested", are learned automatically.
# That's the flexibility logistic regression lacks.
#
# Trees only compare values against thresholds, so no scaling is needed, and
# the 30 ft "no contester" fill from clean_features works as-is: a tree just
# splits "distcont1 < 30".

GBM_NUMERIC = [
    "distance", "three", "corner_three", "shot_angle",
    "closestdefdist",
    "def_dist_1s", "def_dist_075s", "def_dist_05s", "def_dist_025s",
    "contested", "num_contesters", "distcont1", "distcont2",
    "dribblesbefore",
    "shotclock", "shotclock_at_24", "shotclock_at_0",
    "shooterspeed",
]
# LightGBM splits text categories directly (e.g. {dunk, layup} vs the rest),
# so no one-hot encoding is needed here.
GBM_CATEGORICAL = {
    "shottype": [SHOTTYPE_REFERENCE] + SHOTTYPES,
    "gamestate": [GAMESTATE_REFERENCE] + GAMESTATES,
}
# V1: shot, defense and context features only.
GBM_V1_FEATURES = GBM_NUMERIC + list(GBM_CATEGORICAL)
# Experiment: raw court coordinates on top of distance and angle. They added
# nothing (distance + angle already describe location), so V2 leaves them out.
GBM_COORD_FEATURES = GBM_V1_FEATURES + ["locationx", "locationy"]
# V2 (final model) adds who is shooting and defending, encoded as how much
# better or worse than expected their shots go in (see step 6).
ID_ENCODED = ["shooter_id", "closestdef_id", "def_team_id"]
ID_FEATURES = [f"{c}_above_exp" for c in ID_ENCODED]
GBM_V2_FEATURES = GBM_V1_FEATURES + ID_FEATURES

GBM_PARAMS = {
    "objective": "binary",        # predict P(make) by minimizing log-loss
    "metric": "binary_logloss",
    "learning_rate": 0.05,        # shrink each tree's contribution; lower = more trees needed
    # Overfitting controls (limit how specific each tree can get):
    "num_leaves": 31,             # max end-nodes per tree
    "max_depth": 8,               # max splits from root to leaf
    "min_data_in_leaf": 200,      # every leaf must cover >= 200 shots
    "lambda_l2": 1.0,             # penalty that shrinks leaf values toward 0
    "feature_fraction": 0.8,      # each tree sees a random 80% of the features
    "seed": 42,
    "verbose": -1,
}
GBM_MAX_ROUNDS = 3000             # upper limit; early stopping picks the real number
GBM_EARLY_STOP = 100              # stop after 100 rounds with no improvement
GBM_STOP_FRACTION = 0.10          # share of the fit set held back for early stopping


def gbm_matrix(df: pl.DataFrame, features: list[str]) -> pd.DataFrame:
    """Select LightGBM features as pandas, with categories fixed in advance.

    Fixing the category list means "dunk" gets the same internal code in
    every season, so splits learned on one season apply correctly to another.
    """
    X = df.select(features).to_pandas()
    for col, categories in GBM_CATEGORICAL.items():
        if col in features:
            X[col] = pd.Categorical(X[col], categories=categories)
    return X


def fit_lightgbm(fit_set: pl.DataFrame, features: list[str],
                 n_rounds: int | None = None) -> lgb.Booster:
    """Train LightGBM on fit_set using the given feature list.

    With n_rounds=None, early stopping picks the number of trees: a random 10%
    of fit_set is held back, and training stops once log-loss on that slice
    hasn't improved for GBM_EARLY_STOP rounds. This slice comes from the fit
    set, never the season being scored, so the scored season stays unseen.

    With n_rounds given, trains exactly that many trees on all of fit_set
    (used for the final model, once validation has told us a good count).
    """
    if n_rounds is not None:
        data = lgb.Dataset(gbm_matrix(fit_set, features), fit_set[TARGET].to_numpy())
        return lgb.train(GBM_PARAMS, data, num_boost_round=n_rounds)

    is_stop = np.random.default_rng(42).random(fit_set.height) < GBM_STOP_FRACTION
    train_part = fit_set.filter(~is_stop)
    stop_part = fit_set.filter(is_stop)

    train_data = lgb.Dataset(gbm_matrix(train_part, features), train_part[TARGET].to_numpy())
    stop_data = lgb.Dataset(gbm_matrix(stop_part, features), stop_part[TARGET].to_numpy(),
                            reference=train_data)
    return lgb.train(
        GBM_PARAMS,
        train_data,
        num_boost_round=GBM_MAX_ROUNDS,
        valid_sets=[stop_data],
        callbacks=[lgb.early_stopping(GBM_EARLY_STOP, verbose=False)],
    )


def make_lightgbm(features: list[str], id_expected_by: str | None = None) -> FitPredict:
    """Build a LightGBM model function for evaluate() with a given feature set.

    evaluate() expects a function of (fit_set, val_set). This returns one
    with the settings already filled in, so every LightGBM version goes
    through the same validation loop.

    id_expected_by: None = no ID features. Otherwise "shot" or "league",
    passed to add_id_features (see step 6).
    """
    def fit_predict(fit_set: pl.DataFrame, val_set: pl.DataFrame) -> np.ndarray:
        if id_expected_by is not None:
            fit_set, val_set = add_id_features(fit_set, val_set, id_expected_by)
        booster = fit_lightgbm(fit_set, features)
        print(f"    early stopping chose {booster.best_iteration} trees")
        return booster.predict(gbm_matrix(val_set, features),
                               num_iteration=booster.best_iteration)
    return fit_predict


# ---------------------------------------------------------------------------
# Step 6: Player and team IDs as "make rate above expected"
# ---------------------------------------------------------------------------
# IDs are random strings, so a model can't use them directly. Each ID becomes
# a number: how much better or worse than expected that player's shots go in
# (or, for defenders and teams, the shots they defend).
#
# Why "above expected" and not a plain make rate: a plain rate mostly reflects
# shot selection. A center makes a high share of shots because many are
# dunks, which the model already knows from shottype and distance. So each
# shot's expected make rate is the league rate for that shot type in that
# 2 ft distance band, and an ID's value is its average of (made - expected).
# A center making 70% of dunks where the league makes 88% scores *below*
# average. (Tested: plain make rates added nothing; this version helps.)
#
# Two safeguards keep this honest:
#
# 1. Smoothing. 5 shots aren't enough to judge a shooter. The value is
#        sum(made - expected) / (shots + K)
#    which is the plain average pulled toward 0 (= "as expected"), with the
#    pull fading as shots grow. With K = 100: 100 shots -> half strength,
#    1,000 shots -> ~91%. IDs never seen in the fit data (rookies) get 0.
#
# 2. Out-of-fold values for training rows. If a shot's own outcome were in
#    its shooter's value, the value would partly contain the answer, and the
#    model would learn to over-trust it. So the fit set is split into 5
#    folds, and each fold's values come only from the other 4. Validation
#    (and test) rows get values from the whole fit set, which never includes
#    their own outcomes.

ID_SMOOTHING = 100
ID_FOLDS = 5
EXPECTED_DIST_BAND_FT = 2


def expected_make_rate(source: pl.DataFrame, target: pl.DataFrame,
                       expected_by: str) -> np.ndarray:
    """Expected make rate for each target shot, learned from source shots.

    expected_by="shot":   league rate for the shot's type and distance band.
    expected_by="league": the overall league rate for every shot. This turns
                          "above expected" back into a plain make rate, kept
                          so the comparison can be reproduced.
    """
    league_rate = source[TARGET].mean()
    if expected_by == "league":
        return np.full(target.height, league_rate)

    band = (pl.col("distance") // EXPECTED_DIST_BAND_FT).alias("band")
    rates = (source.group_by("shottype", band)
             .agg(expected=pl.col(TARGET).mean()))
    looked_up = (target.select("shottype", band)
                 .join(rates, on=["shottype", "band"], how="left", maintain_order="left"))
    # A type/band combination missing from source falls back to the league rate.
    return looked_up["expected"].fill_null(league_rate).to_numpy()


def above_expected(source: pl.DataFrame, target: pl.DataFrame, col: str,
                   expected_by: str) -> np.ndarray:
    """Smoothed (made - expected) per `col` value, learned on source, looked up for target."""
    residual = source[TARGET].cast(pl.Float64).to_numpy() - expected_make_rate(source, source, expected_by)
    stats = (source.select(col).with_columns(residual=residual)
             .group_by(col).agg(total=pl.col("residual").sum(), shots=pl.len()))
    looked_up = target.select(col).join(stats, on=col, how="left", maintain_order="left")
    total = looked_up["total"].fill_null(0).to_numpy()
    shots = looked_up["shots"].fill_null(0).to_numpy()
    return total / (shots + ID_SMOOTHING)


def add_id_features(fit_set: pl.DataFrame, val_set: pl.DataFrame,
                    expected_by: str = "shot"):
    """Add a `<id>_above_exp` column for each ID to both sets (see notes above)."""
    fold = np.random.default_rng(42).integers(0, ID_FOLDS, fit_set.height)

    fit_values, val_values = {}, {}
    for col in ID_ENCODED:
        name = f"{col}_above_exp"
        # Validation rows: values from the whole fit set.
        val_values[name] = above_expected(fit_set, val_set, col, expected_by)

        # Fit rows: each fold's values come from the other folds only.
        values = np.empty(fit_set.height)
        for k in range(ID_FOLDS):
            in_fold = fold == k
            values[in_fold] = above_expected(fit_set.filter(~in_fold),
                                             fit_set.filter(in_fold), col, expected_by)
        fit_values[name] = values

    return fit_set.with_columns(**fit_values), val_set.with_columns(**val_values)


# ---------------------------------------------------------------------------
# Step 7: Calibration and SHAP importance (figures for the writeup)
# ---------------------------------------------------------------------------

# Chart colors: two series max, from a colorblind-checked palette.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style_axes(ax) -> None:
    """Quiet axes: light grid, no top/right borders, muted labels."""
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def calibration_table(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 20) -> pl.DataFrame:
    """Group shots into equal-size bins by predicted probability and compare
    the average prediction to the actual make rate in each bin.

    A well-calibrated model has these match: of shots it calls 30%, about
    30% go in. This matters here because we're scored against another
    model's probabilities, not just right/wrong.
    """
    return (
        pl.DataFrame({"pred": probs, "made": outcomes.astype(float)})
        .with_columns(bin=pl.col("pred").qcut(n_bins, allow_duplicates=True, labels=None))
        .group_by("bin")
        .agg(predicted=pl.col("pred").mean(), actual=pl.col("made").mean(), shots=pl.len())
        .sort("predicted")
    )


def plot_calibration(results: list[dict], train: pl.DataFrame, names: list[str]) -> None:
    """Calibration curves (out-of-season predictions) for the chosen models."""
    outcomes = train[TARGET].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Perfect calibration")

    for result, color in zip([r for r in results if r["model"] in names], (ORANGE, BLUE)):
        table = calibration_table(result["oof"], outcomes)
        ax.plot(table["predicted"], table["actual"], color=color, linewidth=2,
                marker="o", markersize=5, label=result["model"])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted make probability (bin average)", color=INK)
    ax.set_ylabel("Actual make rate", color=INK)
    ax.set_title("Calibration on held-out seasons (20 equal-size bins)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "calibration.png", dpi=150)
    plt.close(fig)


def shap_values(booster: lgb.Booster, X: pd.DataFrame) -> pd.DataFrame:
    """Per-shot SHAP values from LightGBM's built-in TreeSHAP.

    Each value is how much one feature pushed one shot's prediction up or
    down, in log-odds, compared with the average prediction. Across a shot's
    features they add up exactly to its prediction. Unlike gain importance,
    this splits credit fairly between correlated features.
    """
    contrib = booster.predict(X, pred_contrib=True)
    return pd.DataFrame(contrib[:, :-1], columns=X.columns)  # last column = baseline


def plot_shap(shap: pd.DataFrame, X: pd.DataFrame) -> pl.DataFrame:
    """Save two SHAP figures and return the importance table.

    1. shap_importance.png: mean |SHAP| per feature, the average size of each
       feature's effect on a prediction.
    2. shap_dependence.png: how the prediction moves across the range of
       four key features (each dot is one shot).
    """
    importance = (
        pl.DataFrame({"feature": shap.columns, "mean_abs_shap": np.abs(shap.to_numpy()).mean(axis=0)})
        .sort("mean_abs_shap", descending=True)
    )

    # 1. Importance bar chart (largest at top).
    ordered = importance.reverse()
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.barh(ordered["feature"], ordered["mean_abs_shap"], color=BLUE, height=0.7)
    ax.set_xlabel("Mean |SHAP value| (average effect on log-odds of a make)", color=INK)
    ax.set_title("Feature importance, LightGBM V2", color=INK, fontsize=11, loc="left")
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=INK)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "shap_importance.png", dpi=150)
    plt.close(fig)

    # 2. Dependence plots for four features worth explaining.
    panels = [
        ("distance", "Shot distance (ft)"),
        ("closestdefdist", "Closest defender distance (ft)"),
        ("shooter_id_above_exp", "Shooter make rate above expected"),
        ("shotclock", "Shot clock (s)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharey=True)
    for ax, (col, label) in zip(axes.flat, panels):
        x = X[col].to_numpy()
        if col == "closestdefdist":
            keep = x <= 20  # a few tracking outliers stretch the axis
        else:
            keep = np.ones_like(x, dtype=bool)
        ax.scatter(x[keep], shap[col].to_numpy()[keep], s=4, alpha=0.15, color=BLUE, linewidths=0)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xlabel(label, color=INK, fontsize=9)
        style_axes(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("SHAP value (log-odds)", color=INK, fontsize=9)
    fig.suptitle("How key features move the prediction (each dot is one shot)",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "shap_dependence.png", dpi=150)
    plt.close(fig)

    return importance


# ---------------------------------------------------------------------------
# Step 8: Final model and submission
# ---------------------------------------------------------------------------

# Never predict exactly 0 or 1: log-loss would be infinite if wrong.
PROB_CLIP = (0.001, 0.999)


def fit_final_model(train_ids: pl.DataFrame) -> lgb.Booster:
    """Fit V2 on both training seasons.

    1. Early stopping on a 10% slice of training picks the number of trees.
    2. Refit on 100% of training with that many trees, so no data is wasted.
    """
    n_trees = fit_lightgbm(train_ids, GBM_V2_FEATURES).best_iteration
    print(f"Final model: {n_trees} trees, fit on all {train_ids.height:,} training shots")
    return fit_lightgbm(train_ids, GBM_V2_FEATURES, n_rounds=n_trees)


def write_submission(test_ids: pl.DataFrame, probs: np.ndarray) -> None:
    """Fill make_prob in submission.csv, keeping its rows and columns as-is."""
    submission = pl.read_csv(SUBMISSION_PATH)
    # The template's shot_id order must match testing.csv.gz row for row.
    assert submission["shot_id"].equals(test_ids["shot_id"]), "shot_id order mismatch"

    probs = np.clip(probs, *PROB_CLIP)
    assert len(probs) == submission.height and not np.isnan(probs).any()

    submission.with_columns(make_prob=pl.Series(probs)).write_csv(SUBMISSION_PATH)
    print(f"Wrote {len(probs):,} predictions to {SUBMISSION_PATH} "
          f"(mean {probs.mean():.3f}, range {probs.min():.3f}-{probs.max():.3f})")


def main() -> None:
    train = load_data(TRAIN_PATH)
    test = load_data(TEST_PATH)

    # Step 1
    explore(train, test)

    # Step 2
    train = clean_features(train)
    test = clean_features(test)
    print("\nCleaned feature columns:")
    print([c for c in train.columns if c not in ID_COLS + [TARGET]])
    print(train.drop(ID_COLS + [TARGET, "shottype", "gamestate"]).describe())

    # Step 3
    print("\nLeave-one-season-out log-loss (lower is better)")
    results = [evaluate("Constant baseline", constant_baseline, train)]

    # Step 4
    results.append(evaluate("Logistic regression", logistic_fit_predict, train))

    # Step 5 and 6: LightGBM versions. The middle two are experiments kept for
    # the writeup; "LightGBM V2" is the final model.
    results.append(evaluate("LightGBM V1", make_lightgbm(GBM_V1_FEATURES), train))
    results.append(evaluate("V1 + coords", make_lightgbm(GBM_COORD_FEATURES), train))
    results.append(evaluate("V1 + plain ID rates",
                            make_lightgbm(GBM_V2_FEATURES, id_expected_by="league"), train))
    results.append(evaluate("LightGBM V2 (final)",
                            make_lightgbm(GBM_V2_FEATURES, id_expected_by="shot"), train))

    print("\nSummary (leave-one-season-out log-loss):")
    print(pl.DataFrame([{k: v for k, v in r.items() if k != "oof"} for r in results])
          .with_columns(pl.exclude("model").round(4)))

    print("\nLogistic regression coefficients (fit on all training shots):")
    with pl.Config(tbl_rows=-1):
        print(logistic_coefficients(train))

    # Step 8 (fit first, since step 7's SHAP figures explain the final model).
    # ID values for test come from all training shots; training rows get
    # out-of-fold values, exactly as in validation.
    train_ids, test_ids = add_id_features(train, test)
    final_model = fit_final_model(train_ids)
    write_submission(test_ids, final_model.predict(gbm_matrix(test_ids, GBM_V2_FEATURES)))

    # Step 7
    FIGURE_DIR.mkdir(exist_ok=True)
    plot_calibration(results, train, ["Logistic regression", "LightGBM V2 (final)"])
    print("\nCalibration, LightGBM V2 (out-of-season predictions):")
    v2 = next(r for r in results if r["model"] == "LightGBM V2 (final)")
    with pl.Config(tbl_rows=-1):
        print(calibration_table(v2["oof"], train[TARGET].to_numpy()).drop("bin")
              .with_columns(pl.col("predicted", "actual").round(3)))

    # SHAP on a random 20k-shot sample of training (plenty for stable averages).
    sample = train_ids.sample(20_000, seed=42)
    X_sample = gbm_matrix(sample, GBM_V2_FEATURES)
    importance = plot_shap(shap_values(final_model, X_sample), X_sample)
    print("\nSHAP importance, final model (mean |SHAP|, log-odds):")
    with pl.Config(tbl_rows=-1):
        print(importance.with_columns(pl.col("mean_abs_shap").round(4)))
    print(f"\nFigures saved to {FIGURE_DIR}/")


if __name__ == "__main__":
    main()
