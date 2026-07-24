"""Compute per-feature normalization statistics for a neuralpdr dataset.

Builds the training split exactly as `neuralpdr.train.main` does (same
`filter_models_by_series_length` + `shuffle_and_split`, seeded
deterministically), loads it through `PDRLoader` with no precomputed
mean/std so it derives them from the real data, and writes the result out
in the `normalisations_file` TOML schema read by `neuralpdr.config.Norms`.

Usage:
    python scripts/data/compute_normalisations.py configs/v4/base.toml \
        --output configs/v4/features/normalisations/computed.toml
"""

import argparse
from pathlib import Path

import tomlkit
from pydantic import TypeAdapter

from neuralpdr.config import (
    AUXNorm,
    DataNorm,
    Features,
    IVNorm,
    Norms,
    read_as,
    read_conf,
)
from neuralpdr.data import (
    PDRLoader,
    filter_models_by_series_length,
    log_semi_sorter,
    pad_and_stack,
    shuffle_and_split,
    text_from_h5,
)


def _build_train_loader(
    conf, input_features, train_indices, *, iv_eps, aux_eps, data_eps
):
    return PDRLoader(
        dataset_path=conf.dataset_path,
        independent_variable=input_features.iv,
        data_features=input_features.data,
        auxiliary_features=input_features.aux,
        index_range=(conf.start_index, conf.end_index),
        model_indices=train_indices,
        batch_size=conf.batch_size,
        stage="train",
        collate_fn=pad_and_stack,
        batch_permutation_function=log_semi_sorter,
        use_cache=True,
        independent_variable_normalization_kwargs={"eps": iv_eps},
        features_normalization_kwargs={"eps": data_eps},
        auxiliary_features_normalization_kwargs={"eps": aux_eps},
    )


def compute_norms(config_path: Path, eps: float, iv_eps: float | None = None) -> Norms:
    """Compute normalization stats from the real training split.

    `iv_eps`, if given and different from `eps`, is used for the independent
    variable (visual_extinction). Since it's also the first entry of the aux
    feature list, and `PDRLoader.normalize` only accepts one scalar eps per
    call (not a per-column array), getting a different eps for that one aux
    column means running the aux computation twice and splicing the
    visual_extinction column out of the `iv_eps` run.
    """
    conf = read_conf(config_path)
    input_features = read_as(conf.input_features_file, Features)
    split = conf.train_test_val_split
    iv_eps = eps if iv_eps is None else iv_eps

    model_indices = text_from_h5(str(conf.dataset_path), "model_ids")
    model_indices = filter_models_by_series_length(
        conf.minimal_timeseries_length, model_indices, str(conf.dataset_path)
    )
    train_indices, _val_indices, _test_indices = shuffle_and_split(
        model_indices=model_indices,
        train_split=split.train,
        val_split=split.validate,
        test_split=split.test,
    )

    print(
        f"Computing normalization statistics from {len(train_indices)} "
        f"training models (of {len(model_indices)} total)..."
    )
    baseline_loader = _build_train_loader(
        conf, input_features, train_indices, iv_eps=iv_eps, aux_eps=eps, data_eps=eps
    )
    stats = baseline_loader.get_normalization()
    iv_stats, data_stats, aux_stats = stats["iv"], stats["data"], stats["aux"]

    aux_eps = [float(aux_stats["eps"])] * len(input_features.aux)
    aux_mean = list(aux_stats["mean"])
    aux_std = list(aux_stats["std"])

    if iv_eps != eps and input_features.iv in input_features.aux:
        av_idx = input_features.aux.index(input_features.iv)
        print(
            f"Recomputing aux column {av_idx!r} ({input_features.iv}) with eps={iv_eps}"
        )
        override_loader = _build_train_loader(
            conf,
            input_features,
            train_indices,
            iv_eps=iv_eps,
            aux_eps=iv_eps,
            data_eps=eps,
        )
        override_aux_stats = override_loader.get_normalization()["aux"]
        aux_eps[av_idx] = float(iv_eps)
        aux_mean[av_idx] = float(override_aux_stats["mean"][av_idx])
        aux_std[av_idx] = float(override_aux_stats["std"][av_idx])

    return Norms(
        iv=IVNorm(
            eps=float(iv_stats["eps"]),
            mean=float(iv_stats["mean"]),
            std=float(iv_stats["std"]),
        ),
        data=DataNorm(
            eps=float(data_stats["eps"]),
            mean=data_stats["mean"].tolist(),
            std=data_stats["std"].tolist(),
        ),
        aux=AUXNorm(eps=aux_eps, mean=aux_mean, std=aux_std),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=str,
        help="Path to the base TOML config (e.g. configs/v4/base.toml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to write the normalisations TOML (defaults to "
        "<input_features_file's dir>/normalisations/computed.toml)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-20,
        help="Epsilon added before the log10 transform (uniform across features)",
    )
    parser.add_argument(
        "--iv-eps",
        type=float,
        default=None,
        help="Separate epsilon for the independent variable (visual_extinction), which "
        "is exactly zero at the start of every series and so needs a much larger "
        "floor than the other features. Defaults to --eps (no override).",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    conf_preview = read_conf(config_path)
    output_path = (
        Path(args.output).resolve()
        if args.output
        else conf_preview.input_features_file.parent
        / "normalisations"
        / "computed.toml"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    norms = compute_norms(config_path, args.eps, iv_eps=args.iv_eps)

    data = TypeAdapter(Norms).dump_python(norms, mode="json", exclude_none=True)
    output_path.write_text(tomlkit.dumps(data))
    print(f"Wrote normalization statistics to {output_path}")


if __name__ == "__main__":
    main()
