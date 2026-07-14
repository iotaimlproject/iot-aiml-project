from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, LeakyReLU, Activation
from tensorflow.keras.optimizers import Adam

from ml_model.train.config import M1_CONFIG, N_FEATURES


def _add_activation(model, cfg):
    name = cfg["activation"]
    if name == "leaky_relu":
        model.add(LeakyReLU(0.1))
    else:
        model.add(Activation(name))


def _build(cfg, output_units, compile_kwargs):
    model = Sequential()
    model.add(Input(shape=(N_FEATURES,)))
    for i, (units, rate) in enumerate(zip(cfg["layers"], cfg["dropout"])):
        model.add(Dense(units))
        _add_activation(model, cfg)
        if cfg.get("use_batch_norm"):
            model.add(BatchNormalization())
        if rate > 0:
            model.add(Dropout(rate))
    model.add(Dense(output_units, activation=cfg["output_activation"]))
    model.compile(
        optimizer=Adam(learning_rate=cfg["learning_rate"]),
        **compile_kwargs,
    )
    return model


def build_m1_regressor():
    return _build(
        M1_CONFIG,
        output_units=1,
        compile_kwargs={
            "loss": M1_CONFIG["loss"],
            "metrics": ["mae"],
        },
    )



