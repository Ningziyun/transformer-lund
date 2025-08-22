import numpy as np
from graphnet import GraphNet
from data_loader import get_config, load_data
import os
import tensorflow as tf
import time

start = time.time()
tfk = tf.keras

gpu = tf.config.list_physical_devices("GPU")
assert len(gpu) > 0, f"No GPU found, abort"

def train_model(model, data, labels, train_params, logfolder, mask):
    optimizer = tfk.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer,
        loss="categorical_crossentropy",
        metrics=[tfk.metrics.AUC(), "acc"],
    )
    if mask:
        mask = data[:, :, 2] != 0
        train_data = [data[:, :, :2], data, mask]
    else:
        train_data = [
            data[:, :, :2],
            data,
        ]

    # Class weights
    class_weight = {
        0: len(labels) / np.sum(labels == 0),
        1: len(labels) / np.sum(labels == 1),
    }

    tmp = min(class_weight.values())
    class_weight = {key: class_weight[key] / tmp for key in class_weight}
    print(f"Class weights: {class_weight}")

    fit = model.fit(
        train_data,
        tfk.utils.to_categorical(labels, 2),
        callbacks=[
            tfk.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=8, verbose=1),
            tfk.callbacks.EarlyStopping(
                monitor="val_loss", patience=12, verbose=1, restore_best_weights=True
            ),
            tfk.callbacks.ModelCheckpoint(
                os.path.join(logfolder, "checkpoint_{epoch:02d}"),
                save_best_only=False,
                save_freq=data.shape[0] // train_params["batch_size"] * 5,
                save_weights_only=True,
            ),
        ],
        class_weight=class_weight,
        **train_params,
    )
    np.savez(os.path.join(logfolder, "training.npz"), **fit.history)


def check_activation(config):
    if config["graphnet"]["activation"] == "ReLU":
        config["graphnet"]["activation"] = tfk.layers.ReLU()
    elif config["graphnet"]["activation"] == "LeakyReLU":
        config["graphnet"]["activation"] = tfk.layers.LeakyReLU(alpha=0.1)
    else:
        tmp = config["graphnet"]["activation"]
        raise SystemExit(f"{tmp} activation not implemented")
    return config


def main():
    config = check_activation(get_config())
    model = GraphNet(**config["graphnet"])
    start = time.time()
    data, true_label = load_data(
        config["data"],
        plot_dists=os.path.join(config["logging"]["logfolder"], "distributions.png"),)
    print(f"First labels {true_label[:15]}\n")

    train_model(
        model=model,
        data=data,
        labels=true_label,
        train_params=config["training"],
        logfolder=config["logging"]["logfolder"],
        mask=config["mask"],
    )
    model.save_weights(os.path.join(config["logging"]["logfolder"], "model_weights.h5"))
    if config["mask"]:
        mask = data[:, :, 2] != 0
        test_data = [data[:, :, :2], data, mask]
    else:
        test_data = [data[:, :, :2], data]

    predictions = model.predict(test_data, batch_size=1024, verbose=1)

    np.savez(
        os.path.join(config["logging"]["logfolder"], "predictions.npz"),
        predictions=predictions,
        region=true_label,
        true_label=true_label,
        seed=config["data"]["seed"],
    )

    checker = model.predict(
        [
            np.ones_like(data[:10]),
            np.ones_like(data[:10]),
            np.ones(data[:10].shape[:-1]),
        ]
    )
    np.save(os.path.join(config["logging"]["logfolder"], "check.npy"), checker)
    print(f"Script took {int(time.time() - start)} seconds")


if __name__ == "__main__":
    main()
