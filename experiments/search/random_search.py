import random

SEARCH_SPACE = {
    "rna_encoder": [
        "pca",
        "scvi",
    ],
    "image_encoder": [
        "imagenet_vit_l",
    ],
    "fusion": [
        "concat",
    ],
    "aggregation": [
        "singleprofile",
        "avgpool",
    ],
}


def sample_configuration():
    return {
        key: random.choice(values)
        for key, values in SEARCH_SPACE.items()
    }


if __name__ == "__main__":

    NUM_EXPERIMENTS = 10

    for i in range(NUM_EXPERIMENTS):

        config = sample_configuration()

        if config["aggregation"] == "singleprofile":
            experiment = (
                f"train_on_{config['rna_encoder']}"
                f"_with_{config['fusion']}_{config['image_encoder']}"
            )

        else:
            experiment = (
                f"train_avgpool_on_{config['rna_encoder']}"
                f"_with_{config['fusion']}_{config['image_encoder']}"
            )

        print(f"\nExperiment {i+1}")
        print(f"experiment={experiment}")