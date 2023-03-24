import sys
import error_sampler as err
import datasets

if __name__ == "__main__":
    args = sys.argv
    user_predictor = args[1].lower()
    user_binary = args[2].lower()
    user_seed = int(sys.argv[3])
    user_path = args[4]

    # set random seed
    err.set_random_seed(user_seed)

    metadata_path, dataset_path = datasets.get_dataset_metadata_and_path(
        user_predictor,
        user_binary,
        user_path
    )

    # download data
    datasets.download_data()

    #
    # metadata_path = ".dataset_downloads/medisynth-nn-continuous.json"

    generator = datasets.DatasetGenerator.from_json(metadata_path)
    dataset = generator.make_dataset()

    # dataset.save_dataset(dataset_path)
    print("Dataset sampling completed.")
