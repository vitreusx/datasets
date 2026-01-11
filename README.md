# Datasets

This repo contains [DVC](https://dvc.org/)-versioned datasets, along with a
Python library for loading them.

List of datasets:

- [CIFAR-10 & CIFAR-100](./datasets/cifar)
- [COCO 2017](./datasets/coco)
- [HD-VILA-100M](./datasets/hdvila)
- [ImageNet](./datasets/imagenet) and [ImageNet-100](./datasets/imagenet-100)
- [MNIST](./datasets/mnist)
- [Monocular Visual Odometry (MonoVO)](./datasets/mono-vo)
- [NYU Depth V2](./datasets/nyu-depth-v2)
- [Open Images V7](./datasets/open-images-v7)
- [Wikipedia Dumps](./datasets/wikipedia)

## Usage guide

### Setting up a remote

You can set up a default remote from which to get the files with:

```shell
dvc remote add --default <name> <path>
```

See docs for `dvc remote` for more details. This is useful for e.g. operating on
very large datasets.

### Setting up a dataset

You can set up one of the datasets (let's say `imagenet`) by running:

```shell
dvc repro --pull imagenet
```

If you get an error with `"missing data 'source'`, it means that a dependency
needs to be downloaded (e.g. from S3) - `dvc repro`, by default, doesn't set up
files added with `dvc import-url`. To do this for all the files, run:

```shell
find imagenet -name '*.dvc' \
    -exec dvc update {} \;
```

or an equivalent command. If the dataset won't fit on the computer, you can
download the files directly into the remote:

```shell
find imagenet -name '*.dvc' \
    -exec dvc update --to-remote -r "<name of the remote>" {} \;
```
