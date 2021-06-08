#!/usr/bin/env python3

import os
from pong_screen_gen import PGen
from pong_runner import PRun

# Example running Pong screen generation, training, testing, and playing


if __name__ == "__main__":
    # Setup
    path_parent = "/projects/pong_example"
    path_train = os.path.join(path_parent, "train")
    path_test = os.path.join(path_parent, "test")
    path_models = os.path.join(path_parent, "models")
    model_basename = "mymodel_01"
    model_file = os.path.join(path_models, model_basename + "_model.pt")

    # Create directories
    if not os.path.isdir(path_parent):
        os.mkdir(path_parent)

    if not os.path.isdir(path_models):
        os.mkdir(path_models)

    # Generate some screens for training
    # [up threshold, down threshold, number of games]
    # Only if directory not present
    if not os.path.isdir(path_train):
        os.mkdir(path_train)
        dtrain = [[3, 5, 75], [2, 4, 75], [2, 5, 75]]
        print("Training screens being written to " + path_train)
        for args in dtrain:
            print(
                "Training - up:"
                + str(args[0])
                + " down:"
                + str(args[1])
                + " games:"
                + str(args[2])
            )
            pgener = PGen(args[0], args[1], args[2], path_train)
            pgener.generate()

    # # Generate some screens for testing
    # [up threshold, down threshold, number of games]
    # Only if directory not present
    if not os.path.isdir(path_test):
        os.mkdir(path_test)
        dtest = [[5, 5, 25]]
        print("Testing screens being written to " + path_test)
        for args in dtest:
            print(
                "Testing - up:"
                + str(args[0])
                + " down:"
                + str(args[1])
                + " games:"
                + str(args[2])
            )
            pgener = PGen(args[0], args[1], args[2], path_test)
            pgener.generate()

    # Train the model
    print("Training model to " + path_models)
    prunner = PRun(screenscale=2, showtesting=False)
    if (
        prunner.setvars(
            subcommand="train",
            numpoints=500,
            loadpath=path_train,
            learnrate=0.001,
            savepath=path_models,
            savebasename=model_basename,
            randfiles=True,
            forcecpu=True,
        )
        == False
    ):
        # Method will output error
        exit
    if prunner.run() == False:
        # Method will output error
        exit

    # Test the model
    print("Testing model " + model_file)
    prunner = PRun(screenscale=2, showtesting=False)
    if (
        prunner.setvars(
            subcommand="test",
            numpoints=25,
            loadpath=path_test,
            loadmodel=model_file,
            randfiles=True,
            forcecpu=True,
        )
        == False
    ):
        # Method will output error
        exit
    if prunner.run() == False:
        # Method will output error
        exit

    print("Play model " + model_file)
    prunner = PRun(screenscale=2, showtesting=False)
    if (
        prunner.setvars(
            subcommand="play",
            numgames=2,
            loadmodel=model_file,
            rendergame=True,
            forcecpu=False,
        )
        == False
    ):
        # Method will output error
        exit
    if prunner.run() == False:
        # Method will output error
        exit
    print(
        f'Won: {prunner.results["correct"]:6d}  Lost: {prunner.results["incorrect"]:6d}  Percent Won: {prunner.results["perccorrect"]:3.3f}'
    )
