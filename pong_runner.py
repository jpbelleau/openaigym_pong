#!/usr/bin/env python3

import argparse
import gym
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw, pause
from random import choice
from pong_player import Player

# Runs Pong training, testing, and playing

# Step returns:
#     Num   Description
#     0     Observation - screen (210, 160, 3)
#     1     Reward (-1 [lost point], 0, or 1 [won point])
#     2     Game over (boolean)
#     3     {'ale.lives': 0}

# Actions:
#     Type: Discrete(5)
#     Num   Action
#     0     NOOP
#     1     FIRE
#     2     RIGHT (for Pong - up)
#     3     LEFT (for Pong - down)
#     4     RIGHTFIRE
#     5     LEFTFIRE


class PRun:
    def __init__(self, screenscale=2, showtesting=False):
        # Reduce screen dimensions by this scale factor
        # Reduce details for processing speed
        self.screenscale = screenscale

        # Show screens while running
        self.showtesting = showtesting

        # Number of output classes
        self.num_classes = 3

        # Number of stacked LSTM layers
        self.num_layers = 1

        # Zero disables
        self.dropout = 0

        # How many frames to run before training / playing
        # At the start of a game
        self.startup_frames = 10

        # Current results - correct == wins / incorrect == losses
        self.results = {"correct": 0, "incorrect": 0, "perccorrect": 0, "loss": 0.0}

    def reformat(self, x1, fullscreen=False):
        # Screens from files are partly reformatted already

        if fullscreen == True:
            # Trim based on fullscreen
            # Summing all the colors, shape is (210,160)
            y1 = x1.sum(axis=2)

            # True/False, background is False, 233 is background color
            y2 = y1 != 233
            y3 = y2.astype(int)
            x1 = y3[
                34:194,
            ]

        y4 = x1[:, 16:144]  # original shape is 160,128 - this has opp paddle
        # y4 = x1[:,20:144]   # shape is 160,124 - no opp paddle
        r, c = y4.shape
        y5 = y4.reshape(
            r // self.screenscale,
            self.screenscale,
            c // self.screenscale,
            self.screenscale,
        )
        y6 = y5.transpose([0, 2, 1, 3])
        y7 = y6.sum(axis=(2, 3))
        y8 = y7 != 0
        y9 = y8.astype(float)
        return y9

    def startup(self, env, n, player, reset=True, lastscreen=None):
        # Run a few frames (for the beginning of the point)
        # Keep these for normalizing the number of sequences later

        # Only reset if no previous screen (new game)
        if reset == False:
            x1 = lastscreen
        else:
            x1 = env.reset()

        for _ in range(n):
            act = choice([0, 2, 3])
            # Associate the screen we saw with the action taken
            player.store(self.reformat(x1, fullscreen=True), act)
            x1, _, _, _ = env.step(act)

        # Return the last screen
        return x1

    def setvars(
        self,
        subcommand,
        numpoints=0,
        loadpath=None,
        learnrate=0.001,
        savepath=None,
        savebasename=None,
        loadmodel=None,
        randfiles=False,
        numgames=0,
        rendergame=False,
        forcecpu=False,
    ):
        # Bring in settings
        self.subcommand = subcommand
        self.numpoints = numpoints
        self.loadpath = loadpath
        self.learnrate = learnrate
        self.savepath = savepath
        self.savebasename = savebasename
        self.loadmodel = loadmodel
        self.randfiles = randfiles
        self.numgames = numgames
        self.rendergame = rendergame
        self.forcecpu = forcecpu

        self.numepochs = 0

        if self.numpoints > 0:
            self.numepochs = self.numpoints
        elif self.numgames > 0:
            self.numepochs = self.numgames
        else:
            print("Both number of points and number of games cannot be zero")
            return False
        return True

    def run(self):
        # Run the proper function based on the subcommand

        # So far, no frame skip has performed better than the standard
        # In test / train this is used to get the original screen size
        env = gym.make("PongNoFrameskip-v0")

        # Instantiate player
        pplayer = Player()
        numimgseq = pplayer.numimgseq

        if self.startup_frames < numimgseq:
            print(
                "Startup frames needs to be larger than the number of sequences per batch"
            )
            return False

        screen = self.startup(env, self.startup_frames, pplayer)
        imgw = pplayer.mem[0][0].shape[0]
        imgh = pplayer.mem[0][0].shape[1]

        if self.subcommand != "play":
            env.close()

        hidden_size = int(imgw * imgh)
        pplayer.initnet(
            num_classes=self.num_classes,
            imgw=imgw,
            imgh=imgh,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            learnrate=self.learnrate,
            modelfile=self.loadmodel,
            forcecpu=self.forcecpu,
        )

        if self.showtesting == True:
            im = plt.imshow(self.reformat(screen, fullscreen=True))

        if self.subcommand != "play":
            # Load training data from files
            if pplayer.loaddata(self.loadpath, self.randfiles, self.numpoints) == False:
                return False

        righttot = wrongtot = 0
        pointsfortot = pointsvstot = 0

        if self.subcommand == "train":
            print(
                "Num point    Right   Wrong   Ratio   RightTotal   WrongTotal   RatioTotal   Alpha    Loss"
            )
        elif self.subcommand == "train":
            print(
                "Num point    Right   Wrong   Ratio   RightTotal   WrongTotal   RatioTotal"
            )
        elif self.subcommand == "play":
            print(
                "Num game    Me   Opp   MeTotal   OppTotal   RatioTotal  MvHoldTotal   MvUpTotal   MvDownTotal"
            )

        # Loop per point (train / test) or game (play) depending on command
        for i in range(self.numepochs):
            numpoints = i + 1

            if self.subcommand != "play":
                # Prepare data
                # Need to separate inputs (screens) and answers (actions)
                meminputs = []
                memanswers = []

                for (img, act) in pplayer.alldata[i]:
                    if self.showtesting == True:
                        im.set_data(img)
                        draw()
                        pause(0.001)

                    meminputs.append(self.reformat(img))
                    memanswers.append(act)

                # Run this point screen sequence
                right, wrong = pplayer.replayscreens(
                    meminputs, memanswers, mode=self.subcommand
                )

                righttot += right
                wrongtot += wrong

                rat = right / (right + wrong)
                rattot = righttot / (righttot + wrongtot)

                if self.subcommand == "train":
                    print(
                        f"{numpoints:5d} {right:6d} {wrong:6d} {rat:3.3f} {righttot:9d} {wrongtot:6d} {rattot:5.3f} {pplayer.getalpha():6.3f} {pplayer.currloss:6.3f}"
                    )
                    self.results["loss"] = pplayer.currloss
                elif self.subcommand == "test":
                    print(
                        f"{numpoints:5d} {right:6d} {wrong:6d} {rat:3.3f} {righttot:9d} {wrongtot:6d} {rattot:5.3f}"
                    )

                self.results["correct"] = right
                self.results["incorrect"] = wrong
                self.results["perccorrect"] = rat
            else:
                # Gotta be play
                pointsfor = pointsvs = 0
                movedict = {"net": [0, 0, 0]}

                # Point loop
                while True:
                    screen = self.reformat(screen, fullscreen=True)

                    # Need to combine into a sequence
                    # Create as a subarray of a sequence of images
                    # Basically make this a batch size of 1
                    mvimgseq = []
                    mvimgseq.append([])
                    mvimgseq[0].append(screen)
                    for i in range(numimgseq - 1):
                        mvimgseq[0].insert(0, (pplayer.mem[i][0]))

                    if self.showtesting == True:
                        for img in mvimgseq:
                            im.set_data(img)
                            draw()
                            pause(0.001)

                    # Get the move form the neural net
                    act = pplayer.getmove(mvimgseq)

                    # Keeping track for how many hold, up, down actions
                    if act > 1:
                        movedict["net"][act - 1] += 1
                    else:
                        movedict["net"][act] += 1

                    # Store this screen to be used in future sequences
                    pplayer.store(screen, act)

                    # Render game if desired
                    if self.rendergame == True:
                        env.render()

                    # Send our action to the env and get the output screen
                    screen, x2, x3, _ = env.step(act)

                    # Point over?
                    if int(x2) != 0:
                        # End of the game?
                        if x3 == False:
                            if x2 < 0:
                                pointsvs += 1
                            else:
                                pointsfor += 1

                        # Clear the screen memory
                        pplayer.mem = []

                        print(
                            f'{numpoints:5d} {pointsfor:6d} {pointsvs:6d} ------ ------ --- {movedict["net"][0]:4d} {movedict["net"][1]:4d} {movedict["net"][2]:4d}'
                        )

                        # Prep for the next point
                        screen = self.startup(
                            env, numimgseq, pplayer, reset=False, lastscreen=screen
                        )

                        # End of the game?
                        if x3 == True:
                            pointsfortot += pointsfor
                            pointsvstot += pointsvs
                            rat = pointsfortot / (pointsfortot + pointsvstot)
                            print(
                                f'{numpoints:5d} {pointsfor:6d} {pointsvs:6d} {pointsfortot:9d} {pointsvstot:6d} {rat:3.3f} {movedict["net"][0]:4d} {movedict["net"][1]:4d} {movedict["net"][2]:4d}'
                            )
                            self.results["correct"] = pointsfortot
                            self.results["incorrect"] = pointsvstot
                            self.results["perccorrect"] = rat
                            break

                # Prepare for next game
                screen = self.startup(env, numimgseq, pplayer)

        if self.savepath != None:
            # Save the model
            pplayer.savenet(self.savepath, self.savebasename)

        if self.rendergame == True:
            env.close()

        return True


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Runs Pong training, testing, and playing"
    )

    subparsers = parser.add_subparsers(help="Mode subcommands", dest="subcommand")

    # Subcommand train
    parser_train = subparsers.add_parser("train", help="Training commands")
    parser_train_r = parser_train.add_argument_group("required arguments")

    parser_train_r.add_argument(
        "--numpoints",
        required=True,
        type=int,
        help="Number of point files to use for training",
    )
    parser_train_r.add_argument(
        "--loadpath", required=True, help="Directory to load screen sequences"
    )
    parser_train_r.add_argument(
        "--learnrate", required=True, type=float, help="Learning rate"
    )
    parser_train.add_argument("--savepath", help="Directory to save model")
    parser_train.add_argument(
        "--savebasename", help="Base file name for model and optium saves"
    )
    parser_train.add_argument(
        "--randfiles", action="store_true", help="Randomize file selection"
    )
    parser_train.add_argument("--forcecpu", action="store_true", help="Only use CPU")

    # Subcommand test
    parser_test = subparsers.add_parser("test", help="Testing commands")
    parser_test_r = parser_test.add_argument_group("required arguments")

    parser_test_r.add_argument(
        "--numpoints",
        required=True,
        type=int,
        help="Number of point files to use for testing",
    )
    parser_test_r.add_argument(
        "--loadpath", required=True, help="Directory to load screen sequences"
    )
    parser_test_r.add_argument(
        "--loadmodel", required=True, help="Full path to model to test"
    )
    parser_test.add_argument(
        "--randfiles", action="store_true", help="Randomize file selection"
    )
    parser_test.add_argument("--forcecpu", action="store_true", help="Only use CPU")

    # Subcommand play
    parser_play = subparsers.add_parser("play", help="Play commands")
    parser_play_r = parser_play.add_argument_group("required arguments")
    parser_play_r.add_argument(
        "--numgames", required=True, type=int, help="Number of games to play"
    )
    parser_play_r.add_argument(
        "--loadmodel", required=True, help="Full path to model to play"
    )
    parser_play.add_argument(
        "--rendergame", action="store_true", help="Render the game while playing"
    )
    parser_play.add_argument("--forcecpu", action="store_true", help="Only use CPU")

    args = parser.parse_args()
    dargs = vars(args)

    # Instantiate a runner
    prunner = PRun(screenscale=2, showtesting=False)

    # Setup variables
    if prunner.setvars(**dargs) == False:
        exit

    # Run the command
    if prunner.run() == False:
        exit
