#!/usr/bin/env python3

import argparse
import gym
import numpy as np
import os
import sys
from time import time

# Generates screens for Pong training and testing


class PGen:
    def __init__(self, upthreshold, dnthreshold, numgames, svparentpath=None):
        self.d1 = upthreshold
        self.d2 = dnthreshold
        self.numgames = numgames
        self.svparentpath = svparentpath

    def findme(self, monochromefield):
        # 140 to 143 is our paddle
        subfield = monochromefield[:, 140:]
        locs = list(np.where(subfield != 0))
        x = locs[0]
        if len(x) == 0:
            return None
        return x[0]

    def findball(self, monochromefield):
        # Ballfield - no paddles
        subfield = monochromefield[:, 20:140]
        locs = list(np.where(subfield != 0))
        x = locs[0]
        if len(x) == 0:
            return None
        return x[0]

    def reformat(self, x1):
        # Summing all the colors, shape is (210,160)
        y1 = x1.sum(axis=2)

        # True/False, background is False, 233 is background color
        y2 = y1 != 233
        y3 = y2.astype(int)
        y4 = y3[34:194, :]
        return y4

    def generate(self):
        # Setup save paths if needed
        self.savepath = None
        self.epoch = int(time())
        if self.svparentpath is not None:
            self.savepath = os.path.join(self.svparentpath, f"{self.epoch:010}")
            os.mkdir(self.savepath)

        # So far, no frame skip has performed better than the standard Pong-v0
        env = gym.make("PongNoFrameskip-v0")
        pointsfortot = pointsvstot = 0

        for gamenum in range(1, self.numgames + 1):
            pointsfor = pointsvs = vollystot = 0
            mvright = mvleft = mvstay = 0

            x1 = env.reset()
            pointframes = []

            act = env.action_space.sample()
            x1, x2, x3, _ = env.step(act)
            y = self.reformat(x1)

            # Point loop
            while True:
                me = self.findme(y)
                ball = self.findball(y)

                if ball != None and me != None:
                    # Find distance between our paddle and ball
                    diff = ball - me

                    # Figure out which move to make
                    if diff < self.d1:
                        act = 2  # move right (UP)
                        mvright += 1
                    elif diff > self.d2:
                        act = 3  # move left (DOWN)
                        mvleft += 1
                    else:
                        act = 0
                        mvstay += 1
                else:
                    act = 0
                    mvstay += 1

                # Store this frame and action performed
                pointframes.append([y, act])
                env.render()

                # Perform the action
                x1, x2, x3, _ = env.step(act)

                # Point over?
                if int(x2) != 0:
                    vollystot += len(pointframes)

                    if x2 < 0:
                        # Did not win the point - throw it away
                        pointsvs += 1
                        pointframes = []
                    else:
                        # We won the point - yay!
                        pointsfor += 1

                        if self.savepath is not None:
                            filename = f"pongv0_{self.epoch:010}_{self.d1:02}_{self.d2:02}_{gamenum:05}_{pointsfor:02}.npy"
                            pointframesnp = np.array(pointframes, dtype=object)
                            np.save(
                                os.path.join(self.savepath, filename),
                                pointframesnp,
                                allow_pickle=True,
                                fix_imports=False,
                            )

                        pointframes = []

                    # End of the game?
                    if x3 == True:
                        pointsfortot += pointsfor
                        pointsvstot += pointsvs
                        rat = pointsfortot / (pointsfortot + pointsvstot)
                        print(
                            f"{pointsfor:6d} {pointsvs:6d} {pointsfortot:9d} {pointsvstot:6d} {rat:3.3f} {vollystot:9d} {mvright:4d} {mvleft:4d} {mvstay:4d} {gamenum:5d}"
                        )
                        break
                y = self.reformat(x1)

        env.close()


if __name__ == "__main__":
    # Setup and parse command line arguments
    parser = argparse.ArgumentParser(description="Generate Pong screens")
    parser.add_argument(
        "up_threshold", type=int, help="Threshold (less than) to move up"
    )
    parser.add_argument(
        "dn_threshold", type=int, help="Threshold (greater than) to move down"
    )
    parser.add_argument("num_games", type=int, help="Number of games ot play")
    parser.add_argument(
        "--savepath",
        help="Directory to save screen sequences (will create a subfolder per game)",
    )
    args = parser.parse_args()

    # Instantiate a generator
    pgener = PGen(args.up_threshold, args.dn_threshold, args.num_games, args.savepath)

    # Generate screens
    pgener.generate()
