#!/usr/bin/env python3

import os
import numpy as np
from random import shuffle
import sys
import torch
import torch.nn as nn
from pong_net_lstm import Net

# Pong player - holds state and configures network


class Player:
    def __init__(self):

        # Number of images per sequence (current + N previous)
        self.numimgseq = 2

        # Holds image sequences per point
        self.mem = []

        # Holds all loaded screens
        self.alldata = []

    def initnet(
        self,
        num_classes,
        imgw,
        imgh,
        hidden_size,
        num_layers,
        dropout,
        learnrate,
        modelfile=None,
        forcecpu=False,
    ):
        self.net = Net(
            num_classes=num_classes,
            imgw=imgw,
            imgh=imgh,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            numimgseq=self.numimgseq,
            forcecpu=forcecpu,
        )

        # Learning rate
        self.alpha = learnrate

        self.dev = self.net.getdev()
        self.criterion = nn.BCELoss()
        self.optim = torch.optim.AdamW(self.net.parameters(), lr=self.alpha)

        # Holds current loss / error
        self.currloss = 0

        # Load the model if passed
        if modelfile != None:
            print("Loading model from " + modelfile)
            self.net.load_state_dict(torch.load(modelfile))

        print(self.net)

    def onehotencode(self, act):
        # Action:
        # 0 - stay
        # 2 - right / up
        # 3 - left / down
        if act >= 2:
            act -= 1
        ret = [0, 0, 0]
        ret[act] = 1
        return ret

    def decode(self, arg):
        # Decode from 0 - 2 to 0, 2, 3
        if arg > 0:
            arg += 1
        return arg

    def loaddata(self, loadpath, randkeys, numpointsload):
        # Load Pong screen sequences
        allfiles = {}
        for (dirpath, _, filenames) in os.walk(loadpath):
            if len(filenames) >= 0:
                for filename in filenames:
                    # Format is pongv0_1622112535_02_06_00001_01.npy
                    (_, nmrundate, _, _, nmgame, nmpoint) = (
                        filename.rpartition(".")[0]
                    ).split("_")
                    key = nmrundate + "_" + nmgame + "_" + nmpoint
                    fullpath = os.path.join(dirpath, filename)

                    if key not in allfiles:
                        allfiles[key] = fullpath

        # Hopefully we have some data to load
        if len(allfiles) == 0:
            print("No files found in " + loadpath)
            return False

        # Pull out keys - maybe shuffled
        keys = list(allfiles.keys())

        # Randomize keys if desired
        if randkeys == True:
            shuffle(keys)

        # Pull out number of request keys / sequences
        for i, key in enumerate(keys):
            fullpath = allfiles[key]

            data = np.load(fullpath, allow_pickle=True)
            datanew = []

            # Change action to one host encoding
            for (screen, action) in data:
                datanew.append([screen, self.onehotencode(action)])

            self.alldata.append(datanew)

            if len(self.alldata) % 100 == 0:
                print("Loaded so far", len(self.alldata))

            if numpointsload != -1 and i == (numpointsload - 1):
                break

        return True

    def datasize(self):
        # Return the number of point image sequences loaded
        return len(self.alldata)

    def store(self, x, act):
        # Store scrren and action taken
        self.mem.insert(0, [x, self.onehotencode(act)])

    def storesize(self):
        # Return the number of screens
        return len(self.mem)

    def getalpha(self):
        # Return current alpha / learning rate
        return self.optim.param_groups[0]["lr"]

    def replayscreens(self, meminputs, memanswers, mode="train"):
        # Remove the first screen(s) so the total is mod of the numimgseq
        leftovr = len(meminputs) % self.numimgseq
        meminputs = meminputs[leftovr:]
        memanswers = memanswers[leftovr:]

        if len(meminputs) != len(memanswers):
            # Something is amiss
            print("Input and answers arrays length do not match")
            sys.exit(1)

        # Loop through the inputs and
        # create a new array of batches of screen sequences
        # and coresponding answers
        allbatchesinp = []
        allbatchesans = []
        for i, inp in enumerate(meminputs):
            if i + (self.numimgseq - 1) < len(meminputs):
                tmp = []
                tmpa = []
                tmp.append(inp)
                tmpa.append(memanswers[i])
                for j in range(1, self.numimgseq):
                    tmp.append(meminputs[i + j])
                    tmpa.append(memanswers[i + j])
                allbatchesinp.append(tmp)

                # Use the last answer
                allbatchesans.append(tmpa[-1])

        return self.sendtomodel(allbatchesinp, allbatchesans, mode=mode)

    def sendtomodel(self, inputs, answers, mode="train"):
        # Send batches of screen sequences to the model
        # Shape like (131, 3, 19840)
        inputs = torch.from_numpy(np.array(inputs, dtype=np.float)).float().to(self.dev)

        # Shape like (131, 3)
        answers = (
            torch.from_numpy(np.array(answers, dtype=np.float)).float().to(self.dev)
        )

        # Send to the net
        output = self.net.forward(inputs)

        if mode == "train":
            loss = self.criterion(output, answers)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            # For checking / display
            self.currloss = loss.item()

        # Get right / wrong
        right = wrong = 0
        for i, outitem in enumerate(output):
            guess = torch.argmax(outitem).item()
            answer = torch.argmax(answers[i]).item()

            if answer == guess:
                right += 1
            else:
                wrong += 1

        return right, wrong

    def getmove(self, inputs):
        # Very similar to sendtomodel
        # Send to net and return move
        mv = 0
        inputs = torch.from_numpy(np.array(inputs, dtype=np.float)).float().to(self.dev)
        output = self.net.forward(inputs)

        # Grab the last output and convert to expect action number (0, 2, or 3)
        mv = self.decode(torch.argmax(output[-1]).item())
        return mv

    def savenet(self, svpath, filenamebase):
        # Save model and optim
        torch.save(
            self.net.state_dict(), os.path.join(svpath, filenamebase + "_model.pt")
        )
        # Not used yet
        # torch.save(self.optim.state_dict(), os.path.join(svpath, filenamebase + "_optim.pt"))


if __name__ == "__main__":
    pass
