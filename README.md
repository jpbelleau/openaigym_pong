# OpenAI GYM - Pong Example

Example running screen generation, training, testing, and playing
Pong using an [OpenAI Gym](https://gym.openai.com/envs/Pong-v0/) environment.

## Installation

Tested with Python 3.8 and NVIDIA GeForce RTX 2070 SUPER GPU.
The version off `torch` included in the requirements file may need to be changed depending on your setup.

```bash
git clone https://github.com/jpbelleau/openaigym_pong.git
cd openaigym_pong
pip install -r requirements.txt
```

## Neural Network

A Long short-term memory (LSTM) neural network is used along with
two fully connected layers.

```bash
Net(
  (lstm): LSTM(5120, 5120, batch_first=True)
  (fc1): Linear(in_features=5120, out_features=1689, bias=True)
  (fcO): Linear(in_features=1689, out_features=3, bias=True)
  (relu): ReLU()
)
```

Output nodes are treated as a one-hot encoded array (hold, up, down)

## Usage

The main components are:

* pong_screen_gen.py - generates screens files for training and testing

* pong_runner.py - executes training, testing, and playing a neural network model

* pong_example.py - shows calling as modules instead of the command line

### pong_screen_gen.py

Generates screen files for training and testing saved per won point.
File sizes can vary from 25MB to over 600MB.
Thresholds define when to move (up or down) and the wider the spread the
better the chance the paddle will stay in the same position.
Screens are converted to monochrome and the top and bottom are cropped.
Additional screen processing occurs in other scripts and the generation is left at
the minimum to allow flexibility to change later.

```bash
usage: pong_screen_gen.py [-h] [--savepath SAVEPATH] up_threshold dn_threshold num_games

positional arguments:
  up_threshold         Threshold (less than) to move up
  dn_threshold         Threshold (greater than) to move down
  num_games            Number of games ot play

optional arguments:
  -h, --help           show this help message and exit
  --savepath SAVEPATH  Directory to save screen sequences (will create a subfolder per game)
```

### pong_runner.py

Executes training, testing, and playing the neural network model.
Screens are further modified by cropping the left and right sides
and scaled down by a factor of 2.
The main subcommands are: train, test, and play.

```bash
usage: pong_runner.py [-h] {train,test,play} ...

positional arguments:
  {train,test,play}  Mode subcommands
    train            Training commands
    test             Testing commands
    play             Play commands

optional arguments:
  -h, --help         show this help message and exit
```

#### train

Trains a neural network model based on generated Pong screens.

```bash
usage: pong_runner.py train [-h] --numpoints NUMPOINTS --loadpath LOADPATH --learnrate LEARNRATE [--savepath SAVEPATH] [--savebasename SAVEBASENAME] [--randfiles] [--forcecpu]

optional arguments:
  -h, --help                   show this help message and exit
  --savepath SAVEPATH          Directory to save model
  --savebasename SAVEBASENAME  Base file name for model and optium saves
  --randfiles                  Randomize file selection
  --forcecpu                   Only use CPU

required arguments:
  --numpoints NUMPOINTS        Number of point files to use for training
  --loadpath LOADPATH          Directory to load screen sequences
  --learnrate LEARNRATE        Learning rate
```

#### test

Tests a neural network model based on generated Pong screens.

```bash
usage: pong_runner.py test [-h] --numpoints NUMPOINTS --loadpath LOADPATH --loadmodel LOADMODEL [--randfiles] [--forcecpu]

optional arguments:
  -h, --help             show this help message and exit
  --randfiles            Randomize file selection
  --forcecpu             Only use CPU

required arguments:
  --numpoints NUMPOINTS  Number of point files to use for testing
  --loadpath LOADPATH    Directory to load screen sequences
  --loadmodel LOADMODEL  Full path to model to test
```

#### play

Plays Pong using a neural network model.

```bash
usage: pong_runner.py play [-h] --numgames NUMGAMES --loadmodel LOADMODEL [--rendergame] [--forcecpu]

optional arguments:
  -h, --help             show this help message and exit
  --rendergame           Render the game while playing
  --forcecpu             Only use CPU

required arguments:
  --numgames NUMGAMES    Number of games to play
  --loadmodel LOADMODEL  Full path to model to play
```

### pong_example.py

Example running Pong screen generation, training, testing, and playing
based on calling as modules.  Script will need to be modified for your
environment.

```bash
usage: pong_example.py
```
