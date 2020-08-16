# theedhum-nandrum (தீதும் நன்றும்)
A sentiment classifier on mixed language (and mixed script) reviews in Tamil, Malayalam and English

## Installation
## Pre-requisites
* Python 3.7 or above
## Getting the code
----------------
* `cd /path/to/parent/`
* `git clone https://github.com/oligoglot/theedhum-nandrum.git`
* `cd theedhum-nandrum`

## Setting up dev environment
----------------
* `virtualenv venv_tn`
* `source venv_tn/bin/activate`
* `pip install -r requirements.txt `

## How to run playground files
* You need to activate the virtualenv
    * `source venv_tn/bin/activate`
* `cd src/playground`
* `python classify.py`

# Steps
## Pre-processing
### Noise removal
1. Remove irrelevant parts of the data, like html tags

### Language identification
1. If the text is a different language, need to output "Not tamil"

# Attributions
1. Spelling Corrector in Python 3; see http://norvig.com/spell-correct.html
    Copyright (c) 2007-2016 Peter Norvig
    MIT license: www.opensource.org/licenses/mit-license.php
2. Module to convert Unicode Emojis to corresponding Sentiment Rankings.
    Based on the research by Kralj Novak P, Smailović J, Sluban B, Mozetič I
    (2015) on Sentiment of Emojis.
    Journal Link:
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0144296
    CSV Data acquired from CLARIN repository,
    Repository Link: http://hdl.handle.net/11356/1048
