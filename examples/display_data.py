# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Basic example which iterates through the tasks specified and prints them out.
Used for verification of data loading and iteration.

For example, to make sure that bAbI task 1 (1k exs) loads one can run and to
see a few of them:
`python examples/display_data.py -t babi:task1k:1`
"""

from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task

import json
import logging
import random
import sys

logger = logging.getLogger()

def display_data(opt):
    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)

    try:
        # print dataset size if available
        print('[loaded {} episodes with a total of {} examples]'.format(
            world.num_episodes(), world.num_examples()
        ))
    except KeyboardInterrupt:
        print('Interrupted manually')
        pass

    # Show some example dialogs.
    with world:
        for _ in range(opt['num_examples']):
            world.parley()
            print(world.display() + '\n~~')
            if world.epoch_done():
                print('EPOCH DONE')
                break


def main():
    random.seed(42)

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))


    # Get command line arguments
    parser = ParlaiParser()
    parser.add_argument('-n', '--num-examples', default=10, type=int)
    opt = parser.parse_args()
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(opt, indent=4, sort_keys=True))

    display_data(opt)

if __name__ == '__main__':
    main()
