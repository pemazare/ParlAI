from argparse import ArgumentParser
import json

from parlai.projects.metadialog.utils import (
    Parley,
    extract_parlai_episodes,
    add_person_tokens,
)

def setup_args():
    argparser = ArgumentParser()
    argparser.add_argument('-if', '--infile', type=str)
    argparser.add_argument('-of', '--outfile', type=str)
    config = vars(argparser.parse_args())

    return config

def main(config):
    """Creates .unfiltered files from .sliced files

    input: a .sliced file of logs (in ParlaiDialog format) from Mturk task 1, each of
        which starts with an initial prompt or topic request, and ends with a y_exp
    output: a .unfiltered file (in metadialog format) with every utterance output by
        bot used as a label (i.e., act as though the bot was a human and we want to
        train in a normal supervised way).
    """
    examples = []
    episodes = [e for e in extract_parlai_episodes(config['infile'])]
    for episode in episodes:
        history = []
        num_parleys = len(episode)
        for i, parley in enumerate(episode):
            if i == 0:  # Don't include the topic request
                history.append(parley.response)
                continue
            elif i == num_parleys - 1:
                # ultimate turn was correction request and explanation
                continue
            else:
                example = Parley(
                    context=add_person_tokens(history, last_speaker=1),
                    response=parley.context,  # What the bot said
                )
                examples.append(example)
                history.append(parley.context)
                history.append(parley.response)

    # Write new episodes to metadialog format
    with open(config['outfile'], 'w') as outfile:
        for ex in examples:
            outfile.write(json.dumps(ex.to_dict()) + '\n')

    print(f"Extracted {len(examples)} metadialog episodes out of "
          f"{len(episodes)} parlai episodes and wrote them to {config['outfile']}.")

if __name__ == '__main__':
    config = setup_args()
    assert(config['infile'].endswith('.sliced'))
    assert(config['outfile'].endswith('.unfiltered'))
    main(config)
