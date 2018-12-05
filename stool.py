#!/usr/bin/python

import sys
import os
import argparse
import itertools
import json
import random
import subprocess


def parse_json(grid):
    perms = list(itertools.product(*grid.values()))
    random.shuffle(perms)
    commands = []

    for p in perms:
        argstr = ""
        for i, k in enumerate(grid.keys()):
            if type(p[i]) is int or type(p[i]) is float:
                v = str(p[i])
            else:
                assert '"' not in p[i]
                v = '"%s"' % p[i]
            key = str(k)
            if len(key) > 2:
                argstr += " --%s %s" % (str(k), v)
            else:
                argstr += " -%s %s" % (str(k), v)
        commands.append(argstr)
    return commands


COMMANDS = ['run', 'info', 'kill']

if (len(sys.argv) < 2) or (sys.argv[1] not in COMMANDS):
    print('Error: recognized commands: %s' % ' '.join(COMMANDS))
    exit(1)
COMMAND = sys.argv[1]
SCRIPT = sys.argv[-1] + ' '

parser = argparse.ArgumentParser(description='slurm jobs')

parser.add_argument('--name', type=str, required=True)
parser.add_argument('--ngpu', type=int, default=8)
parser.add_argument('--nodes', type=int, default=1)
parser.add_argument('--sweep', type=str, default='')
parser.add_argument('--exclude', type=str, default=None)

args = parser.parse_args(sys.argv[2:-1])

if COMMAND == 'info':
    STRING = 'squeue -u `whoami` --name %s' % args.name
    subprocess.call(STRING, shell=True)

if COMMAND == 'kill':
    STRING = 'scancel --name %s' % args.name
    subprocess.call(STRING, shell=True)

if COMMAND == 'run':
    STRING = 'sbatch --partition=learnfair --ntasks-per-node=1 --cpus-per-task=15 --open-mode=append '
    #STRING = 'sbatch --partition=short --nodes=1 --ntasks-per-node=1 --time=5 '
    STRING += '--nodes=%s ' % args.nodes
    STRING += '--job-name=%s ' % args.name
    STRING += '--output=/checkpoint/%%u/%s_%%j.out ' % args.name
    STRING += '--error=/checkpoint/%%u/%s_%%j.err ' % args.name

    if args.ngpu > 0:
        STRING += '--gres=gpu:%s ' % args.ngpu

    if args.exclude is not None:
        STRING += '--exclude=%s ' % args.exclude

    STRING += '--wrap="srun --label %s' % SCRIPT.replace('"', '\\"')

    if args.sweep == '':
        subprocess.call(STRING + '"', shell=True)
    else:
        if os.path.isfile(args.sweep) == False:
            print('Error: sweep file does not exist')
            exit(1)

        with open(args.sweep, 'r') as f:
            sweep_commands = parse_json(json.loads(f.read()))

        #subprocess.call(STRING + sweep_commands[0], shell=True)

        for command in sweep_commands:
            subprocess.call(STRING + command.replace('"', '\\"') + '"', shell=True)
            #print(STRING + command.replace('"', '\\"'))
