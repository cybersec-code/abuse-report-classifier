#!/usr/bin/env python3

import os
import logging
import argparse
import pandas as pd

from arc import arc

info = logging.info
debug = logging.debug
warn = logging.warning
error = logging.error

# Default config file
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../data/')
file_config = os.path.join(data_dir, 'prompt.config')

def main():

    prog = 'AbuseReportClassifier'
    version = "1.0.0"
    usage = (
        f"{prog} is a text classifier used to assign an abuse type to abuse"
        "reports written by victims of cryptocurrency abuse, using LLMs."
    )
    changes = (
        f"\n\nReleased version " + version + f"\n"
    )

    parser = argparse.ArgumentParser(description=usage+changes, prog=prog)

    # General arguments
    choices = ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo-preview']

    parser.add_argument('-v', '--version', action='version', version=version)

    parser.add_argument('-C', '--config', type=str, default=file_config,
            help=f"Path to the config file (default={file_config})")

    parser.add_argument('-D', '--dataset', type=str, default=None,
            help=f"Path to a dataset to classify")

    parser.add_argument('-N', '--name', default=None,
            help=f"Name for the resulting dataset")

    parser.add_argument('-S', '--simplify', action="store_true",
            help=f"Simplify the dataset transforming non-UNICODE chars")

    parser.add_argument('-m', '--model', default='gpt-4o-mini', choices=choices,
            help='GPT model in short name (default=gpt-4o-mini)')

    parser.add_argument('-c', '--classify', default='top-to-bottom',
            choices=['top-to-bottom', 'bottom-to-top', 'one-query', 'ablation'],
            help='Classification technique (default=top-to-bottom)')

    parser.add_argument('-o', '--output', type=str, default='./_arc_',
            help='Output files will share this name')

    parser.add_argument('-t', '--text', type=str, default=None,
            help='Classify this text')

    parser.add_argument('-T', '--text-file', type=str, default=None,
            help='Classify this text file')

    parser.add_argument('-e', '--evaluate', action="store_true",
            help='Evaluate the provided dataset (category/predicted columns)')

    # Parse arguments
    args = parser.parse_args()

    if args.output:
        odir = os.path.dirname(args.output)
        odir = odir if odir else './'
        if not os.path.isdir(odir):
            parser.error("Directory not found: %s" % odir)
        elif not os.access(odir, os.W_OK):
            parser.error(f"No write permissions for %s" % odir)

    log_format = '[ARC|%(levelname)s]: %(message)s'
    fname = f"{args.output}.log"
    logging.basicConfig(filename=fname, format=log_format, level=logging.DEBUG)
    # Turn-off matplotlib debug messages
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)
    ######
    debug(f"%s version %s", prog, version)
    debug(args)

    e = set()

    valid_config_path = args.config and os.path.isfile(args.config)
    if not valid_config_path:
        e.add("Config file not found.")

    if args.evaluate and not args.dataset:
        e.add("Must specify a dataset to evaluate")

    elif args.dataset:

        if not os.path.isfile(args.dataset):
            e.add("File not found: %s" % args.dataset)
        else:
            try:
                examples = pd.read_csv(args.dataset)
                if args.name is None:
                    dataset_name = os.path.basename(args.dataset).split('.')[0]
                    args.name = dataset_name
                args.dataset = examples
            except Exception as ex:
                e.add("Not a valid dataset: %s (%s)" % (args.dataset, ex))

    if args.text_file and not os.path.isfile(args.text_file):
        e.add("File not found: %s" % args.text_file)
    elif args.text_file and os.path.isfile(args.text_file):
        try:
            with open(args.text_file) as f:
                text = '\n'.join(f.readlines())

            args.text = text if args.text in [None, ''] else f"{args.text}\n{text}"

        except Exception as ex:
            e.add("Couldn't read the text file %s: %s" % args.text_file, ex)

    if args.text is None and args.dataset is None:
        e.add("Use either --text or --text-file or --dataset to classify some")

    if e:
        parser.error("\n".join(list(e)))
    else:
        arc.process(args)

if __name__ == "__main__":
    main()
