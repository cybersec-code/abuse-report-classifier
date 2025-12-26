#!/usr/bin/env python
# coding: utf-8

import os
import re
import time
import json
import hashlib
import logging
import unicodedata
import configparser
import openai
import tiktoken
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

info = logging.info
debug = logging.debug
warn = logging.warning
error = logging.error

class AbuseReportClassifier:
    ''' AbuseReportClassifier: assign an abuse type to abuse reports '''

    @classmethod
    def load_from_configfile(cls, file_config):

        # Load default config
        config = configparser.ConfigParser()
        try:
            config.read(file_config)

            # Load taxonomy and definitions files
            if 'FILES' in config:
                files_conf = config['FILES']
                file_taxonomy = files_conf.get('file_taxonomy')
                file_definitions = files_conf.get('file_definitions')
                file_definitions_other = files_conf.get('file_definitions_other')
            else:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                data_dir = os.path.join(script_dir, '../data/')
                file_taxonomy = os.path.join(data_dir, 'default.taxonomy')
                file_definitions = os.path.join(data_dir, 'definitions.csv')
                file_definitions_other = os.path.join(data_dir, 'definitions_other.csv')

            # Verify some parameters
            columns_must = {'text', 'sha_text'}
            columns_config = set(config['DATASET']['COLUMNS'].split(', '))
            if len(columns_must & columns_config) != len(columns_must):
                missed = columns_must - columns_config
                error("Columns %s are not in the config file." % missed)
                return None

            none = [None, 'None']
            if 'KEY' not in config['GPT'] or config['GPT']['KEY'] in none:
                error('There is no API key.')
                return None

        except Exception as ex:
            error("TagZilla didn't load using %s: %s", file_config, ex)
            return None

        # Try to load AbuseReportClassifier using the given files
        r = cls._load(file_taxonomy, file_definitions, file_definitions_other)

        if r is None:
            return None

        return cls(config, *r)

    @classmethod
    def _load(cls, file_taxonomy, file_definitions, file_definitions_other):

        # Try to load the entries of the taxonomy file
        try:
            with open(file_taxonomy) as f:
                entries = [l.lower().strip() for l in f.readlines()]
        except Exception as ex:
            msg = "Failed to load entries of the taxonomy using file %s: %s"
            error(msg, file_taxonomy, ex)
            return None

        # Build the taxonomy from the file
        tax = defaultdict(dict)
        cat_levels = defaultdict(set)
        for entry in entries:
            entry_labels = entry.split(':')
            entry_levels = len(entry_labels)

            for level in range(1, entry_levels):
                cat_levels[level].add(entry_labels[level])
                # Parent self-reference
                tax[level-1].update({entry_labels[level-1]: entry_labels[level-1]})
                # Reference for all the children
                for i in range(level, entry_levels):
                    tax[level-1].update({entry_labels[i]: entry_labels[level-1]})

        debug("Taxonomy:")
        for level, categories in tax.items():
            debug("Level %s", level)
            for category, parent in categories.items():
                debug("\t%s:%s", category, parent)
        debug("\n")

        # Try to load the definitions
        try:
            data = [
                pd.read_csv(file_definitions),
                pd.read_csv(file_definitions_other)
            ]
            definitions_all = pd.concat(data)
        except Exception as ex:
            msg = "Failed to load definitions using %s and %s: %s"
            error(msg, file_definitions, file_definitions_other, ex)
            return None

        # Build the definitions by levels
        definitions = defaultdict()
        for level, cats in cat_levels.items():
            definitions[level] = {
                e['category']: e['definition']
                for e in definitions_all.to_dict('records')
                    if e['category'] in cat_levels[level]
            }

        for i in definitions:
            level_defs = ', '.join(definitions[i].keys())
            debug("Level %d definitions: %s", i, level_defs)
        debug("\n")

        definitions_all = {
            e['category']: e['definition']
            for e in definitions_all.to_dict('records')
        }
        return tax, definitions, definitions_all

    def __init__(self, config, tax, definitions, definitions_all):
        # Parse GPT section
        # MAX_TOKENS and CTX_WIN are the default for gpt-4o-mini
        self.config = config
        self.KEY = config['GPT']['KEY']
        self.TEMPERATURE = float(config['GPT'].get('TEMPERATURE', 0))
        self.MAX_TOKENS = int(config['GPT'].get('MAX_TOKENS', 16384))
        self.TOP_P = float(config['GPT'].get('TOP_P', 1.0))
        self.FREQUENCY_PENALTY = float(config['GPT'].get('FREQUENCY_PENALTY', 0.0))
        self.PRESENCE_PENALTY = float(config['GPT'].get('PRESENCE_PENALTY', 0.0))
        self.CTX_WIN = int(config['GPT'].get('CTX_WIN', 128000))
        self.model = config['GPT'].get('MODEL', None)
        self.ablation = False
        # Parse FILES section
        self.tax = tax
        self.definitions = definitions
        self.definitions_all = definitions_all
        # Parse DATASET section
        self.COLUMNS = config['DATASET']['COLUMNS'].split(', ')

    ### Chat GPT functions ###

    def chat_gpt_query(self, messages):
        chat = openai.chat
        openai.api_key = self.KEY

        try:
            return chat.completions.create(model=self.model,
                              response_format={"type": "json_object"},
                              messages=messages,
                              temperature=self.TEMPERATURE,
                              max_tokens=self.MAX_TOKENS,
                              # frequency_penalty=self.FREQUENCY_PENALTY,
                              # presence_penalty=self.PRESENCE_PENALTY,
                              timeout=10)
        except openai.OpenAIError as ex:
            error("Exception when using chat_gpt_query (status_code=%s): %s", ex.status_code, ex.message)
            if ex.status_code != 200:
                error("Exiting due status code")
                exit()

    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError as e:
            warn("Model not found: %s. Using o200k_base instead.", self.model)
            encoding = tiktoken.get_encoding("o200k_base")
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def get_real_cost(self, tk):
        if self.model == 'gpt-3.5-turbo':
            tk_cost_input = 0.5/1e6
            tk_cost_output = 1.5/1e6
        elif self.model == 'gpt-4-turbo-preview':
            tk_cost_input = 10/1e6
            tk_cost_output = 30/1e6
        elif self.model == 'gpt-4o':
            tk_cost_input = 2.5/1e6
            tk_cost_output = 10/1e6
        elif self.model == 'gpt-4o-mini':
            tk_cost_input = 0.15/1e6
            tk_cost_output = 0.6/1e6
        else:
            info(f"Unknown model. Setting price to 0.0")
            tk_cost_input = 0
            tk_cost_output = 0
        cost = (tk_cost_input * tk['prompt']) + (tk_cost_output * tk['completion'])
        return cost

    def parse_json_response(self, resp):
        try:
            rcontent = resp.choices[0].message.content.lower()
            j = json.loads(rcontent)
            answer, reasoning = j['answer'], j['reasoning']
        except Exception as ex:
            info(f"Incomplete response?:\n|{rcontent}|")
            finish_reason = resp.choices[0].finish_reason
            info(f"Finish reason: {finish_reason}")
            # rcontent misses ""}
            if len(rcontent.split('"')) == 7:
                rcontent += '"[...]"}'
            # rcontent misses "}
            elif len(rcontent.split('"')) == 8:
                rcontent += '[...]"}'
            # rcontent misses }
            elif len(rcontent.split('"')) == 9:
                rcontent += '}'
            try:
                j = json.loads(rcontent)
                answer, reasoning = j['answer'], j['reasoning']
            except Exception as jex:
                error(f"Falied to re-parse as JSON:\n|{rcontent}|")
                answer = "unknown"
                reasoning = rcontent
                # We raise the exception so the same query is redo?
                # raise jex

        tokens = {
            'completion': resp.usage.completion_tokens,
            'prompt': resp.usage.prompt_tokens,
            'total': resp.usage.total_tokens
        }

        return answer, reasoning, tokens

    def build_ba_prompt(self, user_text, definitions, class_list):
        conf = self.config['GPT.PROMPT.BA']
        body = conf.get('PRIMING')

        if self.ablation:
            question_key = 'GPT.PROMPT.BA.ABLATION'
        else:
            body += ' ' + conf.get('LIST_DEFINITIONS').format(definitions)
            question_key = 'GPT.PROMPT.BA'

        question = self.config[question_key]['QUESTION'].format(class_list)
        question += ' ' + conf.get('STRUCTURE')

        prompt = conf.get('PROMPT').format(body, user_text, question)
        prompt = prompt.replace('\\n', '\n')
        debug("PROMPT built: \n%s", prompt)

        return prompt

    def direct_class_w_mult_def(self, abuse_definitions: dict, user_text: str,
                                debug=False) -> (str,str,int):

        definitions = ''
        class_list = ''
        for a, d in abuse_definitions.items():
            definitions += f'''{a}: {d}\n\n'''
            class_list += f'''{a}, '''
        class_list = class_list[:-2]

        prompt = self.build_ba_prompt(user_text, definitions, class_list)

        messages = [{'role': 'user', 'content': prompt}]
        str_message = ' '.join([x['content'] for x in messages])

        tk = self.num_tokens_from_string(str_message)

        if tk > self.CTX_WIN - self.MAX_TOKENS:
            info(f"Request to big! Tokens: {tk} | Message: {str_message}")
            info('Trying with simplify_text()')
            prompt = self.simplify_text(prompt)
            messages = [{'role': 'user', 'content': prompt}]
            str_message = ' '.join([x['content'] for x in messages])
            tk = self.num_tokens_from_string(str_message)
            if tk > self.CTX_WIN - self.MAX_TOKENS:
                tokens = { 'completion': 0, 'prompt': 0, 'total': 0}
                return 'STOP', "MESSAGE_TOO_LONG", tokens
            else:
                info(f"New number of tokens: {tk} | Message: {str_message}")

        resp = self.chat_gpt_query(messages)
        # verify that the response is always in JSON format
        answer, reasoning, real_tk = self.parse_json_response(resp)

        if debug:
            info(f"TOKENS: aprox: {tk} | real: {real_tk['total']}")
            info(f"\tprompt: {real_tk['prompt']}")
            info(f"\tcompletion: {real_tk['completion']}")
            info(f"RESP: {resp}")
            info(f"MODEL: {resp.model}")
            info(f"MESSAGES: {messages}")
            info('Motivation: %s\n' % reasoning)

        return answer, reasoning, real_tk

    def get_gpt_multiclass_reponses(self, examples, definitions: dict,
                                    debug=False, test=False):
        res = []
        whys = []
        ntokens = 0
        nqueries = 0
        total_cost = 0
        for r in examples.to_dict('records'):
            done = False

            text = r['text']
            if debug:
                if len(text) > 400:
                    excerpt = "%s [...] %s" % (text[:200], text[-200:])
                else:
                    excerpt = text
                info("Text excerpt: %s\n", excerpt)

            query_cost = 0
            while not done:
                try:
                    resp, why, tk = self.direct_class_w_mult_def(definitions, text, debug=debug)
                    done = True
                    query_cost += self.get_real_cost(tk)
                    ntokens += tk['total']
                    nqueries += 1
                except Exception as ex:
                    info("Exception when using direct_class_w_mult_def: %s", ex)
                    # raise ex
                    if done:
                        break
                    time.sleep(3)
                    info('Retry')
                    done = False

            if debug:
                info(f"resp: {resp}")

            # Clean the answer, e.g., when it has multiple words
            if len(resp.split(' ')) > 1:
                cats = set()
                for cat in definitions:
                    if cat in resp:
                        cats.add(cat)
                # There is exactly one category mentioned in the response
                if len(cats) == 1:
                    resp = cats.pop()
                # The resp doesn't contain any valid category or it contains more than one
                else:
                    # This should be 'other_something' for consistency
                    # resp = 'other' #used to be 'unknown'
                    other_def = [i for i in definitions if i.startswith('other_')]
                    if other_def:
                        resp = other_def[0]
                    else:
                        resp = 'other' # Just for control, it shouldn't happen

            res.append(resp)
            whys.append(why)
            msg = 'category: %s\tpredicted: %s\ttokens %d\tcost %f$\n'
            info(msg, r['category'], resp, tk['total'], query_cost)
            total_cost += query_cost

            if why == "MESSAGE_TOO_LONG":
                break

            if test:
                break

        return res, whys, ntokens, nqueries, total_cost


    def num_tokens_from_messages(self, messages):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError as e:
            warn("Model not found: %s. Using o200k_base instead.", self.model)
            encoding = tiktoken.get_encoding("o200k_base")

        num_tokens = 0
        for message in messages:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                # if there's a name, the role is omitted
                if key == "name":
                    # role is always required and always 1 token
                    num_tokens += -1
        # every reply is primed with <im_start>assistant
        num_tokens += 2
        return num_tokens

    # Verify if a node is the child of another at certain level
    def is_child(self, child, parent, level=1):
        return level in self.tax and child in self.tax[level] and self.tax[level][child] == parent

    # Get the node and all its childs
    def get_node_and_children(self, node, level=1):
        childs = {node}
        if level not in self.tax:
            return childs
        for c, p in self.tax[level].items():
            if p == node:
                childs.add(c)
        return childs

    # Get the child node and its parent
    def get_child_node_and_parent(self, node, level=1):
        return [node, self.tax[level][node]] if level in self.tax and node in self.tax[level] else [node]


    ### Classification functions ###

    # Hierarchical classification
    # by_levels=True produces a classification by levels, independent from the results at previous levels.
    # by_levels=False produces a classification from top to bottom, which outputs an extra _global.csv file with the global results.
    def classify_top_to_bottom(self, dataset, examples, by_levels=False, cols=None):
        cols = self.COLUMNS if cols is None else cols
        global_results = defaultdict(dict)
        global_examples = None
        category = 't'
        gntokens = 0
        gnqueries = 0
        gcost = 0

        for level in range(1, 4):
            info(f"Classification level {level}")
            info("\n")

            if level > 1:

                # Traverse the parents
                for category in self.definitions[level-1]:

                    # Select definitions under this category
                    subdefs = {
                        c: d
                        for c, d in self.definitions[level].items()
                            if (self.tax[level-1][c] == category
                                    and not c.startswith('btt'))
                    }

                    if not subdefs:
                        continue

                    # Select the elements that should be classified at this level
                    # If by_levels==False, we select the elements based on the predicted category
                    # Otherwise, we select the elements based on the true category
                    if not by_levels:
                        # Select the elements using the predicted category from the last classification round (the key is the grandparent)
                        next_iter = global_results[level-1][self.tax[level-2][category]]
                        # Select reports classified with this category
                        next_iter = next_iter[next_iter['predicted']==category].copy()
                    else:
                        # Select the children categories of the parent, and get the elements from the original dataset
                        valid_cats = self.get_node_and_children(category, level-1)
                        next_iter = examples[examples['category'].isin(valid_cats)].copy()

                    if next_iter.empty:
                        global_results[level].update({category: next_iter})
                        continue

                    cats = sorted([(cat, group.shape[0]) for cat, group in next_iter.groupby('category')])
                    info(f"Parent category: {category} (Level {level-1})")
                    info(f"Children categories (classification labels):\t{sorted(subdefs.keys())}")
                    info(f"True categories:\t{cats}")
                    info("\n")

                    fname = f"responses_ttb_{self.model}_{dataset}_level{level}_{category}{'_by-levels' if by_levels else ''}.csv"
                    fname = f"{self.output}{fname}"

                    if os.path.isfile(fname):
                        next_iter = pd.read_csv(fname, index_col=0)
                    else:
                        res, whys, ntokens, nqueries, tot_cost = self.get_gpt_multiclass_reponses(next_iter, subdefs, debug=True, test=False)

                        # Use the category of the parent instead of other_'parent'
                        next_iter['predicted'] = [category if e == f"ttb_other_{category}" else e for e in res]
                        next_iter['motivation'] += [f"[level:{level}]={w}| " for w in whys]
                        next_iter[cols].to_csv(fname)
                        gntokens += ntokens
                        gnqueries += nqueries
                        gcost += tot_cost

                        info(f"Level {level} category {category}:: Tokens={ntokens} Queries={nqueries} Cost={tot_cost}")

                    global_results[level].update({category: next_iter})

                    # Update the labels in the global dataset using the results from this level
                    if not by_levels:
                        global_examples.loc[next_iter.index, 'predicted'] = next_iter.predicted
                        global_examples.loc[next_iter.index, 'motivation'] = next_iter.motivation
                continue

            fname = f"responses_ttb_{self.model}_{dataset}_level{level}_{category}.csv"
            fname = f"{self.output}{fname}"
            # info(f"Searching for cached results using {fname}")

            if os.path.isfile(fname):
                examples = pd.read_csv(fname, index_col=0)
            else:
                res, whys, ntokens, nqueries, tot_cost = self.get_gpt_multiclass_reponses(examples, self.definitions[level], debug=True, test=False)

                examples['predicted'] = [category if e == f"ttb_other_{category}" else e for e in res]
                examples['motivation'] = [f"[level:{level}]={w}| " for w in whys]
                examples[cols].to_csv(fname)
                gntokens += ntokens
                gnqueries += nqueries
                gcost += tot_cost

                info(f"Level {level} category {category}:: Tokens={ntokens} Queries={nqueries} Cost={tot_cost}")

            if not by_levels:
                global_examples = examples.copy()

            global_results[level] = {category: examples}

            # # Check if the prediction resulted in notabuse for all reports
            # if len(set(global_examples.predicted.unique()) - {'notabuse'}) == 0:
            #     info(f"Only notabuse reports. Exiting.")
            #     break

        if not by_levels:
            fname = f"responses_ttb_{self.model}_{dataset}_global.csv"
            fname = f"{self.output}{fname}"
            if not os.path.isfile(fname):
                global_examples[cols].to_csv(fname)

        return global_results, global_examples, gntokens, gnqueries, gcost

    # Hierarchical bottom-to-top classification
    def classify_bottom_to_top(self, dataset, examples):
        global_results = defaultdict(dict)
        global_examples = None
        gntokens = 0
        gnqueries = 0
        gcost = 0

        for level in range(3, 0, -1):
            info(f"Classification level {level}")
            info("\n")

            # Select definitions under this level
            subdefs = {
                c: d
                for c, d in self.definitions[level].items()
                    if not c.startswith('ttb')
            }
            if not subdefs:
                continue

            fname = f"responses_btt_{self.model}_{dataset}_level{level}.csv"
            fname = f"{self.output}{fname}"

            if level < 3:
                # Select the elements predicted as btt_other_abuse in the lower level
                next_iter = global_results[level+1]
                # Select reports classified with this category
                next_iter = next_iter[next_iter['predicted']=='btt_other_abuse'].copy()

                if next_iter.empty:
                    # Nothing to do after
                    debug("BTT classification ends at level %d.", level)
                    for l in range(level, 0, -1):
                        global_results[level] = next_iter
                    break

            else:
                next_iter = examples.copy()

            cats = sorted([(cat, group.shape[0]) for cat, group in next_iter.groupby('category')])
            # info(f"Parent category: {category} (Level {level-1})")
            info(f"Children categories (classification labels):\t{sorted(subdefs.keys())}")
            info(f"True categories:\t{cats}")
            info("\n")

            if os.path.isfile(fname):
                info(f"Using cached results from {fname}")
                next_iter = pd.read_csv(fname, index_col=0)
            else:
                res, whys, ntokens, nqueries, tot_cost = self.get_gpt_multiclass_reponses(next_iter, subdefs, debug=True, test=False)

                # Use the category of the parent instead of other_'parent'
                next_iter['predicted'] = res
                next_iter['motivation'] = [f"[level:{level}]={w}| " for w in whys]
                next_iter[self.COLUMNS].to_csv(fname)
                gntokens += ntokens
                gnqueries += nqueries
                gcost += tot_cost

                info(f"Level {level}:: Tokens={ntokens} Queries={nqueries} Cost={tot_cost}")

            if level > 2:
                global_examples = next_iter

            global_results[level] = next_iter
            # Update the labels with the new results
            global_examples.loc[next_iter.index, 'predicted'] = next_iter.predicted
            global_examples.loc[next_iter.index, 'motivation'] = next_iter.motivation

        fname = f"responses_btt_{self.model}_{dataset}_global.csv"
        fname = f"{self.output}{fname}"
        if not os.path.isfile(fname):
            global_examples[self.COLUMNS].to_csv(fname)

        return global_results, global_examples, gntokens, gnqueries, gcost

    # One-query classification
    # Uses all the classes as a multi-class classifier
    def classify_one_query(self, dataset, examples):
        global_results = defaultdict(dict)
        global_examples = None

        # Gather all definitions and drop categories _other_
        subdefs = {
            c: d
            for l, dd in self.definitions.items()
                for c, d in dd.items()
                    if not '_other_' in c
        }

        fname = f"responses_one-query_{self.model}_{dataset}.csv"
        fname = f"{self.output}{fname}"
        next_iter = examples.copy()

        cats = sorted([(cat, group.shape[0]) for cat, group in next_iter.groupby('category')])
        info(f"Classification labels:\t{sorted(subdefs.keys())}")
        info(f"True categories:\t{cats}")
        info("\n")

        if os.path.isfile(fname):
            info(f"Using cached results from {fname}")
            next_iter = pd.read_csv(fname, index_col=0)
        else:
            res, whys, ntokens, nqueries, tot_cost = self.get_gpt_multiclass_reponses(next_iter, subdefs, debug=True, test=False)

            next_iter['predicted'] = res
            next_iter['motivation'] = whys
            next_iter[self.COLUMNS].to_csv(fname)
            info(f":: Tokens={ntokens} Queries={nqueries} Cost={tot_cost}")

        global_examples = next_iter
        global_results[0] = next_iter

        return global_results, global_examples, ntokens, nqueries, tot_cost

    # One-query classification without definitions
    # Uses all the classes as a multi-class classifier
    def classify_ablation(self, dataset, examples):
        global_results = defaultdict(dict)
        global_examples = None

        # Gather all definitions and drop categories _other_
        subdefs = {
            c: _
            for l, dd in self.definitions.items()
                for c, _ in dd.items()
                    if not '_other_' in c
        }

        fname = f"responses_ablation_{self.model}_{dataset}.csv"
        fname = f"{self.output}{fname}"
        next_iter = examples.copy()

        cats = sorted([(cat, group.shape[0]) for cat, group in next_iter.groupby('category')])
        info(f"Classification labels:\t{sorted(subdefs.keys())}")
        info(f"True categories:\t{cats}")
        info("\n")

        if os.path.isfile(fname):
            info(f"Using cached results from {fname}")
            next_iter = pd.read_csv(fname, index_col=0)
        else:
            res, whys, ntokens, nqueries, tot_cost = self.get_gpt_multiclass_reponses(next_iter, subdefs, debug=True, test=False)

            next_iter['predicted'] = res
            next_iter['motivation'] = [f"[level:1]={w}| " for w in whys]
            next_iter[self.COLUMNS].to_csv(fname)

            info(f":: Tokens={ntokens} Queries={nqueries} Cost={tot_cost}")

        global_examples = next_iter
        global_results[0] = next_iter

        return global_results, global_examples, ntokens, nqueries, tot_cost

    def simplify_text(self, text):
        # Some reports may contain chars like:
        #         I wἱll ďеlеtе аll tհἱѕ ďἱгtу ѕtսḟḟ ᴦἱɡհt аwау.
        # So we replace special Unicode characters with ASCII equivalents
        simplified_text = unicodedata.normalize('NFKD', text)\
                            .encode('ascii', 'ignore')\
                            .decode('utf-8')

        # Remove extra whitespaces and newline characters
        simplified_text = re.sub(r'\s+', ' ', simplified_text).strip()

        return simplified_text

    def simplify_dataset(self, data, text_id='sha_text'):
        changed = 0
        examples = data.copy()
        tk_saved = 0
        toobig = set()
        largest = 0
        for sha, text in examples.loc[:, [text_id, 'text']].values:
            simple_text = self.simplify_text(text)
            tk = self.num_tokens_from_string(text)
            simple_tk = self.num_tokens_from_string(simple_text)
            if tk > 1000 and simple_tk < tk:
                changed += 1
                lt = len(text)
                lst = len(simple_text)
                if lst < lt*0.9:
                    info(f"Too many chars were lost. Using original text for ID {sha}.")
                    info(f"simple text sample: {simple_text[:100]}")
                    if tk > self.CTX_WIN - self.MAX_TOKENS:
                        toobig.add(sha)
                    continue

                info(f"Simplifying text of ID {sha}: From {tk} tokens to {simple_tk}; from {lt} chars to {lst}")
                info(f"text: {text[:100]}")
                info(f"simple text: {simple_text[:100]}")

                examples.loc[examples[text_id]==sha, 'text'] = simple_text
                tk_saved += (tk - simple_tk)
                largest = largest if largest >= simple_tk else simple_tk
                if simple_tk > self.CTX_WIN - self.MAX_TOKENS:
                        toobig.add(sha)

        info(f"Changed {changed} samples")
        info(f"Saved {tk_saved} tokens")
        info(f"The largest report contains {largest} tokens")
        info(f"Reports still too big for querying: {sorted(toobig)}")

        return examples

    def verify_num_tokens_from_string(self, text):
        tk = self.num_tokens_from_string(text)
        if tk > self.CTX_WIN - self.MAX_TOKENS:
            debug(f"Request to big! Tokens: {tk}")
            debug('Trying with simplify_text()')
            text_ = self.simplify_text(text)
            tk = self.num_tokens_from_string(text_)
            if tk > self.CTX_WIN - self.MAX_TOKENS:
                error(f"Message still too long :( New number of tokens: {tk}")
                return tk, text
            else:
                info(f"Text was transformed, new number of tokens: {tk}")
                return tk, text_
        else:
            info("The text will consume %d tokens", tk)
            return tk, text

    def evaluate(self, dataset, level=None, name="confusion_matrix.png"):
        if level:
            y = [self.taxonomy[level][e] if e in self.taxonomy[level] else e for e in dataset.category]
        else:
            y = dataset.category

        y_pred = dataset.predicted
        display_labels = sorted(set(y).union(y_pred))

        cf = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(cf, display_labels=display_labels)
        disp.plot(include_values=True, cmap="viridis", ax=None, xticks_rotation="vertical")
        plt.savefig(name)

        info("Classification results:")
        info(classification_report(y, y_pred, target_names=display_labels))

def sha256(text):
    m = hashlib.sha256()
    m.update(text.encode())
    return m.hexdigest()

def process(args):
    ## Build TagZila and make some configurations ##
    arc = AbuseReportClassifier.load_from_configfile(file_config=args.config)
    if arc:
        info("AbuseReportClassifier was loaded using %s", args.config)
        if arc.model is None:
            info("Setting model to %s", args.model)
            arc.model = args.model
        if args.classify == 'ablation':
            arc.ablation = True
        if args.output:
            arc.output = args.output
        else:
            arc.output = './'
    else:
        return

    ## Read or build a dataset ##
    dataset = args.name if args.name else "text_response"
    if args.dataset is not None:
        samples = args.dataset

        if 'sha_text' not in samples.columns:
            samples['sha_text'] = samples.text.apply(sha256)
    else:
        # Try the message before include it in the prompt
        tk, args.text = arc.verify_num_tokens_from_string(args.text)
        if tk > arc.CTX_WIN - arc.MAX_TOKENS:
            return

        sha_text = sha256(args.text)
        data = [[sha_text, 'no_category', args.text]]
        columns = ['sha_text', 'category', 'text']
        samples = pd.DataFrame(data=data, columns=columns)

    # In case it contains rare characters
    if args.simplify:
        samples = arc.simplify_dataset(samples)

    if args.evaluate:
        arc.evaluate(samples)
        return

    # Classify the dataset
    if args.classify == 'top-to-bottom':
        r = arc.classify_top_to_bottom(dataset, samples)
    elif args.classify == 'bottom-to-top':
        r = arc.classify_bottom_to_top(dataset, samples)
    elif args.classify == 'one-query':
        r = arc.classify_one_query(dataset, samples)
    else:
        r = arc.classify_ablation(dataset, samples)

    results, samples, tokens, nqueries, cost = r

    info(f"Total queries: {nqueries:,}")
    info(f"Total tokens: {tokens:,}")
    info(f"Total cost: ${cost:.2f}")


