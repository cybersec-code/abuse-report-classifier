# Abuse Report Classifier

**abuse-report-classifier** (ARC) is an LLM-based unsupervised classifier for abuse reports. Given an input TEXT (e.g., an abuse report), ARC uses a hierarchical abuse-type taxonomy, made of abuse-types with their definitions, and a query approach to classify the input TEXT. ARC can use four approaches to query the LLM, which will interpret and classify the given TEXT using a subset of abuse types and definitions:

- **Top-to-bottom** (default): ARC traverses the taxonomy from top to bottom, selecting the definitions to use by levels, in cascade mode. This implies querying several times to the LLM for the same text to refine the response using more fine-grained abuse types.
- Bottom-to-top: Same as Top-to-Bottom, but in the inverse direction (from coarse-grained to fine-grained).
- One-query: All the definitions are used at the same time for the classification.
- Ablation: The prompt uses no definitions, so the LLM will use it's own definitions for the classification.

The prompt, the abuse types, and the definitions are defined in `src/data/`.

# Install

To install ARC in a conda environment, you can follow the next steps:

1. Create the environment:

```
conda create env --name arc
```

2. Activate it:
```
conda activate arc
```

3. Install the dependencies:
```
conda install --file requirements.txt 
```

4. Build and install the package using pip:
```
cd /path/to/arc
~/path/to/env/arc/bin/pip3 install -e .
```

5. Verify the installation:
```
abuse-report-classifier --version
```

# Usage

ARC uses a prompt.config file (default is in src/data/prompt.config) to configure the queries made to OpenAI's ChatGPT. Once configured with an API KEY, you can try it using:

```
abuse-report-classifier --text "This BTC address 1HdxNqmMfZWAe8Qw57C3AVZ4xrprS9Vb3T receives ransom deposits from victims paying to decrypt their data."
```

ARC will classify the input text using the default model (gpt-4o-mini, you can change it using `--model`) and the default approach (top-to-bottom, use `--classify` to select a different one). It will output the classification results in CSV format (`*.csv`), as well as the LOG file (`.log`). By default, the files will be prepended with the string `_arc_` (you can modify it using `--output`):

```
_arc_.log
_arc_responses_ttb_gpt-4o-mini_text_response_global.csv
_arc_responses_ttb_gpt-4o-mini_text_response_level1_t.csv
_arc_responses_ttb_gpt-4o-mini_text_response_level2_abuse.csv
_arc_responses_ttb_gpt-4o-mini_text_response_level3_ransom.csv
```

Each dataset contains the responses of the model for the levels of the taxonomy used by the classifier, one row per query. The columns are:

* category: The true category of the abuse report (e.g., for evaluation).
* predicted: The abuse type selected by the classifier.
* sha_text: The sha256() of the text classified.
* text: The text classified.
* motivation: The motivation of the model for selecting the abuse type.

To clasify a dataset called `ba_samples.csv` (one report per row), using *one-query* classification, with *gpt-4o*, use:

```
abuse-report-classifier --dataset ba_samples.csv --classify one-query --model gpt-4o
```

The columns of the resulting dataset are defined in the config file (by default [category, predicted, sha_text, text, motivation]). The only mandatory column is `text`. Appart from it, the dataset can have any columns.

# Taxonomy

The taxonomy must have the format:

```
t
t:abuse
t:abuse:abuse1
t:abuse:abuse1:abuse1.1
t:abuse:abuse1:ttb_other_abuse1
t:abuse:abuse2
t:abuse:ttb_other_abuse
t:abuse:btt_other_abuse:btt_other_abuse
t:notabuse
```

The first level must be abuse vs notabuse. The second level may have different abuse types, and must have two additional `ttb_other_abuse` and `btt_other_abuse` fall-back definitions for the top-to-bottom and bottom-to-top classifications. A third level may be defined for each abuse type, along with a pair of `ttb_other_*` and `btt_other_abuse` entries. 
Using this structure we can create a taxonomy for classifying abuse, cryptoransom and scam, by replacing abuse1=ransom, abuse1.1=cryptoransom, and abuse2=scam:

```
0|   1 |    2 |              3 |         <- LEVEL
t
t:abuse                                  <- general abuse, level 1
t:abuse:ransom                           <- ransom abuse, level 2
t:abuse:ransom:cryptoransom              <- fine-grained abuse type cryptoransom, level 3
...
t:abuse:ransom:ttb_other_ransom          <- fall-back for top-to-bottom classification, level 3
t:abuse:scam                             <- scam abuse, level 2
...
t:abuse:ttb_other_abuse                  <- fall-back for top-to-bottom classification, level 2
t:abuse:btt_other_abuse:btt_other_abuse  <- fall-back for bottom-to-top classification must be put together, levels 2 and 3
t:notabuse
```

# Cite

If you use this tool, please cite our work using:

```
@article{gomez2025cleanUpTheMess,
title = {Clean Up the Mess: Addressing Data Pollution in Cryptocurrency Abuse Reporting Services},
journal = {Future Generation Computer Systems},
volume = {179},
pages = {108313},
year = {2026},
issn = {0167-739X},
doi = {https://doi.org/10.1016/j.future.2025.108313},
url = {https://www.sciencedirect.com/science/article/pii/S0167739X25006077},
author = {Gibran Gomez and Kevin {Van Liebergen} and Davide Sanvito and Giuseppe Siracusano and Roberto Gonzalez and Juan Caballero},
keywords = {Cryptocurrency Abuse Reporting Services, Bitcoin, Cryptocurrencies, LLM-based classification},
}
```
