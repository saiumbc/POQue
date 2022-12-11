# POQue: Asking Participant-specific Outcome Questions for a Deeper Understanding of Complex Events

POQue is a crowdsourced dataset consisting of ~ 8K annotations of complex events in ~ 4K stories.  Each story is annotated from the perspective of a single participant.  For the situation described in a story, an annotation is partially tailored to the identified participant taking on an "Agent-like" and "Patient-like" semantic role. Each annotation consists of several parts -- a high level process summary of the situation, its outcome and the changes resulting from it.  More details on the annotations are presented below.  

Here we release the collected annotations and the training, validation and test splits of the dataset.  We also release the source code used for training and testing the BART and T5 models for which we reported results in the paper.
Our paper "POQue: Asking Participant-specific Outcome questions for a Deeper Understanding of Complex Events" is published in the Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (http://https://aclanthology.org/2022.emnlp-main.pdf).
The camera ready version is available on ArXiv (https://arxiv.org/abs/... [Will add this tomorrow when Arxiv releases the submission).

## Dataset Information

| Split | Total  | Agent | Patient |
| :---: | :----: | :---: | :-----: |
| Train |  5590  |  2796 |  2794   |
| Valid |   986  |   493 |   493   |
| Test  |  1196  |   607 |   589   |
| Total |  7772  |  3896 |  3876   |

The models -- BART and T5 -- presented in the paper were trained on the combined Train and Validation data splits and tested on the test split.

## Building the Dataset

### Motivation

For this data collection, we took a holistic view of a complex event annotating various aspects of the complex event and its outcome as changes in participants.  We show that while achieving a computational understanding of complex events is not simple, it can be made possible through semantically grounding the outcome of the complex event in the various changes of state in participants.  

### Preparing Stories

We selected stories from 3 English language datasets:

1) ROCStories  (Mostafazadeh et al, 2016, https://www.aclweb.org/anthology/N16-1098/).  

2) Stories used in the ESTER dataset (Han et al, 2021, https://aclanthology.org/2021.emnlp-main.597.pdf).

3) Heuristically salient portions of Annotated New York Times newswire articles (Sandhaus, 2008, https://catalog.ldc.upenn.edu/LDC2008T19)

We resolved coreferent mentions and selected the largest cluster of mentions.  The shortest mention containing a name from the SSA names collection is identified as the Participant who is assigned either an "agent-like" or "patient-like" semantic role.

| Split | Total  | Agent | Patient |
| :---: | :----: | :---: | :-----: |
| ROC |   1383  | 1364 |   1356   |
| ESTER  |  1275  |  1237 |   1218   |
| NYT |  1343  |  1295 |  1302   |
| Total |  4001  |  3896 |  3876   |

### Crowdsourcing Annotations

Annotators were presented a story and asked to fill out 4 steps in a HIT, where they identified the following aspects of the described situation:
1) A high level process summary.
2a) An endpoint for the situation.
2b) Whether the endpoint described is stated or implied by the story (A third option of unsure was also provided).
3a) Whether the identified Participant caused or experienced the situation.
3b) A summary of changes resulting from the situation.
3c) Various change modes undergone by the story partipants as a result of the situation.
4) Factors contributing to the endpoint and changes resulting from it.

### Validating Annotations

1545 Random annotations were evaluated by 3 crowd workers and 2 experts.  Each step of the 4 HITs were evaluated seperately without any cascading errors resulting in inter-annotator agreement scores of 0.74-0.96 using the weighted Fleiss's Kappa.   

### Annotations Released

The data folder contains the following data files:
all_train_data.json, all_valid_data.json, all_test_data.json

Each annotation is assigned a unique id which is a number in the range from 0 to 8219 (the total annotations collected with our AMT HIT).  We discarded 447 annotations from 26 workers who did not follow our HIT instructions, and the remaining 7772 are distributed in the above 3 files.  The fields collected for each annotation are as follows:

|# | field | Annotation|
|:---:|:---:|:--:|
|1 | uid  | A unique_id for each annotation.
|2 | story_id | The id of the ROC Story or newswire story.
|3 | story_version | The version of the annotation where the participant is assigned the "Agent-like" or "Patient-like" semantic role.
|4 | story_source | Identifies the type of story (ROC Story, ESTER or ANYT newswire) and the subgroup (ROC CATERS,  ESTER Subevent, ANYT Financial/National/Foreign desk stories).
|5 | identified_participant | The highlighted participant in the story.  The shortest mention matches with a name in the SSA database.  In most instances this is the name of a person.
|6 | original_story_text | The story text with the identified participant highlighted.  While ROC and ESTER stories are complete stories, NYT stories are limited to about 100-150 tokens.
|7 | story_text | The original story text without the html tags.
|8| process_summary | A high level process summary of the story
|9| all_process_summaries | The process summaries from both the agent and patient versions of the story
|10| endpoint_description | The description of an endpoint in the story.
|11| all_endpoint_descriptions | The endpoint descriptions from both the agent and the patient versions of the story.
|9 | change_summary | A summary of changes experienced by the story participants.
|10| factors | A list of factors contributing to the situation and changes.
|11| endpoint_anchoring | An id (stated/implied/neither) and value indicating whether in the story, the endpoint was explicitly mentioned, implied, or unsure.
|12| participant_involvement | An id and value indicating the participant's involvement in the situation with a 1-5 likert score for Not Likely to Very Likely involved.
|13| change_modes | A list of 0 to 5 change modes experienced by participants.  These are listed below:
|||Existence -- indicates a change in existence for a participant.
|||Feeling -- indicates a change in cognition such as a change in feeling, |||emotion, perception, belief, thought.
|||Location -- indicates a change in location.
|||Possession -- indicates a change in possession
|||Other_Way -- indicates a change relating to something other than Location/Existence/Feeling/Possession.

## Source code for Tasks & Evaluation

The src_code directory contains the following files used to train and evaluate models models on the 5 task formulations:
model.py, task.py, test.py, train.py

This dataset was primarily built to advance research in understanding complex events.  We formulated 5 tasks aimed at understanding various aspects of a complex event.  We benchmarked BART and T5 models on these tasks using various automated and human evaluation metrics.  Human evaluation consisted of 3 crowd workers assigning a 1-5 likert value for the evaluated attributes such as abstractness, factuality and salience.  We report a few of the results here.  Please refer to the paper for more results, our baselines and result analysis.


### Task 1: Generating a Process Summary

We fine-tuned base and large models to generate a high level abstract summary given either a story or factors.  We evaluated this task on abstractness as determined by the average of human annotator likert scores and compared it with automated measurements of average length (the number of tokens in the endpoint description) and average extractiveness (the percentage of of trigrams in the summary that match with the story trigrams) for both reference and generated summaries.  Please refer to the paper for automated evaluation scores such as Rouge, BLEU, etc.

| Model      | Input | Abstractiveness     | Length | Extractiveness |
|     :---:  |   :---:    |  :---: | :---: | :---: |
| Reference |  story | 3.57 | 3.6 | .13 |
| Bart-base  | story | 2.77 | 4.0 | .46 |
| Bart-base  | factors | 3.22 | 4.2 | .33 |
| T5-base    | story | 2.32  | 10.0 | .60 |
| T5-base    | factors | 2.73  | 9.9 | .56 |
| T5-large   | story | 2.13 | 6.9 | .63 |

### Task 2a: Generating an Endpoint of a Complex Event

We fine-tuned base and large models to generate the description of an endpoint given either a story or factors.  We evaluated this task on factuality and salience as determined by the average of human annotator likert scores and compared these with automated measurements of average length (the number of tokens in the endpoint description) and average extractiveness (the percentage of of trigrams in the summary that match with the story trigrams) for both generated and reference factors.   Please refer to the paper for automated evaluation scores such as Rouge, BLEU, etc.

| Model      |  Input |Factuality | Salience     | Length | Extractiveness |
|     :---:  |   :---: |:---:   |  :---: | :---: | :---: |
| Reference | story | 4.15 | 3.46 | 7.9 | .27 |
| Bart-base | story |  4.66 | 3.97 | 11 | .72 |
| Bart-base | factors |  4.11 | 3.28 | 7.3 | .49 |
| Bart-large| story | 4.59| 3.81 | 10.3 | .63 |
| T5-base   | story | 4.71  | 4.03 | 13.6 | .70 |
| T5-base   | factors | 4.11  | 3.28 | 7.3 | .47 |
| T5-large  | story | 4.71 | 4.23 | 12.9 | .67 |

### Task 2b: Generating factors that explain a Complex Event

We finetuned base and large models to generate the various factors that explain the complex event given the story and endpoint.  We evaluated this task on brevity, factuality and salience as determined by the average of human annotator likert scores and compared these with automated measurements of average number of factors and average number of tokens across all factors for both generated and reference factors.   Please refer to the paper for automated evaluation scores such as Rouge, BLEU, etc.

| Model      | Brevity | Factuality | Salience     |  # Factors | # Tokens |
|     :---:  |  :---:    |  :---:    |  :---: | :---: | :---: |
| Reference | 3.35 | 3.25 | 3.04 | 3.6 | 8.3 |
| Bart-base  | 2.54 | 3.49 | 3.57 | 3.5 | 14.0 |
| Bart-large | 2.12 | 3.20| 3.31 | 3.6 | 13.7 |
| T5-base    | 3.23 | 3.69  | 4.01 | 2.6 | 19.6 |
| T5-large   | 2.85 | 3.80 | 3.96 | 3.7 | 13.9 |

### Task 3: Generating a Change summary for changes resulting from a Complex Event

We finetuned base and large models to generate the various factors that explain the complex event given the story and endpoint.  We evaluated this task on factuality and salience as determined by the average of human annotator likert scores for both the reference and generated summaries.  Please refer to the paper for automated evaluation scores such as Rouge, BLEU, etc.

| Model      | Factuality | Salience     |  
|     :---:  |  :---:    |  :---:    |  
| Reference | 3.36 | 3.32 |
| Bart-base  | 3.03 | 2.93 |  
| Bart-large | 2.99 | 3.05|  
| T5-base    | 3.74 | 3.23 |  
| T5-large   | 3.81 | 3.53 |  

### Task 4: Identifying the various change modes experienced by participants in a Complex Event

We finetuned models to classify the various change modes experienced by participants given the story and the changes resulting from the endpoint.  We tested the T5 model with a classification head on top of the Encoder and in a text-to-text Encoder-Decoder setting.  We evaluated this multi-label task on Subset Accuracy, Hamming Score and macro F1.

| Model      | Subset Accuracy  | Hamming Score    | macro F1  |
|     :---:  |  :---:    |  :---:    | :---:     |
| Bart-base  | 64.6 | 71.7 | 61.2 |
| T5-base (Enc Only)   | 59.4 | 66.1 | 50.3 |
| T5-base (Enc-Dec)  | 65.8 | 67.3 | 62.0 |

### Task 5: Identifying a participants involvment in causing or experiencing a Complex Event

We finetuned base and large models to identify whether a participant was involved in causing or experiencing the event, given the story and the changes resulting from the endpoint.  We also compared models trained seperately on agent and patient versions of the dataset.  We evaluated this task using Accuracy and macro F1.


| Model      | Combined  | Agent     | Patient   |
|     :---:  |  :---:    |  :---:    | :---:     |
| Bart-base  | 82.7/76.8 | 80.7/75.2 | 84.2/76.4 |
| Bart-large | 76.0/43.2 | 75.5/49.1 | 79.2/50.8 |
| T5-base    | 82.6/76.5 | 80.1/73.2 | 84.2/77.4 |
| T5-large   | 83.0/77.4 | 80.3/74.3 | 84.5/78.1 |


## Citation

Please use the following BibTex entry to cite our work.
We will update this when the formal proceedings are published.

```
@inproceedings{vallurupalli-etal-2022-poque,
    title = "{POQ}ue: Asking Participant-specific Outcome Questions for a Deeper Understanding of Complex Events",
    author = "Vallurupalli, Sai  and
      Ghosh, Sayontan  and
      Erk, Katrin  and
      Balasubramanian, Niranjan and
      Ferraro, Frank",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = Dec,
    year = "2022",
    address = "Online and Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://preview.aclanthology.org/emnlp-22-ingestion/2022.emnlp-main.594/",
    doi = "",
    pages = "",
}
```

## Contributors

[Sai Vallurupalli], [Sayontan Ghosh], [Katrin Erk](https://www.cs.utexas.edu/people/faculty-researchers/katrin-erk), [Niranjan Balasubramanian](https://www3.cs.stonybrook.edu/~niranjan/), [Frank Ferraro](https://redirect.cs.umbc.edu/~ferraro/)
