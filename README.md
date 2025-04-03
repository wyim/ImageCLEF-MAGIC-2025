# IMAGECLEF MEDIQA MAGIC 2025

This repository includes the evaluation code used for the ImageCLEF - MEDIQA Multimodal And Generative TelemedICine (MAGIC) 2025 (https://www.imageclef.org/2025/medical/mediqa), ongoing from February 10 to May 09.

To register please visit the ImageCLEF website and follow instructions on signing up to the evaluation lab:
https://www.imageclef.org/2025

## Task Description

In the 2nd MEDIQA-MAGIC task, we will extend on the previous year’s dataset and challenge based on multimodal dermatology response generation. Participants will be given a clinical narrative context along with accompanying images. The task is divided into two relevant sub-parts: (i) segmentation of dermatological problem regions, and (ii) providing answers to closed-ended questions.

In the first sub-task, given each image and the clinical history, participants will need to generate segmentations of the regions of interest for the described dermatological problem. In the second sub-task, participants will be given a dermatological query, its accompanying images, as well as a closed-question with accompanying choices – the task is to select the correct answer to each question.

The dataset is created by using real consumer health users’ queries and images; the question schema was created in collaboration with two certified dermatologists. Segmentation will be evaluated against common metrics such as Jaccard or Dice. Closed question-answering will be evaluated using accuracy.

## Data

### Segmentation

| Split | Queries | Images | Masks |
| ------ | ------|---|-------|
| Train | 842 | 2474 | 7448 |
| Valid | 56 | 157 | 472 |
| Test | 100 | 314 | 944 |

Masks are saved as binary tiff files. You can can load them as follows:
```
import tifffile
mask = tifffile.imread('dermavqa-segmentations/valid/IMG_ENC00863_00009_mask_ann3.tiff')
```

The naming convention is as follows: IMG_{ENCOUNTERID}\_{IMAGEID}\_mask\_{ANNOTATOR#}.tiff
Each image has 3 segmentations coming from 4 different annotators {ann0,ann1,ann2,ann3}

To visualize the images/mask:
```
#visualize image
img = np.asarray(Image.open(img_path))
imgplot = plt.imshow(img)
#visualize mask
plt.imshow(mask)
```

The expected format of the system outputs will be the same type of tiff images. Please name with convention IMG_{ENCOUNTERID}\_{IMAGEID}\_mask\_{SYSNAME}.tiff.


### Closed QA


| Split | Queries |
| ------ | ------|
| Train | 300 |
| Valid | 56 |
| Test | 100 |

The ClosedQA task requires answering a set of previously defined clinically relevant questions with multiple choice answers using both images and the clinical history.

You are provided a list of 27 questions defined in 'closedquestions_definitions_imageclef2025.json', with both English and Chinese translations. Please see below to download the original images and clinical history queries.

Some questions are repeated questions, used when multiple problems or sites are discussed. For example, 
```
{
        "qid": "CQID011-001",
        "question_type_en": "Site Location",
        "question_type_zh": "部位位置",
        "question_category_en": "General",
        "question_category_zh": "综合",
        "question_en": "1 Where is the affected area?",
        "question_zh": "1 受影响的区域在哪里？",
        "options_en": [
            "head",
            "neck",
            "upper extremities",
            "lower extremities",
            "chest/abdomen",
            "back",
            "other (please specify)",
            "Not mentioned"
        ],
        "options_zh": [
            "头部",
            "颈部",
            "上肢",
            "下肢",
            "腹部",
            "背部",
            "其他（请注明）",
            "无法得知"
        ]
    }
```
There is a nearly identifical CQID011-002, where the question_en is "2 Where is the affected area?"/"2 受影响的区域在哪里？" to account for multiple sites.

In the evaluation, these questions are grouped together. Compared to the gold standard, each of these grouped instances will get a partial score of number\_matches/max(gold\_inputs,system\_inputs). Please see the evaluation code for exact details.

The expected format will be a json file with a list of objects - each object with the encounter id and a key-value pair using the QID and the assigned option value:
```
{
        "encounter_id": "ENC00001",
        "CQID010-001": 1,
        "CQID011-001": 5,
        "CQID011-002": 7,
        "CQID011-003": 7,
        "CQID011-004": 7,
        "CQID011-005": 7,
        "CQID011-006": 7,
        "CQID012-001": 1,
        "CQID012-002": 3,
        "CQID012-003": 3,
        "CQID012-004": 3,
        "CQID012-005": 3,
        "CQID012-006": 3,
        "CQID015-001": 6,
        "CQID020-001": 3,
        "CQID020-002": 9,
        "CQID020-003": 9,
        "CQID020-004": 9,
        "CQID020-005": 9,
        "CQID020-006": 9,
        "CQID020-007": 9,
        "CQID020-008": 9,
        "CQID020-009": 9,
        "CQID025-001": 2,
        "CQID034-001": 8,
        "CQID035-001": 1,
        "CQID036-001": 1
    }
```

### Original Images and Queries

The images and consumer health queries can be found in the following links.

Original Images: https://osf.io/p8bfu

| Split | Query File |
| ------ | --------------------|
| Train | https://osf.io/jgbtm |
| Valid | https://osf.io/krj8g |
| Test | https://osf.io/y82bn |

Note the links to these valid and test files are updated from the original 2024 challenge files to account for corrections in some translations. The original valid and test files used in the challenges are as follows: https://osf.io/h573a and https://osf.io/83crj

The dataset was part of a Shared Task:
```
@inproceedings{mediqa-m3g-2024,
  author    = {Asma {Ben Abacha} and
               Wen{-}wai Yim and
               Yujuan Fu and
               Zhaoyi Sun and
               Fei Xia and
               Meliha Yetisgen and
               Martin Krallinger
              }, 
  title     = {Overview of the MEDIQA-M3G 2024 Shared Tasks on Multilingual Multimodal Medical Answer Generation},
  booktitle = {NAACL-ClinicalNLP 2024},
  year      = {2024}
}
```
The dataset construction is described in this MICCAI paper:
```
@inproceedings{Yim2024DermaVQAAM,
  title={DermaVQA: A Multilingual Visual Question Answering Dataset for Dermatology},
  author={Wen-wai Yim and Yujuan Fu and Zhaoyi Sun and Asma Ben Abacha and Meliha Yetisgen-Yildiz and Fei Xia},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2024}
}
```

## Code

The evaluation code can be found in the evaluation folder.

For the segmentation scoring, please use "score_segmentations.py". Your system output is expected to follow the gold labels where a folder contains a number of mask images with the following naming convention IMG_{ENCOUNTERID}\_{IMAGEID}\_mask\_{SYSNAME}.tiff. You can pass in your SYSNAME on evaluation or use the default "sys" suffix.
```
python score_segmentations.py <reference-directory> <system-directory> <score-output-directory> <(optional)suffix>
```

For the closed QA portion, please use the "score_cvqa.py"; where you can pass in the gold/system file names
```
python score_cvqa.py <reference-jsonfile> <system-jsonfile> <score-output-directory>
```

The evaluation script for the shared task will just use the "run_segandcvqa_scoring.py" which calls the previous two scripts.

## Organizers

- Asma Ben Abacha, Microsoft
- Wen-wai Yim, Microsoft
- Noel Codella, Microsoft
- Roberto Andres Novoa, Stanford University
- Josep Malvehy, Hospital Clinic of Barcelona
