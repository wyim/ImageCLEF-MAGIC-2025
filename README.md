# IMAGECLEF MEDIQA MAGIC 2025

This repository includes the evaluation code used for the ImageCLEF - MEDIQA Multimodal And Generative TelemedICine (MAGIC) 2025 (https://www.imageclef.org/2025/medical/mediqa), ongoing from February 10 to May 09.

To register please visit the ImageCLEF website and follow instructions on signing up to the evaluation lab:
https://www.imageclef.org/2025

## Task Description

In the 2nd MEDIQA-MAGIC task, we will extend on the previous year’s dataset and challenge based on multimodal dermatology response generation. Participants will be given a clinical narrative context along with accompanying images. The task is divided into two relevant sub-parts: (i) segmentation of dermatological problem regions, and (ii) providing answers to closed-ended questions.

In the first sub-task, given each image and the clinical history, participants will need to generate segmentations of the regions of interest for the described dermatological problem. In the second sub-task, participants will be given a dermatological query, its accompanying images, as well as a closed-question with accompanying choices – the task is to select the correct answer to each question.

The dataset is created by using real consumer health users’ queries and images; the question schema was created by two certified dermatologists. Segmentation will be evaluated against common metrics such as Jaccard or Dice. Closed question-answering will be evaluated using metrics accuracy.

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

### Closed QA

Closed question answering data will be organized in {train,valid,test}_cvqa.json files as a list of objects with the following structure:

```
{
    "encounter_id": "ENC0001",
    "{QID1}": "{QID1-OPTION-INDEX}",
    "{QID2}": "{QID2-OPTION-INDEX}",
    ...
}
```
 
A list of common clinical dermatology questions are specified in the closedquestions_definitions.json file. An example is shown here:
```
{
  "qid": "CQID015-001",
  "question_type_en": "Onset",
  "question_type_zh": "发作时间",
  "question_category_en": "General",
  "question_category_zh": "综合",
  "question_en": "When did the patient first notice the issue?",
  "question_zh": "病人第一次注意到这个问题是什么时候？",
  "options_en": [
      "within hours",
      "within days",
      "within weeks",
      "within months",
      "over a year",
      "multiple years"
  ],
  "options_zh": [
      "数小时内",
      "数天内",
      "数周内",
      "数月内",
      "超过一年",
      "多于一年"
  ]
}
```

For the challenge, only a subset of the entire list of questions are considered for evaluation. These are specified in the evaluation code.


### Original Images and Queries

The images and consumer health queries can be found in the following links.

Original Images: https://osf.io/p8bfu

| Split | Query File |
| ------ | --------------------|
| Train | https://osf.io/jgbtm |
| Valid | https://osf.io/h573a |
| Test | https://osf.io/9v83a |

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

The evaluation script for the shared task will just use the "run_segandcvqa_scoring.py" which calls the previous to scripts.

## Organizers

- Asma Ben Abacha, Microsoft
- Wen-wai Yim, Microsoft
- Noel Codella, Microsoft
- Roberto Andres Novoa, Stanford University
- Josep Malvehy, Hospital Clinic of Barcelona
