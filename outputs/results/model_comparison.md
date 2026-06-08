## Per-(model, dataset) results

| Model | Dataset | Split | Images | IoU | F1 | Precision | Recall |
|---|---|---|--:|--:|--:|--:|--:|
| hybridnets | bdd100k_lane | val | 10000 | 0.2291 | 0.3727 | 0.2496 | 0.7355 |
| hybridnets | culane | test | 34680 | 0.1982 | 0.3309 | 0.5972 | 0.2288 |
| hybridnets | curvelanes | valid | 20000 | 0.3267 | 0.4925 | 0.5950 | 0.4201 |
| hybridnets | tusimple | test | 2782 | 0.4721 | 0.6414 | 0.7763 | 0.5464 |
| yolopx | bdd100k_lane | val | 10000 | 0.2006 | 0.3342 | 0.2058 | 0.8893 |
| yolopx | culane | test | 34680 | 0.2626 | 0.4159 | 0.4616 | 0.3785 |
| yolopx | curvelanes | valid | 20000 | 0.3845 | 0.5554 | 0.5292 | 0.5844 |
| yolopx | tusimple | test | 2782 | 0.4960 | 0.6631 | 0.6340 | 0.6950 |

## Table 12. Preliminary model comparison

| Candidate Model | Mean IoU | Mean F1 | Mean Precision | Mean Recall | Cross-Dataset Consistency | Output Usability | Preliminary Assessment |
|---|--:|--:|--:|--:|--:|---|---|
| hybridnets | 0.3065 | 0.4594 | 0.5545 | 0.4827 | 0.7373 | - | - |
| yolopx | 0.3359 | 0.4922 | 0.4576 | 0.6368 | 0.7430 | - | - |

_n_datasets per model varies; Mean = average across that model's datasets. Cross-Dataset Consistency = 1 - (std/mean) of F1 across those datasets (1.0 = identical across domains). Output Usability / Preliminary Assessment left blank for manual judgement._
