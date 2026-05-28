
# Manifest methods and definition

## Manifest generated data

Dataset-named adapter input file: 
1. grab links to or generate ground truth masks
2. JSON array of image notes including: 
  a. image location file link
  b. image dimensions
  c. global identifier for this image/datapoint
  d. dataset name
  e. Count of images
  f. mask location file link

Questions for other people:
1. Is it like WAY DUMB to save whole images for inference vs. evaluating them on the fly?
  a. It would probably be doable to attach one of two live evaluators to each adapter
2. JSON vs. CSV
  a. JSON is probably just nicer to look at as a human?
3. For the evaluation, should we 
4. Should we pick a minimum height (measured top down) to have detections? Most models and datasets seem to have *some* kind of threshold (but it would change based on the camera height and pitch)

## Adapter outputs

The adapters generate:
- JSON of list of images processed
AND EITHER (both?):
1. Links to resulting image masks (and the masks themselves ofc)
2. list of the JSON lane line points

## Evaluator inputs

Needs: 
1. Links to BOTH types of transformed ground truth
2. Indicator for more natural type of ground truth
3. Processed JSON/masks from the target adapter

## Manifest generator tasks

1. Find/load in label for test/training sets
2. Create adapter inputs (mostly a list of image locations)
3. Compute, store, and add the more natural ground truth method (maybe slight transformation)
4. Compute, store, and add the less natural ground truth method (probably a lot of transformation)

## Sample manifest

```
{
  "metadata": {
    "dataset": "tusimple",
    "num_samples": 2000
  },
  "samples": [
    {
      "sample_id": "clips__0530__1492626760788443246_0__20",
      "image_path": "/shared/data/archive/TUSimple/test_set/clips/0530/1492626760788443246_0/20.jpg",
      "width": 1280,
      "height": 720,
      "ground_truth": {
        "mask_path": "/shared/data/archive/TUSimple/test_set/segmentation_masks/clips__0530__1492626760788443246_0__20.png",
        "lane_json": {
          // Samples should be in increments of 10 pixels, skipping 0 and image_height
          "h_samples": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710], 
          "lanes": [
            // horizontal position of each lane line, ordered to match with the heights in h_samples
            [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 632, 625, 617, 609, 601, 594, 586, 578, 570, 563, 555, 547, 539, 532, 524, 516, 508, 501, 493, 485, 477, 469, 462, 454, 446, 438, 431, 423, 415, 407, 400, 392, 384, 376, 369, 361, 353, 345, 338, 330, 322, 314, 307, 299], 
            [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 719, 734, 748, 762, 777, 791, 805, 820, 834, 848, 863, 877, 891, 906, 920, 934, 949, 963, 978, 992, 1006, 1021, 1035, 1049, 1064, 1078, 1092, 1107, 1121, 1135, 1150, 1164, 1178, 1193, 1207, 1221, 1236, 1250, 1265, -2, -2, -2, -2, -2], 
            [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 532, 503, 474, 445, 416, 387, 358, 329, 300, 271, 241, 212, 183, 154, 125, 96, 67, 38, 9, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2], 
            [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 781, 822, 862, 903, 944, 984, 1025, 1066, 1107, 1147, 1188, 1229, 1269, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2]
          ]
        }
      }
    }
  ]
}
```
