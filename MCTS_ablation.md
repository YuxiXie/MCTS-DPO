## SQA

K: simulation (iteration) times
$b_1$: initial step's search breadth
$b_2$: next steps' search breadth
w/o G.T.: only using self-evaluation

|   Settings                     |   ARC-C  | AI2S-M  |
| :-                             | :-:      | :-:     |
| K=5,$b_1$=2,$b_2$=2            |   82.5   |  87.3   |
| K=10,$b_1$=3,$b_2$=3           |   89.8   |  92.9   |
| K=16,$b_1$=3,$b_2$=3           |   91.1   |  95.4   |
| K=10,$b_1$=4,$b_2$=2           |   89.8   |  94.6   |
| K=16,$b_1$=4,$b_2$=2           |   88.9   |  93.9   | **


## GSM8K

K: simulation (iteration) times
$b_1$: initial step's search breadth
$b_2$: next steps' search breadth
w/o G.T.: only using self-evaluation

|   Settings                     | Accuracy |
| :-                             | :-:      |
| K=5,$b_1$=2,$b_2$=2            | 85.8     |
| K=5,$b_1$=2,$b_2$=2 (w/o G.T.) | 80.1     |
| K=5,$b_1$=5,$b_2$=3            | 88.1     |
| K=16,$b_1$=5,$b_2$=3           | 89.8     |
| K=32,$b_1$=5,$b_2$=3           | 89.4     | **
| K=16,$b_1$=5,$b_2$=4           | 90.8     |


## MATH

K: simulation (iteration) times
$b_1$: initial step's search breadth
$b_2$: next steps' search breadth
w/o G.T.: only using self-evaluation

|   Settings                     | Accuracy |
| :-                             | :-:      |
| K=20,$b_1$=5,$b_2$=3           | 47.3     |
| K=32,$b_1$=5,$b_2$=3           | 48.2     |
