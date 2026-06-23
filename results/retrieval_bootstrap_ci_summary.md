# Retrieval: MRR and Hit@1 (bootstrap 95% CI)

Pool all models per question (bench_index). Gold: 3 models × 1 run. Noise: 3 models × 3 runs. Per question — mean RR and Hit@1 over the pool; bootstrap over questions. n — number of unique questions after pooling.

| Bench | Mode | n questions | MRR | 95% CI MRR | Hit@1 | 95% CI Hit@1 |
| --- | --- | ---: | ---: | --- | ---: | --- |
| gold | baseline | 453 | 0.9251 | [0.9048; 0.9447] | 0.8911 | [0.8624; 0.9183] |
| gold | full | 453 | 0.9055 | [0.8828; 0.9279] | 0.8661 | [0.8352; 0.8962] |
| noise | baseline | 310 | 0.7494 | [0.7111; 0.7872] | 0.6290 | [0.5774; 0.6839] |
| noise | full | 310 | 0.8023 | [0.7638; 0.8397] | 0.7290 | [0.6806; 0.7774] |

Source: `results/retrieval_bootstrap_ci.json`
