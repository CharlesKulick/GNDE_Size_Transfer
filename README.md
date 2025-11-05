# Size-Transferability of Graph Neural Differential Equations

Read our paper here: [arXiv](https://arxiv.org/abs/2510.03923) [OpenReview](https://openreview.net/pdf?id=x7zp0X5zfM)

> Graph Neural Differential Equations (GNDEs) combine the structural inductive bias of Graph Neural Networks (GNNs) with the continuous-depth architecture of Neural ODEs, offering an effective framework for modeling dynamics on graphs. In this paper, we present the first rigorous convergence analysis of GNDEs with time-varying parameters in the infinite-node limit, providing theoretical insights into their size transferability. We introduce Graphon Neural Differential Equations (Graphon-NDEs) as the infinite-node limit of GNDEs and establish their well-posedness. Leveraging tools from graphon theory and dynamical systems, we prove the trajectory-wise convergence of GNDE solutions to Graphon-NDE solutions. Moreover, we derive explicit convergence rates for GNDEs over weighted graphs sampled from Lipschitz-continuous graphons and unweighted graphs sampled from {0, 1}-valued (discontinuous) graphons. We further obtain size transferability bounds, providing theoretical justification for the practical strategy of transferring GNDE models trained on moderate-sized graphs to larger, structurally similar graphs without retraining. Numerical experiments support our theoretical findings.

The code in this repository fully replicates the experiments showcased in our paper. We also provide code for experiments on GNDE reaction-diffusion dynamics transfer with full training pipelines. The corresponding work is under review.

## Installation 

To run these experiments, first install packages specified in the [requirements.](requirements.txt)

The DGL library must also be installed; for the correct version of DGL, run:

```pip install  dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html```

## GNDE Size Transferability Usage

To run our suite of GNDE size transferability examples from [our original paper](https://arxiv.org/abs/2510.03923), the easiest method is executing [Graphon_NDE_Convergence.sh](Graphon_NDE_Convergence.sh). This will automatically run all three convergence examples sequentially. To do this, first make the bash file executable from your terminal:

```chmod +x Graphon_NDE_Convergence.sh```

Then simply executing the file will run the experiments:

```./Graphon_NDE_Convergence.sh```

For more control over specifics, run the [convergence experiment source file](convergence_experiment.py) directly, passing in arguments in the following pattern:

```bash
python convergence_experiment.py --graphon <graphon_type> --graphon_parameter <parameter_value> --num_random_inits <num_inits>
```

See also the hyperparameters located in the associated [config file](configs/convergence_config.yaml).

## Transferability of Dynamics Usage

To run complete experiments training GNDE models on graph reaction-diffusion dynamical systems, as in our work under review, the easiest method is executing [Dynamics_Convergence.sh](Dynamics_Convergence.sh). This bash script runs through a list of training graph sizes and seeds, and trains a model on the specified dynamics for each setup. Model checkpoints are automatically saved.

To run this, first make the file executable:

```chmod +x Dynamics_Convergence.sh```

Then run the file:

```./Dynamics_Convergence.sh```

For basic changes, such as choosing the type of dynamics, graphon of interest, range of graph sizes, and amount of training data, modify the top of the bash script. Many more tunable hyperparameters are also located in the associated [config file](configs/dynamics_config.yaml).

To run a transfer analysis on the results of the GNDE dynamics learning code, run [dynamics_transfer.py](dynamics_transfer.py). Provide matchng arguments as used for GNDE training to ensure the correct checkpoints and data generation are used, with the following pattern:

```bash
python dynamics_transfer.py --dynamics <dynamics_type> --nvals <training_graph_sizes> --seeds <seeds_for_training> --N <large_graph_size> --use_weighted <weighted_flag> --graphon <graphon_type> --graphon_parameter <parameter_value> --num_train_trajectories <training_trajectories> --fourier_degree <degree_for_initial_conditions> --num_test_conditions <test_ics_for_analysis> --dropout <dropout_value>
```

## References

Our [GNDE models](models/model_defs.py) are heavily inspired by [torchgde](https://github.com/Zymrael/gde/) with some custom extensions:

```
@article{poli2019graph,
  title={Graph Neural Ordinary Differential Equations},
  author={Poli, Michael and Massaroli, Stefano and Park, Junyoung and Yamashita, Atsushi and Asama, Hajime and Park, Jinkyoo},
  journal={arXiv preprint arXiv:1911.07532},
  year={2019}
}
```

## Cite us!

If our work proved useful in your own project, please cite us!

```
@misc{yan2025convergence,
      title={On the Convergence and Size Transferability of Continuous-depth Graph Neural Networks}, 
      author={Mingsong Yan and Charles Kulick and Sui Tang},
      year={2025},
      eprint={2510.03923},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.03923}, 
}
```
