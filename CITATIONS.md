# Citations — BibTeX

Papers cited by tasks whose work has started: **Task 01** (3D occupancy world model)
and **Task 02** (traffic spatiotemporal forecasting, Phase 0 scaffolding).
Remaining planned tasks (cross-embodiment navigation, LiDAR point cloud) will add
their references when work starts. See role labels in
[`README.md`](README.md#references) / [`README.ko.md`](README.ko.md).

## Anchor — DreamerV3
```bibtex
@article{hafner2025dreamerv3,
  title   = {Mastering diverse control tasks through world models},
  author  = {Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal = {Nature},
  volume  = {640},
  pages   = {647--653},
  year    = {2025},
  doi     = {10.1038/s41586-025-08744-2}
}
```

## Bridge — OccWorld
```bibtex
@inproceedings{zheng2024occworld,
  title         = {OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving},
  author        = {Zheng, Wenzhao and Chen, Weiliang and Huang, Yuanhui and Zhang, Borui and Duan, Yueqi and Lu, Jiwen},
  booktitle     = {European Conference on Computer Vision (ECCV)},
  year          = {2024},
  doi           = {10.1007/978-3-031-72624-8_4},
  eprint        = {2311.16038},
  archivePrefix = {arXiv}
}
```

## Dataset — nuScenes
```bibtex
@inproceedings{caesar2020nuscenes,
  title     = {nuScenes: A Multimodal Dataset for Autonomous Driving},
  author    = {Caesar, Holger and Bankiti, Varun and Lang, Alex H. and Vora, Sourabh and Liong, Venice Erin and Xu, Qiang and Krishnan, Anush and Pan, Yu and Baldan, Giancarlo and Beijbom, Oscar},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {11618--11628},
  year      = {2020},
  doi       = {10.1109/CVPR42600.2020.01164}
}
```

## Occupancy labels — Occ3D
```bibtex
@article{tian2023occ3d,
  title         = {Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving},
  author        = {Tian, Xiaoyu and Jiang, Tao and Yun, Longfei and Wang, Yue and Wang, Yilun and Zhao, Hang},
  journal       = {arXiv preprint arXiv:2304.14365},
  year          = {2023},
  eprint        = {2304.14365},
  archivePrefix = {arXiv}
}
```

---

# Task 02 — Traffic Spatiotemporal Forecasting (STGNN)

## Anchor — GraphCast
```bibtex
@article{lam2023graphcast,
  title   = {Learning skillful medium-range global weather forecasting},
  author  = {Lam, Remi and Sanchez-Gonzalez, Alvaro and Willson, Matthew and Wirnsberger, Peter and Fortunato, Meire and Alet, Ferran and Ravuri, Suman and Ewalds, Timo and Eaton-Rosen, Zach and Hu, Weihua and Merose, Alexander and Hoyer, Stephan and Holland, George and Vinyals, Oriol and Stott, Jacklynn and Pritzel, Alexander and Mohamed, Shakir and Battaglia, Peter},
  journal = {Science},
  volume  = {382},
  number  = {6677},
  pages   = {1416--1421},
  year    = {2023},
  doi     = {10.1126/science.adi2336}
}
```

## Bridge — DCRNN
```bibtex
@inproceedings{li2018dcrnn,
  title     = {Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
  author    = {Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2018},
  eprint    = {1707.01926},
  archivePrefix = {arXiv}
}
```

## Bridge — Graph WaveNet
```bibtex
@inproceedings{wu2019graphwavenet,
  title     = {Graph WaveNet for Deep Spatial-Temporal Graph Modeling},
  author    = {Wu, Zonghan and Pan, Shirui and Long, Guodong and Jiang, Jing and Zhang, Chengqi},
  booktitle = {International Joint Conference on Artificial Intelligence (IJCAI)},
  pages     = {1907--1913},
  year      = {2019},
  doi       = {10.24963/ijcai.2019/264}
}
```

## Datasets — METR-LA / PEMS-BAY
> Processed datasets and adjacency matrices are distributed via the DCRNN repository
> (github.com/liyaguang/DCRNN); the underlying loop-detector data is provided by Caltrans PeMS.
> Cite the DCRNN paper above (`li2018dcrnn`) as the source of the processed benchmark splits.
