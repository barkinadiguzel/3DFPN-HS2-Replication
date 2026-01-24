# 🌌 3DFPN-HS²-Replication — 3D Lung Nodule Detection

This repository provides a **PyTorch-based replication** of  
**3DFPN-HS²: 3D Feature Pyramid Network with High Sensitivity & Specificity for Pulmonary Nodule Detection**,  
implemented as a **modular, research-friendly detection framework**.

The project translates the paper’s **3D FPN, HS² refinement, and multi-scale candidate extraction**  
into a clean, extendable codebase.

- Enables **high-sensitivity nodule detection from 3D CT volumes** 🫁  
- Implements **multi-scale feature fusion via 3D FPN blocks** 🔮  
- Incorporates **HS² network for false positive suppression** 🛡️  
- Designed for **reproducible and efficient experimentation** ⚙️  

**Paper reference:** [3DFPN-HS²: 3D Feature Pyramid Network for Pulmonary Nodule Detection — Liu et al., 2019](https://arxiv.org/abs/1906.03467) 📄

---

## 🝆 Overview — 3D Multi-Scale Detection

![3DFPN Overview](images/figmix.jpg)

> Pulmonary nodules vary widely in **size, density, and shape**, requiring multi-scale volumetric reasoning.  

The network learns a mapping:

$$
f_\theta : \mathbb{R}^{D \times H \times W} \rightarrow \mathbb{R}^{D \times H \times W}
$$

where the output is a **3D probability map of nodule candidates** $\hat{Y}$ for a given CT volume $V$.

The architecture combines **3D FPN encoding–decoding** with **HS² refinement**,  
enabling robust detection of both small and large nodules while reducing false positives.

---

## 🧠 Architectural Principle — 3DFPN-HS²

- **Encoder**: 3D convolution blocks C2–C5 with progressive downsampling  
- **3D Feature Pyramid**: P2–P5 with lateral & top-down connections  
- **HS² Network**: 2 convolution layers + 3 fully connected layers for false positive reduction  
- **Output**: Refined nodule probability map  

Mathematically, for encoder features $[C2, C3, C4, C5]$, the 3D FPN produces fused features:

$$
[P2, P3, P4, P5] = \text{FPN3D}([C2, C3, C4, C5])
$$

and HS² refines the candidate regions via **Location History Images** (LHI):

$$
\hat{Y}_{\text{refined}} = \text{HS²}(LHI(P2, P3, P4, P5))
$$

---

## 🔬 Loss Function — Focal / Weighted BCE

To handle class imbalance between nodules and background:

$$
\mathcal{L} = - \frac{1}{N} \sum_i \alpha (1-\hat{y}_i)^\gamma y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)
$$

where $\alpha$ is the **positive class weight**, $\gamma$ the focusing parameter, $y_i$ the ground-truth, and $\hat{y}_i$ the predicted probability.

---

## 🩻 Data Handling

- **Dataset**: LUNA16 3D CT volumes  
- **Augmentation**: 3D random flip, rotation, elastic deformation, intensity noise  
- **Normalization**: Each volume scaled to $[0,1]$  

This improves **training stability** and **model generalization**.

---

## 🧪 What the Model Learns

- Detect **multi-scale nodules** with varying density 🌫️  
- Preserve **edge geometry** via skip & lateral connections 🝀  
- Fuse features **across scales** using 3D FPN 🔮  
- Suppress false positives from vessels & airway structures 🛡️  

Detection becomes a **context-aware volumetric reasoning task** rather than simple voxel-wise classification.

---

## 📦 Repository Structure

```bash
3DFPN-HS2-Replication/
├── src/
│   ├── model/
│   │   ├── encoders.py           # 3D convolution encoder blocks (C2–C5)
│   │   ├── feature_pyramid.py    # 3D Feature Pyramid (P2–P5)
│   │   ├── hs2_net.py            # HS² network (2 conv + 3 FC)
│   │   └── fpn_hs2_model.py      # Full 3DFPN-HS² assembly
│
│   ├── dataset/
│   │   └── luna16_loader.py      # LUNA16 volume loader
│
│   ├── augmentation/
│   │   └── transforms.py         # 3D flips, rotation, elastic, noise
│
│   ├── loss/
│   │   └── focal_loss.py         # Focal / weighted BCE loss
│
│   ├── utils/
│   │   ├── metrics.py            # Sensitivity, specificity, FP per scan
│   │   ├── visualization.py      # 3D volume overlay
│   │   └── lhi_utils.py          # Functions to compute LHIs
│
│   └── config.py                 # Model params, pyramid layers, LHI τ, thresholds
│
├── images/
│   └── figmix.jpg
│
├── requirements.txt
└── README.md
```
---


## 🔗 Feedback

For questions or feedback, contact: [barkin.adiguzel@gmail.com](mailto:barkin.adiguzel@gmail.com)
