ConditionalSurfaceDecoder: Physics-Aware 3D Inverse Design

ConditionalSurfaceDecoder is a continuous 3D surface auto-decoder network tailored for inverse physical design. It maps multimodal constraints—including continuous numeric tensors, bounding spatial limits, and categorical properties—into continuous Signed Distance Fields (SDFs).

Intellectual Property Notice & Academic Showcase
This repository serves exclusively as an academic showcase and architectural demonstration of the underlying machine learning engineering. The full source code, including the proprietary mathematical formulations for spatial domain mapping, physical boundary estimation, and constraint verification algorithms, is currently under intellectual property protection for an ongoing patent and academic submission.

The code provided here highlights the public network topology, the Test-Time Optimization (TTO) pipeline structure, and the software engineering methodology.

Core Methodology

This architecture bypasses traditional iterative geometric optimization by directly projecting constraints into a learned latent space. The pipeline features:

Multimodal Conditioning: Fusing high-frequency spatial coordinates, projected constraint vectors, spatial bounds, and category embeddings into a unified representation.

Curriculum Learning for Constraints: Gradually introducing constraint penalties during inference to regularize the pure geometric representation.

Test-Time Optimization (TTO): Iteratively optimizing a stochastic latent representation to satisfy multimodal conditions for unseen target geometries without requiring discrete surface gradients.

Architecture Overview

(Insert an image of the neural network architecture block diagram here, highlighting Frequency Encoding, Latent Lookup, and Constraint Projection)

Interactive Demonstration

The demo/demo_inference.ipynb notebook provides a self-contained, interactive walk-through of the Test-Time Optimization (TTO) process using a generalized target condition.

Visualizing the Optimization

During inference, the latent representation is refined iteratively. The model aligns the physical volume of the generated SDF with the target constraints.

(Insert an animated GIF here showing a 3D mesh morphing and optimizing its shape over several epochs)

Quick Start

1. Clone the repository:

git clone [https://github.com/yourusername/InertiaSDF-Showcase.git](https://github.com/yourusername/InertiaSDF-Showcase.git)
cd InertiaSDF-Showcase


2. Install dependencies:
Ensure you have a modern PyTorch environment configured.

pip install -r requirements.txt


3. Run the interactive notebook:
Launch the demo to execute the pipeline locally.

jupyter notebook demo/demo_inference.ipynb


Repository Structure

├── README.md
├── LICENSE                             
├── requirements.txt
├── demo/
│   ├── demo_inference.ipynb            # Interactive TTO pipeline showcase
│   └── assets/                         # Media for visualization
├── inertiasdf/                         
│   ├── __init__.py                     
│   ├── model_architecture.py           # Network topology and feature encoding
│   ├── inference_api.py                # TTO loop with abstracted constraint logic
│   └── geometry_utils.py               # 3D surface extraction via Marching Cubes
└── weights/
    └── dummy_weights.pth               # Pre-trained checkpoint for synthetic demo


AI Assistance Disclosure

In the interest of academic and professional transparency, Artificial Intelligence tools (Large Language Models) were utilized during the preparation of this showcase repository. These tools assisted strictly in software engineering tasks, including code refactoring, semantic obfuscation to protect proprietary intellectual property, and documentation drafting. The core physical formulations, mathematical derivations, and architectural concepts remain entirely original work.

License

This showcase repository is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. See the LICENSE file for details. Commercial use, reproduction of the full training loops, or reverse engineering of the abstracted physical constraint modules without explicit authorization is strictly prohibited.
