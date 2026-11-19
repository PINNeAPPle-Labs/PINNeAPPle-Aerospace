# Physics AI for Aerospace: A Comprehensive Reference

*A living reference document on physics-informed and scientific machine learning applied to aerospace engineering — aircraft, spacecraft, launch vehicles, propulsion, aerodynamics, and structures. Compiled September 2026 to seed an open-source knowledge hub.*

---

## Table of Contents

1. [Academic Research Landscape](#1-academic-research-landscape)
2. [Model Architectures](#2-model-architectures)
3. [Commercial Products & Startups](#3-commercial-products--startups)
4. [Numerical Methods & Solvers](#4-numerical-methods--solvers)
5. [Geometry, CAD & Mesh Resources](#5-geometry-cad--mesh-resources)
6. [Datasets & Data Generators](#6-datasets--data-generators)
7. [Industry Standards & Certification](#7-industry-standards--certification)
8. [Open Challenges & Gaps](#8-open-challenges--gaps)
9. [Roadmap: Building the Reference Hub](#9-roadmap-building-the-reference-hub)
10. [Where this reference meets the PINNeAPPle codebase](#10-where-this-reference-meets-the-pinneapple-codebase)
11. [Contributing](#11-contributing)
12. [Sources](#sources)

---

## 1. Academic Research Landscape

Physics AI for aerospace sits at the intersection of scientific machine learning (SciML), computational fluid dynamics (CFD), computational structural mechanics, and applied mathematics. The field has matured rapidly since 2019, moving from proof-of-concept PDE solvers to industrial-scale surrogate models trained on production CFD data.

### Landmark papers

- **Raissi, Perdikaris & Karniadakis (2019)**, *"Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,"* Journal of Computational Physics, 378:686–707. This is the paper that coined "PINNs" and established the now-standard practice of embedding PDE residuals directly into a neural network's loss function via automatic differentiation. Code: [github.com/maziarraissi/PINNs](https://github.com/maziarraissi/PINNs).
- **Lu, Jin & Karniadakis (2019/2021)**, *"Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators,"* originally arXiv:1910.03193, published in *Nature Machine Intelligence* 3:218–229 (2021). Introduced the branch-net/trunk-net architecture for learning solution *operators* (function-to-function maps) rather than single PDE solutions — a foundational shift toward operator learning.
- **Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart & Anandkumar (2020–2021)**, the **Fourier Neural Operator (FNO)** line of work out of Caltech, parameterizing integral kernels in Fourier space for resolution-invariant PDE surrogates.
- **Li, Kovachki, Choy, Li, Kossaifi, Otta, Nabian, Stadler, Hundt, Azizzadenesheli & Anandkumar (2023)**, *"Geometry-Informed Neural Operator for Large-Scale 3D PDEs"* (NeurIPS 2023, arXiv:2309.00583) — the **GINO** architecture, developed largely at NVIDIA/Caltech, combining a graph neural operator front-end (for irregular point clouds/geometries via signed distance functions) with an FNO core on a regular latent grid. Validated on an industrial 3D vehicle-aerodynamics dataset (Ahmed-body-like geometries, Re up to 5 million), reporting roughly a **26,000× speed-up** over GPU-based CFD for drag-coefficient prediction.
- **Pfaff, Fortunato, Sánchez-González & Battaglia (2021)**, *"Learning Mesh-Based Simulation with Graph Networks"* (ICLR 2021), the **MeshGraphNets** paper from DeepMind — message-passing GNNs operating directly on simulation meshes, with adaptive remeshing support, applied to Navier–Stokes flow (cylinder flow), cloth, and structural mechanics.
- **Takamoto et al. (2022)**, *"PDEBench: An Extensive Benchmark for Scientific Machine Learning"* (NeurIPS 2022 Datasets & Benchmarks) — a standardized benchmark spanning multiple PDE families with FNO/U-Net/PINN baselines.
- **Bonnet et al. (2022)**, *"AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier–Stokes Solutions"* (NeurIPS 2022 Datasets & Benchmarks) — the leading open aerospace-specific CFD ML benchmark (see Section 6).
- **Holzschuh & Thuerey (2025)**, *"PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations"* (ICML 2025) — adapts diffusion-transformer architectural advances into a general-purpose backbone for physics foundation models.

### Notable research groups

- **Brown University — Division of Applied Mathematics**: George Em Karniadakis leads one of the field's most influential groups; co-inventor of PINNs and DeepONet, director of the DOE **PhILMs** (Physics-Informed Learning Machines) center, and PI on an OSD/AFOSR MURI on PINNs for PDEs.
- **Caltech — Computing + Mathematical Sciences**: Anima Anandkumar's group originated neural operators (FNO, GINO) and has applied them to weather forecasting (with NVIDIA, e.g. the FourCastNet lineage), plasma/fusion modeling, and 3D aerodynamics; close collaboration with NVIDIA Research (Kamyar Azizzadenesheli, Nikola Kovachki, Zongyi Li are recurring co-authors across the FNO/GINO papers).
- **Stanford — Aerospace Design Lab (Department of Aeronautics & Astronautics)**: originators and maintainers of the open-source **SU2** adjoint-CFD/design-optimization suite, a common baseline/hybridization target for ML-augmented shape optimization.
- **Stanford — Center for Turbulence Research (CTR)**: long-standing home of turbulence-modeling research (RANS/LES closures), increasingly intersecting with data-driven and ML-augmented turbulence closure work.
- **DLR (German Aerospace Center) — Institute of Aerodynamics and Flow Technology**: active PINN research for transonic/inviscid airfoil flows (Wassing, Langer, Bekemeyer), including work presented at AIAA SciTech 2025 on adapting CFD numerical concepts (e.g., upwinding, limiters) into PINN training.
- **IRT SystemX / Extrality (France)**: creators and maintainers of the AirfRANS dataset and the NeurIPS ML4CFD competition series, bridging French aerospace industry (Safran, Airbus-adjacent ecosystem) and academic ML benchmarking.
- Other active academic centers referenced in current literature include groups working on hypersonic surrogate modeling and UQ (e.g., University of Illinois Urbana-Champaign / UT Austin Oden Institute collaborations on hypersonic aerodynamic sensing), and numerous groups publishing at AIAA SciTech on ML-augmented turbomachinery, aeroelasticity, and structural optimization (including NASA-affiliated technical reports, e.g., NASA Technical Reports Server entries on ML surrogate models for turbomachinery rotor structural optimization).

### Key venues

- **NeurIPS** — the "Machine Learning and the Physical Sciences" (**ML4PS**) workshop is the central cross-disciplinary venue; NeurIPS also hosts the **Datasets & Benchmarks** track (source of PDEBench and AirfRANS) and recurring **ML4CFD** competitions (2024 edition centered on airfoil design using AirfRANS).
- **AIAA SciTech Forum** — the primary aerospace-specific venue; features dedicated ML/AI tracks, short courses ("Machine Learning for Aerospace Engineering," a recurring 2-day short course), and paper sessions on PINNs, surrogate aerodynamics, and ML-augmented turbomachinery/aeroelasticity design.
- **ICLR** — frequent venue for neural-operator and GNN-simulator papers (MeshGraphNets was an ICLR 2021 paper).
- **ICML** — increasingly hosts physics-ML architecture papers (e.g., PDE-Transformer, ICML 2025).
- Domain journals: *Journal of Computational Physics*, *AIAA Journal*, *Journal of Propulsion and Power*, *Physics of Fluids* (AIP), and *Nature Machine Intelligence*.

---

## 2. Model Architectures

### Physics-Informed Neural Networks (PINNs)
Neural networks whose loss function includes PDE residuals (computed via automatic differentiation) alongside data/boundary/initial-condition terms, allowing mesh-free solution of forward and inverse PDE problems from sparse data. Strengths: natural handling of inverse/parameter-identification problems, no mesh required. Weaknesses well-documented in aerospace CFD contexts: difficulty with high-Reynolds-number turbulent, shock-dominated, or transonic/supersonic flows due to stiff, multi-scale loss landscapes; DLR's 2025 work on inviscid transonic airfoil flow and work on PINNs for inverse problems in supersonic flows are examples of researchers actively adapting CFD numerical concepts (upwinding, entropy conditions, adaptive weighting) to make PINNs viable for compressible aerospace flows.

### Neural Operators
A family of models that learn mappings between infinite-dimensional function spaces (e.g., geometry/boundary-condition → flow field), rather than single PDE solutions, enabling generalization across new geometries/conditions without retraining:
- **DeepONet** (branch-trunk architecture; Lu, Jin, Karniadakis) and its multi-input extension **MIONet**.
- **Fourier Neural Operator (FNO)**: learns convolution kernels in Fourier space; resolution-invariant but natively restricted to regular grids (requires FFT).
- **Geo-FNO**: extends FNO to irregular/deformed geometries via a learned coordinate transformation to a regular latent grid.
- **Graph Neural Operator (GNO)**: kernel integral operators implemented via message passing on graphs, naturally handling unstructured meshes and point clouds.
- **Geometry-Informed Neural Operator (GINO)**: NVIDIA/Caltech hybrid combining a GNO encoder (irregular input geometry, via signed distance functions) with an FNO core (regular latent grid) and a GNO decoder — designed explicitly for large-scale 3D industrial geometries such as full vehicle/aircraft surfaces.
- Emerging variants: KAN-based operators (e.g., DeepOKAN), optimal-transport-based geometric operator learning, and discretization-convergent operators that generalize across mesh resolutions.

### Graph Neural Networks for Meshes
- **MeshGraphNets** (DeepMind, Pfaff et al. 2021): encode-process-decode message-passing GNNs operating directly on the simulation mesh graph, trained to predict per-step state updates (auto-regressive rollout); support learned adaptive remeshing. Widely used as a baseline/backbone for CFD-on-mesh surrogate tasks (e.g., the open `meshgraphnet-cfd-surrogate` reimplementations trained on DeepMind's `cylinder_flow` dataset).
- Follow-on work targets generalization to **unseen geometries** (a key open problem — see Section 8) and multiscale/adaptive-mesh-refinement GNNs to reduce the cost of fine local mesh resolution.
- GNN approaches applied specifically to turbomachinery (e.g., **GraphRotorNet** on NASA's Rotor37 benchmark, using sub-graph clustering to scale to full 3D multi-physics snapshots).

### Transformer-Based Foundation Models for PDEs/Simulation
- **PDE-Transformer** (Holzschuh & Thuerey, ICML 2025): diffusion-transformer-derived backbone intended as a general-purpose, scalable architecture for physics foundation models, benchmarked across 16 PDE types.
- **Transolver** and its scaling successor referenced in recent literature ("Transolver-3: Scaling Up Transformer Solvers to Industrial-Scale Geometries") — transformer solvers targeting industrial-scale unstructured geometries (explicitly benchmarked against CRM-class wing geometries).
- **AeroTransformer** / **SMART** ("Scalable Mesh-free Aerodynamic Simulations from Raw Geometries using a Transformer-based Surrogate Model") and a 2026 AIAA Journal paper, *"Toward a Foundation-Model Paradigm for Aerodynamic Prediction in Three-Dimensional Design,"* reflect an active push toward aerodynamics-specific foundation models trained across many geometries/conditions rather than one-off surrogates.
- **Unisolver**: a PDE-conditional transformer aiming at a universal neural PDE solver, conditioning a shared backbone on the governing equation/parameters.

### Diffusion & Generative Models for Physics
Increasingly used for **turbulence super-resolution**, **forecasting**, and **sparse flow reconstruction**: e.g., spectrally decomposed diffusion models for turbulence recovery, conditional flow matching for near-wall turbulence with uncertainty quantification, and physics-informed diffusion models for flow-field reconstruction (published in *Journal of Computational Physics*). These are positioned as generative alternatives/complements to deterministic surrogates, particularly valuable where uncertainty quantification (ensemble sampling) is required — directly relevant to aerospace UQ needs.

### Hybrid Physics-ML / Differentiable Solvers
End-to-end differentiable CFD frameworks allow gradients to flow through the *entire* numerical solver, enabling gradient-based design optimization and learned sub-grid closures trained jointly with the solver:
- **JAX-Fluids**: fully differentiable high-order compressible (two-phase capable) CFD solver in JAX.
- **JAX-CFD**, **PhiFlow**: differentiable incompressible-flow solvers widely used as SciML research substrates.
- **JAX-FVM**: differentiable entropy-stable finite-volume solver on unstructured meshes for compressible flows.
- **Diff-FlowFSI**: GPU-optimized differentiable CFD platform (JAX) for turbulence and fluid-structure interaction, using immersed-boundary methods for complex geometry.
This category is where "hybrid physics-ML" is most concrete for aerospace: a classical numerical scheme provides guaranteed conservation/stability properties while a learned component (turbulence closure, sub-grid model, or geometry-parameterization) is trained via backpropagation through the solver.

### Reduced-Order Models (ROM) for Aero-Structures
- **Proper Orthogonal Decomposition (POD)** combined with regression/neural-network coefficient prediction remains a workhorse for parametric aerodynamic/aeroelastic ROMs.
- **Autoencoder-based ROMs**, including LSTM-autoencoders for flutter/limit-cycle-oscillation prediction, and dynamics-based autoencoders explicitly benchmarked in a 2024 *AIAA Journal* paper on airfoil flutter reduced-order modeling.
- **Operator inference** for data-driven model reduction of coupled aeroelastic flutter systems.
- Gaussian Process / Kriging-based surrogates remain the standard for low-data, UQ-critical regimes such as hypersonic aerothermodynamics, where training sets can be as small as ~200 samples.

---

## 3. Commercial Products & Startups

The commercial "Physics AI" category has seen major capital inflows and M&A activity through 2025–2026, alongside significant defense/aerospace-specific deployment.

- **NVIDIA PhysicsNeMo** (formerly **Modulus**): open-source framework providing optimized training pipelines and reference AI-physics architectures (FNO, GINO, DoMINO, etc.) for surrogate modeling of vehicle/aircraft aerodynamics and structures. Integrated into the **NVIDIA Omniverse Blueprint for Real-Time Digital Twins**, which combines CUDA-X accelerated solvers, PhysicsNeMo, and Omniverse/OpenUSD rendering; NVIDIA cites roughly **1,200× faster** simulation-plus-visualization workflows for CFD digital twins in reference demonstrations. NVIDIA markets a broader aerospace stack spanning external aerodynamics, thermal management, propulsion, structures, and system-level optimization.
- **Luminary Cloud** (San Mateo, CA): a cloud-native "Physics AI Platform" combining GPU-accelerated CFD with trained Physics AI models. Announced at AIAA SciTech 2026 (per Aerospace America) new defense-focused models — **SHIFT-CCA** (Collaborative Combat Aircraft), **SHIFT-Submarine**, **SHIFT-Pump** — built via its "Physics AI Factory," an end-to-end platform for training/deploying Physics AI inside secure, U.S.-controlled environments for DoD use.
- **PhysicsX** (London): an AI-native engineering platform unifying simulation, physics-AI models, data, and applications; raised a **$300M Series C** (led by Temasek) at a reported **$2.4B valuation**, explicitly targeting compression of engineering design cycles "from weeks to seconds." Works across aerospace, automotive, and energy engineering.
- **Neural Concept** (Switzerland/France): deep-learning-based generative/predictive design platform for engineering geometries; raised a **$100M Series C** in 2025, applied to aerodynamic and thermal design in aerospace and automotive.
- **Ansys SimAI** (now part of **Synopsys**, following the Ansys–Synopsys merger): a cloud AI platform that trains ML models on prior 3D Ansys simulation results to predict physical fields (pressure, stress, temperature) on *new* geometries in minutes; explicitly cited aerospace applications include electromagnetic signature analysis for antennas and structural design of jet-engine brackets. As of the 2026 portfolio, includes **SimAI Pro** (local/workstation GPU training) and **SimAI Premium** (cloud-scale).
- **Cadence** (Design Systems): announced a **~€2.7B acquisition of Hexagon's Design & Engineering business** (including **MSC Software** — Nastran, Adams) in September 2025, explicitly framed as expanding "Physical AI and System Design and Analysis," bringing Nastran-class structural solvers together with Cadence's Reality/CFD and AI simulation stack; deal expected to close ~Q1 2026. Cited customers spanning aerospace/defense include Lockheed Martin, BAE Systems, and Boeing.
- **Altair Engineering**: acquired by **Siemens** for **$10B** (closed March 2025), integrating Altair's simulation, data science, and AI capabilities (including its HyperWorks suite and AI-driven design exploration tools) into the Siemens Xcelerator platform — consolidating a major structural/multiphysics simulation vendor with AI tooling under a large industrial software umbrella.
- **Monolith AI**: applies machine learning to physical test data (not only simulation) for engineering design, positioned as complementary to CAE-trained AI platforms; used across automotive and aerospace test/validation workflows (specific aerospace case studies less publicly documented than automotive ones — flagged here as a gap for the repo to fill with primary sourcing).
- **Rescale**: cloud HPC orchestration platform (ScaleX) rather than an ML-surrogate vendor per se, but a critical enabler — notably powering **Boom Supersonic's** almost entirely simulation-driven aircraft development process (200M+ cell unstructured Navier–Stokes CFD runs for supersonic vortex-lift flight regimes) and an earlier collaboration on Airbus's **Project Vahana** eVTOL program.
- **Boom Supersonic**: while not itself an "AI company," Boom's public case studies (via Rescale) illustrate the *demand side* — extremely compute-intensive CFD across the flight envelope for supersonic aircraft design, the kind of workload ML surrogates are increasingly targeted at accelerating.
- **SpaceX / Blue Origin**: internal simulation and ML-surrogate practices are not publicly documented in detail (proprietary); this reference explicitly avoids speculation here and flags it as an area where only conference talks or hiring-page signals (not verifiable technical claims) are publicly available as of this writing.
- Broader landscape: market maps (e.g., Axomap's "AI-Native Simulation Landscape") show a fast-growing set of players spanning **Ansys/Synopsys SimAI, Cadence, PhysicsX, Neural Concept, Luminary Cloud, NVIDIA PhysicsNeMo, Monolith AI**, and others — signaling that "Physics AI" is consolidating into (a) large CAE incumbents bolting on AI (Ansys, Cadence, Siemens/Altair) and (b) venture-backed "AI-native" simulation platforms (PhysicsX, Luminary, Neural Concept).

---

## 4. Numerical Methods & Solvers

ML surrogates in aerospace are trained against, validated against, or hybridized with classical numerical solvers. Familiarity with these tools is prerequisite context for any physics-AI-for-aerospace reference:

- **OpenFOAM**: open-source finite-volume CFD toolbox; the most common source of open training data for community ML-CFD projects (e.g., DeepMind's `cylinder_flow` dataset for MeshGraphNets was generated with COMSOL, but OpenFOAM is the dominant open-source data generator for RANS/LES aerospace datasets such as AirfRANS).
- **SU2**: open-source (Stanford Aerospace Design Lab-originated, now community-maintained) unstructured finite-volume RANS/Euler solver with a *built-in discrete adjoint* for PDE-constrained shape optimization — a natural target for ML-hybridization research (learned closures, learned adjoints, surrogate-in-the-loop optimization) and used in turbomachinery adjoint design (e.g., multistage turbomachinery design papers in *Journal of Propulsion and Power*).
- **CFD++, STAR-CCM+ (Siemens)**: commercial RANS/LES/DES solvers widely used in industry as the ground-truth data source for commercial AI-surrogate products (e.g., Ansys SimAI trains on prior Ansys Fluent/CFX simulation results analogously).
- **Cadence Fidelity CFD (formerly Pointwise mesh generation + CFD++ lineage)**: Cadence's CFD/meshing stack, now organizationally adjacent to Nastran/Adams following the Hexagon D&E acquisition — positioning Cadence to offer a combined mesh-to-structure-to-AI pipeline.
- **Nek5000 / NekRS**: high-order spectral-element solvers (Argonne National Laboratory lineage) used for DNS/LES-fidelity turbulence data generation, increasingly used as a high-fidelity data source for training turbulence-closure or super-resolution ML models given their spectral accuracy.
- **High-order Discontinuous Galerkin (DG) solvers**: used in research contexts requiring high-order accuracy for LES/DNS-quality training data, and increasingly explored with differentiable-programming wrappers (e.g., fast automated adjoints for spectral PDE solvers).
- **Finite element structural solvers — Nastran, Abaqus**: the structural-analysis backbone of aerospace certification workflows; MSC Nastran (now under Cadence via the Hexagon acquisition) and Abaqus (Dassault Systèmes/SIMULIA) remain the dominant certified structural solvers that AI surrogates (e.g., Ansys SimAI's jet-engine-bracket example) are trained to approximate for rapid design-iteration loops, with the certified solver retained as the final verification step.
- **Panel methods**: low-fidelity potential-flow methods (e.g., vortex-lattice/panel codes used in early-stage conceptual design, such as within OpenVSP/SUAVE workflows) remain relevant as fast, physically-grounded baselines and as one "fidelity level" in multi-fidelity ML training pipelines.
- **RANS / LES / DNS context**: Reynolds-Averaged Navier–Stokes (RANS) remains the industrial workhorse (cheapest, requires turbulence closure modeling — a classic ML-augmentation target for closure discovery); Large-Eddy Simulation (LES) and Direct Numerical Simulation (DNS) provide progressively higher-fidelity but far more expensive training data, typically reserved for canonical/benchmark flows rather than full-aircraft configurations.
- **Adjoint methods**: the standard technique for gradient-based aerodynamic shape optimization (implemented in SU2, and used throughout industrial design optimization); a major point of hybridization interest, since neural operators that are themselves differentiable can, in principle, provide gradients directly without a separate adjoint solve — an active research direction rather than a solved industrial capability.

---

## 5. Geometry, CAD & Mesh Resources Specific to Aerospace

- **OpenVSP** (Open Vehicle Sketch Pad): NASA-originated, now community-maintained (github.com/OpenVSP/OpenVSP) open-source parametric aircraft geometry tool; released as open source under NOSA 1.3 in January 2012; a common geometry-generation front end for aerospace ML pipelines that need parametrically varied aircraft shapes at scale.
- **CPACS** (Common Parametric Aircraft Configuration Schema): a shared XML-based data schema (originated at DLR) enabling interoperability between aircraft design tools — used to integrate OpenVSP, CEASIOM, and VAMPzero, among others, into common multidisciplinary design workflows; a natural standard for structuring parametric-geometry ML training datasets.
- **SUAVE** (Stanford): an open-source conceptual aircraft design and analysis environment for "unconventional vehicle designs," using OpenVSP for geometry generation; relevant for generating physically consistent aircraft configurations and low/mid-fidelity performance data at the scale ML training requires.
- **Standard CFD validation/benchmark geometries**:
  - **ONERA M6 wing** — a cornerstone transonic-aerodynamics benchmark for validating shock-capture, 3D flow, and shock/boundary-layer interaction accuracy; reference geometry hosted via NASA's Turbulence Modeling Resource.
  - **NASA Common Research Model (CRM)** — a public, representative long-range twin-aisle transport configuration (fuselage + wing + horizontal tail + nacelle) used as the standard test case in the **AIAA Drag Prediction Workshop** series since 2008; increasingly used as an industrial-scale geometry benchmark for neural-operator/transformer surrogate papers (e.g., Transolver-3, and a 2025 "ONERA CRM Wall Distribution" neural-field prediction challenge).
  - **RAE2822 airfoil** — a standard 2D transonic airfoil benchmark case.
  - **NASA Rotor37 / Rotor67** — standard open turbomachinery (axial compressor/fan) benchmark geometries, now used for GNN/neural-operator turbomachinery research (e.g., GraphRotorNet).
- **Mesh generation tools**: Pointwise (now part of Cadence Fidelity), commercial meshers integrated into STAR-CCM+/Ansys Fluent workflows, and open unstructured mesh generators used to produce AirfRANS-style anisotropic boundary-layer meshes — mesh quality/topology remains a major source of the "unstructured, variable-connectivity" data challenge that motivates GNN- and point-cloud-based (rather than grid-based) ML architectures in aerospace specifically.

---

## 6. Datasets & Data Generators

- **AirfRANS** (Bonnet et al., NeurIPS 2022 Datasets & Benchmarks; IRT SystemX/Extrality): 1,000 steady-state, high-fidelity incompressible RANS simulations over 2D NACA 4- and 5-digit airfoils in the subsonic regime, with variable angle of attack and inlet velocity. Inputs include anisotropic unstructured meshes and surface-distance fields; outputs include lift/drag coefficients and full pressure/velocity/turbulent-viscosity fields. Deliberately reflects industrial mesh refinement and *data scarcity* (only ~1,000 samples) rather than idealized abundant-data settings. Library: `airfrans` (PyPI) / [github.com/Extrality/airfrans_lib](https://github.com/Extrality/airfrans_lib). Documentation: [airfrans.readthedocs.io](https://airfrans.readthedocs.io/).
- **NeurIPS ML4CFD Competition (2024)** and its predecessor "Machine Learning for Physical Simulation" challenge (IRT SystemX): built directly on AirfRANS, providing a standardized competitive leaderboard format for airfoil-design CFD surrogate models — a template worth emulating for a broader aerospace leaderboard.
- **PLAID** (2025): "A Unified Data Model for Machine Learning on Heterogeneous Physics Simulations" — a newer data-format effort aimed at standardizing how heterogeneous CFD/structural simulation datasets (mesh + field + scalar outputs) are represented for ML, directly relevant to unifying aerospace datasets like AirfRANS across tools.
- **PDEBench** (Takamoto et al., NeurIPS 2022): broad multi-PDE benchmark (compressible/incompressible Navier–Stokes, diffusion-reaction, shallow water, etc.) with pre-generated datasets and FNO/U-Net/PINN/gradient-based-inverse baselines; not aerospace-specific but widely used as a generic architecture benchmark that aerospace-focused papers often cite alongside domain-specific datasets.
- **PDEArena**: a complementary modern PDE surrogate-modeling benchmark (incompressible Navier–Stokes, shallow water, Maxwell's equations).
- **DeepMind `cylinder_flow` / MeshGraphNets datasets**: COMSOL-generated incompressible Navier–Stokes flow-over-cylinder trajectories, the standard baseline dataset for mesh-based GNN simulator research (not aerospace-specific geometry, but the canonical starting benchmark for the architecture class).
- **NASA Rotor37 / Rotor67 CFD datasets**: used in multiple 2024–2025 papers (RANS steady-state simulations via commercial 3D CFD solvers) as an open turbomachinery benchmark for graph/operator learning.
- **NASA Turbulence Modeling Resource** (Langley Research Center): hosts reference geometries (including ONERA M6) and RANS turbulence-model validation cases; a valuable, authoritative, publicly available data source for validation-grade aerospace CFD ML work.
- **AIAA Drag Prediction Workshop / Common Research Model data**: CRM-based multi-solver comparison data generated across workshop iterations since 2008 — a rich but historically fragmented source of cross-code CFD comparison data that a curated repo could help unify and index.
- **Synthetic data-generation pipelines**: several recent papers (e.g., the AirfRANS successors and neural-field surrogate papers) emphasize *parametric* geometry sampling (via OpenVSP/CPACS-driven variation) combined with automated meshing + SU2/OpenFOAM batch runs as the emerging standard pattern for generating aerospace-scale ML training sets — this pattern (parametric geometry → automated mesh → batch CFD → dataset) is itself a reusable pipeline blueprint the repo should document and template.

---

## 7. Industry Standards & Validation/Certification Context

Aerospace is a safety-critical, heavily regulated industry, and this is the single largest structural difference between "Physics AI" here versus in other engineering domains — a fast, accurate surrogate is necessary but not sufficient; it must fit into a certification and verification framework.

- **DO-178C / ED-12C** ("Software Considerations in Airborne Systems and Equipment Certification"): the foundational standard by which the FAA, EASA, and Transport Canada approve commercial airborne software, defining process rigor by Design Assurance Level (A, most critical, through E). It is *not* written with machine-learning systems in mind, and industry/academic consensus (e.g., a 2024 Frontiers in Aerospace Engineering review, "ML meets aerospace: challenges of certifying airborne AI") is that DO-178C alone is insufficient for ML-based airborne software.
- **DO-330 / ED-215** ("Software Tool Qualification Considerations"): governs qualification of *development/verification tools* (including, potentially, ML-based design tools used to generate or verify certified artifacts); relevant to physics-AI surrogates used inside a certified design pipeline, even when the surrogate itself is a *design aid* rather than flight software.
- Industry position: a 2023 AIA (Aerospace Industries Association) white paper, *"Securing Artificial Intelligence/Machine Learning in Aviation,"* argues existing DO-178C/DO-330 guidance "may support" but is "insufficient for complete protection" against AI/ML-specific risks, calling for new dedicated processes.
- **EASA AI Roadmap**: EASA published **AI Roadmap 1.0** in February 2020, targeting approval of human-AI-teaming systems by 2025 and fully autonomous commercial air transport by 2035. EASA's first concrete ML certification deliverable was **guidance for Level 1 ML applications** (human-assistance, non-autonomous) released December 2021, revised February 2023 to cover Level 1 & 2. **EASA AI Roadmap 2.0** (2026) broadens scope to hybrid/neurosymbolic AI approaches and strengthens cybersecurity/AI-safety-assurance guidance, all organized around a "human-centric" trustworthiness framework covering trustworthiness analysis, learning assurance, dataset assurance, explainability, and safety-risk mitigation.
- **MLEAP** (Machine Learning Application Approval): an EASA-initiated, Horizon-Europe-funded research project producing methods/tools for ensuring robustness, data completeness, and generalization of ML models intended for aviation use — directly relevant to how a physics-AI surrogate's training data and generalization envelope would need to be documented for eventual certification consideration.
- **AIAA**: runs short courses and technical committee activity on ML for aerospace engineering (e.g., recurring SciTech short courses), though — distinct from EASA's binding regulatory guidance — AIAA's role here is more professional/technical-consensus-building than regulatory.
- **Practical implication for physics-AI surrogates today**: across essentially all commercial and research literature reviewed, ML/AI surrogate models (FNO, GINO, PINNs, GNNs, transformer surrogates) are positioned as **design-exploration and early-stage analysis accelerators**, not as replacements for certified analysis. The certified solver (Nastran, Abaqus, a qualified CFD code) remains the final verification step before certification submission; the surrogate's value is compressing the *design iteration* loop (thousands of candidate evaluations) before a small number of designs proceed to full certified analysis. This "surrogate-for-exploration, solver-for-certification" pattern is the de facto industry norm as of 2026, not yet formally codified in binding certification standards for physics/CFD/structural AI specifically (as opposed to flight-control or perception AI, which is the primary focus of current EASA/FAA AI guidance).

---

## 8. Open Challenges & Gaps

- **Generalization across geometries.** Most neural-operator/GNN results (including MeshGraphNets follow-up work explicitly titled "Generalization capabilities of MeshGraphNets to unseen geometries") show meaningful accuracy degradation when a model trained on one geometry family is evaluated on a substantially different, unseen geometry. GINO and Geo-FNO are steps toward geometry-agnostic operators, but robust, out-of-the-box generalization to novel full-aircraft configurations (not just interpolation within a parametric family) remains unsolved.
- **Uncertainty quantification (UQ).** Deterministic point-prediction surrogates dominate the literature, but aerospace design and certification fundamentally require calibrated uncertainty bounds (especially for hypersonic and other data-scarce, extreme-flight-regime problems). Gaussian-process/Bayesian-neural-network approaches exist (e.g., multi-fidelity Bayesian NNs for transonic aerodynamic load UQ) but are not yet standard practice alongside the higher-capacity neural-operator/transformer models, which typically lack native, trustworthy confidence estimates.
- **Multi-physics coupling.** Aerospace problems are inherently multi-physics — aero-structural (aeroelasticity), aero-thermal (hypersonics, propulsion), aero-acoustic (sonic boom, noise), and aero-propulsive (integrated propulsion-airframe) coupling. Most current ML surrogates target a single physics domain (aerodynamics *or* structures *or* thermal) in isolation; end-to-end differentiable *coupled* multi-physics ML surrogates are an active but early-stage research area.
- **Real-time control applications.** Surrogate speed-ups (reported up to tens of thousands of times faster than CFD) open the door to model-predictive-control and in-the-loop applications (active flow control, adaptive aeroelastic control, guidance), but robustness, verification, and low-latency deployment (including on flight-qualified hardware) for such real-time uses remain largely unaddressed in the literature reviewed.
- **Out-of-distribution robustness and extreme flight regimes.** Hypersonics is repeatedly flagged (e.g., UIUC/Oden Institute work on hypersonic sensing via SciML) as a domain where **training data is inherently scarce** — ground-test facilities cannot fully replicate flight conditions, and flight-test data is expensive/rare — making standard supervised ML approaches fundamentally data-starved. This motivates continued reliance on physics-informed/hybrid approaches (embedding conservation laws directly) over purely data-driven ones for these regimes, and active-learning/optimal-experimental-design strategies to make the most of scarce data.
- **Data and benchmark fragmentation.** Unlike computer vision or NLP, aerospace CFD/ML lacks a single dominant, continuously maintained, large-scale open benchmark analogous to ImageNet. AirfRANS (2D, 1,000 samples), PDEBench (generic PDEs), and scattered CRM/Rotor37/ONERA-M6-based studies each use different formats, fidelities, and metrics — a clear gap a curated open reference/benchmark hub can address (see Roadmap).
- **Mesh/geometry representation standardization.** No consistent standard yet for representing heterogeneous simulation data (mesh + fields + scalar outputs) for ML consumption, though efforts like **PLAID** (2025) are emerging; most groups still build bespoke data loaders per dataset.
- **Certification pathway for AI-based design/analysis tools.** As discussed in Section 7, current DO-178C/DO-330/EASA guidance targets flight software and human-AI teaming, not physics-simulation surrogates used purely in the design loop — leaving an open question of how (or whether) such tools will eventually need formal qualification even when the certified solver remains the final gate.
- **Industrial-scale validation transparency.** Much of the most industrially relevant work (full-aircraft-scale, proprietary-geometry surrogate validation at OEMs) is not publicly published in detail; publicly available academic benchmarks (AirfRANS, CRM-based studies) are still predominantly 2D or simplified 3D relative to full production aircraft complexity.

---

## 9. Roadmap: Building the Reference Hub

A phased plan for growing this repository into the leading open reference for Physics AI in Aerospace over roughly 1–3 years.

### Phase 0 (Months 0–2): Foundation
- Publish this document as the repo's core `README`/`docs` entry point; version it and open it to community PRs.
- Stand up a **curated, tagged bibliography** (BibTeX + a lightweight searchable index) covering the papers named in Section 1, expandable via community submission with a lightweight review/tagging process (architecture type, application area, venue, year, code-availability flag).
- Create a **glossary** (PINN, FNO, DeepONet, GNO, GINO, ROM, POD, RANS/LES/DNS, adjoint, DAL, DO-178C, etc.) — this reference document is a strong seed for it.
- Set contribution guidelines emphasizing *verifiability*: no unsourced claims, every factual assertion linked to a paper, product page, or primary source.

### Phase 1 (Months 2–6): Curated Knowledge Base
- Build a structured, filterable **paper database** (architecture × application domain × fidelity × dataset used × code availability), likely as a static site (e.g., a searchable table generated from a structured YAML/JSON source of truth) so it stays maintainable by non-engineers via PR.
- Build a **tools & solvers directory**: OpenFOAM, SU2, Nek5000/NekRS, Nastran/Abaqus, OpenVSP/CPACS/SUAVE, PhysicsNeMo, JAX-Fluids/JAX-CFD/PhiFlow — each with a one-paragraph description, links, license, and "known aerospace ML use cases."
- Build a **datasets directory**: AirfRANS, PDEBench, PDEArena, Rotor37/67-based datasets, CRM/ONERA-M6/RAE2822 reference geometry sources, with direct download links, format documentation, and known baseline results.
- Publish a first set of **tutorial notebooks**: e.g., "Train an FNO on AirfRANS," "Reproduce a MeshGraphNets cylinder-flow rollout," "Adjoint shape optimization with SU2," "PINN for RAE2822 transonic flow." These should be runnable (Colab/Binder-compatible) and version-pinned.

### Phase 2 (Months 6–14): Benchmarks & Community
- Launch an **aerospace physics-AI leaderboard**, initially wrapping existing benchmarks (AirfRANS/ML4CFD-style tasks, PDEBench aero-relevant subsets, a Rotor37-based task) with a standardized submission format and public results table — explicitly positioned as complementary to, not competing with, the NeurIPS ML4CFD competition series, ideally in direct coordination with IRT SystemX/Extrality.
- Formalize a **dataset contribution pipeline blueprint** (documented in Section 6): parametric geometry (OpenVSP/CPACS) → automated meshing → batch OpenFOAM/SU2 runs → standardized (PLAID-compatible where possible) output format — and use it to generate at least one genuinely new open 3D aerospace dataset (e.g., a parametric wing or turbomachinery family beyond what currently exists openly) to fill the "2D/simplified-3D only" gap identified in Section 8.
- Open a **community forum/discussion space** (GitHub Discussions or Discord) and a lightweight **monthly "what's new in physics AI for aerospace" digest**, aggregating new arXiv papers, product announcements, and funding events (this document's Sections 1–3 are the seed content for the first several digest issues).
- Begin outreach to academic groups identified in Section 1 (Brown/Karniadakis, Caltech/Anandkumar, Stanford ADL/CTR, DLR, IRT SystemX) and companies in Section 3 (NVIDIA PhysicsNeMo team, Luminary Cloud, PhysicsX, Neural Concept) to invite guest tutorials, dataset contributions, or advisory involvement — framed as mutually beneficial visibility, not a request for proprietary IP.

### Phase 3 (Months 14–24): Standards & Trustworthiness Layer
- Publish a dedicated, continuously updated **"Certification & Trustworthy AI for Aerospace Simulation" tracker**, summarizing DO-178C/DO-330, EASA AI Roadmap 1.0/2.0, MLEAP outputs, and any new FAA/AIAA guidance as it emerges — this is a distinctive value-add few existing ML resources provide, since it requires aerospace-regulatory literacy rather than pure ML expertise.
- Start a **"UQ and generalization" benchmark track** specifically targeting the gaps in Section 8 (out-of-distribution geometry generalization, calibrated uncertainty) rather than only in-distribution accuracy — scoring submissions on OOD test splits by construction.
- Publish reference **hybrid/differentiable-solver templates** (JAX-Fluids, JAX-CFD, PhiFlow-based) showing how to embed a learned closure/operator inside a classical solver, positioned as the practical on-ramp for "hybrid physics-ML" adoption in real aerospace design pipelines.

### Phase 4 (Months 24–36): Ecosystem Maturity
- Pursue formal **university/lab partnerships** (co-hosted tutorials at AIAA SciTech / NeurIPS ML4PS, joint dataset releases) and **industry advisory input** from commercial players (without compromising the repo's vendor-neutral, open-source positioning).
- Expand the leaderboard to a **second physics domain beyond aerodynamics** — e.g., aeroelasticity/structures (Nastran/Abaqus-adjacent) or propulsion/turbomachinery (Rotor37/67-based) — to avoid the repo being perceived as "just a CFD benchmark."
- Consider a lightweight **certification-readiness checklist** (non-binding, community-maintained) for teams wanting to understand what documentation/UQ/validation evidence a physics-AI surrogate would plausibly need to support eventual use in a certified design workflow, explicitly caveated as informational, not regulatory guidance.
- Reassess scope annually against the fast-moving commercial landscape (the Cadence/Hexagon and Siemens/Altair consolidations, and PhysicsX/Luminary funding rounds tracked in Section 3, show this space changes quickly) and keep the repo's positioning as the **neutral, open, continuously-updated reference** that no single vendor has an incentive to maintain.

---

---

## 10. Where this reference meets the PINNeAPPle codebase

*Added November 2026.* Everything above describes the field. This section is the other half: the places inside the PINNeAPPle organisation where each topic has something runnable, so a claim here can be checked against a case rather than taken on trust. Links are to live issues and finished experiments, not aspirations.

### Use cases and open work in [`PINNeAPPle`](https://github.com/PINNeAPPle-Labs/PINNeAPPle)

| Topic from this document | Where it lives |
|---|---|
| Section 2 — neural operators on geometry; Section 6 — Ahmed body data | [#241](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/241) Ahmed body: validate `pp.cfd.ExternalFlow` and train a surrogate on AhmedML |
| Section 6 — AirfRANS as the reference aero dataset | [#242](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/242) reproduce the four AirfRANS tasks with MeshGraphNet |
| Section 8 — open gap: a benchmark protocol others can rerun | [#1408](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1408) benchmark problem P12, external aerodynamics, airfoil RANS (AirfRANS) |
| Section 4 — panel and low-fidelity methods as a rung | [#492](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/492) NeuralFoil and AeroSandbox as the fast rung of the multi-fidelity ladder |
| Section 4 — rotorcraft and turbomachinery baselines | [#270](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/270) small-UAV propeller performance against BEMT and rotating-frame OpenFOAM |
| Section 4 — adjoint-driven shape optimisation | [#490](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/490) DAFoam bridge; [#296](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/296) MDO with surrogates and the airliner problem |
| Section 5 — geometry parameterisation for design spaces | [#292](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/292) FFD, RBF morphing and shape modes on STL meshes |
| Section 2 — the aircraft as a module, not a script | [#1122](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1122) aerodynamics and aircraft: viscous airfoils, compressible flow, aeroelasticity, rotors, performance, sizing, handling qualities |
| Section 8 — morphing and mission-adaptive design | [#1292](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1292) AERIS, an aircraft whose geometry changes in flight; [#1299](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1299) optimiser-found rotor blades |

### Finished experiments in [`PINNeAPPle-Benchmark`](https://github.com/PINNeAPPle-Labs/PINNeAPPle-Benchmark)

Each of these ships its own runnable code and a technical paper generated from its own results.

| Experiment | What it measures |
|---|---|
| [`08_supersonic_shock_train`](https://github.com/PINNeAPPle-Labs/PINNeAPPle-Benchmark/tree/main/experiments/08_supersonic_shock_train) | A supersonic shock train in a duct with a fixed-pressure outlet — the compressible regime Section 2 names as a PINN weakness. |
| [`09_equal_budget_pinn_fno_deeponet`](https://github.com/PINNeAPPle-Labs/PINNeAPPle-Benchmark/tree/main/experiments/09_equal_budget_pinn_fno_deeponet) | When PINNs beat data-driven surrogates and when they do not, at equal budget, on Burgers, Darcy, Navier–Stokes and **the RANS airfoil** — directly relevant to the architecture comparison in Section 2. |
| [`05_thermal_channel_twin`](https://github.com/PINNeAPPle-Labs/PINNeAPPle-Benchmark/tree/main/experiments/05_thermal_channel_twin) | Heated 2-D channel with a CFD reference, FNO/DeepONet/POD surrogates and EnKF — the conjugate-heat rung below a full thermal digital twin. |
| [`10_multifidelity_capsule`](https://github.com/PINNeAPPle-Labs/PINNeAPPle-Benchmark/tree/main/experiments/10_multifidelity_capsule) | How many expensive runs a cheap model replaces, measured on a capsule aerothermodynamics corpus (1,024 modified-Newtonian cases, 64 with Eilmer Navier–Stokes at Mach 6–15). |

### Related repositories

- [`PINNeAPPle-CFD`](https://github.com/PINNeAPPle-Labs/PINNeAPPle-CFD) — the end-to-end CFD workflow this document assumes in Section 4: problem → equations → mesh → solver → dataset → model → verification.
- [`PINNeAPPle`](https://github.com/PINNeAPPle-Labs/PINNeAPPle) — the library itself: PINNs, neural operators, solvers, geometry, validation.

If a row here points at an issue that has since closed, the code is the source of truth; the issue is cited for the reasoning, not as a substitute for it.

---

## 11. Contributing

This document is curated, and corrections are the most valuable contribution it receives.

1. Open an issue or a pull request with the specific addition, correction or removal.
2. **Cite a real, checkable source** — a paper, an official repo, a vendor page. Anything not verifiable is marked as such rather than stated as fact.
3. Keep entries factual and neutral; no marketing language for commercial products, including PINNeAPPle's own.
4. When adding a dataset, tool or solver, say what it is, who maintains it and where it lives.
5. Flag what has gone stale — funding rounds, acquisitions and renamed products move quickly in this field.

The full conventions are in [CONTRIBUTING.md](../CONTRIBUTING.md). If you are adding something that has a runnable counterpart in the PINNeAPPle organisation, add a row to Section 10 as well so the claim stays checkable.

---

## Sources

**Foundational papers**
- [Physics-informed neural networks (Raissi, Perdikaris, Karniadakis, 2019) — ADS/JCP](https://ui.adsabs.harvard.edu/abs/2019JCoPh.378..686R/abstract) · [Code](https://github.com/maziarraissi/PINNs)
- [DeepONet (Lu, Jin, Karniadakis) — arXiv:1910.03193](https://arxiv.org/abs/1910.03193) · [Nature Machine Intelligence version](https://www.nature.com/articles/s42256-021-00302-5)
- [Geometry-Informed Neural Operator (GINO) — arXiv:2309.00583](https://arxiv.org/pdf/2309.00583) · [NeurIPS 2023 page](https://neurips.cc/virtual/2023/poster/72670)
- [MeshGraphNets project page](https://sites.google.com/view/meshgraphnets) · [DeepMind research code](https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/README.md)
- [PDE-Transformer — arXiv:2505.24717](https://arxiv.org/abs/2505.24717)
- [PDEBench — arXiv:2210.07182](https://arxiv.org/abs/2210.07182)
- [AirfRANS (NeurIPS 2022) — proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/94ab7b23a345f93333eac8748a66c763-Paper-Datasets_and_Benchmarks.pdf) · [Docs](https://airfrans.readthedocs.io/) · [GitHub](https://github.com/Extrality/airfrans_lib)

**Research groups & academic context**
- [George Karniadakis — Brown University Applied Mathematics](https://appliedmath.brown.edu/people/george-em-karniadakis)
- [Anima Anandkumar — Caltech CMS](https://www.cms.caltech.edu/people/anima)
- [DLR PINNs for compressible flows](https://elib.dlr.de/222474/1/final_elib_document.pdf)
- [SU2 — Stanford Aerospace Design Lab](http://adl2.stanford.edu/papers/AIAA-2014-0243.pdf)
- [NeurIPS 2024 ML4CFD Competition](https://neurips.cc/virtual/2024/competition/84799) · [IRT SystemX challenge page](https://www.irt-systemx.fr/en/ml4cfd-neurips-2024-challenge-harnessing-machine-learning-for-computational-fluid-dynamics-in-airfoil-design/)
- [AIAA Machine Learning for Aerospace Engineering short course](https://aiaa.org/courses/machine-learning-2-day-course-at-scitech/)

**Commercial / industry**
- [NVIDIA PhysicsNeMo blog](https://developer.nvidia.com/blog/transforming-cfd-simulations-with-ml-using-nvidia-physicsnemo/)
- [NVIDIA Omniverse digital twins for fluid simulation (GitHub)](https://github.com/NVIDIA-Omniverse-blueprints/digital-twins-for-fluid-simulation)
- [Luminary Cloud — Aerospace America](https://aerospaceamerica.aiaa.org/institute/advancing-the-field-luminary-cloud-announces-new-physics-ai-models-at-aiaa-scitech-forum/) · [Luminary Cloud site](https://luminary.ai/)
- [PhysicsX Series C — TAMradar](https://www.tamradar.com/funding-rounds/physicsx-series-c-300m) · [AWS Startups profile](https://aws.amazon.com/aws-startups/learn/physicsx-accelerating-engineering-innovation-with-generative-ai/)
- [Ansys SimAI](https://ansys.synopsys.com/products/ai/simai) · [Explaining SimAI](https://www.ansys.com/blog/explaining-simai)
- [Cadence to acquire Hexagon's Design & Engineering business](https://www.cadence.com/en_US/home/company/newsroom/press-releases/pr/2025/cadence-to-acquire-hexagons-design--engineering-business.html) · [Engineering.com coverage](https://www.engineering.com/cadence-closes-on-nastran-adams-and-more-from-hexagon/)
- [Siemens completes Altair acquisition context — Verdantix](https://www.verdantix.com/client-portal/blog/hexagon-refocuses-strategy-with-sale-of-cae-software-portfolio-to-cadence)
- [Boom Supersonic + Rescale case study](https://rescale.com/resources/cloud-hpc-simulation-enables-booms-supersonic-passenger-jet-to-take-off/) · [Rescale/Vahana announcement](https://www.digitalengineering247.com/article/rescale-announces-boom-supersonic-partnership-project-vahana-collaboration/cloud-computing)
- [Axomap AI-Native Simulation Landscape](https://axomap.com/maps/ai-for-engineering-product-development-ai-native-simulatio)

**Geometry, benchmarks & datasets**
- [OpenVSP — NASA GitHub](https://github.com/nasa/OpenVSP) · [Wikipedia overview](https://en.wikipedia.org/wiki/OpenVSP)
- [SUAVE — Stanford paper](https://suave.stanford.edu/publications/suave-hi-fi.pdf)
- [ONERA M6 wing validation](https://www.researchgate.net/publication/399723142_Aerodynamic_Validation_of_the_ONERA_M6_Wing_Using_Transonic_CFD_Simulation)
- [NASA Common Research Model design optimization — arXiv:2507.10488](https://arxiv.org/pdf/2507.10488)
- [Transolver-3 (industrial-scale geometries) — arXiv:2602.04940](https://arxiv.org/pdf/2602.04940)
- [PLAID unified data model — arXiv:2505.02974](https://arxiv.org/pdf/2505.02974)
- [Rotor37 GraphRotorNet — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0360544225017256)

**Certification & standards**
- [ML meets aerospace: certifying airborne AI — Frontiers in Aerospace Engineering (2024)](https://www.frontiersin.org/journals/aerospace-engineering/articles/10.3389/fpace.2024.1475139/full)
- [AIA — Securing AI/ML in Aviation (white paper)](https://www.aia-aerospace.org/wp-content/uploads/Securing-Artificial-Intelligence-Machine-Learning-Aviation.pdf)
- [EASA AI Roadmap 2.0 (PDF)](https://sassofia.com/wp-content/uploads/2026/04/EASA-Artificial-Intelligence-Roadmap-2.0.pdf)
- [EASA MLEAP project page](https://www.easa.europa.eu/en/research-projects/machine-learning-application-approval)
- [EASA's approach to AI in aviation — ICAO DGCA info paper](https://www.icao.int/sites/default/files/APAC/Meetings/DGCA/59th%20DGCA%20Meeting-%20Cebu%20Philipines-%2014%20to%2018%20October%202024/03-%20Information%20Papers/IP-03-05-EASA-EASA-S-APPROACH-TO-ARTIFICIAL-INTELLIGENCE-IN-AVIATION.pdf)
- [Cross-regional review of AI safety regulations in commercial aviation — arXiv:2503.04767](https://arxiv.org/pdf/2503.04767)

**Open challenges**
- [MeshGraphNets generalization to unseen geometries — arXiv:2408.06101](https://arxiv.org/html/2408.06101)
- [Hypersonic aerodynamic sensing via SciML — Oden Institute](https://kiwi.oden.utexas.edu/papers/hypersonic-sensing-scientific-machine-learning-Pham-Willcox.pdf)
- [Multi-fidelity Bayesian NN for transonic aerodynamic load UQ — arXiv:2407.05684](https://arxiv.org/pdf/2407.05684)
- [Data-driven aerospace engineering: reframing the industry with ML — arXiv:2008.10740](https://arxiv.org/pdf/2008.10740)

---

*This document was compiled through web research in September 2026 and reflects the state of the field at that time; Sections 10 and 11 were added in November 2026 to tie each topic to the runnable part of the PINNeAPPle organisation. It should be treated as a living document — the roadmap in Section 9 explicitly assumes ongoing revision as the commercial and academic landscape evolves.*
