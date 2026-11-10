# Generative Rule Learning Dynamics

Official project repository for **The two clocks and the innovation window: When and how generative models learn rules**.

We study when generative models learn rules, memorize training examples, and generate novel valid samples. The experiments compare diffusion and autoregressive models on controlled discrete-rule tasks, with extensions toward more naturalistic rendered images and language.

## Research themes

- **Two learning clocks:** the onset of rule learning and the onset of memorization.
- **The innovation window:** the period during which models generate valid examples beyond the training set.
- **Rule complexity and representation:** how interaction order, model architecture, and image tokenization affect learning.
- **Mechanisms:** analyses of diffusion denoiser landscapes across training and noise scales.

## Tasks

- Discrete rules: group parity, exact-K, Latin squares, and row-K constraints.
- Rendered images: `Block`, `Orb`, and `Flower` tasks.
- Language: the `Light-switch` task with four levels of textual dressing.

## Code access

This repository currently provides the project overview. The existing research implementation, including dataset generators, training scripts, and analysis tools, is available in [DiffusionAttnConsistency](https://github.com/Animadversio/DiffusionAttnConsistency).

See its [README](https://github.com/Animadversio/DiffusionAttnConsistency#readme) for the code structure and main entry points. A complete reproduction environment and exact commands for every reported result are not yet provided.
