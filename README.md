# Generative Rule Learning Dynamics

Official project repository for **The two clocks and the innovation window: When and how generative models learn rules**.

We study when generative models learn rules, memorize training examples, and generate novel valid samples. The experiments compare diffusion and autoregressive models on controlled discrete-rule tasks, with extensions toward more naturalistic rendered images and language.

<p align="center">
  <a href="assets/landscape-dynamics.gif"><img src="assets/landscape-dynamics.gif" alt="Evolution of the diffusion score and denoiser fields across training, on a plane containing training, valid-novel, and invalid examples at four noise scales." width="100%"></a>
</p>

**How does the denoising landscape change as a model learns?** Training dynamics on a two-dimensional slice for group parity ($G=3$, $D=36$, $N=4096$). Columns show noise scales $\sigma=0.2,0.5,1,2$; rows show score magnitude and a projected denoiser component, with arrows indicating the corresponding fields. Cyan, green, and red stars mark training, valid-novel, and invalid examples. [View full size and provenance](assets/README.md).

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
