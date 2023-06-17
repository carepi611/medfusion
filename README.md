# medfusion
Medfusion - Medical Denoising Diffusion Probabilistic Model 
=============

Install
=============

Create virtual environment and install packages: \
`python -m venv venv` \
`source venv/bin/activate`\
`pip install -e .`


Get Started 
=============

1 Prepare Data
-------------

* Create data files, datasets, and store medical image data of different types.


2 Train Autoencoder 
----------------
* Go to [scripts/train_latent_embedder_2d.py](scripts/train_latent_embedder_2d.py) and import your Dataset. 
* Load your dataset with eg. `SimpleDataModule` 
* Customize `VAE` to your needs 

2.1 Evaluate Autoencoder 
----------------
* Use [scripts/evaluate_latent_embedder.py](scripts/evaluate_latent_embedder.py) to evaluate the performance of the Autoencoder. 


3 Train Diffusion 
----------------
* Go to [scripts/train_diffusion.py](scripts/train_diffusion.py) and import/load your Dataset as before.
* Load your pre-trained VAE or VAEGAN with `latent_embedder_checkpoint=...` 
* Use `cond_embedder = LabelEmbedder` for conditional training, otherwise  `cond_embedder = None`  

3.1 Evaluate Diffusion 
----------------
* Go to [scripts/sample.py](scripts/sample.py) to sample a test image.
* Go to [scripts/helpers/sample_dataset.py](scripts/helpers/sample_dataset.py) to sample a more reprensative sample size.
* Use [scripts/evaluate_images.py](scripts/evaluate_images.py) to evaluate performance of sample (FID, Precision, Recall)
