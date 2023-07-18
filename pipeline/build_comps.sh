docker login gitlab.kiss.space.unibw-hamburg.de:4567 -u diag-tcn-token -p  glpat-6UsubUx6rAC4m5UY9WtB
poetry run kfp component build ./pipeline/src --component-filepattern train_diag_tcn_vae_model_comp.py
