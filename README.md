# ProteinGym GFP Variant Effect Prediction with ESM-2 and LoRA

This project explores GFP variant-effect prediction using the **ProteinGym** benchmark and the ESM-2 protein language model. It computes zero-shot mutation scores and fine-tunes ESM-2 with a hand-implemented LoRA module to predict experimentally measured `DMS_score` values.

> This notebook focuses on a single GFP assay, `GFP_AEQVI_Sarkisyan_2016`. Results should not be interpreted as evidence that the model generalizes to all proteins or experimental assays.

## Project Overview

Notebook: `ProteinGym.ipynb`

The workflow includes:

1. Retrieve the GFP wild-type reference sequence from ProteinGym resources.
2. Download the ProteinGym substitution dataset and load `GFP_AEQVI_Sarkisyan_2016.csv`.
3. Validate mutation annotations, protein sequences, and experimental scores.
4. Compute ESM-2 masked-marginal zero-shot mutation scores.
5. Create training, validation, and test splits.
6. Add a hand-written LoRA module to ESM-2 attention projection layers and train a regression head to predict `DMS_score`.
7. Evaluate the fine-tuned model on the test set and compare it with zero-shot scores.
8. Compare ESM-2 8M and 35M models and run LoRA configuration ablations.
9. Save checkpoints, training histories, test predictions, and plots.

## Dataset

- **Source:** ProteinGym substitution benchmark.
- **Assay:** `GFP_AEQVI_Sarkisyan_2016`.
- **Key fields:**
  - `mutant`: mutation annotation, for example, `A1P:D2N`.
  - `mutated_sequence`: amino acid sequence of the variant.
  - `DMS_score`: experimentally measured variant-effect score and the supervised regression target.

The notebook checks required fields, missing scores, sequence lengths, and valid amino acid characters.

ProteinGym scores follow the definitions of the original assay. Score scales can differ across assays and should not be compared directly without appropriate processing.

## Methods

### 1. ESM-2 Zero-Shot Mutation Scoring

The notebook uses `esm2_t6_8M_UR50D`. For each position in the wild-type sequence, the residue at that position is masked, and the conditional log-probability difference between a mutant residue and the wild-type residue is calculated:

\[
S_i = \log P(a_i^{mut}\mid x_{\setminus i}) -
      \log P(a_i^{WT}\mid x_{\setminus i})
\]

For multi-site variants, the scores at individual mutated positions are summed as an additive approximation. This does not explicitly model epistasis between mutation sites and should not be interpreted as an exact prediction of combinatorial mutation effects.

### 2. Supervised Fine-Tuning with Hand-Written LoRA

The notebook implements LoRA without using PEFT. A low-rank trainable update is added to a frozen pretrained linear layer:

\[
W' = W + \frac{\alpha}{r}BA
\]

where `r` is the LoRA rank and `alpha` is the scaling parameter.

The main configuration includes:

- LoRA rank: 4 (other configurations are evaluated in ablation experiments)
- LoRA alpha: 8
- LoRA dropout: 0.05
- Optimizer: AdamW
- Loss: mean squared error (MSE)
- Default training epochs: 5
- Early stopping: validation RMSE, with patience of 2 epochs

The notebook includes ESM-2 8M and ESM-2 35M workflows. The 35M workflow uses the representation from layer 12, mean-pools residue representations while excluding BOS, EOS, and padding tokens, and uses a regression head to predict the assay score. Refer to the notebook cells for implementation details.

### 3. Evaluation Metrics

- **Spearman correlation:** Measures rank agreement between predicted and experimental scores.
- **Pearson correlation:** Measures linear correlation between predictions and experimental scores.
- **RMSE:** Measures prediction error relative to the experimental score scale.

Zero-shot scores are log-probability differences, whereas the fine-tuned model predicts regression values. Spearman correlation is therefore particularly useful for comparing ranking performance. RMSE should be interpreted with care when the score scales are not directly comparable. The notebook also generates scatter plots, metric comparisons, and per-variant prediction plots.

## Requirements

Google Colab with a GPU runtime is recommended.

Main dependencies include:

- Python
- PyTorch
- fair-esm
- NumPy
- pandas
- requests
- SciPy
- scikit-learn
- Matplotlib
- tqdm

The notebook includes a dependency-installation cell similar to:

```bash
pip install fair-esm scipy scikit-learn pandas requests
```

Other dependencies are commonly available in Colab. For local execution, install any additional packages required by the notebook imports.

## How to Run

### Google Colab

1. Upload `ProteinGym_GFP_ESM2_LoRA.ipynb` to Google Colab.
2. Select a GPU runtime under **Runtime** settings.
3. Run all cells in order, from top to bottom.
4. Allow the notebook to download the ProteinGym reference files and substitution dataset.
5. Review the evaluation metrics, plots, and saved output files.

A working internet connection is required to access the relevant GitHub and ProteinGym data-hosting resources. The first run also downloads pretrained ESM-2 weights. Runtime depends on network speed, hardware, and dataset size.

### Execution Notes

- Run the notebook from the beginning because later cells reuse datasets, DataLoaders, model objects, and evaluation variables created earlier.
- Running only selected later cells may result in missing-variable errors.
- CPU execution may be substantially slower, especially for zero-shot scoring and model training.
- Record the dataset version, random seed, hardware, and software versions to improve reproducibility.
- Use the validation set for model selection and hyperparameter tuning. Reserve the test set for final evaluation.

## Data Splits and Generalization

The notebook evaluates the model using training, validation, and test sets and includes safeguards against leakage from identical mutation sequences. This does **not** establish strict generalization to unseen mutation sites, unseen combinations of mutations, new protein families, or independent assays.

Because the current experiment uses a single GFP assay, results should be interpreted only within the context of this dataset and experimental system. Broader generalization claims require additional proteins, independent assays, or external test sets designed around mutation sites or mutation combinations.

## Output Files

The notebook writes results under `/content/` in Colab. Main outputs include:

| File or directory | Description |
|---|---|
| `GFP_ESM2_zero_shot_results.csv` | Original data with zero-shot scores |
| `GFP_ESM2_LoRA/lora_checkpoint.pt` | ESM-2 8M LoRA trainable parameters, normalization parameters, and related training metadata |
| `GFP_ESM2_LoRA/training_history.csv` | Training history for the 8M model |
| `GFP_ESM2_LoRA/test_predictions.csv` | Test-set predictions from the 8M model |
| `GFP_ESM2_35M_LoRA/lora_checkpoint_35M.pt` | ESM-2 35M LoRA checkpoint and evaluation information |
| `GFP_ESM2_35M_LoRA/training_history_35M.csv` | Training history for the 35M model |
| `GFP_ESM2_35M_LoRA/test_predictions_35M.csv` | Test-set predictions from the 35M model |

**Checkpoint note:** The checkpoint stores trainable parameters and relevant metadata; it does not contain the full pretrained ESM-2 base-model weights. To restore a model, load the matching pretrained ESM-2 model, inject LoRA using the same architecture, and then restore the saved parameters.

## Recommended Reporting

When publishing results, consider documenting:

- Number of records and sample counts in the training, validation, and test sets
- Spearman, Pearson, and RMSE for zero-shot, ESM-2 8M, and ESM-2 35M models
- LoRA target modules, rank, alpha, dropout, learning rate, and number of epochs
- Random seed, GPU model, and software versions
- Test-set prediction plots and error analysis
- Limitations of the split strategy and whether independent external validation was performed

Performance values are intentionally not pre-filled in this README. Report the metrics obtained from an actual notebook run.

## Limitations

1. The experiment covers a single GFP assay and does not establish cross-protein or cross-assay generalization.
2. Multi-site zero-shot scores are calculated by summing individual mutation scores and may miss interactions between sites.
3. Fine-tuning results depend on data splits, random seeds, training settings, and hardware.
4. Potential leakage from duplicate records, near-identical variants, or other sources should be checked carefully.
5. For a formal benchmark, fix the dataset version and split strategy in advance and report the final test-set results without using the test set for model selection.

## References

- ProteinGym GitHub: <https://github.com/OATML-Markslab/ProteinGym>
- ProteinGym data portal: <https://marks.hms.harvard.edu/proteingym/>
- ESM / fair-esm GitHub: <https://github.com/facebookresearch/esm>

## License and Acknowledgments

This project uses ProteinGym data and the ESM-2 pretrained model. Follow the licensing, citation, and usage terms of the upstream dataset and model projects when using or redistributing their data or weights.

This notebook is intended for research and educational use. It is not a validated protein-engineering or clinical decision-making tool.
