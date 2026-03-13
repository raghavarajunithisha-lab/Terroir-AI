"""
Terroir AI — Phase 2: Generate Realistic OTU Abundance Table
=============================================================
Creates a simulated OTU abundance matrix based on the paper's findings:
  - 96 samples (3 cultivars × 4 organs × 4 replicates × 2 kingdoms)
  - ~160 OTUs (39 core + 120 pathogens)
  - Key patterns from the paper embedded:
    * Pseudomonas high in Monterey (disease protection)
    * Botrytis low in Monterey (suppressed)
    * Elsanta has more Bacillus (correlates with sweetness)
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ─── Load real OTU names from extracted tables ───────────────────────
base = "d:/ChitoseProject/terroir_ai/supplementary/extracted"

core = pd.read_csv(f"{base}/TableS8_core_microbiome.csv")
# Remove duplicate header rows
core = core[core['OTU'] != 'OTU'].reset_index(drop=True)

pathogens = pd.read_csv(f"{base}/TableS4_fungal_pathogens.csv")
pathogens = pathogens[pathogens['OTU'] != 'OTU'].reset_index(drop=True)

# Combine all unique OTUs
all_otus = list(core['OTU'].unique()) + list(pathogens['OTU'].unique())
all_otus = list(dict.fromkeys(all_otus))  # remove duplicates, preserve order
print(f"Total unique OTUs: {len(all_otus)}")

# ─── Define sample metadata ─────────────────────────────────────────
cultivars = ['Darselect', 'Elsanta', 'Monterey']
organs = ['bulk_soil', 'rhizosphere', 'root', 'leaves']
kingdoms = ['bacteria', 'fungi']
replicates = [1, 2, 3, 4]

samples = []
for cv in cultivars:
    for organ in organs:
        for kingdom in kingdoms:
            for rep in replicates:
                samples.append({
                    'sample_id': f"{cv}_{kingdom}_{organ}_rep{rep}",
                    'cultivar': cv,
                    'organ': organ,
                    'kingdom': kingdom,
                    'replicate': rep
                })

meta_df = pd.DataFrame(samples)
print(f"Total samples: {len(meta_df)}")

# ─── Generate abundance values ──────────────────────────────────────
# Base abundance: random counts following negative binomial (realistic for amplicon data)
n_samples = len(meta_df)
n_otus = len(all_otus)

# Start with baseline abundances
abundance = np.random.negative_binomial(n=2, p=0.01, size=(n_samples, n_otus))

# ─── Embed known biological patterns from the paper ─────────────────

# Helper: find column index for an OTU
def otu_idx(name):
    try:
        return all_otus.index(name)
    except ValueError:
        return None

# Helper: find rows for a cultivar
def cv_rows(cultivar):
    return meta_df[meta_df['cultivar'] == cultivar].index.tolist()

# Helper: find rows for a cultivar + organ
def cv_organ_rows(cultivar, organ):
    mask = (meta_df['cultivar'] == cultivar) & (meta_df['organ'] == organ)
    return meta_df[mask].index.tolist()

# Pattern 1: Pseudomonas (Otu000003) high in Monterey, moderate in Darselect, low in Elsanta
idx = otu_idx('Otu000003')
if idx is not None:
    for r in cv_rows('Monterey'):
        abundance[r, idx] = np.random.normal(2100, 300)
    for r in cv_rows('Darselect'):
        abundance[r, idx] = np.random.normal(1500, 200)
    for r in cv_rows('Elsanta'):
        abundance[r, idx] = np.random.normal(800, 150)

# Pattern 2: Botrytis (Otu0003) high in Elsanta (susceptible), low in Monterey (resistant)
idx = otu_idx('Otu0003')
if idx is not None:
    for r in cv_rows('Elsanta'):
        abundance[r, idx] = np.random.normal(500, 100)
    for r in cv_rows('Darselect'):
        abundance[r, idx] = np.random.normal(300, 80)
    for r in cv_rows('Monterey'):
        abundance[r, idx] = np.random.normal(50, 20)

# Pattern 3: Pseudarthrobacter (Otu000001) — core bacterium, similar across all
idx = otu_idx('Otu000001')
if idx is not None:
    for r in range(n_samples):
        abundance[r, idx] = np.random.normal(1200, 200)

# Pattern 4: Higher microbial diversity in rhizosphere vs bulk soil
for i, row in meta_df.iterrows():
    if row['organ'] == 'rhizosphere':
        abundance[i, :] = (abundance[i, :] * 1.5).astype(int)
    elif row['organ'] == 'root':
        abundance[i, :] = (abundance[i, :] * 0.7).astype(int)  # fewer species inside root
    elif row['organ'] == 'leaves':
        abundance[i, :] = (abundance[i, :] * 0.4).astype(int)  # fewest on leaves

# Pattern 5: Saccharomyces (Otu0011) — higher in Elsanta (sweeter fruit, more sugar)
idx = otu_idx('Otu0011')
if idx is not None:
    for r in cv_rows('Elsanta'):
        abundance[r, idx] = np.random.normal(400, 80)
    for r in cv_rows('Darselect'):
        abundance[r, idx] = np.random.normal(200, 50)
    for r in cv_rows('Monterey'):
        abundance[r, idx] = np.random.normal(150, 40)

# Pattern 6: Plectosphaerella (Otu0007) — pathogen, more in Darselect
idx = otu_idx('Otu0007')
if idx is not None:
    for r in cv_rows('Darselect'):
        abundance[r, idx] = np.random.normal(600, 100)
    for r in cv_rows('Elsanta'):
        abundance[r, idx] = np.random.normal(400, 80)
    for r in cv_rows('Monterey'):
        abundance[r, idx] = np.random.normal(200, 60)

# Ensure no negative values
abundance = np.clip(abundance, 0, None).astype(int)

# ─── Build the final DataFrame ──────────────────────────────────────
otu_df = pd.DataFrame(abundance, columns=all_otus)
otu_df.insert(0, 'sample_id', meta_df['sample_id'])
otu_df.insert(1, 'cultivar', meta_df['cultivar'])
otu_df.insert(2, 'organ', meta_df['organ'])
otu_df.insert(3, 'kingdom', meta_df['kingdom'])
otu_df.insert(4, 'replicate', meta_df['replicate'])

# ─── Save ────────────────────────────────────────────────────────────
out_dir = "d:/ChitoseProject/terroir_ai/pipeline/data"
os.makedirs(out_dir, exist_ok=True)

otu_path = f"{out_dir}/otu_abundance_table.csv"
meta_path = f"{out_dir}/sample_metadata.csv"

otu_df.to_csv(otu_path, index=False)
meta_df.to_csv(meta_path, index=False)

print(f"\nSaved OTU abundance table: {otu_path}")
print(f"  Shape: {otu_df.shape} (samples × OTUs)")
print(f"Saved sample metadata: {meta_path}")
print(f"\nPreview of abundances (first 5 samples, first 5 OTUs):")
print(otu_df.iloc[:5, 5:10].to_string())
print(f"\nCultivar sample counts:")
print(meta_df['cultivar'].value_counts().to_string())
