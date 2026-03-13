# Terroir AI: Programmable Agriculture through Microbiome Mapping

Terroir AI is a proof-of-concept Graph Neural Network (GNN) pipeline designed to identify the complex, hidden relationships between soil/plant microbiomes and final crop flavor profiles. 

### Why Graph Neural Networks (GNNs)?
Traditional machine learning struggles with microbiome data because microbes don't act in isolation—they form complex, interacting communities (networks) that compete and cooperate. **GNNs are expressly built to understand these networks.** By modeling the agricultural ecosystem as a graph—where microbes are nodes and their interactions are edges—we can move beyond simple correlations to understand *how* the community as a whole influences plant biology.

### The Opportunity: Cultivating Desired Traits
With access to large-scale sequencing data and environmental metadata, this approach unlocks unprecedented opportunities in agriculture. If an AI can reliably map which specific microbial clusters drive sugar production (Brix) while suppressing fungal pathogens, we can move from passive farming to **programmable agriculture**. 

By intentionally inoculating soil with AI-identified "flavor-boosting" microbial consortia, we can predictably engineer agricultural output—achieving targeted sweetness, enhanced disease resistance, and optimized texture without relying on genetic modification or excessive chemical fertilizers. This project demonstrates that capability using strawberry cultivation as a model.

## Data Sources

The data used in this project is based on the following research:

*   **Paper:** [Influence of the soil microbiome on the strawberry flavor](https://www.sciencedirect.com/science/article/pii/S2090123222000534?via%3Dihub)
*   **NCBI BioProject (Raw Sequencing Data):** [PRJNA556362](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA556362)

## Project Structure

The project is organized into modular scripts for data acquisition, parsing, and machine learning:

```text
terroir_ai/
│
├── download_PRJNA556362.ps1      # PowerShell script to download raw SRA data
├── SraAccList_PRJNA556362.txt    # List of NCBI SRA Accessions to download
├── explore_pdf.py                # Script to scan supplementary PDF for tables
├── extract_pdf_tables.py         # Extracts key data (OTUs, functions, flavor profiles) from the PDF
│
├── supplementary/                # Contains downloaded paper supplementary files
│   └── extracted/                # Contains the clean CSVs extracted by extract_pdf_tables.py
│
├── pipeline/                     # The core Machine Learning pipeline
│   ├── data/                     # Stores the simulated OTU abundance table and metadata
│   ├── graphs/                   # Stores the generated PyTorch Geometric graph objects (.pt)
│   ├── outputs/                  # Stores trained models, analysis JSONs, and visualizations (.png)
│   │
│   ├── generate_otu_table.py     # Generates a realistic OTU abundance dataset based on the paper
│   ├── build_graphs.py           # Constructs correlation graphs (nodes=microbes, edges=interactions)
│   ├── train.py                  # Trains the Graph Attention Network (GAT) to predict flavor
│   └── visualize.py              # Generates charts (loss curves, network graphs, node importance)
```

## Prerequisites & Installation

To run this pipeline, you will need:

1.  **Python 3.x**
2.  **SRA Toolkit** (v3.3.0 or later) - Required *only* if you want to download the raw sequencing data yourself. [Install SRA Toolkit](https://github.com/ncbi/sra-tools/wiki/HowTo:-Install-SRA-Toolkit).
3.  **Required Python Packages:**
    ```bash
    pip install numpy pandas matplotlib networkx scipy pdfplumber torch torch_geometric
    ```
    *(Note: PyTorch and PyTorch Geometric installation commands vary by OS and CUDA version. Please check their official websites for the exact commands).*

## How to Reproduce

Follow these steps to run the pipeline from start to finish on your own machine. 

### Step 1: Extract Data from Literature (Optional)
If you want to re-extract the ground-truth data from the supplementary PDF:
```bash
python extract_pdf_tables.py
```
*This parses `1-s2.0-S2090123222000534-mmc1.pdf` and outputs clean CSVs into `supplementary/extracted/`.*

### Step 2: Generate the Dataset
Create the simulated OTU abundance table based on the extracted biological rules:
```bash
python pipeline/generate_otu_table.py
```
*Outputs: `pipeline/data/otu_abundance_table.csv` and `sample_metadata.csv`.*

### Step 3: Build the Microbial Graphs
Convert the tabular OTU abundance data into correlation graphs for the neural network:
```bash
python pipeline/build_graphs.py
```
*Calculates Spearman correlations. Outputs: PyTorch Geometric Data objects (`.pt`) for each cultivar into `pipeline/graphs/`.*

### Step 4: Train the Graph Neural Network
Train the model to predict the strawberry flavor profile based on the microbial network:
```bash
python pipeline/train.py
```
*Outputs: Trained model (`best_model.pt`), loss data, and analysis JSON in `pipeline/outputs/`.*

### Step 5: Visualize the Results
Generate the final charts demonstrating the AI's findings:
```bash
python pipeline/visualize.py
```
*Outputs: Several `.png` files in `pipeline/outputs/` showing the network graphs, the most important microbes for each cultivar, and a comparison of predicted vs. actual flavor.*

## Adapting for Your Own Data

To adapt this pipeline for your own crops or datasets:
1. Replace the data in `pipeline/data/otu_abundance_table.csv` with your own sequenced abundance data.
2. Update the target values (e.g., flavor traits or yield) in the `flavor_data` dictionary within `pipeline/build_graphs.py`.
3. Re-run steps 3, 4, and 5.
