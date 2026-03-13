"""
Terroir AI - Extract key tables from mmc1.pdf
Extracts Table S7 (mineral composition) and Table S12 (fruit quality)
"""
import pdfplumber
import csv
import os

pdf_path = "d:/ChitoseProject/terroir_ai/supplementary/1-s2.0-S2090123222000534-mmc1.pdf"
out_dir = "d:/ChitoseProject/terroir_ai/supplementary/extracted"
os.makedirs(out_dir, exist_ok=True)

with pdfplumber.open(pdf_path) as pdf:

    # =========================================================
    # TABLE S7 - Chemical composition (Page 18, 0-indexed: 17)
    # =========================================================
    print("=" * 60)
    print("Extracting Table S7: Chemical composition of strawberry organs")
    page_s7 = pdf.pages[17]  # page 18
    text_s7 = page_s7.extract_text()
    tables_s7 = page_s7.extract_tables()
    
    if tables_s7:
        table = tables_s7[0]
        csv_path = os.path.join(out_dir, "TableS7_chemical_composition.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in table:
                cleaned = [str(c).replace('\n', ' ').strip() if c else '' for c in row]
                writer.writerow(cleaned)
        print(f"  Saved: {csv_path}")
        print(f"  Rows: {len(table)}, Cols: {len(table[0])}")
        print("  Preview:")
        for r in table[:5]:
            print(f"    {[str(c)[:25] if c else '' for c in r]}")
    else:
        print("  WARNING: No table found on page 18, extracting raw text instead")
        txt_path = os.path.join(out_dir, "TableS7_raw_text.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text_s7)
        print(f"  Saved raw text: {txt_path}")

    # =========================================================
    # TABLE S12 - Fruit quality parameters (Page 23, 0-indexed: 22)
    # =========================================================
    print()
    print("=" * 60)
    print("Extracting Table S12: Quality parameters of strawberry fruits")
    page_s12 = pdf.pages[22]  # page 23
    text_s12 = page_s12.extract_text()
    tables_s12 = page_s12.extract_tables()
    
    if tables_s12:
        table = tables_s12[0]
        csv_path = os.path.join(out_dir, "TableS12_fruit_quality.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in table:
                cleaned = [str(c).replace('\n', ' ').strip() if c else '' for c in row]
                writer.writerow(cleaned)
        print(f"  Saved: {csv_path}")
        print(f"  Rows: {len(table)}, Cols: {len(table[0])}")
        print("  Preview:")
        for r in table[:5]:
            print(f"    {[str(c)[:25] if c else '' for c in r]}")
    else:
        print("  WARNING: No structured table found, extracting raw text")
        txt_path = os.path.join(out_dir, "TableS12_raw_text.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text_s12)
        print(f"  Saved raw text: {txt_path}")

    # =========================================================
    # TABLE S4 - Fungal pathogens OTUs (Pages 13-16)
    # =========================================================
    print()
    print("=" * 60)
    print("Extracting Table S4: Fungal pathogen OTUs")
    all_rows_s4 = []
    for pg_idx in [12, 13, 14, 15]:  # pages 13-16
        page = pdf.pages[pg_idx]
        tables = page.extract_tables()
        for table in tables:
            if table and table[0] and 'OTU' in str(table[0][0]):
                for row in table:
                    cleaned = [str(c).replace('\n', ' ').strip() if c else '' for c in row]
                    all_rows_s4.append(cleaned)
    
    if all_rows_s4:
        csv_path = os.path.join(out_dir, "TableS4_fungal_pathogens.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(all_rows_s4)
        print(f"  Saved: {csv_path} ({len(all_rows_s4)} rows)")

    # =========================================================
    # TABLE S8 - Core microbiome OTUs (Pages 19-20)
    # =========================================================
    print()
    print("=" * 60)
    print("Extracting Table S8: Core microbiome OTUs")
    all_rows_s8 = []
    for pg_idx in [18, 19]:  # pages 19-20
        page = pdf.pages[pg_idx]
        tables = page.extract_tables()
        for table in tables:
            if table and table[0]:
                for row in table:
                    cleaned = [str(c).replace('\n', ' ').strip() if c else '' for c in row]
                    all_rows_s8.append(cleaned)
    
    if all_rows_s8:
        csv_path = os.path.join(out_dir, "TableS8_core_microbiome.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(all_rows_s8)
        print(f"  Saved: {csv_path} ({len(all_rows_s8)} rows)")

    # =========================================================
    # TABLE S9 - Bacterial/fungal functions (Page 21)
    # =========================================================
    print()
    print("=" * 60)
    print("Extracting Table S9: Bacterial and fungal functions")
    page_s9 = pdf.pages[20]
    tables_s9 = page_s9.extract_tables()
    if tables_s9:
        csv_path = os.path.join(out_dir, "TableS9_microbial_functions.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in tables_s9[0]:
                cleaned = [str(c).replace('\n', ' ').strip() if c else '' for c in row]
                writer.writerow(cleaned)
        print(f"  Saved: {csv_path} ({len(tables_s9[0])} rows)")

    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE!")
    print(f"All files saved to: {out_dir}")
    print("=" * 60)
