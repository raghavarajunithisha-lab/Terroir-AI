"""
Terroir AI - PDF Supplementary Table Explorer v2
Writes results to a text file for easy reading
"""
import pdfplumber

pdf_path = "d:/ChitoseProject/terroir_ai/supplementary/1-s2.0-S2090123222000534-mmc1.pdf"
out_path = "d:/ChitoseProject/terroir_ai/supplementary/pdf_scan_results.txt"

with open(out_path, 'w', encoding='utf-8') as out:
    with pdfplumber.open(pdf_path) as pdf:
        out.write(f"Total pages: {len(pdf.pages)}\n")
        out.write("=" * 70 + "\n")
        
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            tables = page.extract_tables()
            
            preview = text[:300].replace('\n', ' | ') if text else "(no text)"
            out.write(f"\n--- Page {i+1} ---\n")
            out.write(f"  Text preview: {preview}\n")
            out.write(f"  Tables found: {len(tables)}\n")
            
            for t_idx, table in enumerate(tables):
                if table and len(table) > 0:
                    ncols = len(table[0]) if table[0] else 0
                    out.write(f"  Table {t_idx+1}: {len(table)} rows x {ncols} cols\n")
                    if table[0]:
                        headers = [str(c)[:35] if c else '' for c in table[0]]
                        out.write(f"    Headers: {headers}\n")
                    if len(table) > 1 and table[1]:
                        row1 = [str(c)[:35] if c else '' for c in table[1]]
                        out.write(f"    Row 1:   {row1}\n")

print(f"Results written to {out_path}")
