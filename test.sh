source activate pytorch

python seq_wc.py --input_file corpus/BC2GM-IOBES/test.tsv --scope GENE
python seq_wc.py --input_file corpus/BC4GMEMD-IOBES/test.tsv --scope Chemical
python seq_wc.py --input_file corpus/BC5CDR-chem-IOBES/test.tsv --scope Chemical
python seq_wc.py --input_file corpus/BC5CDR-disease-IOBES/test.tsv --scope Disease
python seq_wc.py --input_file corpus/BC5CDR-IOBES/test.tsv --scope Chemical Disease
python seq_wc.py --input_file corpus/NCBI-disease-IOBES/test.tsv --scope Disease
python seq_wc.py --input_file corpus/JNLPBA-IOBES/test.tsv --scope protein, DNA, cell_line, cell_type, RNA
