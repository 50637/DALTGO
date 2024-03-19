DALTGO (A novel deep architecture combining BiLSTM and Transformer for protein function prediction)  
To enhance accurate annotation of protein functions, we developed a protein function prediction model called DALTGO. DALTGO first uses a pre-trained protein language model to characterize the input amino acid sequence and introduces an attention mechanism and a bidirectional LSTM model to obtain functional residues and global pattern in the sequence. In addition, DALTGO uses graph sampling and aggregation information transfer methods to extract GO term features in the basic ontology network and uses convolutional networks to extract potential connections between amino acid residues and GO terms to achieve accurate predictions of protein functions.   
[1] The folders in the DALTGO package:
data: This folder contains two benchmark datasets, including CAFA3 and ATGO2022.  
encode:This folder contains the program that encodes the protein.  
[2] Scripts:
train.py - This script is the main program.  
metric.py - This script is used to evaulate the model.  
hparam.py - This script is used to initialize parameters.      
get_GOfeatures.py - This script is used to generate the features of GOterms.  
[3] Datasets:
Each dataset contains two types of files:  
(1)Files contain the sequence of the protein and the corresponding annotations.   
(2)A file that contains the features of all GO term in current dataset.  
[3] Running:     
--Command line:  
run train.py script  
--Model output: will generate a file called results.csv  
[4] Installation:    
git clone https://github.com/50637/DALTGO.git    
