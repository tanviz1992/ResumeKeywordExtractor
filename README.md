## **Resume Keyword Extractor**

The ld_resume.py script extracts the top 5 keywords from a folder containing resumes.
The script accepts a wide range of input arguments, so as the user can update and train the model to get the desired results.

### The repository contains the below files :

* **README.md** - Readme markdown file
* **ld_resume.py** - Python File to extract keywords from a folder containing resumes
* **ld_resume_nb.ipynb** - Notebook document file with the ld_resume.py python script loaded and sample run with output
* **ResumeKeywordExtractor.pdf** - Project Report containing description, Quickstart, Steps and 
* **result.json** - Sample JSON output being created by the program

### **Usage**

Usage: python ld_resume.py [ARGS] 
**Required Arguments:**
   **--dir </dir/to/folder/>**             Path to folder containing resumes 
   **--result-file <output json file>**    Path to Output Json File 
              
**Optional Arguments:**
   **--min-df <min_df value>**             Min DF value for the CountVectorizer - default : 0.2
   **--max-df <max_df value>**             Max DF value for the CountVectorizer - default : 0.8
   **--stop-words <stop words>**           Custom Stop Words to include if any  - default : [] - No default stop words
   **--num-common-words-ignore <Number>**  Number of common words to ignore     - default : 20
   **--num-keywords <Number>**             Number of keywords to extract to JSON. default = 5 

   **--help-print**                        Prints the usage section of the script
            
**Example Usage:** 
python ld_resume.py --dir /path/to/folder_containing_resumes/ --result-file /path/output/result.json

