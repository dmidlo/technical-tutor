# nlp-technology_text_human_readabilty_index

## Abstract

Given a piece of text written for technologists, how difficult will it be for the reader to comprehend?

the goal is to give a "Cognitive Ease" score to a given text based it's "absolute" difficulty or complexity.

"Absolute" cognitive ease would be defined as the generic score given to a text that gives the text and places the ease score against against traditional readability and complexity score algorithms and 

## Method

Ensemble method using a diverse collection of 6617 technical publicationsBooks, articles, papers, posts as training set.

Initial training data is stored in pdf documents.


## Problems

### Metadata Extraction

pdf as a very flexible format, and as such, it's also can be a beast to parse.  At the document level, the first problem here is simply the Title (which, in the end probably won't be a highly weighted feature). PDF metadata is not required by the specification, so many pdfs do not have Title info.

### Taxonomies

#### Code Block or Not
`{"code", "not code"}`

#### Programming Language
`{"Python", "C++", "Bash", ...}`

#### Needs OCR
`{"Needs OCR", "Does Not Need OCR", "Already OCRed"}`

#### Page Types

```Python
{"Cover Page", "Title Page", "Publisher Page", "Blank Page", "Preface Page", "Appendix Page", "Chapter Start Page",
 "Chapter End Page", "Chapter Transition Page", "Biography Page", "Advertisement Page", "Forward Page", "ToC Page" ...}
```

#### Text Block

```Python
{"Document", "Topic", "Sub-Topic", "Chapter", "Section", "Sub-Section", "Article", "Figure Description", "Table", 
 "Paragraph", "Sub-Paragraph", "Sentence", "Phrase", "Word"}
```

## Bibliography

### References

- https://prolingo.com/blog/what-readability-algorithm-scores-fail-to-tell-you/
- http://cs231n.stanford.edu/reports/2015/pdfs/kggriswo_FinalReport.pdf
- https://www.diva-portal.org/smash/get/diva2:721646/FULLTEXT01.pdf
- https://en.wikipedia.org/wiki/Readability
- https://readabilityformulas.com/search/pages/Readability_Formulas/
- https://en.wikipedia.org/wiki/HOCR
- https://www.deeplearning.ai/the-batch/how-metas-llama-nlp-model-leaked/
- https://machinelearningmastery.com/best-practices-document-classification-deep-learning/
- https://ieeexplore.ieee.org/document/8125990
- https://maelfabien.github.io/machinelearning/NLP_5/
- https://dylancastillo.co/text-classification-using-python-and-scikit-learn/
- https://www.atmosera.com/blog/text-classification-with-neural-networks/
- https://www.educative.io/answers/text-classification-in-nlp
- https://txt.cohere.ai/10-must-read-text-classification-papers/
- https://few-shot-text-classification.fastforwardlabs.com/
- https://stackabuse.com/text-classification-with-python-and-scikit-learn/
- https://www.mdpi.com/2078-2489/10/4/150
- http://www.scholarpedia.org/article/Text_categorization
- https://keras.io/examples/nlp/text_classification_from_scratch/
- https://www.kaggle.com/code/matleonard/text-classification
- https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
- https://docs.uipath.com/ai-center/automation-cloud/latest/user-guide/text-classification
- https://nlp.stanford.edu/IR-book/html/htmledition/text-classification-and-naive-bayes-1.html
- https://autokeras.com/tutorial/text_classification/
- https://medium.com/text-classification-algorithms/text-classification-algorithms-a-survey-a215b7ab7e2d
- https://neptune.ai/blog/text-classification-tips-and-tricks-kaggle-competitions
- https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
- https://developer.apple.com/documentation/naturallanguage/creating_a_text_classifier_model
- https://www.sciencedirect.com/topics/computer-science/text-classification
- https://realpython.com/python-keras-text-classification/
- https://paperswithcode.com/task/text-classification
- https://www.datacamp.com/tutorial/text-classification-python
- https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
- https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
- http://cs231n.stanford.edu/reports/2015/pdfs/kggriswo_FinalReport.pdf
- https://aclanthology.org/N04-1042.pdf
- https://www.microsoft.com/en-us/research/publication/automatic-extraction-of-titles-from-general-documents-using-machine-learning/
- https://docear.org/papers/SciPlore%20Xtract%20--%20Extracting%20Titles%20from%20Scientific%20PDF%20Documents%20by%20Analyzing%20Style%20Information%20(Font%20Size)-preprint.pdf
- https://clgiles.ist.psu.edu/papers/JCDL-2003-automata-metdata.pdf
- https://ieeexplore.ieee.org/document/1204842
- https://www.researchgate.net/publication/262171677_Docear's_PDF_inspector_Title_extraction_from_PDF_files

### Github Topics

- pdf-files
- text-classification

### Google Searches

- grade level reading algroithms
- exploring pdf files in python
- scrape title of pdf
- text classification
- readability algorithms
- weebit corpora
- Newsela corpus
- text classification semi-supervised learning
- extract title from pdf
- extract title from pdf python
- python modify pdf metadata
- ocr scanned multi-page pdf python
- python tesseract sandwhich
- pdf renderer sandwich
- pathlib name without extension
- stanford alpaca 7b download
- normal distribution python
- text classification semi-supervised learning
- partial training machine learning
- extract title from pdf python

### StackExchange Titles

- using pytesseract to generate a PDF from image
- Python with pytesseract - How to get the same output for pytesseract.image_to_data in a searchable PDF?
- How do I get the filename without the extension from a path in Python?
- How to extract the title of a PDF document from within a script for renaming?
- Extracting titles from PDF files?
- Extract titles from each page of a PDF?
- Extracting the actual in-text title from a PDF

### Tools

- (pypdfium2)[https://github.com/pypdfium2-team/pypdfium2]
- (pdfplumber)[https://github.com/jsvine/pdfplumber]
- (pdftotext)[https://github.com/jalan/pdftotext]
- (pikepdf)[https://github.com/pikepdf/pikepdf]
- (tabula-py)[https://github.com/chezou/tabula-py]
- (fpdf2)[https://github.com/PyFPDF/fpdf2]
- (pdfminer.six)[https://github.com/pdfminer/pdfminer.six]
- (pypdf)[https://github.com/py-pdf/pypdf]
- (pdftitle)[https://pypi.org/project/pdftitle/]

- (gTTS)[https://github.com/pndurette/gTTS]
- (robot)[https://github.com/robotframework/robotframework]
- (robotframework-pdf2textlibrary)[https://github.com/qahive/robotframework-pdf2textlibrary]
- (pathlib)[https://docs.python.org/3/library/pathlib.html]
- (python sh)[https://github.com/amoffat/sh]
- (python re)[https://docs.python.org/3/library/re.html]
- (pillow)[https://github.com/python-pillow/Pillow]
- (python io)[https://docs.python.org/3/library/io.html]
- (pytesseract)[https://github.com/madmaze/pytesseract]
- (wand)[https://github.com/emcconville/wand]
- (doctr)[https://github.com/mindee/doctr]
- (hocr-tools)[https://github.com/ocropus/hocr-tools]
- (pypdfocr)[https://pypi.org/project/pypdfocr/]
- (ocrmypdf)[https://ocrmypdf.readthedocs.io/en/latest/index.html]
- (unpaper)[https://github.com/Flameeyes/unpaper]
- (pdfsandwich)[http://www.tobias-elze.de/pdfsandwich/]
- (imagemagik -convert)[http://www.imagemagick.org/script/convert.php]

- (pdfviewer)[https://github.com/naiveHobo/pdfviewer]
- (xpdf)[https://github.com/ecatkins/xpdf_python]
- (pdfquery)[https://github.com/jcushman/pdfquery]
- (reportlab)[https://docs.reportlab.com/install/open_source_installation/]
- (pdfrw)[https://github.com/pmaupin/pdfrw]
- (slate)[https://github.com/timClicks/slate]
- (pdflib)[https://github.com/alephdata/pdflib]

- (paperwork)[https://gitlab.gnome.org/World/OpenPaperwork/paperwork]
- (ocrfeeder)[https://gitlab.gnome.org/GNOME/ocrfeeder]
- (teedy docs)[https://github.com/sismics/docs]
- (Papermerge)[https://github.com/ciur/papermerge]
- (Mayan EDMS)[https://gitlab.com/mayan-edms/mayan-edms]
- (paperless ngx)[https://github.com/paperless-ngx/paperless-ngx]
- (docspell)[https://github.com/eikek/docspell]

- https://nlp.gsu.edu/
- (nltk)[https://www.nltk.org/]
- (spaCy)[https://spacy.io/]
- (SciPy)[https://scipy.org/]