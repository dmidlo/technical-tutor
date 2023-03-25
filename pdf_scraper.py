from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
import tempfile

from rich import print

from pdfminer.pdfparser import PDFParser, PSSyntaxError, PSEOF
from pdfminer.pdfdocument import PDFDocument, PDFEncryptionError, PDFNoValidXRef

import pikepdf
from pikepdf._core import PdfError

from bs4 import UnicodeDammit

import sh
from sh import pdftitle
from sh import pdfsandwich

from numpy import array
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt

books_directory = Path(Path.home(), "Books")
damaged_directory = Path(Path.home(), "damaged_pdfs")
books = list(books_directory.glob('*.pdf'))

title_lengths = []

analysis_metadata_boilerplate = {
        "Publisher": None,
        "Producer": None,
        "CreationDate": None,
        "ModDate": None,
        "PublicationDate": None,
        "Comments": None,
        "Edition": None,
        "ISBN": None,
        "Topics": None,
        "Author": None,
        "Title": None,
        "pdftitle_title_original": None,
        "pdftitle_title_max2": None,
        "pdftitle_title_eliot": None,
        "Filename": None,
        "PageCount": None,
        "AbsCogEase": None,
        "RelCogEase": None,
        "Read": False,
        "RetentionScore": None,
        "RetentionPressure": None,
        "FleschReadingEaseScore": None,
        "FleschKincaidReadabilityTest": None,
        "GunningFog": None,
        "SMOGgrade": None, # Simple Measure of Gobbledygook
        "LinsearWriteFormula": None,
        "ColemanLiauIndex": None,
        "AutomatedReadabilityIndex": None,
        "LasbarhetsIndex": None,
        "DaleChallFormula": None,
        "NewDaleChallFormula": None,
        "FryReadabilityGraph": None,
        "FORCASTFormula": None,
        "GolubSyntacticDensityScore": None,
        "ClozeDeletionTest": None,
        "LixReadabilityFormula": None,
        "BormuthReadabilityIndex": None,
        "PowersSumnerKearlReadabilityFormula": None,
        "RaygorEstimateGraph": None,
        "RaygorReadabilityFormula": None,
        "SPACHEReadabilityFormula": None,
        "LexileFramework": None,
        "ATOSReadabilityFormula": None,
        "CohMetrixPsycholinguisticsMeasurements": None,
    }

def decode_str(string: str) -> str:
    try:
        return UnicodeDammit(string).unicode_markup
    except:
        return None

def decrypt_pdf(pdf_path: Path):
    with pikepdf.open(pdf_path, password="", allow_overwriting_input=True) as pdf:
        pdf.save(pdf_path)

def get_pdf_version(pdf_path: Path) -> str:
    with pikepdf.open(pdf_path) as pdf:
        return pdf.pdf_version

def normalize_xmp_types(pdf_path: Path) -> None:
    with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        with pdf.open_metadata() as meta:
            null_set = {None}

            if meta.get("dc:creator") == null_set:
                meta.update({"dc:creator": [""]})

        pdf.save()

def initialize_analysis_json(pdf_path: Path, pdf_metadata: dict, overwrite: bool=False) -> None:
    with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        with pdf.open_metadata() as meta:
            if not meta.get("pdf:Json") or overwrite:
                analysis_metadata_boilerplate.update(pdf_metadata)
                json_meta = json.dumps(analysis_metadata_boilerplate)
                meta.update({"pdf:Json": json_meta})
        pdf.save()

def get_analysis_dict(pdf_path: Path) -> dict:
    try:
        with pikepdf.open(pdf_path) as pdf:
            with pdf.open_metadata() as meta:
                analysis_json = meta.get("pdf:Json")
                
                return json.loads(analysis_json)
            
    except PdfError:
        pdf_path.rename(Path(damaged_directory, pdf_path.name))
        return None

def update_analysis_json(pdf_path: Path, analysis_dict: dict):
    dict_meta = {}
    with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        with pdf.open_metadata() as meta:
            dict_meta = get_analysis_dict(pdf_path)

            if not dict_meta:
                return None

            dict_meta.update(analysis_dict)
            json_meta = json.dumps(dict_meta)

            meta.update({"pdf:Json": json_meta})
        pdf.save()
    return dict_meta if dict_meta else None

def scrape_pdf_metadata(pdf_path: Path, parser: str) -> dict:
    pdf_metadata = {}

    match parser:
        case "pdfminer":
            with pdf_path.open(mode="rb") as pdf:
                try:
                    parse = PDFParser(pdf)
                    doc = PDFDocument(parse)

                    if len(doc.info) == 1:
                        info = doc.info[0]

                        title = None
                        if decode_str(info.get('Title')) and len(decode_str(info.get('Title'))) < 1000:
                            title = decode_str(info.get('Title'))

                        pdf_metadata |= {
                            "Publisher": decode_str(info.get('Publisher')),
                            "Producer": decode_str(info.get('Producer')),
                            "CreationDate": decode_str(info.get('CreationDate')),
                            "ModDate": decode_str(info.get('ModDate')),
                            "PublicationDate": decode_str(info.get('PublicationDate')),
                            "Comments": decode_str(info.get('Comments')),
                            "Edition": decode_str(info.get('Edition')),
                            "ISBN": decode_str(info.get('ISBN')),
                            "Topics": decode_str(info.get('Topics')),
                            "Author": decode_str(info.get('Author')),
                            "Title":  title,
                            "Filename": pdf_path.name,
                            "PageCount": decode_str(info.get('PageCount')),
                            "PdfVersion": get_pdf_version(pdf_path)
                        }

                    normalize_xmp_types(pdf_path)
                    initialize_analysis_json(pdf_path, pdf_metadata, True)
                    
                except PDFEncryptionError:
                    decrypt_pdf(pdf_path)
                    scrape_pdf_metadata(pdf_path, parser)
                    return
                except PSSyntaxError:
                    decrypt_pdf(pdf_path)
                    scrape_pdf_metadata(pdf_path, parser)
                    return
                except PSEOF:
                    # repair
                    return
                except PDFNoValidXRef:
                    # repair
                    return
                

    return pdf_metadata

def get_pdftitle_titles(pdf_path: Path, analysis_dict: dict):
    page_number = 0

    def get_original(pdf_path: Path, page_number: int):
        return ("pdftitle_title_original", pdftitle(pdf=pdf_path, title_case=True, page_number=page_number))
    
    def get_max2(pdf_path: Path, page_number: int):
        return ("pdftitle_title_max2", pdftitle(pdf=pdf_path, algo="max2", title_case=True, page_number=page_number))

    def get_eliot(pdf_path: Path, page_number: int):
        return ("pdftitle_title_eliot", pdftitle(pdf=pdf_path, algo="eliot", title_case=True, page_number=page_number))
    
    def ocr_pages(pdf_path: Path, start: int, stop: int):
        with tempfile.TemporaryDirectory() as tmpdirname:

            source = pikepdf.Pdf.open(pdf_path, allow_overwriting_input=True)

            sliced_pdf = pikepdf.Pdf.new()
            temp_pdf_path = Path(tmpdirname, "temp.pdf")
            temp_ocr_path = Path(tmpdirname, "temp_ocr.pdf")
            
            pages = source.pages[start:stop]
            sliced_pdf.pages.extend(pages)
            sliced_pdf.save(temp_pdf_path)

            pdfsandwich("-lang", "eng+equ+osd", temp_pdf_path)
            temp_ocr_path.rename(temp_pdf_path)
            pdfsandwich("-lang", "eng+equ+osd", temp_pdf_path)

            ocr_pdf = pikepdf.Pdf.open(temp_ocr_path)
            for i, page in enumerate(ocr_pdf.pages):
                source.pages.append(page)
                source.pages[i].emplace(source.pages[-1])
                del source.pages[-1]

            source.save()

    def run_pdftitle_algo(func, pdf_path: Path, ocr_already_attempted: bool = False):
        page_scan_count = 5
        retries = 5
        attempts = 0

        for page_number in range(0, page_scan_count):
            try:
                title: tuple = func(pdf_path, page_number)

                if len(title[1]) > 1000:
                    raise OverflowError

                title_lengths.append(len(title[1]))
                analysis_dict.update({title[0]: title[1]})
                break
            except sh.ErrorReturnCode:
                attempts += 1
                continue
            except OverflowError:
                attempts += 1
                continue

        if attempts == retries and not ocr_already_attempted:
            ocr_pages(pdf_path, start=0, stop=page_scan_count)
            run_pdftitle_algo(func, pdf_path, True)

    run_pdftitle_algo(get_original, pdf_path)
    run_pdftitle_algo(get_max2, pdf_path)
    run_pdftitle_algo(get_eliot, pdf_path)

    update_analysis_json(pdf_path, analysis_dict)
    return analysis_dict

def scrape(book):
    # print("+" * 30, "\n", book.name, "\n", "+" * 30)

    scrape_pdf_metadata(book, "pdfminer")
    analysis_dict = get_analysis_dict(book)

    if not analysis_dict:
        return None
    
    analysis_dict = get_pdftitle_titles(book, analysis_dict)

    if analysis_dict["Title"] and len(analysis_dict["Title"]) < 1000:
        title_lengths.append(len(analysis_dict["Title"]))

def run() -> None:

    # book = "A Systematic Comparison of Various Statistical Alignment Models.pdf"
    # pdf_path = Path(Path.home(), "Books", book)

    # scrape(pdf_path)

    for book in books:
        scrape(book)

    lengths = array(title_lengths)

    sns.histplot(lengths)

    print(lengths)

    # with ProcessPoolExecutor(max_workers=8) as executor:
    #     futures = []

    #     for book in books:
    #         future = executor.submit(scrape, book)
    #         futures.append(future)

    #     for future in as_completed(futures):
    #         futures.remove(future)


if __name__ == "__main__":
    run()