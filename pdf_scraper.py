from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
from datetime import datetime

from rich import print

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfparser import PSSyntaxError
from pdfminer.pdfparser import PSEOF
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfdocument import PDFEncryptionError
from pdfminer.pdfdocument import PDFNoValidXRef

import pikepdf
from pikepdf._core import PdfError
from pikepdf import Stream, Dictionary, Page
from pikepdf.models.metadata import encode_pdf_date, decode_pdf_date

from bs4 import UnicodeDammit

import sh
from sh import pdftitle  # pylint: disable=E0611
from sh import pdfsandwich  # pylint: disable=E0611


#########################################
# SETTINGS

# Run
RUN_ONE_BOOK = False
RUN_ONE_BOOK_TITLE = (
    "3D Game Programming for Kids - Create Interactive Worlds with JavaScript.pdf"
)
RUN_USE_MULTIPROCESSING = False

# Output
OUTPUT_STOUT_TITLES = True

# Directories
BOOKS_DIRECTORY = Path(Path.home(), "Books")
DAMAGED_DIRECTORY = Path(Path.home(), "damaged_pdfs")

# Multiprocessing
MAX_SUBPROCESSES = 8

# Document Level Metadata
DOC_ANALYSIS_JSON_OVERWRITE_WHEN_INITIALIZE = False

# Title Discovery
TITLE_MAX_ALLOWED_LENGTH = 1000
TITLE_MAX_ALLOWED_PARSE_RETRIES = 5
TITLE_LEADING_PAGES_SCAN_COUNT = 5

BOOKS = list(BOOKS_DIRECTORY.glob("*.pdf"))

ANALYSIS_METADATA_BOILERPLATE = {
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
    "SMOGgrade": None,  # Simple Measure of Gobbledygook
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
        pdf.save(pdf_path, encryption=False)


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


def initialize_analysis_json(
    pdf_path: Path, pdf_metadata: dict, overwrite: bool = False
) -> None:
    with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        with pdf.open_metadata() as meta:
            if not meta.get("pdf:Json") or overwrite:
                ANALYSIS_METADATA_BOILERPLATE.update(pdf_metadata)
                json_meta = json.dumps(ANALYSIS_METADATA_BOILERPLATE)
                meta.update({"pdf:Json": json_meta})
        pdf.save()


def get_analysis_dict(pdf_path: Path) -> dict:
    try:
        with pikepdf.open(pdf_path) as pdf:
            with pdf.open_metadata() as meta:
                analysis_json = meta.get("pdf:Json")

                return json.loads(analysis_json)

    except PdfError:
        pdf_path.rename(Path(DAMAGED_DIRECTORY, pdf_path.name))
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
    analysis_dict = get_analysis_dict(pdf_path)

    print(analysis_dict)



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
                        if (
                            decode_str(info.get("Title"))
                            and len(decode_str(info.get("Title")))
                            < TITLE_MAX_ALLOWED_LENGTH
                        ):
                            title = decode_str(info.get("Title"))

                        pdf_metadata |= {
                            "Publisher": decode_str(info.get("Publisher")),
                            "Producer": decode_str(info.get("Producer")),
                            "CreationDate": decode_str(info.get("CreationDate")),
                            "ModDate": decode_str(info.get("ModDate")),
                            "PublicationDate": decode_str(info.get("PublicationDate")),
                            "Comments": decode_str(info.get("Comments")),
                            "Edition": decode_str(info.get("Edition")),
                            "ISBN": decode_str(info.get("ISBN")),
                            "Topics": decode_str(info.get("Topics")),
                            "Author": decode_str(info.get("Author")),
                            "Title": title,
                            "Filename": pdf_path.name,
                            "PageCount": decode_str(info.get("PageCount")),
                            "PdfVersion": get_pdf_version(pdf_path),
                        }

                    normalize_xmp_types(pdf_path)
                    initialize_analysis_json(pdf_path, pdf_metadata, DOC_ANALYSIS_JSON_OVERWRITE_WHEN_INITIALIZE)

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


def initialize_page_pieceinfo(page: Page, app_name: str):
    if app_name[0] != "/":
        app_name = f"/{app_name}"

    if not app_name[1].isupper():
        app_name = f"{app_name[0]}{app_name[1].capitalize()}{app_name[2:]}"

    pieceinfo = page.get("/PieceInfo")

    if not pieceinfo:
        page.PieceInfo = Dictionary()

    page.PieceInfo[app_name] = Dictionary(LastModified=encode_pdf_date(datetime.now()))

def initialize_pages_metadata(pdf: Path, app_name: str):
    doc = pikepdf.open(pdf, allow_overwriting_input=True)

    for page in doc.pages:
        initialize_page_pieceinfo(page, app_name)

    doc.save()


def ocr_pages(pdf_path: Path, start: int, stop: int):
    with tempfile.TemporaryDirectory() as tmpdirname:
        print("ocrd")
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


def get_pdftitle_titles(pdf_path: Path, analysis_dict: dict):
    page_number = 0

    def get_original(pdf_path: Path, page_number: int):
        return (
            "pdftitle_title_original",
            pdftitle(pdf=pdf_path, title_case=True, page_number=page_number),
        )

    def get_max2(pdf_path: Path, page_number: int):
        return (
            "pdftitle_title_max2",
            pdftitle(
                pdf=pdf_path, algo="max2", title_case=True, page_number=page_number
            ),
        )

    def get_eliot(pdf_path: Path, page_number: int):
        return (
            "pdftitle_title_eliot",
            pdftitle(
                pdf=pdf_path, algo="eliot", title_case=True, page_number=page_number
            ),
        )

    def run_pdftitle_algo(func, pdf_path: Path, ocr_already_attempted: bool = False):
        page_scan_count = TITLE_LEADING_PAGES_SCAN_COUNT
        retries = TITLE_MAX_ALLOWED_PARSE_RETRIES
        attempts = 0

        for page_number in range(0, page_scan_count):
            try:
                title: tuple = func(pdf_path, page_number)

                if len(title[1]) > TITLE_MAX_ALLOWED_LENGTH:
                    raise OverflowError

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
    if OUTPUT_STOUT_TITLES:
        print("+" * 30, "\n", book.name, "\n", "+" * 30)

    scrape_pdf_metadata(book, "pdfminer")
    analysis_dict = get_analysis_dict(book)

    if not analysis_dict:
        return None

    analysis_dict = get_pdftitle_titles(book, analysis_dict)

    # initialize_pages_metadata(book, "analysis")


def run() -> None:
    if RUN_ONE_BOOK:
        book = RUN_ONE_BOOK_TITLE
        pdf_path = Path(Path.home(), "Books", book)
        scrape(pdf_path)
    else:
        if RUN_USE_MULTIPROCESSING:
            with ProcessPoolExecutor(max_workers=MAX_SUBPROCESSES) as executor:
                futures = []

                for book in BOOKS:
                    future = executor.submit(scrape, book)
                    futures.append(future)

                for future in as_completed(futures):
                    futures.remove(future)
        else:
            for book in BOOKS:
                scrape(book)


if __name__ == "__main__":
    run()
