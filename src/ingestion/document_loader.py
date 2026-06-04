# src/ingestion/document_loader.py

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(data_folder: str) -> list:
    """
    # data folder-la irukkura ella documents-aiyum padikkrom
    # PDF, TXT files support pannrom
    """

    all_documents = []  # ella documents store panna oru list
    data_path = Path(data_folder)

    # Folder-la irukkura ella files-aiyum loop pannu
    for file_path in data_path.rglob("*"):

        if file_path.suffix == ".pdf":
            # PDF file-aa irundha PyPDFLoader use pannu
            print(f"PDF padikkrom: {file_path.name}")
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            all_documents.extend(docs)

        elif file_path.suffix == ".txt":
            # Text file-aa irundha TextLoader use pannu
            print(f"Text file padikkrom: {file_path.name}")
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            all_documents.extend(docs)

    print(f"Motham {len(all_documents)} pages/sections padichom")
    return all_documents


def split_documents(documents: list,
                    chunk_size: int = 500,
                    chunk_overlap: int = 50) -> list:
    """
    # Document-ai chinna chinna pieces-aa cut pannrom
    # Enna na LLM-ku ore nerathula full document kudukka mudiyadhu

    # chunk_size = 500 → ovvoru piece-layum 500 words mattum
    # chunk_overlap = 50 → pieces-kku idaiyila 50 words overlap irukkum
    #                      (context pogama irukka)
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Indha characters-la paathu split pannum - paragraph, sentence, word order-la
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Documents-ai {len(chunks)} chunks-aa split pannom")
    return chunks