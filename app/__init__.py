import os
import sys
import types

os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

try:
    import langchain.docstore.document
except ImportError:
    class FakeDocument:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    docstore_mod = types.ModuleType("langchain.docstore")
    document_mod = types.ModuleType("langchain.docstore.document")
    document_mod.Document = FakeDocument
    docstore_mod.document = document_mod

    sys.modules["langchain.docstore"] = docstore_mod
    sys.modules["langchain.docstore.document"] = document_mod

try:
    import langchain.text_splitter
except ImportError:
    class FakeTextSplitter:
        def __init__(self, *args, **kwargs):
            pass
        def split_documents(self, docs):
            return docs

    text_splitter_mod = types.ModuleType("langchain.text_splitter")
    text_splitter_mod.RecursiveCharacterTextSplitter = FakeTextSplitter

    sys.modules["langchain.text_splitter"] = text_splitter_mod
