import lucene, glob

from java.io import StringReader
from java.nio.file import Path, Paths

from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import \
        Document, Field, StoredField, StringField, TextField, FieldType
from org.apache.lucene.index import \
        IndexOptions, IndexWriter, IndexWriterConfig, DirectoryReader, \
            MultiFields, Term
from org.apache.lucene.store import MMapDirectory, SimpleFSDirectory



def getWriter(store, analyzer=None, create=False):
    if analyzer is None:
        analyzer = WhitespaceAnalyzer()
    analyzer = LimitTokenCountAnalyzer(analyzer, 10000000)
    config = IndexWriterConfig(analyzer)
    if create:
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
    print(store, config)
    writer = IndexWriter(store, config)
    return writer

def getStore(INDEX_DIR):
    return SimpleFSDirectory(Paths.get(INDEX_DIR))

def getAnalyzer():
    return StandardAnalyzer()

def getDoclist(DOCUMENTS_DIR):
    return glob.glob(DOCUMENTS_DIR+'*.txt')


def main():
    INDEX_DIR = "full_index"
    DOCUMENTS_DIR = "/media/joseph/Windows8_OS/Users/Joseph/AppData/Local/lxss/home/jwymbs23/data_science_projects/french_pamphlets/frc-data-master/OCR_text/"
    # Initialize lucene and JVM
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print("lucene version is:", lucene.VERSION, '\n')
    
    store = getStore(INDEX_DIR)

    analyzer = getAnalyzer()
    
    writer = getWriter(store = store, analyzer=analyzer, create = True)
    
    #get list of documents
    doc_list = getDoclist(DOCUMENTS_DIR)


    ftype = FieldType()
    ftype.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
    ftype.setTokenized(True)
    ftype.setStoreTermVectors(True)
    ftype.freeze()
    
    for cd, doc_name in enumerate(doc_list):
        if not cd%1000:
            print(cd, '--', len(doc_list))
        with open(doc_name, 'r') as d:
            doc_lines = d.readlines()
            full_text = ''.join([i.strip() for i in doc_lines]).lower()
            try:
                # create a document that would we added to the index
                doc = Document()
                
                # Add fields to this document
                #could process fname here instead of in the dataframe later 
                doc.add(Field("identifier", doc_name.split('/')[-1], TextField.TYPE_STORED))#Store.YES))#, Field.Index.ANALYZED))
                doc.add(Field("text", full_text, ftype))#TextField.TYPE_STORED, TermVector.YES, ))#Store.YES))#, Field.Index.ANALYZED))
                
                # Add the document to the index
                writer.addDocument(doc)
            except:
                print("Failed in indexDocs: ", doc_name)
    #writer.optimize()
    writer.commit()


if __name__ == "__main__":
    main()
