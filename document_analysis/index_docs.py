import lucene
from lucene import IndexWriter, StandardAnalyzer, Document, Field
import glob


INDEX_DIR = "./full_index"

# Initialize lucene and JVM
lucene.initVM()

print "lucene version is:", lucene.VERSION

# Get the analyzer
analyzer = lucene.StandardAnalyzer(lucene.Version.LUCENE_CURRENT)
#will split the text myself
#tokenizer = lucene.StandardTokenizer(lucene.Version.LUCENE_CURRENT, reader)
# Get index storage
store = lucene.SimpleFSDirectory(lucene.File(INDEX_DIR))

# Get index writer
writer = lucene.IndexWriter(store, analyzer, True, lucene.IndexWriter.MaxFieldLength.UNLIMITED)

#get list of documents
f_list = glob.glob('./OCR_text/*.txt')

for fname in f_list:
    with open(fname, 'r') as f:
        lines = f.readlines()
        full_text = ''.join([i.strip() for i in lines]).lower()
        try:
            # create a document that would we added to the index
            doc = lucene.Document()
            
            # Add fields to this document
            #could process fname here instead of in the dataframe later 
            doc.add(lucene.Field("identifier", fname, Field.Store.YES, Field.Index.ANALYZED))
            doc.add(lucene.Field("text", full_text, Field.Store.YES, Field.Index.ANALYZED))
            
            # Add the document to the index
            writer.addDocument(doc)
        except Exception, e:
            print "Failed in indexDocs:", e
writer.optimize()
writer.commit()
