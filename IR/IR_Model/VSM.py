import time

from gensim import corpora, models, matutils

from IR.IR_Model.Model import Model


class VSM(Model):
    def __init__(self, fo_lang_code=None):
        super().__init__(fo_lang_code)
        self.name = "VSM"
        self.tfidf_model = None

    def build_model(self, docs):
        start = time.time()
        print("Building VSM model...")
        docs_tokens = []
        for doc_text in docs:
            docs_tokens.append(doc_text.split())
        print("Building dictionary...")
        dictionary = corpora.Dictionary(docs_tokens)
        print("Building corpus...")
        corpus = [dictionary.doc2bow(token) for token in docs_tokens]
        print("Building model...")
        self.tfidf_model = models.TfidfModel(corpus, id2word=dictionary)
        print("Finish building VSM model in {:.4f}s.".format(time.time() - start))

    def get_doc_similarity(self, doc1_tokens, doc2_tokens):
        doc1_vec = self.tfidf_model[self.tfidf_model.id2word.doc2bow(doc1_tokens)]
        doc2_vec = self.tfidf_model[self.tfidf_model.id2word.doc2bow(doc2_tokens)]
        return matutils.cossim(doc1_vec, doc2_vec)
