class VadChunk(object):
    """this class holds the incoming snippet text and stores the encoding of each snippet"""

    def __init__(self, sid, from_time, to_time, speaker, text, confidence, questions=None, q_encoding=None,
                 encoding_method=None):
        self.sid = sid
        self.from_time = from_time
        self.to_time = to_time
        self.speaker = speaker
        self.text = text
        self.questions = questions
        self.confidence = confidence
        self.q_encoding = q_encoding
        self.encoding_method = encoding_method

    def set_sid(self, sid):
        self.sid = sid

    def set_questions(self, questions):
        self.questions = questions

    def set_question_encoding(self, encoding, encoding_method):
        self.q_encoding = encoding
        self.encoding_method = encoding_method


class FacetSignal(object):
    """This class stores the embedding of all the facet_signal """

    def __init__(self, fsid, text, embedding=None, embedding_method=None):
        self.fsid = fsid
        self.text = text
        self.embedding = embedding
        self.embedding_method = embedding_method

    def set_embedding(self, embedding, embedding_method):
        self.embedding = embedding
        self.embedding_method = embedding_method

    


class Facet(object):
    """ This class represents the facets that are defined for lead qualification, the facets have names(
    eg:authority,budget etc) , the list of facet_signal object and the id of facet signals caught for the particular
    facet type """

    def __init__(self, fid, name, facet_signals):
        self.fid = fid
        self.name = name
        self.facet_signals = facet_signals

    def set_id(self, fid, name):
        self.fid = fid
        self.name = name

    def set_facet_signals(self, facet_signal):
        self.facet_signals = facet_signal


class CaughtFacetSignals(object):
    def __init__(self, snippet, snippet_text, snippet_question, facet_name, facet_signal, facet_signal_text, score,
                 method):
        self.snippet = snippet
        self.snippet_text = snippet_text
        self.snippet_question = snippet_question
        self.facet_name = facet_name
        self.facet_signal = facet_signal
        self.facet_signal_text = facet_signal_text
        self.score = score
        self.method = method
