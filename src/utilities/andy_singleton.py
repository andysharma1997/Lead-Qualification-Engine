from src.utilities import andy_logger
from laserembeddings import Laser

logger = andy_logger.get_logger("andy_singleton")


class Singletons:
    __instance = None
    laser_embedder = cached_lq_facets = cached_snippets = None

    @staticmethod
    def get_instance():
        """Static access method"""
        if Singletons.__instance is None:
            logger.info("Calling private constructor for embedder initialization ")
            Singletons()
        return Singletons.__instance

    def __init__(self):
        """Virtual private constructor"""
        if Singletons.__instance is not None:
            raise Exception("The singleton is already initialized you are attempting to initialize it again get lost")
        else:
            logger.info("Initializing laser embedder")
            self.laser_embedder = Laser()
            self.cached_lq_facets = {}
            self.cached_snippets = []
            Singletons.__instance = self

    def perform_laser_embedding(self, all_sentences):
        """
        This method embeds all the sentences passed using Laser embedder
        :param all_sentences:
        :return: list of sentence embeddings
        """
        if self.laser_embedder is not None:
            sentence_embeddings = self.laser_embedder.embed_sentences(all_sentences, ["en"] * len(all_sentences))
            return sentence_embeddings
        else:
            logger.info("the embedder is not set please restart the service")

    def get_cached_lq_facets(self):
        """
        :return: the dictionary of cached facets
        """
        return self.cached_lq_facets

    def get_cached_snippets(self):
        return self.cached_snippets

    def set_cached_snippet(self, vad_chunk):
        self.cached_snippets.append(vad_chunk)
